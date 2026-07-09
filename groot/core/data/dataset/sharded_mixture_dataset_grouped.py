# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Grouped mixture dataset with per-step rejection sampling.

Subclass of ``ShardedMixtureDataset`` that adds priority-weighted rejection
sampling.  Used exclusively by the V2 quality-weighted factory when
``speedup=True``.  Base class behavior is completely unchanged.
"""

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from groot.core.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from groot.core.data.types import BaseProcessor, ShardedDataset


class ShardedMixtureDatasetGrouped(ShardedMixtureDataset):
    """Mixture dataset with rejection sampling for priority-weighted iteration.

    Extends the base class by:
    - Accepting ``per_dataset_group_weight_maps`` and ``per_dataset_group_histograms``
    - Adjusting schedule weights by mean acceptance rate
    - Applying per-step rejection sampling in ``_iter_single_shard``
    - ``[V2_SAMPLING]`` logging via ``V2IterLoggingMixin``
    """

    def __init__(
        self,
        datasets: list[ShardedDataset],
        weights: list[float],
        processor: BaseProcessor,
        seed: int = 42,
        training: bool = True,
        num_shards_per_epoch: int = int(1e5),
        override_pretraining_statistics: bool = False,
        group_ids: list[int] | None = None,
        per_dataset_group_weight_maps: list[dict[int, float]] | None = None,
        per_dataset_group_histograms: list[dict[int, int]] | None = None,
    ):
        # Compute rejection sampling params BEFORE super().__init__() because
        # it calls generate_shard_sampling_schedule() which needs these.
        if per_dataset_group_weight_maps is not None:
            self.per_dataset_group_weight_maps = per_dataset_group_weight_maps
        else:
            self.per_dataset_group_weight_maps = [None] * len(datasets)

        self._max_group_weights = [
            max(m.values()) if m is not None else 1.0
            for m in self.per_dataset_group_weight_maps
        ]

        self._mean_acceptance_rates = []
        for i, wmap in enumerate(self.per_dataset_group_weight_maps):
            if wmap is None:
                self._mean_acceptance_rates.append(1.0)
            else:
                hist = per_dataset_group_histograms[i] if per_dataset_group_histograms else None
                if hist is not None:
                    max_w = self._max_group_weights[i]
                    total = sum(hist.values())
                    if total > 0 and max_w > 0:
                        rate = sum(
                            (wmap.get(g, 0.0) / max_w) * count
                            for g, count in hist.items()
                        ) / total
                    else:
                        rate = 1.0
                    self._mean_acceptance_rates.append(rate)
                else:
                    self._mean_acceptance_rates.append(1.0)

        # Log rejection sampling config
        for i, wmap in enumerate(self.per_dataset_group_weight_maps):
            if wmap is not None:
                ds_path = getattr(datasets[i], "dataset_path", f"idx={i}")
                print(
                    f"[V2_SAMPLING] REJECTION_INIT dataset={ds_path} "
                    f"weight_map={wmap} "
                    f"max_w={self._max_group_weights[i]:.2f} "
                    f"mean_accept={self._mean_acceptance_rates[i]:.4f}"
                )

        super().__init__(
            datasets=datasets,
            weights=weights,
            processor=processor,
            seed=seed,
            training=training,
            num_shards_per_epoch=num_shards_per_epoch,
            override_pretraining_statistics=override_pretraining_statistics,
        )
        self.dataset_id_to_path = {
            i: str(getattr(ds, "dataset_path", f"dataset_{i}"))
            for i, ds in enumerate(self.datasets)
        }
        if group_ids is not None:
            self.dataset_id_to_group_id = dict(enumerate(group_ids))
        else:
            self.dataset_id_to_group_id = dict.fromkeys(range(len(self.datasets)), 0)

    def __iter__(self):
        yield from self._iter_single_shard()

    @staticmethod
    def _attach_dataset_metadata(sample, dataset_index: int, group_index: int) -> None:
        """Attach dataset provenance when the processor output is mutable dict data."""
        if isinstance(sample, tuple):
            for s in sample:
                if isinstance(s, dict):
                    s["dataset_id"] = np.array([dataset_index], dtype=np.int64)
                    s["group_id"] = np.array([group_index], dtype=np.int64)
        elif isinstance(sample, dict):
            sample["dataset_id"] = np.array([dataset_index], dtype=np.int64)
            sample["group_id"] = np.array([group_index], dtype=np.int64)

    def _accept_sample(
        self,
        dataset_index: int,
        sample,
        rng: np.random.Generator,
    ) -> tuple[bool, int | None]:
        """Apply this dataset's priority rejection rule to one sample."""
        weight_map = self.per_dataset_group_weight_maps[dataset_index]
        if weight_map is None:
            return True, None

        raw = sample[0] if isinstance(sample, tuple) else sample
        if "step_group" not in raw:
            return True, None

        group = int(raw["step_group"])
        max_w = self._max_group_weights[dataset_index]
        accept_prob = weight_map.get(group, 0.0) / max_w if max_w > 0 else 0.0
        return rng.random() <= accept_prob, group

    def generate_shard_sampling_schedule(self) -> list[tuple[int, int]]:
        """Override to adjust weights by mean acceptance rate."""
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch)

            average_shard_sizes = []
            for dataset in self.datasets:
                average_shard_size = sum(
                    dataset.get_shard_length(i) for i in range(len(dataset))
                ) / len(dataset)
                average_shard_sizes.append(average_shard_size)

            # Adjust weights by mean acceptance rate when rejection sampling is active.
            #
            # Goal: expected yield fraction from dataset i = raw_w[i].
            # Yield proportional to P[i] * avg_shard_size * accept_rate[i]
            #      proportional to effective_w[i] * accept_rate[i]   (after /avg_shard_size cancels)
            # For yield proportional to raw_w[i]:  effective_w[i] = raw_w[i] / accept_rate[i]
            effective_weights = [
                w / self._mean_acceptance_rates[i] if self._mean_acceptance_rates[i] > 0 else 0.0
                for i, w in enumerate(self.weights)
            ]
            for i, w in enumerate(self.weights):
                if self._mean_acceptance_rates[i] < 1.0:
                    ds_path = getattr(self.datasets[i], "dataset_path", f"idx={i}")
                    print(
                        f"[V2_SAMPLING] EFFECTIVE_WEIGHT dataset={ds_path} "
                        f"raw_w={w:.6f} "
                        f"accept_rate={self._mean_acceptance_rates[i]:.4f} "
                        f"eff_w={effective_weights[i]:.6f}"
                    )

            normalized_weights = np.array(
                [w / s for w, s in zip(effective_weights, average_shard_sizes)]
            )
            normalized_weights = normalized_weights / normalized_weights.sum()

            dataset_sampling_schedule = rng.choice(
                len(self.datasets), size=self.num_shards_per_epoch, p=normalized_weights
            )

            shard_sampling_schedule = []
            shards_to_sample = []
            for dataset in self.datasets:
                shard_ids = list(range(len(dataset)))
                rng.shuffle(shard_ids)
                shards_to_sample.append(shard_ids)

            for i in dataset_sampling_schedule:
                if len(shards_to_sample[i]) == 0:
                    shard_ids = list(range(len(self.datasets[i])))
                    rng.shuffle(shard_ids)
                    shards_to_sample[i] = shard_ids
                shard_idx = shards_to_sample[i].pop(0)
                shard_sampling_schedule.append((i, shard_idx))
        else:
            shard_sampling_schedule = []
            for i, dataset in enumerate(self.datasets):
                shard_sampling_schedule.extend([(i, j) for j in range(len(dataset))])
        return shard_sampling_schedule

    def _iter_single_shard(self):
        """Override to add rejection sampling + V2 logging."""
        self._executor = ThreadPoolExecutor(max_workers=1)

        self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed + self.epoch)

        # Cumulative counters for [V2_SAMPLING] logging
        _cumulative_yielded_per_ds: dict[int, int] = {}
        _cumulative_yielded_per_group: dict[int, int] = {}
        _shards_processed = 0
        _total_yielded = 0

        worker_profiler = None
        if self._enable_worker_profiling and self._worker_profile_dir is not None:
            from pathlib import Path

            import torch.profiler

            worker_id = self.worker_id if self.worker_id is not None else 0
            trace_dir = Path(self._worker_profile_dir) / f"rank_{self.rank}_worker_{worker_id}"
            trace_dir.mkdir(parents=True, exist_ok=True)
            worker_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(
                    skip_first=self._worker_profile_skip_first,
                    wait=0,
                    warmup=1,
                    active=self._worker_profile_active_steps,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
                record_shapes=False,
                with_stack=False,
            )
            worker_profiler.start()

        samples_yielded = 0

        while True:
            self.curr_shard_index += 1

            wait_start = time.time()
            self.finish_cache_shard()
            wait_end = time.time()

            dataset_index, shard_index = self.worker_shard_sampling_schedule[self.curr_shard_index]
            dataset_path = getattr(
                self.datasets[dataset_index], "dataset_path", f"idx={dataset_index}"
            )
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: "
                f"Wait for shard {shard_index} in dataset {dataset_path} "
                f"in {wait_end - wait_start:.2f} seconds, shard_len={len(self.curr_shard)}"
            )

            self.cache_next_shard()

            assert self.curr_shard is not None
            indices_in_shard = np.arange(len(self.curr_shard))
            rng.shuffle(indices_in_shard)
            group_index = self.dataset_id_to_group_id.get(dataset_index, 0)
            accepted_count = 0
            rejected_count = 0
            per_group_accepted: dict[int, int] = {}
            per_group_rejected: dict[int, int] = {}

            for index in indices_in_shard:
                sample = self.curr_shard[index]
                accepted, step_group = self._accept_sample(dataset_index, sample, rng)
                if not accepted:
                    rejected_count += 1
                    if step_group is not None:
                        per_group_rejected[step_group] = per_group_rejected.get(step_group, 0) + 1
                    continue
                accepted_count += 1
                if step_group is not None:
                    per_group_accepted[step_group] = per_group_accepted.get(step_group, 0) + 1
                self._attach_dataset_metadata(sample, dataset_index, group_index)
                yield sample
                samples_yielded += 1
                if worker_profiler is not None:
                    worker_profiler.step()
                    if samples_yielded >= (
                        self._worker_profile_skip_first + 1 + self._worker_profile_active_steps
                    ):
                        worker_profiler.stop()
                        worker_profiler = None

            # [V2_SAMPLING] per-shard logging
            print(
                f"[V2_SAMPLING] SHARD dataset={dataset_path} shard={shard_index} "
                f"shard_len={len(self.curr_shard)} yielded={accepted_count} rejected={rejected_count}"
            )
            if per_group_accepted:
                print(
                    f"[V2_SAMPLING] SHARD_GROUPS "
                    f"accepted_per_group={dict(sorted(per_group_accepted.items()))} "
                    f"rejected_per_group={dict(sorted(per_group_rejected.items()))}"
                )
                for g, cnt in per_group_accepted.items():
                    _cumulative_yielded_per_group[g] = _cumulative_yielded_per_group.get(g, 0) + cnt

            _cumulative_yielded_per_ds[dataset_index] = (
                _cumulative_yielded_per_ds.get(dataset_index, 0) + accepted_count
            )
            _total_yielded += accepted_count
            _shards_processed += 1

            if _shards_processed % 3 == 0:
                total = max(_total_yielded, 1)
                ds_ratios = {k: f"{v / total:.4f}" for k, v in sorted(_cumulative_yielded_per_ds.items())}
                g_total = max(sum(_cumulative_yielded_per_group.values()), 1) if _cumulative_yielded_per_group else 1
                group_ratios = {k: f"{v / g_total:.4f}" for k, v in sorted(_cumulative_yielded_per_group.items())}
                print(
                    f"[V2_SAMPLING] CUMULATIVE shards={_shards_processed} "
                    f"total_yielded={_total_yielded} "
                    f"ds_yield_ratio={ds_ratios} "
                    f"group_yield_ratio={group_ratios}"
                )

            self.delete_cached_shard()

    def shutdown(self, wait: bool = True) -> None:
        """Stop grouped background shard prefetching."""
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None
        self._cache_job = None
        self.curr_shard = None
