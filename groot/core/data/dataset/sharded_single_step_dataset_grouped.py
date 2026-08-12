# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Grouped single-step dataset with per-step priority labels.

Subclass of ``ShardedSingleStepDataset`` that adds ``step_group_labels`` support.
Used exclusively by the V2 quality-weighted factory when ``speedup=True``.
Base class behavior is completely unchanged.
"""

import heapq
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from structlog.contextvars import bound_contextvars

from groot.core.utils.distributed import worker_logging
from groot.core.data.dataset import sharded_single_step_dataset
from groot.core.data.hydrator import client_protocol
from groot.core.data import types


def _compute_valid_anchor_indices(
    episode_len: int,
    min_delta_index: int,
    max_delta_index: int,
    allow_padding: bool,
) -> np.ndarray:
    if episode_len <= 0:
        return np.empty(0, dtype=np.int64)
    if allow_padding:
        return np.arange(episode_len, dtype=np.int64)

    start = max(0, -min_delta_index)
    stop = episode_len - max_delta_index
    if stop <= start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


class ShardedSingleStepDatasetGrouped(sharded_single_step_dataset.ShardedSingleStepDataset):
    """Single-step dataset with per-step group labels for rejection sampling.

    Extends the base class by:
    - Accepting ``step_group_labels`` (parallel to ``step_indices_filter``)
    - Tracking labels through shard assignment (``shard_dataset``)
    - Tagging each datapoint with ``dp["step_group"]`` in ``get_shard``
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: types.EmbodimentTag,
        modality_configs: dict[str, types.ModalityConfig],
        hydrator: client_protocol.HydratorClientProtocol | None = None,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        resample_on_invalid_frames: bool = False,
        shard_size: int = 2**10,
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
        use_lance_backend: bool = False,
        lance_base_path: str | Path | None = None,
        data_filter: types.OnlineDataFilter | None = None,
        step_indices_filter: dict[int, np.ndarray] | None = None,
        step_group_labels: dict[int, np.ndarray] | None = None,
    ):
        # Store labels BEFORE super().__init__ because it calls shard_dataset()
        self.step_group_labels = step_group_labels
        super().__init__(
            dataset_path=dataset_path,
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
            hydrator=hydrator,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            resample_on_invalid_frames=resample_on_invalid_frames,
            shard_size=shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=seed,
            allow_padding=allow_padding,
            data_filter=data_filter,
            step_indices_filter=step_indices_filter,
        )

    def get_valid_step_indices(self, episode_index: int) -> np.ndarray:
        original_length = self.episode_loader.get_episode_length(episode_index)
        min_delta_index = min(min(c.delta_indices) for c in self.modality_configs.values())
        max_delta_index = max(max(c.delta_indices) for c in self.modality_configs.values())
        return _compute_valid_anchor_indices(
            episode_len=original_length,
            min_delta_index=min_delta_index,
            max_delta_index=max_delta_index,
            allow_padding=self.allow_padding,
        )

    def _get_episode_step_count(self, episode_index: int) -> int:
        step_indices, _ = self._get_filtered_valid_step_indices_and_labels(episode_index)
        return len(step_indices)

    def get_datapoint(
        self,
        episode_data: types.EpisodeData,
        step_index: int,
        step_data_id: types.VLAStepDataId | None = None,
    ) -> dict:
        datapoint_t0 = time.perf_counter()
        vla_step_data = self.get_step_data(episode_data, step_index, step_data_id=step_data_id)
        datapoint = self._get_datapoint_from_step_data(vla_step_data)
        structlog.get_logger().debug(
            "get_datapoint_completed",
            num_frames=vla_step_data.num_frames,
            duration_ms=round((time.perf_counter() - datapoint_t0) * 1000, 2),
        )
        return datapoint

    def _get_filtered_valid_step_indices_and_labels(
        self,
        episode_index: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Get filtered valid steps and preserve parallel group labels.

        ``step_indices_filter`` remains the sampler-provided candidate set; any
        anchors outside the configured delta range are dropped with their labels.
        """
        if self.step_indices_filter is None:
            return self.get_valid_step_indices(episode_index), None
        if episode_index not in self.step_indices_filter:
            return np.empty(0, dtype=np.int64), None

        requested = np.asarray(self.step_indices_filter[episode_index], dtype=np.int64)
        valid_steps = self.get_valid_step_indices(episode_index)
        keep_mask = np.isin(requested, valid_steps)
        step_indices = requested[keep_mask]

        labels = None
        if self.step_group_labels is not None:
            assert episode_index in self.step_group_labels, (
                f"Missing step_group_labels for episode {episode_index}"
            )
            episode_labels = np.asarray(self.step_group_labels[episode_index])
            assert len(episode_labels) == len(requested), (
                f"step_group_labels length ({len(episode_labels)}) must match "
                f"step_indices_filter length ({len(requested)}) for episode {episode_index}"
            )
            labels = episode_labels[keep_mask]

        return step_indices, labels

    def shard_dataset(self):
        """Override to track per-step group labels in parallel with step indices."""
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        num_splits = int(1 / self.episode_sampling_rate)

        assert (
            len(shuffled_episode_indices) > 0
        ), f"No valid trajectories found for dataset {self.dataset_path}"

        effective_lengths = {
            idx: self._get_episode_step_count(idx) for idx in shuffled_episode_indices
        }
        valid_episode_indices = np.array(
            [idx for idx in shuffled_episode_indices if effective_lengths[idx] > 0]
        )
        n_skipped = len(shuffled_episode_indices) - len(valid_episode_indices)
        if n_skipped > 0:
            print(
                f"Warning: Skipping {n_skipped} episodes with zero effective length "
                f"in dataset {self.dataset_path}"
            )
        shuffled_episode_indices = valid_episode_indices

        assert (
            len(shuffled_episode_indices) > 0
        ), f"No valid episodes with positive effective length for dataset {self.dataset_path}"

        total_steps = np.sum([effective_lengths[idx] for idx in shuffled_episode_indices]).astype(
            int
        )
        num_shards = np.ceil(total_steps / self.shard_size).astype(int)

        max_shards = len(shuffled_episode_indices) * num_splits
        if num_shards > max_shards:
            print(
                f"Warning: Reducing num_shards from {num_shards} to {max_shards} "
                f"(dataset has only {len(shuffled_episode_indices)} episodes x "
                f"{num_splits} splits = {max_shards} sub-sequences). "
                f"To avoid this, try increasing `shard_size` or decreasing `episode_sampling_rate`."
            )
            num_shards = max_shards

        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)
        shard_heap = [(0, i) for i in range(num_shards)]
        heapq.heapify(shard_heap)

        has_labels = self.step_group_labels is not None
        if has_labels:
            sharded_group_labels = [[] for _ in range(num_shards)]

        for ep_idx in shuffled_episode_indices:
            step_indices, labels = self._get_filtered_valid_step_indices_and_labels(ep_idx)
            if len(step_indices) == 0:
                continue

            # Use permutation-based shuffle to keep labels parallel
            if labels is not None:
                shuffle_order = self.rng.permutation(len(step_indices))
                step_indices = step_indices[shuffle_order]
                labels = labels[shuffle_order]
            else:
                self.rng.shuffle(step_indices)

            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                current_length, shard_index = heapq.heappop(shard_heap)
                sharded_episodes[shard_index].append((ep_idx, split_step_indices))
                if has_labels and labels is not None:
                    sharded_group_labels[shard_index].append(labels[i::num_splits])
                new_length = current_length + len(split_step_indices)
                shard_lengths[shard_index] = new_length
                heapq.heappush(shard_heap, (new_length, shard_index))

        assert all(
            shard_lengths[i] > 0 for i in range(num_shards)
        ), "All shards must have length greater than 0"

        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, "
            f"shard length std: {np.std(shard_lengths)}"
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths
        self.sharded_group_labels = sharded_group_labels if has_labels else None

    def get_shard(self, idx: int) -> list:
        """Override to attach step_group labels to each datapoint."""
        with torch.profiler.record_function(
            f"get_shard [{getattr(self, 'dataset_path', 'unknown')}]"
        ):
            with bound_contextvars(
                component="get_shard",
                dataset_path=str(self.dataset_path),
                shard_idx=int(idx),
                dataset_idx=int(self.dataset_index),
            ):
                start_t = time.perf_counter()
                episodes = self.sharded_episodes[idx]
                ep_indices = [ep_idx for (ep_idx, _) in episodes]
                worker_log = worker_logging.get_unified_logger()
                worker_log.info(
                    "[GET_SHARD] shard_idx=%s start, n_episodes=%s", idx, len(ep_indices)
                )
                self.episode_loader.register_video_files_for_episodes(ep_indices)
                worker_log.info("[GET_SHARD] shard_idx=%s register_video_files done", idx)

                pq_before = getattr(self.episode_loader, "_timing_parquet_cumulative", 0.0)
                vid_before = getattr(self.episode_loader, "_timing_video_cumulative", 0.0)

                shard_labels = (
                    self.sharded_group_labels[idx]
                    if self.sharded_group_labels is not None
                    else None
                )

                datapoints = []
                for sample_idx, (ep_idx, step_indices) in enumerate(episodes):
                    with bound_contextvars(
                        sample_idx=sample_idx,
                        episode_index=int(ep_idx),
                        episode_length=int(self.episode_loader.get_episode_length(ep_idx)),
                    ):
                        sample_t0 = time.perf_counter()
                        try:
                            episode_data = self.episode_loader[ep_idx]
                        except RuntimeError as e:
                            raise RuntimeError(
                                f"Failed to load episode {ep_idx} from dataset {self.dataset_path}: {e}"
                            ) from e
                        labels_for_ep = shard_labels[sample_idx] if shard_labels is not None else None
                        for j, step_index in enumerate(step_indices):
                            step_index = int(step_index)
                            with bound_contextvars(step_index=step_index):
                                step_data_id = types.VLAStepDataId(
                                    dataset_idx=self.dataset_index,
                                    episode_idx=ep_idx,
                                    step_idx=step_index,
                                )
                                dp = self.get_datapoint(
                                    episode_data, step_index, step_data_id=step_data_id
                                )
                                if labels_for_ep is not None:
                                    dp["step_group"] = int(labels_for_ep[j])
                                datapoints.append(dp)
                        structlog.get_logger().debug(
                            "get_sample_completed",
                            step_indices=[int(step_index) for step_index in step_indices],
                            duration_ms=round((time.perf_counter() - sample_t0) * 1000, 2),
                        )

                pq_delta = (
                    getattr(self.episode_loader, "_timing_parquet_cumulative", 0.0) - pq_before
                )
                vid_delta = (
                    getattr(self.episode_loader, "_timing_video_cumulative", 0.0) - vid_before
                )
                n = max(len(datapoints), 1)

                dataset_path = getattr(self, "dataset_path", "unknown")
                print(
                    f"[SHARD_TIMING] dataset={dataset_path} shard={idx} "
                    f"episodes={len(episodes)} samples={len(datapoints)} "
                    f"parquet_s={pq_delta:.2f} video_s={vid_delta:.2f}"
                )

                if shard_labels is not None:
                    group_counts = Counter(dp.get("step_group") for dp in datapoints)
                    print(
                        f"[GROUPED_SHARD] dataset={dataset_path} shard={idx} "
                        f"group_dist={dict(sorted(group_counts.items()))}"
                    )

                per_sample_pq = pq_delta / n
                per_sample_vid = vid_delta / n
                for dp in datapoints:
                    if isinstance(dp, dict):
                        dp["_timing_parquet_s"] = per_sample_pq
                        dp["_timing_video_s"] = per_sample_vid

                self.episode_loader.release_video_files_for_episodes(ep_indices)
                structlog.get_logger().info(
                    "get_shard_completed",
                    shard_length=int(self.shard_lengths[idx]),
                    duration_ms=round((time.perf_counter() - start_t) * 1000, 2),
                )

        return datapoints
