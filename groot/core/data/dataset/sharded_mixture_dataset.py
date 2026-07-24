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

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import contextvars
import copy
import os
import time

from loguru import logger
import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from groot.core.data.types import BaseProcessor, ShardedDataset


def merge_statistics(
    per_dataset_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
    is_relative_stats: bool = False,
    action_horizon: int | None = None,
    dataset_paths: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Compute overall statistics from per-dataset statistics using weighted averaging.

    This function combines statistics from multiple datasets according to their sampling
    weights, computing weighted means and variances while preserving min/max/quantile
    information across all datasets.

    The weighted variance computation uses the formula:
    Var_combined = Σ(w_i * (σ_i² + μ_i²)) - (Σ(w_i * μ_i))²

    Args:
        per_dataset_stats: List of per-dataset statistics dictionaries, one per dataset.
            Each element has structure: {joint_group: {stat_type: values}}
            Example: {"gripper": {"mean": [0.1, 0.2], "std": [0.5, 0.3]}}
            Note: The caller splits by modality before calling this function, so each
            dict contains joint groups for a single modality (not modality-keyed).
        dataset_sampling_weights: Weights for combining dataset statistics.
            Should sum to 1.0 or will be normalized.
        is_relative_stats: Whether the statistics are relative (affects merging logic).
        action_horizon: Required action horizon length. When truncating relative stats
            to a common length, the result must be >= this value or training will fail.
        dataset_paths: Optional dataset path names for error messages.

    Returns:
        Combined statistics dictionary with same structure as input elements:
        {joint_group: {stat_type: values}} containing weighted averages for
        mean/std and global min/max/quantiles across datasets.
    """
    # Normalize sampling weights to sum to 1
    dataset_sampling_weights = np.array(dataset_sampling_weights)
    normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    # Initialize overall statistics dict
    overall_stats: dict[str, dict[str, list[float]]] = {}

    # For relative stats, datasets may have different action horizons (e.g., 50 vs 48).
    # Truncate all stats to the minimum common first dimension so merging works,
    # but only if the truncated length is still >= the training action_horizon.
    if is_relative_stats:
        per_dataset_stats = copy.deepcopy(per_dataset_stats)
        for joint_group in per_dataset_stats[0]:
            shapes = [np.array(ds[joint_group]["mean"]).shape[0] for ds in per_dataset_stats]
            min_len = min(shapes)
            if action_horizon is not None and min_len < action_horizon:
                too_short = [
                    f"  - {dataset_paths[i] if dataset_paths else f'dataset {i}'}: "
                    f"stats length {shapes[i]}"
                    for i in range(len(shapes))
                    if shapes[i] < action_horizon
                ]
                raise ValueError(
                    f"Relative stats for '{joint_group}' are shorter than the "
                    f"action_horizon ({action_horizon}):\n"
                    + "\n".join(too_short)
                    + "\nRegenerate stats with a sufficient action horizon."
                )
            if min_len < max(shapes):
                for ds in per_dataset_stats:
                    for stat_key in ds[joint_group]:
                        arr = np.array(ds[joint_group][stat_key])
                        if arr.ndim >= 1 and arr.shape[0] > min_len:
                            ds[joint_group][stat_key] = arr[:min_len].tolist()

    # Process each joint group (e.g., "left_arm", "right_arm", "gripper")
    for joint_group in per_dataset_stats[0]:
        # Get dimensionality from first dataset (assumed consistent)
        dim = (
            [len(per_dataset_stats[0][joint_group]["mean"])]
            if not is_relative_stats
            else np.array(per_dataset_stats[0][joint_group]["mean"]).shape
        )

        # Initialize accumulators for weighted mean and variance computation
        weighted_means = np.zeros(dim)
        weighted_squares = np.zeros(dim)

        # Collect min/max/quantiles from all datasets for global computation
        min_list = []
        max_list = []
        q01_list = []
        q99_list = []

        # Accumulate weighted statistics across datasets
        for dataset_idx, dataset_stats in enumerate(per_dataset_stats):
            w_i = normalized_weights[dataset_idx]
            stats = dataset_stats[joint_group]
            means = np.array(stats["mean"])
            stds = np.array(stats["std"])

            # Update weighted sums for mean and variance calculation
            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)

            # Collect extremes and quantiles for global computation
            min_list.append(stats["min"])
            max_list.append(stats["max"])
            q01_list.append(stats["q01"])
            q99_list.append(stats["q99"])

        # Compute final combined statistics
        overall_mean = weighted_means.tolist()
        overall_variance = weighted_squares - weighted_means**2
        overall_std = np.sqrt(overall_variance).tolist()

        # Global min/max across all datasets
        overall_min = np.min(np.array(min_list), axis=0).tolist()
        overall_max = np.max(np.array(max_list), axis=0).tolist()

        # Global quantiles (conservative bounds across datasets)
        q01_array = np.array(q01_list)
        q99_array = np.array(q99_list)
        weighted_q01 = np.min(q01_array, axis=0).tolist()
        weighted_q99 = np.max(q99_array, axis=0).tolist()

        # Store combined statistics for this joint group
        overall_stats[joint_group] = {
            "min": overall_min,
            "max": overall_max,
            "mean": overall_mean,
            "std": overall_std,
            "q01": weighted_q01,
            "q99": weighted_q99,
        }

    return overall_stats


class ShardedMixtureDataset(IterableDataset):
    """Iterable dataset that combines multiple sharded datasets with configurable mixing ratios.

    This dataset provides the core functionality for multi-dataset training in VLA systems:
    1. Combines multiple ShardedDataset instances with specified mixing weights
    2. Implements intelligent shard sampling that accounts for dataset sizes
    3. Provides efficient background shard caching for continuous data loading
    4. Handles distributed training across multiple workers and processes
    5. Merges dataset statistics for consistent normalization

    Key features:
    - Weighted sampling across datasets normalized by shard sizes
    - Background shard caching with ThreadPoolExecutor for efficiency
    - Distributed training support with proper shard allocation
    - Automatic epoch management and shard reshuffling
    - Per-embodiment statistics merging for cross-embodiment training

    The sampling strategy ensures that datasets are sampled proportionally to their
    weights while accounting for differences in shard sizes, preventing bias toward
    datasets with smaller shards.

    Args:
        datasets: List of ShardedDataset instances to combine
        weights: Mixing weights for each dataset (will be normalized)
        processor: Data processor to apply to all datasets
        seed: Random seed for reproducible sampling
        training: Whether in training mode (affects sampling strategy)
        num_shards_per_epoch: Number of shards to sample per epoch during training

    Example:
        >>> mixture = ShardedMixtureDataset(
        ...     datasets=[dataset1, dataset2, dataset3],
        ...     weights=[0.5, 0.3, 0.2],
        ...     processor=my_processor,
        ...     num_shards_per_epoch=10000
        ... )
        >>> for batch in mixture:
        ...     # batch contains processed data from mixed datasets
        ...     pass
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
        precomputed_stats_by_embodiment: dict[str, dict] | None = None,
        enable_shard_prefetch_pipeline: bool = False,
        shard_prefetch_depth: int = 1,
        shard_prefetch_workers: int = 1,
    ):
        """Initialize mixture dataset with datasets, weights, and configuration."""
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        self.training = training
        self.num_shards_per_epoch = num_shards_per_epoch
        # Shard prefetch pipeline depth (how many shards to keep building/buffered
        # ahead of the one being served) and the thread-pool width that builds
        # them. Defaults reproduce the historical 1-ahead, single-thread behavior.
        # Env vars override so the pipeline can be tuned without config plumbing.
        self._shard_prefetch_depth = shard_prefetch_depth
        self._shard_prefetch_workers = shard_prefetch_workers
        self.epoch = 0
        self.processor = processor
        self.override_pretraining_statistics = override_pretraining_statistics
        self.precomputed_stats_by_embodiment = precomputed_stats_by_embodiment or {}
        self.enable_shard_prefetch_pipeline = enable_shard_prefetch_pipeline

        # Generate initial shard sampling schedule
        self.shard_sampling_schedule = self.generate_shard_sampling_schedule()

        # Merge statistics across datasets and configure processor
        self.merge_statistics()

        # Initialize distributed training parameters
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

        # Per-worker profiling configuration (wired from trainer)
        self._enable_worker_profiling = False
        self._worker_profile_dir = None
        self._worker_profile_skip_first = 5
        self._worker_profile_active_steps = 2000

        # Initialize shard caching system
        self.curr_shard = None
        self._executor = None
        self._cache_job: Future | None = None
        # FIFO of in-flight/completed prefetched shards (N-ahead pipeline).
        self._prefetch_queue: deque = deque()

    def close(self) -> None:
        """Shut down the background ThreadPoolExecutor so the process can exit cleanly."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def merge_statistics(self):
        """Merge dataset statistics across all datasets, grouped by embodiment.

        Combines statistics from datasets with the same embodiment tag using
        weighted averaging, then configures the processor with merged statistics.
        This ensures consistent normalization across datasets within each embodiment.
        """
        # Group datasets and weights by embodiment
        all_stats_by_emb: dict[str, list] = {}
        weights_by_emb: dict[str, list[float]] = {}
        paths_by_emb: dict[str, list[str]] = {}
        for ds, w in zip(self.datasets, self.weights, strict=True):
            emb = getattr(ds, "embodiment_tag", None)
            if emb is None:
                continue
            emb = emb.value
            if emb in self.precomputed_stats_by_embodiment:
                continue
            if emb not in all_stats_by_emb:
                all_stats_by_emb[emb] = []
                weights_by_emb[emb] = []
                paths_by_emb[emb] = []
            stats = ds.get_dataset_statistics()  # type: ignore[attr-defined]
            all_stats_by_emb[emb].append(stats)
            weights_by_emb[emb].append(w)
            paths_by_emb[emb].append(
                getattr(ds, "dataset_path", f"dataset_{len(paths_by_emb[emb])}")
            )

        # Merge statistics within each embodiment group
        stats_by_emb = copy.deepcopy(self.precomputed_stats_by_embodiment)
        for emb, stats in all_stats_by_emb.items():
            stats_by_emb[emb] = {}
            # Get the action horizon from the processor's modality_configs
            # (which include YAML overrides via base_config deep-merge).
            _mc = self.processor.get_modality_configs()
            emb_action_config = _mc.get(emb, {}).get("action")
            action_horizon = None
            if emb_action_config is not None and emb_action_config.delta_indices:
                action_horizon = len(emb_action_config.delta_indices)
            for modality in ["state", "action", "relative_action"]:
                # Filter to datasets that have this modality
                indices = [i for i, s in enumerate(stats) if modality in s]
                if not indices:
                    continue
                modality_stats = [stats[i][modality] for i in indices]
                modality_weights = [weights_by_emb[emb][i] for i in indices]
                modality_paths = [paths_by_emb[emb][i] for i in indices]
                stats_by_emb[emb][modality] = merge_statistics(
                    per_dataset_stats=modality_stats,
                    dataset_sampling_weights=modality_weights,
                    is_relative_stats=(modality == "relative_action"),
                    action_horizon=action_horizon if modality == "relative_action" else None,
                    dataset_paths=modality_paths,
                )

        # Configure processor and datasets with merged statistics
        self.global_stats = stats_by_emb
        for emb, modalities in stats_by_emb.items():
            for mod, joint_groups in modalities.items():
                for joint_group, stats in joint_groups.items():
                    if "min" in stats and "max" in stats:
                        lo, hi = np.min(stats["min"]), np.max(stats["max"])
                        logger.info(
                            f"merge_statistics | {emb} / {mod} / {joint_group}: "
                            f"range=[{lo:.4f}, {hi:.4f}]"
                        )
        self.processor.set_statistics(
            self.global_stats, override=self.override_pretraining_statistics
        )
        for ds_idx, ds in enumerate(self.datasets):
            ds.dataset_index = ds_idx
            # ShardedMapDataset wraps an inner dataset via .dataset; ShardedSingleStepDataset
            # does not — use getattr to avoid AttributeError on the latter.
            # NOTE: make ShardedDataset subclasses expose a uniform interface for the
            # inner dataset so this workaround is no longer needed.
            inner = getattr(ds, "dataset", None)
            if inner is not None and hasattr(inner, "dataset_idx"):
                inner.dataset_idx = ds_idx
            ds.set_processor(self.processor)

    def get_dataset_statistics(self):
        """Get the merged dataset statistics."""
        return self.global_stats

    def generate_shard_sampling_schedule(self) -> list[tuple[int, int]]:
        """Generate a schedule of (dataset_index, shard_index) pairs for shard sampling.

        For training: Uses weighted random sampling normalized by average shard sizes
        to ensure fair representation regardless of shard size differences.

        For evaluation: Samples every shard from every dataset exactly once
        for comprehensive evaluation coverage.

        Returns:
            List of (dataset_index, shard_index) tuples defining the sampling order
        """
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch)

            # Compute average shard sizes for normalization
            average_shard_sizes = []
            for dataset in self.datasets:
                average_shard_size = sum(
                    dataset.get_shard_length(i) for i in range(len(dataset))
                ) / len(dataset)
                average_shard_sizes.append(average_shard_size)

            # Normalize weights by shard sizes to ensure fair sampling
            normalized_weights = np.array(
                [w / s for w, s in zip(self.weights, average_shard_sizes, strict=True)]
            )
            normalized_weights = normalized_weights / normalized_weights.sum()

            # Sample datasets according to normalized weights
            dataset_sampling_schedule = rng.choice(
                len(self.datasets), size=self.num_shards_per_epoch, p=normalized_weights
            )

            # Generate shuffled shard indices for each dataset. numpy permutation
            # uses 8 bytes per int vs ~28 for a Python list, and lets us track
            # consumption with an O(1) cursor instead of O(N) list.pop(0) (which
            # is O(N^2) per epoch at shard_size=16 where num_shards reaches the
            # hundreds of thousands).
            shard_sampling_schedule = []
            shards_to_sample = [rng.permutation(len(d)) for d in self.datasets]
            shard_cursors = [0] * len(self.datasets)

            for raw_dataset_idx in dataset_sampling_schedule:
                dataset_idx = int(raw_dataset_idx)
                # Reshuffle and refill if dataset shards are exhausted
                if shard_cursors[dataset_idx] >= len(shards_to_sample[dataset_idx]):
                    shards_to_sample[dataset_idx] = rng.permutation(len(self.datasets[dataset_idx]))
                    shard_cursors[dataset_idx] = 0
                shard_idx = int(shards_to_sample[dataset_idx][shard_cursors[dataset_idx]])
                shard_cursors[dataset_idx] += 1
                shard_sampling_schedule.append((dataset_idx, shard_idx))

        else:
            # Evaluation mode: sample every shard exactly once
            shard_sampling_schedule = []
            for i, dataset in enumerate(self.datasets):
                shard_sampling_schedule.extend([(i, j) for j in range(len(dataset))])
        return shard_sampling_schedule

    def filter_shard_sample_schedule(self):
        """Filter the shard sampling schedule for distributed training.

        Distributes shards across world_size processes and num_workers per process,
        ensuring each worker gets a unique subset of shards for parallel processing.

        Returns:
            Filtered list of (dataset_index, shard_index) pairs for this worker
        """
        filtered_schedule = []
        worker_info = get_worker_info()

        # Determine worker configuration
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Cache worker configuration and validate consistency
        if self.worker_id is None:
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert (
                self.worker_id == worker_id and self.num_workers == num_workers
            ), "Worker ID or number of workers has been changed since it was set. This is not allowed."

        # Distribute shards across all workers in all processes
        for i, shard in enumerate(self.shard_sampling_schedule):
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        return filtered_schedule

    def _resolve_prefetch_depth(self) -> int:
        """How many shards to keep building/buffered ahead (env overrides ctor)."""
        env = os.environ.get("GROOT_SHARD_PREFETCH_DEPTH")
        val = int(env) if env not in (None, "") else self._shard_prefetch_depth
        return max(1, val)

    def _resolve_prefetch_workers(self) -> int:
        """Thread-pool width for background shard builds (env overrides ctor)."""
        env = os.environ.get("GROOT_SHARD_PREFETCH_WORKERS")
        val = int(env) if env not in (None, "") else self._shard_prefetch_workers
        return max(1, val)

    def _next_shard_pair(self):
        """Advance the submit cursor to the next shard pair.

        Rolls the epoch + schedule over when the worker's schedule is exhausted.
        Returns ``(dataset_index, shard_index, dataset_path)``.
        """
        self.curr_shard_index += 1
        if self.curr_shard_index >= len(self.worker_shard_sampling_schedule):
            self.epoch += 1
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
            if not self.worker_shard_sampling_schedule:
                self._raise_empty_worker_schedule()
            self.curr_shard_index = 0
        dataset_index, shard_index = self.worker_shard_sampling_schedule[self.curr_shard_index]
        dataset_path = getattr(self.datasets[dataset_index], "dataset_path", f"idx={dataset_index}")
        return dataset_index, shard_index, dataset_path

    def _submit_shard(self):
        """Submit the next shard build into the prefetch queue."""
        dataset_index, shard_index, dataset_path = self._next_shard_pair()
        # Capture the current context so structlog contextvars (worker_type,
        # rank, etc.) propagate into the worker thread.
        ctx = contextvars.copy_context()
        fut = self._executor.submit(ctx.run, self.datasets[dataset_index].get_shard, shard_index)
        self._prefetch_queue.append((fut, dataset_path, shard_index, time.time()))

    def _uses_deltacat_prefetch(self) -> bool:
        """Only DeltaCAT datasets use the new N-ahead shard prefetch pipeline."""
        return self.enable_shard_prefetch_pipeline

    def __iter__(self):
        """Iterate over the mixture dataset with an N-ahead background shard pipeline.

        Implements an efficient iteration strategy:
        1. Filter shards for this worker's portion
        2. Prime the pipeline with ``depth`` background shard builds
        3. For each shard: pop+wait for the oldest, refill to keep ``depth`` in
           flight, then yield the current shard's shuffled timesteps
        4. Shuffle timesteps within each shard for additional randomization
        5. Handle epoch transitions and schedule regeneration

        ``depth``/``workers`` (``GROOT_SHARD_PREFETCH_DEPTH`` /
        ``GROOT_SHARD_PREFETCH_WORKERS``, default 1/1) control how far ahead the
        pipeline runs. With ``depth>1`` the builder banks cheap shards ahead so a
        single expensive whole-episode decode no longer stalls the GPU; ``workers>1``
        builds several shards concurrently (requires a reentrant ``get_shard``).
        """
        if not self._uses_deltacat_prefetch():
            yield from self._iter_legacy_cache()
            return

        depth = self._resolve_prefetch_depth()
        workers = self._resolve_prefetch_workers()

        # Start background thread pool
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._prefetch_queue = deque()

        # Initialize worker-specific shard schedule
        self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
        if not self.worker_shard_sampling_schedule:
            self._raise_empty_worker_schedule()
        self.curr_shard_index = -1
        if self.worker_id in (0, None) and self.rank == 0:
            logger.info(
                f"[shard-prefetch] depth={depth} workers={workers} "
                f"(schedule_len={len(self.worker_shard_sampling_schedule)})"
            )
        # Prime the pipeline with `depth` background shard builds.
        for _ in range(depth):
            self._submit_shard()
        rng = np.random.default_rng(self.seed + self.epoch)

        # Start per-worker profiler if enabled
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

        # Continuous iteration with epoch management
        while True:
            # Pop the oldest prefetched shard and wait for it to finish building.
            fut, dataset_path, shard_index, submit_t = self._prefetch_queue.popleft()
            wait_start = time.time()
            self.curr_shard = fut.result()
            wait_end = time.time()

            # Immediately refill so the pipeline stays `depth` deep while we serve.
            self._submit_shard()

            logger.info(
                f"Rank {self.rank}, Worker {self.worker_id}: "
                f"Wait for shard {shard_index} in dataset {dataset_path} "
                f"in {wait_end - wait_start:.2f} seconds, shard_len={len(self.curr_shard)}, "
                f"build={wait_end - submit_t:.2f}s, qdepth={len(self._prefetch_queue)}"
            )

            # Yield shuffled timesteps from current shard
            assert self.curr_shard is not None
            indices_in_shard = np.arange(len(self.curr_shard))
            rng.shuffle(indices_in_shard)
            for index in indices_in_shard:
                yield self.curr_shard[index]
                samples_yielded += 1
                if worker_profiler is not None:
                    worker_profiler.step()
                    # Stop after schedule completes
                    if samples_yielded >= (
                        self._worker_profile_skip_first + 1 + self._worker_profile_active_steps
                    ):
                        worker_profiler.stop()
                        worker_profiler = None

            # Clean up cached shard to free memory
            self.delete_cached_shard()

    def _iter_legacy_cache(self):
        """Iterate with the historical one-shard-ahead cache path.

        This keeps non-DeltaCAT training on the same iterator control flow as
        main while the DeltaCAT-specific prefetch pipeline soaks separately.
        """
        # Start background thread pool
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Initialize worker-specific shard schedule
        self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed + self.epoch)

        # Start per-worker profiler if enabled
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

        # Continuous iteration with epoch management
        while True:
            self.curr_shard_index += 1

            # Wait for background caching to complete
            wait_start = time.time()
            self.finish_cache_shard()
            wait_end = time.time()

            dataset_index, shard_index = self.worker_shard_sampling_schedule[self.curr_shard_index]
            dataset_path = getattr(
                self.datasets[dataset_index], "dataset_path", f"idx={dataset_index}"
            )
            logger.info(
                f"Rank {self.rank}, Worker {self.worker_id}: "
                f"Wait for shard {shard_index} in dataset {dataset_path} "
                f"in {wait_end - wait_start:.2f} seconds, shard_len={len(self.curr_shard)}"
            )

            # Start caching next shard immediately
            self.cache_next_shard()

            # Yield shuffled timesteps from current shard
            assert self.curr_shard is not None
            indices_in_shard = np.arange(len(self.curr_shard))
            rng.shuffle(indices_in_shard)
            for index in indices_in_shard:
                yield self.curr_shard[index]
                samples_yielded += 1
                if worker_profiler is not None:
                    worker_profiler.step()
                    # Stop after schedule completes
                    if samples_yielded >= (
                        self._worker_profile_skip_first + 1 + self._worker_profile_active_steps
                    ):
                        worker_profiler.stop()
                        worker_profiler = None

            # Clean up cached shard to free memory
            self.delete_cached_shard()

    def cache_next_shard(self):
        """Start background caching of the next shard using ThreadPoolExecutor.

        Handles epoch transitions by regenerating the sampling schedule when
        the current schedule is exhausted.
        """
        assert self._executor is not None
        # Check if epoch is complete and regenerate schedule if needed
        if self.curr_shard_index + 1 >= len(self.worker_shard_sampling_schedule):
            self.epoch += 1
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
            if not self.worker_shard_sampling_schedule:
                self._raise_empty_worker_schedule()
            self.curr_shard_index = -1

        next_dataset_idx, next_shard_idx = self.worker_shard_sampling_schedule[
            self.curr_shard_index + 1
        ]
        # Submit background loading job. Capture the current context so structlog
        # contextvars (worker_type, rank, etc.) propagate into the worker thread.
        ctx = contextvars.copy_context()
        self._cache_job = self._executor.submit(
            ctx.run, self.datasets[next_dataset_idx].get_shard, next_shard_idx
        )

    def finish_cache_shard(self):
        """Wait for the background caching job to complete and retrieve the shard."""
        assert self._cache_job is not None
        self.curr_shard = self._cache_job.result()
        self._cache_job = None

    def delete_cached_shard(self):
        """Delete the current cached shard to free memory."""
        del self.curr_shard

    def _raise_empty_worker_schedule(self):
        worker_id = self.worker_id if self.worker_id is not None else 0
        num_workers = self.num_workers if self.num_workers is not None else 1
        worker_slots = self.world_size * num_workers
        raise ValueError(
            "ShardedMixtureDataset assigned no shards to "
            f"rank={self.rank}, worker_id={worker_id}. "
            f"num_shards_per_epoch={self.num_shards_per_epoch} must cover "
            f"world_size * dataloader_num_workers = {worker_slots}; increase "
            "`data.num_shards_per_epoch` or reduce dataloader workers."
        )

    def reset_seed(self, seed: int):
        """Reset the random seed and regenerate sampling schedules.

        Used for deterministic training restarts or seed changes during training.

        Args:
            seed: New random seed to use
        """
        self.seed = seed
        self.epoch = 0
        self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
        self.curr_shard_index = -1
        self.curr_shard = None
        self._cache_job = None

    def print_dataset_statistics(self):
        """Print formatted dataset statistics for debugging and monitoring."""
        logger.info("=" * 100)
        logger.info("ShardedMixtureDataset Statistics")
        logger.info("=" * 100)

        # Print header
        logger.info(f"{'Dataset Path':<60} {'Length':<10} {'Mix Ratio':<12}")
        logger.info("-" * 100)

        # Print dataset details
        for i, ds in enumerate(self.datasets):
            dataset_path = str(ds.dataset_path)
            # Truncate long paths for better display
            if len(dataset_path) > 55:
                dataset_path = "..." + dataset_path[-52:]

            length = len(ds)
            mix_ratio = self.weights[i] * 100

            logger.info(f"{dataset_path:<60} {length:<10,} {mix_ratio:<12.2f}")

        # Print additional metadata
        embodiments = {
            ds.embodiment_tag.value for ds in self.datasets if hasattr(ds, "embodiment_tag")  # type: ignore[attr-defined]
        }
        logger.info(f"Embodiments: {', '.join(sorted(embodiments))}")
        logger.info(f"Number of datasets: {len(self.datasets)}")
        logger.info("=" * 100)

    def get_initial_actions(self):
        """Collect initial actions from all datasets.

        Returns:
            Combined list of initial actions from all constituent datasets
        """
        initial_actions = []
        for dataset in self.datasets:
            if hasattr(dataset, "get_initial_actions"):
                initial_actions.extend(dataset.get_initial_actions())  # type: ignore[attr-defined]
        return initial_actions
