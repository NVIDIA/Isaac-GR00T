from concurrent.futures import Future, ThreadPoolExecutor
import time

import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from gr00t.data.interfaces import BaseProcessor, ShardedDataset


def merge_statistics(
    per_dataset_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
    is_relative_stats: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """
    Compute overall statistics from per-dataset statistics using weighted averaging.

    This function combines statistics from multiple datasets according to their sampling
    weights, computing weighted means and variances while preserving min/max/quantile
    information across all datasets.

    The weighted variance computation uses the formula:
    Var_combined = Σ(w_i * (σ_i² + μ_i²)) - (Σ(w_i * μ_i))²

    Args:
        per_dataset_stats: List of per-dataset statistics dictionaries.
            Each element has structure: {modality: {joint_group: {stat_type: values}}}
            Example: {"state": {"gripper": {"mean": [0.1, 0.2], "std": [0.5, 0.3]}}}
        dataset_sampling_weights: Weights for combining dataset statistics.
            Should sum to 1.0 or will be normalized.
        is_relative_stats: Whether the statistics are relative (affects merging logic).

    Returns:
        Combined statistics dictionary with same structure as input, containing
        weighted averages for mean/std and global min/max/quantiles across datasets.
    """
    # Normalize sampling weights to sum to 1
    dataset_sampling_weights = np.array(dataset_sampling_weights)
    normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    # Initialize overall statistics dict
    overall_stats: dict[str, dict[str, list[float]]] = {}

    # Process each modality (e.g., "state", "action")
    for modality in per_dataset_stats[0]:
        # Get dimensionality from first dataset (assumed consistent)
        dim = (
            [len(per_dataset_stats[0][modality]["mean"])]
            if not is_relative_stats
            else np.array(per_dataset_stats[0][modality]["mean"]).shape
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
            stats = dataset_stats[modality]
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

        # Store combined statistics for this modality
        overall_stats[modality] = {
            "min": overall_min,
            "max": overall_max,
            "mean": overall_mean,
            "std": overall_std,
            "q01": weighted_q01,
            "q99": weighted_q99,
        }

    return overall_stats


class ShardedMixtureDataset(IterableDataset):
    """
    Iterable dataset that combines multiple sharded datasets with configurable mixing ratios.

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
        ...     num_shards_per_epoch=10000,
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
    ):
        """Initialize mixture dataset with datasets, weights, and configuration."""
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        self.training = training
        self.num_shards_per_epoch = num_shards_per_epoch
        self.epoch = 0
        self.processor = processor
        self.override_pretraining_statistics = override_pretraining_statistics

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

        # Initialize shard caching system
        self.curr_shard = None
        self._executor = None
        self._cache_job: Future | None = None

    def merge_statistics(self):
        """
        Merge dataset statistics across all datasets, grouped by embodiment.

        Combines statistics from datasets with the same embodiment tag using
        weighted averaging, then configures the processor with merged statistics.
        This ensures consistent normalization across datasets within each embodiment.
        """
        # Group datasets and weights by embodiment
        all_stats_by_emb: dict[str, list] = {}
        weights_by_emb: dict[str, list[float]] = {}
        for ds, w in zip(self.datasets, self.weights):
            emb = getattr(ds, "embodiment_tag", None)
            if emb is None:
                continue
            emb = emb.value
            if emb not in all_stats_by_emb:
                all_stats_by_emb[emb] = []
                weights_by_emb[emb] = []
            stats = ds.get_dataset_statistics()  # type: ignore
            all_stats_by_emb[emb].append(stats)
            weights_by_emb[emb].append(w)

        # Merge statistics within each embodiment group
        stats_by_emb = {}
        for emb, stats in all_stats_by_emb.items():
            stats_by_emb[emb] = {}
            for modality in ["state", "action", "relative_action"]:
                if modality in stats[0]:
                    modality_stats = [s[modality] for s in stats]
                    stats_by_emb[emb][modality] = merge_statistics(
                        per_dataset_stats=modality_stats,
                        dataset_sampling_weights=weights_by_emb[emb],
                        is_relative_stats=(modality == "relative_action"),
                    )

        # Configure processor and datasets with merged statistics
        self.global_stats = stats_by_emb
        self.processor.set_statistics(
            self.global_stats, override=self.override_pretraining_statistics
        )
        for ds in self.datasets:
            ds.set_processor(self.processor)

    def get_dataset_statistics(self):
        """Get the merged dataset statistics."""
        return self.global_stats

    def generate_shard_sampling_schedule(self) -> list[tuple[int, int]]:
        """
        Generate a schedule of (dataset_index, shard_index) pairs for shard sampling.

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
                [w / s for w, s in zip(self.weights, average_shard_sizes)]
            )
            normalized_weights = normalized_weights / normalized_weights.sum()

            # Sample datasets according to normalized weights
            dataset_sampling_schedule = rng.choice(
                len(self.datasets), size=self.num_shards_per_epoch, p=normalized_weights
            )

            # Generate shuffled shard indices for each dataset
            shard_sampling_schedule = []
            shards_to_sample = []
            for dataset in self.datasets:
                shard_ids = list(range(len(dataset)))
                rng.shuffle(shard_ids)
                shards_to_sample.append(shard_ids)

            # Create final sampling schedule with shard cycling
            for i in dataset_sampling_schedule:
                # Reshuffle and refill if dataset shards are exhausted
                if len(shards_to_sample[i]) == 0:
                    shard_ids = list(range(len(self.datasets[i])))
                    rng.shuffle(shard_ids)
                    shards_to_sample[i] = shard_ids
                shard_idx = shards_to_sample[i].pop(0)
                shard_sampling_schedule.append((i, shard_idx))

        else:
            # Evaluation mode: sample every shard exactly once
            shard_sampling_schedule = []
            for i, dataset in enumerate(self.datasets):
                shard_sampling_schedule.extend([(i, j) for j in range(len(dataset))])
        return shard_sampling_schedule

    def filter_shard_sample_schedule(self):
        """
        Filter the shard sampling schedule for distributed training.

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
            assert self.worker_id == worker_id and self.num_workers == num_workers, (
                "Worker ID or number of workers has been changed since it was set. This is not allowed."
            )

        # Distribute shards across all workers in all processes
        for i, shard in enumerate(self.shard_sampling_schedule):
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        return filtered_schedule

    def __iter__(self):
        """
        Iterate over the mixture dataset with background shard caching.

        Implements an efficient iteration strategy:
        1. Filter shards for this worker's portion
        2. Start background caching of the first shard
        3. For each shard: wait for cache, start caching next, yield current
        4. Shuffle timesteps within each shard for additional randomization
        5. Handle epoch transitions and schedule regeneration
        """
        # Start background thread pool
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Initialize worker-specific shard schedule
        self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed + self.epoch)

        # Continuous iteration with epoch management
        while True:
            self.curr_shard_index += 1

            # Wait for background caching to complete
            wait_start = time.time()
            self.finish_cache_shard()
            wait_end = time.time()

            dataset_index, shard_index = self.worker_shard_sampling_schedule[self.curr_shard_index]
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: Wait for shard {shard_index} in dataset {dataset_index} in {wait_end - wait_start:.2f} seconds"
            )

            # Start caching next shard immediately
            self.cache_next_shard()

            # Yield shuffled timesteps from current shard
            assert self.curr_shard is not None
            indices_in_shard = np.arange(len(self.curr_shard))
            rng.shuffle(indices_in_shard)
            for index in indices_in_shard:
                yield self.curr_shard[index]

            # Clean up cached shard to free memory
            self.delete_cached_shard()

    def cache_next_shard(self):
        """
        Start background caching of the next shard using ThreadPoolExecutor.

        Handles epoch transitions by regenerating the sampling schedule when
        the current schedule is exhausted.
        """
        assert self._executor is not None
        # Check if epoch is complete and regenerate schedule if needed
        if self.curr_shard_index + 1 >= len(self.worker_shard_sampling_schedule):
            self.epoch += 1
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
            self.curr_shard_index = -1

        print(f"Rank {self.rank}, Worker {self.worker_id}: Caching shard...")
        next_dataset_idx, next_shard_idx = self.worker_shard_sampling_schedule[
            self.curr_shard_index + 1
        ]
        # Submit background loading job
        self._cache_job = self._executor.submit(
            self.datasets[next_dataset_idx].get_shard, next_shard_idx
        )

    def finish_cache_shard(self):
        """Wait for the background caching job to complete and retrieve the shard."""
        assert self._cache_job is not None
        self.curr_shard = self._cache_job.result()
        self._cache_job = None

    def delete_cached_shard(self):
        """Delete the current cached shard to free memory."""
        del self.curr_shard

    def reset_seed(self, seed: int):
        """
        Reset the random seed and regenerate sampling schedules.

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
        print("=" * 100)
        print("ShardedMixtureDataset Statistics")
        print("=" * 100)

        # Print header
        print(f"{'Dataset Path':<60} {'Length':<10} {'Mix Ratio':<12}")
        print("-" * 100)

        # Print dataset details
        for i, ds in enumerate(self.datasets):
            dataset_path = str(ds.dataset_path)
            # Truncate long paths for better display
            if len(dataset_path) > 55:
                dataset_path = "..." + dataset_path[-52:]

            length = len(ds)
            mix_ratio = self.weights[i] * 100

            print(f"{dataset_path:<60} {length:<10,} {mix_ratio:<12.2f}")

        # Print additional metadata
        embodiments = set(
            ds.embodiment_tag.value
            for ds in self.datasets
            if hasattr(ds, "embodiment_tag")  # type: ignore
        )
        print(f"Embodiments: {', '.join(sorted(embodiments))}")
        print(f"Number of datasets: {len(self.datasets)}")
        print("=" * 100)

    def get_initial_actions(self):
        """
        Collect initial actions from all datasets.

        Returns:
            Combined list of initial actions from all constituent datasets
        """
        initial_actions = []
        for dataset in self.datasets:
            if hasattr(dataset, "get_initial_actions"):
                initial_actions.extend(dataset.get_initial_actions())  # type: ignore
        return initial_actions
