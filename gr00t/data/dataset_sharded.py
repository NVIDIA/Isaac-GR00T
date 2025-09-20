import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch.distributed as dist
import yaml
from torch.utils.data import IterableDataset, get_worker_info

from gr00t.utils.video import get_frames_by_timestamps

from .dataset import LeRobotMixtureDataset, LeRobotSingleDataset


class ShardedLeRobotSingleDataset(LeRobotSingleDataset):
    """
    A single dataset with shards.
    """

    def __init__(
        self,
        *args,
        num_steps_per_shard: int = int(1e4),
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.num_steps_per_shard = num_steps_per_shard
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()
        self.frames_to_load = self.get_all_frames_to_load()

        # Set shard caching properties
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self.frame_indices_map: dict[int, dict[str, np.ndarray]] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self.sharded_trajectories)

    def get_all_video_paths(self) -> dict[int, dict[str, Path]]:
        """Get the video paths for all trajectories and all views.

        Returns:
            dict[int, dict[str, Path]]: The video paths for all trajectories.
        """
        video_paths = {}
        for trajectory_id in self.trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                video_paths[trajectory_id][key] = self.get_video_path(
                    trajectory_id, key.replace("video.", "")
                )
        return video_paths

    def get_all_parquet_paths(self) -> dict[int, Path]:
        """Get the parquet paths for all trajectories.

        Returns:
            dict[int, Path]: The parquet paths for all trajectories.
        """
        return {
            trajectory_id: self.get_parquet_path(trajectory_id)
            for trajectory_id in self.trajectory_ids
        }

    def generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        """Generate shards of trajectories. We recommend num_steps_per_shard >> average trajectory length.

        Args:
            num_steps_per_shard (int): The number of steps per shard.

        Returns:
            list[list[str]]: The shards of trajectories.
        """
        sharded_trajectories = [[]]
        curr_num_steps = 0
        curr_shard_index = 0
        trajectory_ids = self.trajectory_ids

        assert (
            len(trajectory_ids) > 0
        ), f"No valid trajectories found for dataset {self.dataset_path}"
        total_steps = np.sum(
            [self.trajectory_lengths[trajectory_id] for trajectory_id in trajectory_ids]
        ).astype(int)
        num_shards = np.ceil(total_steps / self.num_steps_per_shard).astype(int)
        cutoffs = np.linspace(0, total_steps, num_shards + 1)[1:]  # Exclude the first cutoff (0)
        shard_lengths = []
        last_num_steps = 0
        for trajectory_id in trajectory_ids:
            sharded_trajectories[-1].append(trajectory_id)
            curr_num_steps += self.trajectory_lengths[trajectory_id]
            if curr_num_steps > cutoffs[curr_shard_index]:
                sharded_trajectories.append([])
                curr_shard_index += 1
                shard_lengths.append(curr_num_steps - last_num_steps)
                last_num_steps = curr_num_steps
        shard_lengths.append(curr_num_steps - last_num_steps)
        assert (
            curr_num_steps == total_steps
        ), "Total steps not equal to the sum of trajectory lengths"
        assert (
            len(shard_lengths) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        assert (
            len(sharded_trajectories) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        print(f"Generated {len(sharded_trajectories)} shards for dataset {self.dataset_path}")
        return sharded_trajectories, np.array(shard_lengths)

    def get_all_frames_to_load(self):
        """Generate a map of video frame indices to trajectory indices."""
        all_frames_to_load = {}
        for trajectory_id in self.trajectory_ids:
            all_frames_to_load[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                frames_to_load = np.unique(
                    np.concatenate(
                        [i + np.array(self.delta_indices[key]) for i in range(self.trajectory_lengths[trajectory_id])]
                    )
                )
                # Cap within the length of the trajectory and >= 0
                frames_to_load = frames_to_load[
                    (frames_to_load < self.trajectory_lengths[trajectory_id])
                    & (frames_to_load >= 0)
                ]
                all_frames_to_load[trajectory_id][key] = frames_to_load
        return all_frames_to_load

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        frames_to_load: dict[int, dict[str, np.ndarray]],
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
    ) -> tuple[
        dict[str, np.ndarray], dict[int, int], pd.DataFrame, dict[int, dict[str, np.ndarray]]
    ]:
        print("Caching shard")
        start_time = time.time()
        assert "video" in modality_keys, "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        frame_indices_map = {}
        curr_step_index = 0
        cached_df = None
        curr_frame_index = {key: 0 for key in modality_keys["video"]}
        for trajectory_id in trajectory_ids:
            trajectory_start_indices[trajectory_id] = curr_step_index
            parquet_path = parquet_paths[trajectory_id]
            parquet_df = pd.read_parquet(parquet_path)
            # Check timestamps are in sync
            parquet_timestamps = parquet_df["timestamp"].to_numpy()
            trajectory_length = len(parquet_timestamps)
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            frame_indices_map[trajectory_id] = {}
            for key in modality_keys["video"]:
                # Only load the frames that are needed
                this_frames_to_load = frames_to_load[trajectory_id][key]
                if len(this_frames_to_load) == 0:
                    continue
                load_timestamps = parquet_timestamps[this_frames_to_load]
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                # Store a mapping that frame_indices_map[trajectory_id][key][frame_index] = index_in_concat_video_frames
                frame_indices_map[trajectory_id][key] = (
                    np.ones(len(parquet_timestamps), dtype=np.int32) * -1
                )
                frame_indices_map[trajectory_id][key][this_frames_to_load] = np.arange(
                    curr_frame_index[key],
                    curr_frame_index[key] + len(this_frames_to_load),
                    dtype=np.int32,
                )
                curr_frame_index[key] += len(this_frames_to_load)
                if key not in cached_frames:
                    cached_frames[key] = []
                frames = get_frames_by_timestamps(
                    video_paths[trajectory_id][key].as_posix(),
                    timestamps=load_timestamps,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs or {},
                )
                cached_frames[key].append(frames)
            if cached_df is None:
                cached_df = parquet_df
            else:
                cached_df = pd.concat([cached_df, parquet_df])
            curr_step_index += trajectory_length

        # Concatenate the frames
        for key in cached_frames:
            cached_frames[key] = np.concatenate(cached_frames[key], axis=0)
        end_time = time.time()
        print(f"Cached shard in {end_time - start_time:.2f} seconds")
        assert cached_df is not None, "Cached dataframe is None"
        return cached_frames, trajectory_start_indices, cached_df, frame_indices_map

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self.modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.frames_to_load,
            self.video_backend,
            self.video_backend_kwargs,
        )

    def finish_cache_shard(self):
        """Get the cached shard."""
        assert self._cache_job is not None
        self.cached_shard, self.shard_start_indices, self.cached_df, self.frame_indices_map = (
            self._cache_job.result()
        )
        self._cache_job = None  # Clear the future to allow memory to be freed

    def delete_cached_shard(self):
        """Delete the cached shard."""
        del self.cached_shard
        del self.shard_start_indices
        del self.cached_df

    def get_trajectories_in_shard(self) -> list[int]:
        """Get the trajectories in a shard."""
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def get_video(self, trajectory_id: int, key: str, base_index) -> np.ndarray:
        """Get the video frames from cached shards for a trajectory by a base index.

        Args:
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the trajectory index
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        # Calculate the absolute indices
        assert (
            self.shard_start_indices is not None
            and self.cached_shard is not None
            and trajectory_id in self.shard_start_indices
            and self.frame_indices_map is not None
            and trajectory_id in self.frame_indices_map
            and key in self.frame_indices_map[trajectory_id]
        ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        indices_in_shard = self.frame_indices_map[trajectory_id][key][step_indices]
        assert np.all(
            indices_in_shard != -1
        ), f"Indices in shard are not loaded for {trajectory_id=}, {key=}, {step_indices=}"

        return self.cached_shard[key][indices_in_shard]

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the trajectory data."""
        assert self.cached_df is not None, "Cached dataframe is None"
        traj_data = self.cached_df.loc[self.cached_df["episode_index"] == trajectory_id]
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        assert (
            len(traj_data) == trajectory_length
        ), f"Trajectory length mismatch: {len(traj_data)} != {trajectory_length} {self.args} {self.kwargs}"
        indices = traj_data["index"].to_numpy()
        if len(indices) > 0:
            start_index = indices[0]
            expected_indices = np.arange(start_index, start_index + len(indices))
            assert np.array_equal(
                indices, expected_indices
            ), f"[{self}] Index sequence mismatch in trajectory data, {trajectory_id=}"
        return traj_data


class ShardedLeRobotMixtureDataset(LeRobotMixtureDataset, IterableDataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: list[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        shard_sampling_rate: float = 0.5,
        num_shards_to_sample: int = 2**20,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[ShardedLeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __iter__ will yield different samples every epoch; if "val" or "test", __iter__ will yield the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            shard_sampling_rate (float): How much data per shard to sample, in a 0-1 scale.
            num_shards_to_sample (int): The number of shards to sample.
        """
        super().__init__(
            data_mixture=data_mixture,
            mode=mode,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
        )
        # Add type hint
        self.datasets: list[ShardedLeRobotSingleDataset] = self.datasets
        # Set properties
        self.shard_sampling_rate = shard_sampling_rate
        self.num_shards_to_sample = num_shards_to_sample

        # Calculate shard sampling weights
        all_shard_sampling_weights = []
        all_shards = []
        for dataset_id, (dataset, weight) in enumerate(
            zip(self.datasets, self._dataset_sampling_weights)
        ):
            shard_sampling_weights = dataset.shard_lengths / dataset.shard_lengths.sum()
            all_shard_sampling_weights.append(shard_sampling_weights * weight)
            all_shards.extend(
                [(dataset_id, shard_idx) for shard_idx in range(shard_sampling_weights.shape[0])]
            )
        all_shard_sampling_weights = np.concatenate(all_shard_sampling_weights)
        all_shard_sampling_weights /= all_shard_sampling_weights.sum()
        self._shard_sampling_weights = all_shard_sampling_weights
        self._all_shards = all_shards

        # Generate shards sample schedule for all ranks and workers
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

        # Check shard sampling rate
        assert 0 <= shard_sampling_rate <= 1, "Shard sampling rate must be between 0 and 1"

        # Set properties for distributed training
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The dataset sampling weights."""
        return self._dataset_sampling_weights

    @property
    def shard_sampling_weights(self) -> list[np.ndarray]:
        """The weights of each shard."""
        return self._shard_sampling_weights

    @property
    def all_shards(self) -> list[tuple[int, int]]:
        """The shards to sample."""
        return self._all_shards

    @property
    def shards_sample_schedule(self) -> list[tuple[int, int]]:
        """The shards sample schedule.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        assert self._shards_sample_schedule is not None, "Shards sample schedule not set."
        return self._shards_sample_schedule

    @property
    def trajectory_sampling_weights(self):
        """The trajectory sampling weights."""
        raise ValueError("ShardedRobotMixtureDataset does not support trajectory sampling weights.")

    @property
    def primary_dataset_indices(self):
        """The primary dataset indices."""
        raise ValueError("ShardedRobotMixtureDataset does not support primary dataset indices.")

    def reset_seed(self, seed: int):
        self.seed = seed
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

    def generate_shards_sample_schedule(self):
        if self.mode == "train":
            rng = np.random.default_rng(self.seed)
            sampled_shard_ids = rng.choice(
                len(self.all_shards), size=self.num_shards_to_sample, p=self.shard_sampling_weights
            )
            shards_sample_schedule = [self.all_shards[i] for i in sampled_shard_ids]
            rng.shuffle(shards_sample_schedule)
        else:
            shards_sample_schedule = [
                self.all_shards[i % len(self.all_shards)] for i in range(self.num_shards_to_sample)
            ]
        return shards_sample_schedule

    def filter_shards_sample_schedule(self):
        """Filter the shards sample schedule for each worker.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        # Filter shards for each worker
        filtered_schedule = []
        worker_info = get_worker_info()
        # If we have multiple workers, further split shards among them
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if self.worker_id is None:
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert (
                self.worker_id == worker_id and self.num_workers == num_workers
            ), "Worker ID or number of workers has been changed since it was set. This is not allowed."

        for i, shard in enumerate(self.shards_sample_schedule):
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        # print(f"Filtered shards for rank {self.rank}, worker {worker_id}: {filtered_schedule}")
        return filtered_schedule

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            shard_lengths = dataset.shard_lengths
            assert len(shard_lengths.shape) == 1, "Shard lengths must be a 1D array"
            num_shards = shard_lengths.shape[0]
            max_shard_length = int(shard_lengths.max())
            min_shard_length = int(shard_lengths.min())
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
                "Num shards": num_shards,
                "Max shard length": max_shard_length,
                "Min shard length": min_shard_length,
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump(
            {
                "Mixture dataset": dataset_descriptions,
                "Rank": self.rank,
                "World size": self.world_size,
            }
        )

    def __iter__(self):
        """Iterate over the dataset."""

        # Not supported: balance_trajectory_weights=False
        if not self.balance_trajectory_weights:
            raise NotImplementedError(
                "balance_trajectory_weights=False is not supported. Please use balance_dataset_weights=True instead."
            )

        self._shards_sample_schedule = self.filter_shards_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed)
        for i, (dataset_index, shard_index) in enumerate(self.shards_sample_schedule):
            self.curr_shard_index += 1
            assert (
                i == self.curr_shard_index
            ), f"Shard index mismatch: {i} != {self.curr_shard_index}"
            dataset = self.datasets[dataset_index]
            wait_start = time.time()
            dataset.finish_cache_shard()
            wait_end = time.time()
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: Wait for shard {shard_index} in dataset {dataset_index} in {wait_end - wait_start:.2f} seconds"
            )
            # Start caching the next shard immediately
            self.cache_next_shard()
            all_steps: list[tuple[int, int]] = []
            for trajectory_id in dataset.get_trajectories_in_shard():
                trajectory_index = dataset.get_trajectory_index(trajectory_id)
                max_delta_index = dataset.max_delta_index
                trajectory_length = dataset.trajectory_lengths[trajectory_index]
                allowed_length = trajectory_length - max_delta_index
                allowed_indices = range(allowed_length)
                for i in allowed_indices:
                    all_steps.append((trajectory_id, i))
            if self.mode == "train":
                rng.shuffle(all_steps)
            sampled_steps = all_steps[: int(dataset.num_steps_per_shard * self.shard_sampling_rate)]
            for trajectory_id, step_index in sampled_steps:
                yield dataset.transforms(dataset.get_step_data(trajectory_id, step_index))

            # Delete the cached shard and shard start indices to free up memory
            dataset.delete_cached_shard()

    def cache_next_shard(self):
        """Cache the next shard in a background thread."""
        next_dataset_idx, next_shard_idx = self.shards_sample_schedule[self.curr_shard_index + 1]
        self.datasets[next_dataset_idx].start_cache_shard(next_shard_idx)

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(
            "__getitem__ is not supported for CachedRobotMixtureDataset. Please use __iter__ instead."
        )

    def __len__(self) -> int:
        """The length of the dataset."""
        total_length = 0
        for dataset_idx, _ in self.shards_sample_schedule:
            dataset = self.datasets[dataset_idx]
            total_length += int(dataset.num_steps_per_shard * self.shard_sampling_rate)
        return total_length
