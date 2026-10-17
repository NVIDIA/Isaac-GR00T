# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import heapq
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import structlog
from structlog.contextvars import bound_contextvars

from groot.core.data import types
from groot.core.data.dataset.episode_loader_factory import create_episode_loader
from groot.core.data.dataset.step_filter import load_step_indices_filter
from groot.core.data.hydrator import client_protocol

logger = logging.getLogger(__name__)


def extract_step_data(
    episode_data: types.EpisodeData,
    step_index: int,
    modality_configs: dict[str, types.ModalityConfig],
    embodiment_tag: types.EmbodimentTag,
    allow_padding: bool = False,
) -> types.VLAStepData:
    """Extract a single step — delegates to the batch version."""
    return extract_step_data_batch(
        episode_data, [step_index], modality_configs, embodiment_tag, allow_padding
    )[0]


def extract_step_data_batch(
    episode_data: types.EpisodeData,
    step_indices: list[int] | np.ndarray,
    modality_configs: dict[str, types.ModalityConfig],
    embodiment_tag: types.EmbodimentTag,
    allow_padding: bool = False,
) -> list[types.VLAStepData]:
    """Extract multiple steps at once — one gather per column.

    For each (modality, key) column, computes the flat list of all row indices
    needed across all steps (step_index + delta for each step and each delta),
    performs a single ``col.gather(flat_indices).to_numpy()`` call, then
    reshapes into per-step slices. This amortizes the polars→numpy conversion
    cost over the entire batch instead of paying it per step.
    """
    n_rows = episode_data.height
    n_steps = len(step_indices)

    results: list[dict] = [{} for _ in range(n_steps)]
    for i in range(n_steps):
        for modality in modality_configs:
            results[i][modality] = {}

    for modality, config in modality_configs.items():
        horizon = len(config.delta_indices)

        # Build the flat index array: for each step, the delta-offset row indices.
        # When padding, also compute per-step padded masks before clamping.
        flat_indices = []
        for step_idx in step_indices:
            for delta in config.delta_indices:
                idx = step_idx + delta
                if allow_padding:
                    idx = max(0, min(idx, n_rows - 1))
                flat_indices.append(idx)

        if allow_padding:
            # Per-modality padded mask: 1 = frame exceeded episode end (clamped to last frame)
            for i, step_idx in enumerate(step_indices):
                raw_indices = [step_idx + delta for delta in config.delta_indices]
                results[i][f"_padded_mask_{modality}"] = np.array(
                    [1 if raw > n_rows - 1 else 0 for raw in raw_indices],
                    dtype=np.int64,
                )

        assert all(idx >= 0 for idx in flat_indices), (
            f"Indices to load are negative: {flat_indices}. "
            "Please allow padding to resolve this."
        )

        for i in range(n_steps):
            structlog.get_logger().debug(
                "extract_step_data",
                modality=modality,
                indices_to_load=flat_indices[i * horizon : (i + 1) * horizon],
            )

        for key in config.modality_keys:
            col_name = f"{modality}.{key}"
            if col_name not in episode_data.columns:
                assert modality == "rl_info", f"{col_name} not found in episode data"
                continue

            col = episode_data[col_name]
            # One gather for all steps × all deltas.
            gathered = col.gather(flat_indices)

            if modality in ["state", "action", "rl_info", "tactile"]:
                # Array column: .to_numpy() gives (n_steps * horizon, width).
                all_np = gathered.to_numpy().astype(np.float32)
                # Reshape to (n_steps, horizon, width), then assign per step.
                all_np = all_np.reshape(n_steps, horizon, -1)
                for i in range(n_steps):
                    results[i][modality][key] = all_np[i]
            else:
                # Video / language: split the flat list into per-step chunks.
                all_list = gathered.to_list()
                for i in range(n_steps):
                    results[i][modality][key] = all_list[i * horizon : (i + 1) * horizon]

    # Build VLAStepData objects.
    vla_steps = []
    for i in range(n_steps):
        r = results[i]
        language_data = r.get("language", {})
        text = language_data[next(iter(language_data.keys()))][0]

        # Collect padded masks into metadata
        metadata = {}
        for key in list(r.keys()):
            if key.startswith("_padded_mask_"):
                metadata[key] = r[key]
        metadata["reward"] = np.asarray(float(step_indices[i]) / max(n_rows, 1), dtype=np.float32)

        vla_steps.append(
            types.VLAStepData(
                images=r.get("video", {}),
                states=r.get("state", {}),
                actions=r.get("action", {}),
                languages=language_data,
                tactile=r.get("tactile", {}),
                deform=r.get("deform", {}),
                text=text,
                rl_info=r.get("rl_info"),
                embodiment=embodiment_tag,
                metadata=metadata,
            )
        )
    return vla_steps


class ShardedSingleStepDataset(types.ShardedDataset):
    """Single-step dataset that creates shards from individual timesteps across episodes.

    This dataset implementation provides step-level data access for VLA training by:
    1. Loading episodes using LeRobotEpisodeLoader
    2. Splitting episodes into individual timesteps
    3. Organizing timesteps into balanced shards for efficient loading
    4. Supporting episode subsampling for data efficiency

    The sharding strategy ensures balanced shard sizes while maintaining randomization
    across episodes and timesteps within episodes. Each shard contains a mix of
    timesteps from different episodes to improve training diversity.

    Key features:
    - Step-level data access (vs episode-level)
    - Balanced sharding for consistent batch sizes
    - Episode subsampling via sampling rate
    - Integration with LeRobot data format
    - Support for multi-modal data (video, state, action, language)

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        hydrator: Optional hydrator client for remote (S3) file access.
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for video backend
        resample_on_invalid_frames: If True, clamp invalid video frame indices and retry; if False, raise.
        shard_size: Target number of timesteps per shard
        episode_sampling_rate: Fraction of episode timesteps to use (for efficiency)
        seed: Random seed for reproducible sharding and sampling
        allow_padding: Whether to allow padding of indices to valid range [0, max_length - 1]
        data_filter: Optional online data filter applied during episode loading.
        step_indices_filter: Optional per-episode filter specifying which step indices to include.
        episode_loader_kwargs: Optional backend-specific keyword arguments for create_episode_loader.
        enable_batched_step_extraction: Whether get_shard may extract all sampled
            steps from one episode in one batched gather. Enabled only for
            DeltaCAT PackDS by DatasetFactory.
        load_step_filter: Whether to load ``meta/step_filter.jsonl`` when no explicit
            ``step_indices_filter`` is provided.

    Example:
        >>> dataset = ShardedSingleStepDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(delta_indices=list(range(8)), modality_keys=["joint_velocities"])
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed timesteps
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
        shard_size: int = 2**10,  # 1024 steps
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
        data_filter: types.OnlineDataFilter | None = None,
        step_indices_filter: dict[int, np.ndarray] | None = None,
        episode_loader_kwargs: dict[str, Any] | None = None,
        enable_batched_step_extraction: bool = False,
        load_step_filter: bool = False,
    ):
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.hydrator = hydrator
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.resample_on_invalid_frames = resample_on_invalid_frames
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.processor = None
        self.enable_batched_step_extraction = enable_batched_step_extraction
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
        # Max delta index across all modalities so step_index + delta stays in [0, episode_len-1]
        self._max_delta_index = max(max(c.delta_indices) for c in modality_configs.values())
        episode_loader_kwargs = dict(episode_loader_kwargs or {})

        # Initialize episode loader for data access
        self.episode_loader = create_episode_loader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            resample_on_invalid_frames=resample_on_invalid_frames,
            data_filter=data_filter,
            hydrator=hydrator,
            **episode_loader_kwargs,
        )

        # A LanceDS loader serves its view as the filter; a store refuses data.load_step_filter
        # (factory._create_sharded_dataset).
        self.step_indices_filter = (
            step_indices_filter
            if step_indices_filter is not None
            else getattr(self.episode_loader, "step_indices_filter", None)
        )
        if self.step_indices_filter is None and load_step_filter:
            self.step_indices_filter = load_step_indices_filter(dataset_path, self.episode_loader)
        self.dataset_index = -1  # Set by ShardedMixtureDataset after construction

        # Create balanced shards from episode timesteps
        self.shard_dataset()

    def shard_dataset(self):
        """Create balanced shards by distributing episode timesteps across shards.

        The sharding process:
        1. Shuffle episode order for randomization
        2. Split each episode into multiple sub-sequences based on sampling rate
        3. Distribute sub-sequences across shards to balance shard sizes
        4. Use greedy assignment to minimize shard size variance

        This approach ensures:
        - Balanced shard sizes for consistent training batches
        - Diversity within shards (mix of episodes and timesteps)
        - Reproducible sharding based on seed
        """
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        num_splits = int(1 / self.episode_sampling_rate)

        assert (
            len(shuffled_episode_indices) > 0
        ), f"No valid trajectories found for dataset {self.dataset_path}"

        # Filter out episodes with zero effective length (too short for the
        # action horizon / delta indices). Without this, empty sub-sequences
        # get distributed to shards, causing the "All shards must have length
        # greater than 0" assertion to fail.
        if self.step_indices_filter is not None:
            effective_lengths = {
                idx: self._get_episode_step_count(idx) for idx in shuffled_episode_indices
            }
        else:
            effective_lengths = {
                idx: self.get_effective_episode_length(idx) for idx in shuffled_episode_indices
            }
        valid_episode_indices = np.array(
            [idx for idx in shuffled_episode_indices if effective_lengths[idx] > 0]
        )
        n_skipped = len(shuffled_episode_indices) - len(valid_episode_indices)
        if n_skipped > 0:
            logger.warning(
                "Warning: Skipping %s episodes with zero effective length in dataset %s",
                n_skipped,
                self.dataset_path,
            )
        shuffled_episode_indices = valid_episode_indices

        assert (
            len(shuffled_episode_indices) > 0
        ), f"No valid episodes with positive effective length for dataset {self.dataset_path}"

        # Calculate total timesteps and required number of shards
        total_steps = np.sum([effective_lengths[idx] for idx in shuffled_episode_indices]).astype(
            int
        )
        num_shards = np.ceil(total_steps / self.shard_size).astype(int)

        # Ensure num_shards doesn't exceed the number of shards that will be created
        # Otherwise, some shards will remain empty
        max_shards = len(shuffled_episode_indices) * num_splits
        if num_shards > max_shards:
            logger.warning(
                "Warning: Reducing num_shards from %s to %s "
                "(dataset has only %s episodes × %s splits = %s sub-sequences). "
                "To avoid this, try increasing `shard_size` or decreasing `episode_sampling_rate`.",
                num_shards,
                max_shards,
                len(shuffled_episode_indices),
                num_splits,
                max_shards,
            )
            num_shards = max_shards

        # Initialize shard containers
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # Use a min-heap for O(log n) shard assignment instead of O(n) argmin
        # Heap entries: (current_length, shard_index)
        shard_heap = [(0, i) for i in range(num_shards)]
        heapq.heapify(shard_heap)

        # Distribute episode sub-sequences across shards
        for ep_idx in shuffled_episode_indices:
            if self.step_indices_filter is not None:
                if ep_idx in self.step_indices_filter:
                    step_indices = self.step_indices_filter[ep_idx].copy()
                else:
                    continue
            else:
                step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))
            if len(step_indices) == 0:
                continue
            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                if len(split_step_indices) == 0:
                    continue
                # Pop shard with minimum current length (greedy balancing) - O(log n)
                current_length, shard_index = heapq.heappop(shard_heap)
                sharded_episodes[shard_index].append((ep_idx, split_step_indices))
                new_length = current_length + len(split_step_indices)
                shard_lengths[shard_index] = new_length
                # Push back with updated length - O(log n)
                heapq.heappush(shard_heap, (new_length, shard_index))

        # Validate shard creation
        assert all(
            shard_lengths[i] > 0 for i in range(num_shards)
        ), "All shards must have length greater than 0"

        logger.info("Generated %s shards for dataset %s", num_shards, self.dataset_path)
        logger.info(
            "Total steps: %s, average shard length: %s, shard length std: %s",
            total_steps,
            total_steps / num_shards,
            np.std(shard_lengths),
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length so all modalities stay in bounds.

        Without padding: step_index + max(delta_indices) <= episode_len - 1,
        so valid step indices = max(0, episode_len - max_delta_index).

        With allow_padding: all step indices [0, episode_len) are valid.
        Out-of-bound indices are clamped and marked in _padded_mask_{modality}.
        """
        original_length = self.episode_loader.get_episode_length(episode_index)
        if self.allow_padding:
            return original_length
        return max(0, original_length - self._max_delta_index)

    def _get_episode_step_count(self, episode_index: int) -> int:
        """Number of steps for an episode, respecting step_indices_filter if set."""
        if self.step_indices_filter is not None:
            if episode_index in self.step_indices_filter:
                return len(self.step_indices_filter[episode_index])
            return 0
        return self.get_effective_episode_length(episode_index)

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_lengths)

    def get_step_data(
        self,
        episode_data: types.EpisodeData,
        step_index: int,
        step_data_id: types.VLAStepDataId | None = None,
    ) -> types.VLAStepData:
        """Extract a single timestep's VLAStepData from episode data.

        Args:
            episode_data: Complete episode DataFrame from LeRobotEpisodeLoader
            step_index: Timestep index within the episode to extract
            step_data_id: Optional identifier for this datapoint (dataset/shard/datapoint indices)

        Returns:
            VLAStepData for the requested timestep
        """
        # Always allow padding in extraction
        # `allow_padding` only controls how `step_index`` is sampled
        vla_step_data = extract_step_data(
            episode_data, step_index, self.modality_configs, self.embodiment_tag, allow_padding=True
        )
        if step_data_id is not None:
            vla_step_data.metadata[types.VLAStepDataId.STEP_DATA_ID_KEY] = step_data_id
        return vla_step_data

    def _get_datapoint_from_step_data(self, vla_step_data: types.VLAStepData) -> dict:
        """Apply the processor to a VLAStepData to produce a model-ready datapoint."""
        assert self.processor is not None, "Processor must be set before getting datapoints"
        messages = [{"type": types.MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)

    def get_datapoint(self, episode_data: types.EpisodeData, step_index: int) -> dict:
        """Extract and process a single timestep into a model-ready datapoint."""
        vla_step_data = self.get_step_data(episode_data, step_index)
        return self._get_datapoint_from_step_data(vla_step_data)

    def get_shard_length(self, idx: int) -> int:
        """Get the number of timesteps in a specific shard."""
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        """Load and process all timesteps in a specific shard.

        Loads the required episodes and extracts all timesteps assigned to this shard,
        applying the configured processor to each timestep.  Includes profiler
        instrumentation and per-sample timing annotations so callers can diagnose
        parquet vs video loading costs.

        Args:
            idx: Shard index to load

        Returns:
            List of processed timesteps ready for model training

        Raises:
            RuntimeError: If an episode fails to load
        """
        with bound_contextvars(
            component="get_shard",
            dataset_path=str(self.dataset_path),
            shard_idx=idx,
            dataset_idx=self.dataset_index,
        ):
            start_t = time.perf_counter()
            try:
                from groot.core.data.packds import deltacat_sql_loader as _dsl

                _shard_timing = _dsl._shard_timing_enabled()  # noqa: SLF001
            except Exception:  # noqa: BLE001
                _dsl = None
                _shard_timing = False
            if _shard_timing:
                _dsl.timing_snapshot_reset()
            episodes = self.sharded_episodes[idx]
            ep_indices = [ep_idx for (ep_idx, _) in episodes]
            # Prefetch all video files for this shard so downloads run async before we touch each episode
            self.episode_loader.register_video_files_for_episodes(ep_indices)
            _get_num_cameras = getattr(self.episode_loader, "get_num_cameras", None)
            num_cameras = _get_num_cameras() if _get_num_cameras is not None else 0
            datapoints = []
            total_frames = 0
            total_episode_length = 0
            for sample_idx, (ep_idx, step_indices) in enumerate(episodes):
                episode_length = self.episode_loader.get_episode_length(ep_idx)
                with bound_contextvars(
                    sample_idx=sample_idx,
                    episode_index=ep_idx,
                    episode_length=episode_length,
                    num_cameras=num_cameras,
                ):
                    sample_t0 = time.perf_counter()
                    try:
                        episode_data = self.episode_loader[ep_idx]
                        total_episode_length += episode_length
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"Failed to load episode {ep_idx} from dataset {self.dataset_path}: {e}"
                        ) from e
                    step_indices_list = [int(step_index) for step_index in step_indices]
                    if self.enable_batched_step_extraction:
                        vla_step_batch = extract_step_data_batch(
                            episode_data,
                            step_indices_list,
                            self.modality_configs,
                            self.embodiment_tag,
                            allow_padding=True,
                        )
                    else:
                        vla_step_batch = [
                            self.get_step_data(episode_data, step_index)
                            for step_index in step_indices_list
                        ]
                    for step_index, vla_step_data in zip(
                        step_indices_list,
                        vla_step_batch,
                        strict=True,
                    ):
                        with bound_contextvars(step_index=step_index):
                            step_data_id = types.VLAStepDataId(
                                dataset_idx=self.dataset_index,
                                episode_idx=ep_idx,
                                step_idx=step_index,
                                epoch_index=structlog.contextvars.get_contextvars().get(
                                    "epoch_index", 0
                                ),
                                shard_index=idx,
                            )
                            vla_step_data.metadata[types.VLAStepDataId.STEP_DATA_ID_KEY] = (
                                step_data_id
                            )
                            datapoint_t0 = time.perf_counter()
                            total_frames += vla_step_data.num_frames
                            datapoints.append(self._get_datapoint_from_step_data(vla_step_data))
                            structlog.get_logger().debug(
                                "get_datapoint_completed",
                                num_frames=vla_step_data.num_frames,
                                duration_ms=round((time.perf_counter() - datapoint_t0) * 1000, 2),
                            )
                    structlog.get_logger().debug(
                        "get_sample_completed",
                        step_indices=step_indices,
                        duration_ms=round((time.perf_counter() - sample_t0) * 1000, 2),
                    )

            # Release video files now that all episodes in this shard are loaded,
            # allowing LRU eviction of files no longer needed.
            self.episode_loader.release_video_files_for_episodes(ep_indices)
            structlog.get_logger().info(
                "get_shard_completed",
                shard_length=self.shard_lengths[idx],
                total_frames=total_frames,
                total_episode_length=total_episode_length,
                duration_ms=round((time.perf_counter() - start_t) * 1000, 2),
                num_cameras=num_cameras,
            )
            if _shard_timing and _dsl is not None:
                _snap = _dsl.timing_snapshot_reset()
                print(  # noqa: T201
                    f"[shard-timing] episodes={len(ep_indices)} "
                    f"steps={self.shard_lengths[idx]} "
                    f"wall={time.perf_counter() - start_t:.1f}s "
                    f"fetches={_snap['n_fetch']} fetch_sum={_snap['fetch_s']:.1f}s "
                    f"decode_sum={_snap['decode_s']:.1f}s "
                    f"fetch_MB={_snap['fetch_bytes'] / 1024**2:.0f}",
                    flush=True,
                )

        return datapoints

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()

    def set_dataset_statistics(self, statistics: dict[str, Any]) -> None:
        """Install externally computed statistics on the underlying episode loader."""
        self.episode_loader.set_dataset_statistics(statistics)

    def get_initial_actions(self):
        """Get initial actions from the underlying episode loader."""
        return self.episode_loader.get_initial_actions()
