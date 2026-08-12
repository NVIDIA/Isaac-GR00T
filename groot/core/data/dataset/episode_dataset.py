# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Sequence
from typing import Any

import numpy as np
from structlog.contextvars import bound_contextvars
from torch.utils import data

from groot.core.data import types
from groot.core.data.dataset import sharded_single_step_dataset
from groot.core.data.dataset.episode_loader_factory import create_episode_loader
from groot.core.data.hydrator import client_protocol


class EpisodeDataset(data.Dataset):
    """Episode-level dataset that structures robot demonstration episodes as message sequences.

    This dataset transforms LeRobot format data into structured episode sequences suitable
    for training VLA models that process episodes as conversations. Each episode is
    converted into a sequence of messages with clear boundaries and step-by-step content.

    The dataset creates multiple views of each episode based on the action horizon,
    allowing different starting points within episodes for data augmentation. This
    increases the effective dataset size while providing varied training contexts.

    Key features:
    - Episode-to-message sequence conversion with proper formatting
    - Integration with LeRobotEpisodeLoader for multi-modal data access
    - Support for start/step/end episode message types
    - Embodiment tag integration for multi-robot training

    Message structure:
    - START_OF_EPISODE: Marks the beginning of an episode sequence
    - EPISODE_STEP: Contains VLAStepData with multi-modal observations and actions
    - END_OF_EPISODE: Marks the completion of an episode sequence

    Args:
        dataset_path: Path to LeRobot format dataset (local path or S3 URI,
            e.g. s3://bucket/key). For S3, hydrator is required.
        embodiment_tag: Robot embodiment identifier for multi-embodiment training
        modality_configs: Configuration for each data modality (video, state, action, language)
        hydrator: Hydrator for file access (required for S3 dataset_path, optional for local).
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend
        resample_on_invalid_frames: If True, clamp invalid video frame indices and retry; if False, raise.
        filter: Optional filter of the data.
        max_len: Maximum length of a subsequence to be sampled from the dataset.

    Example:
        >>> dataset = EpisodeDataset(
        ...     dataset_path="s3://bucket/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(delta_indices=list(range(8)), modality_keys=["joint_velocities"]),
        ...         "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"])
        ...     },
        ...     hydrator=hydrator,
        ... )
        >>> episode_messages = dataset[0]  # Get first episode as message sequence
        >>> print(f"Episode has {len(episode_messages)} messages")
    """

    def __init__(
        self,
        dataset_path: str,
        embodiment_tag: types.EmbodimentTag,
        modality_configs: dict[str, types.ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        resample_on_invalid_frames: bool = False,
        data_filter: types.OnlineDataFilter | None = None,
        max_len: int | None = None,
        step_stride: int | None = None,
        hydrator: client_protocol.HydratorClientProtocol | None = None,
        episode_loader_kwargs: dict[str, Any] | None = None,
        dataset_idx: int = 0,
    ):
        """Initialize episode dataset with LeRobot data loader and configuration.

        Sets up the underlying data loader and computes the action horizon for
        episode augmentation. The action horizon determines how many different
        starting points are available within each episode.

        Args:
            dataset_path: Path to the dataset root (local or S3,
                e.g. s3://bucket/key). For S3, hydrator is required.
            embodiment_tag: Embodiment identifier for this dataset
            modality_configs: Dictionary mapping modality names to their configurations
            hydrator: Hydrator for file access (required for S3, optional for local).
            video_backend: Backend for video decoding
            video_backend_kwargs: Additional backend-specific arguments
            resample_on_invalid_frames: If True, clamp invalid video frame indices
                and retry; if False, raise.
            data_filter: Optional online filter applied to the data.
            max_len: Maximum length of a subsequence to be sampled from the dataset.
            step_stride: Step interval between trajectory samples. Defaults to the
                action horizon, preserving non-overlapping action sequences.
            episode_loader_kwargs: Extra keyword arguments forwarded to the episode loader.
            dataset_idx: Index of this dataset within a multi-dataset mixture.
        """
        self.dataset_path = dataset_path
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.max_len = max_len
        self.dataset_idx = dataset_idx

        # Initialize the underlying episode loader. ``episode_loader_kwargs``
        # carries backend-specific options (e.g. DeltaCAT episode_query /
        # steps_query / key_remap) through to create_episode_loader, mirroring
        # ShardedSingleStepDataset so trajectory mode can read DeltaCAT tables.
        episode_loader_kwargs = dict(episode_loader_kwargs or {})
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

        # Compute action horizon for episode formatting
        self.action_horizon = len(self.modality_configs["action"].delta_indices)
        self.step_stride = self.action_horizon if step_stride is None else step_stride
        if self.step_stride <= 0:
            raise ValueError(f"step_stride must be positive, got {self.step_stride}")

        # Lazy window index: instead of materializing one (episode_idx, start_idx)
        # tuple per stride-1 window (O(total_frames) Python tuples -> multi-GB
        # host RAM at init for long-episode trajectory datasets), store only two
        # per-episode int arrays (O(num_episodes)) plus a prefix-sum, and resolve
        # a global window index -> (episode_idx, start_idx) on demand via
        # searchsorted (see _lookup_window).
        num_episodes = len(self.episode_loader.episodes_metadata)
        ep_num_windows = np.empty(num_episodes, dtype=np.int64)
        # Per-episode "recorded length" (max_len for windowed episodes, else the
        # raw episode length), shared by all of an episode's windows.
        ep_recorded_len = np.empty(num_episodes, dtype=np.int64)
        for episode_idx, ep_meta in enumerate(self.episode_loader.episodes_metadata):
            ep_len = int(ep_meta["length"])
            if self.max_len is None:
                ep_num_windows[episode_idx] = 1
                ep_recorded_len[episode_idx] = ep_len
            else:
                effective_len = max(0, ep_len - self.action_horizon)
                if effective_len > self.max_len:
                    ep_num_windows[episode_idx] = effective_len - self.max_len + 1
                    ep_recorded_len[episode_idx] = self.max_len
                else:
                    ep_num_windows[episode_idx] = 1
                    ep_recorded_len[episode_idx] = ep_len
        self._ep_recorded_len = ep_recorded_len
        # Prefix sum over per-episode window counts: window range for episode e is
        # [_cum_windows[e], _cum_windows[e + 1]). Size num_episodes + 1.
        self._cum_windows = np.zeros(num_episodes + 1, dtype=np.int64)
        np.cumsum(ep_num_windows, out=self._cum_windows[1:])
        self._map_len = int(self._cum_windows[-1])

    def _lookup_window(self, map_idx: int) -> tuple[int, int]:
        """Resolve a global window index -> (episode_idx, start_idx) lazily.

        Equivalent to indexing the old materialized ``idx_to_episode_and_start_map``
        (episodes in order, start_idx 0..num_windows-1), but O(log num_episodes)
        with O(num_episodes) memory instead of O(num_windows).
        """
        episode_idx = int(np.searchsorted(self._cum_windows, map_idx, side="right") - 1)
        start_idx = int(map_idx - self._cum_windows[episode_idx])
        return episode_idx, start_idx

    def __len__(self):
        """Return total number of episode sequences available.

        The total count is number_of_episodes * step_stride, as each episode
        can be started from multiple different timestep phases based on the step stride.

        Returns:
            Total number of episode sequences (episodes * step_stride)
        """
        return self._map_len * self.step_stride

    def preload_indices(self, indices: Sequence[int]) -> None:
        preload_episodes = getattr(self.episode_loader, "preload_episodes", None)
        if not callable(preload_episodes):
            return
        episode_indices = []
        for idx in indices:
            map_idx = int(idx) // self.step_stride
            if map_idx < 0 or map_idx >= self._map_len:
                continue
            episode_idx, _start_idx = self._lookup_window(map_idx)
            episode_indices.append(int(episode_idx))
        preload_episodes(list(dict.fromkeys(episode_indices)))

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        """Get a complete episode sequence as a list of formatted messages.

        Converts a raw episode into a structured message sequence with proper
        start/step/end markers. The index determines both which episode to load
        and which timestep offset to start from within that episode.

        The message sequence structure:
        1. START_OF_EPISODE message (empty content)
        2. Multiple EPISODE_STEP messages with VLAStepData content
        3. END_OF_EPISODE message (empty content)

        Args:
            idx: Global index across all episode sequences

        Returns:
            List of message dictionaries, each containing 'type' and 'content' fields.
            EPISODE_STEP messages contain VLAStepData with multi-modal observations.

        Example:
            >>> messages = dataset[42]
            >>> for msg in messages:
            ...     if msg["type"] == MessageType.EPISODE_STEP.value:
            ...         step_data = msg["content"]  # VLAStepData object
            ...         print(f"Step has {len(step_data.images)} camera views")
        """
        # Calculate global index into episode and offset within episode
        map_idx = idx // self.step_stride
        episode_idx, start_idx = self._lookup_window(map_idx)
        offset = idx % self.step_stride
        episode_length = self.get_effective_episode_length(episode_idx)

        with bound_contextvars(
            episode_index=int(episode_idx),
            effective_episode_length=int(episode_length),
            start_idx=int(start_idx),
            offset=int(offset),
            action_horizon=int(self.action_horizon),
            step_stride=int(self.step_stride),
        ):
            messages = []

            # Start of episode marker
            messages.append(
                {
                    "type": types.MessageType.START_OF_EPISODE.value,
                    "content": "",
                }
            )

            # Load the episode (window). When the loader supports window-scoped reads
            # (DeltaCAT) and there is no episode-level filter, decode ONLY the
            # [start_idx, start_idx+max_len) window instead of the whole episode --
            # byte-identical output, but ~14x less video decode for long episodes.
            # This is the N2-TTT trajectory fix: whole-episode decode made shard
            # builds ~21 min and the run died in NCCL-barrier timeout at warmup.
            # start_idx > 0 only occurs when effective_len > max_len (see __init__),
            # so for short episodes start_idx == 0 and the window == the full episode.
            use_window_scoped = (
                self.max_len is not None
                and getattr(self.episode_loader, "filter", None) is None
                and hasattr(self.episode_loader, "get_episode_window")
            )
            if use_window_scoped:
                episode_data = self.episode_loader.get_episode_window(
                    episode_idx, start_idx, self.max_len
                )
            else:
                episode_data = self.episode_loader[episode_idx]
                # Handle max_len capping: select a contiguous subsequence
                if self.max_len is not None and len(episode_data) > self.max_len:
                    episode_data = episode_data.slice(start_idx, self.max_len)

            # Guard against metadata/data length mismatch: clamp episode_length to
            # actual rows so extract_step_data never sees n_rows=0 with step_idx>0.
            # Metadata (episodes.jsonl) can overcount if parquet is truncated or
            # a filter removed rows after the window was planned.
            episode_length = min(
                episode_length, max(0, len(episode_data) - self.action_horizon + 1)
            )

            # Generate episode steps starting from the specified offset. By default
            # steps are spaced by action_horizon to provide non-overlapping action
            # sequences; step_stride can intentionally sample denser overlapping
            # chunks.
            empty_episode = True
            for step_idx in range(offset, episode_length, self.step_stride):
                with bound_contextvars(step_index=int(step_idx)):
                    empty_episode = False
                    content = sharded_single_step_dataset.extract_step_data(
                        episode_data,
                        step_idx,
                        self.modality_configs,
                        self.embodiment_tag,
                        allow_padding=True,
                    )
                    content.metadata[types.VLAStepDataId.STEP_DATA_ID_KEY] = types.VLAStepDataId(
                        dataset_idx=self.dataset_idx,
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                    )
                    messages.append(
                        {
                            "type": types.MessageType.EPISODE_STEP.value,
                            "content": content,
                        }
                    )

            if empty_episode:
                # Do not return any messages if the episode is empty.
                return []

            # End of episode marker
            messages.append(
                {
                    "type": types.MessageType.END_OF_EPISODE.value,
                    "content": "",
                }
            )

            return messages

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length accounting for action horizon."""
        original_length = int(self._ep_recorded_len[episode_index])
        action_delta_indices = self.modality_configs["action"].delta_indices
        action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
        return max(0, original_length - action_horizon + 1)

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader.

        Provides access to normalization statistics computed from the underlying
        LeRobot dataset, enabling consistent data preprocessing across different
        dataset implementations.

        Returns:
            Dictionary containing dataset statistics with structure:
            {modality: {joint_group: {stat_type: values}}}
        """
        return self.episode_loader.get_dataset_statistics()

    def set_dataset_statistics(self, statistics: dict) -> None:
        """Install externally computed statistics on the underlying episode loader.

        Lets the DeltaCAT sampled-stats preflight push q01/q99 percentile stats
        into the loader for trajectory-mode training (mirrors
        ShardedSingleStepDataset.set_dataset_statistics).
        """
        self.episode_loader.set_dataset_statistics(statistics)
