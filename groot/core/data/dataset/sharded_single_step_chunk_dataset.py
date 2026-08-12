# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Chunk-based single-step dataset for DreamZero-style training.

This module provides a dataset class that samples multi-chunk temporal windows
with aligned video, action, and state data across modalities.
"""

from typing import Any

import numpy as np

from groot.core.data import types as data_types
from groot.core.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from groot.core.data.types import (
    EmbodimentTag,
    ModalityConfig,
    VLAStepData,
)


def extract_chunk_data(
    episode_data: data_types.EpisodeData,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    num_chunks: int = 16,
) -> VLAStepData:
    """
    Extract chunk-based data from an episode with cross-modality alignment.

    This implements the sampling logic from ShardedLeRobotGlobalLangSingleActionChunkDataset:
    - Uses delta_indices from each modality config to define per-chunk sampling pattern
    - Expands bidirectionally from step_index to collect up to num_chunks chunks
    - Video is processed first to determine actual chunk count (due to kn+1 constraint)
    - All other modalities are limited to the same number of chunks for alignment

    The chunk_length is inferred from the action modality's delta_indices span.

    Args:
        episode_data: DataFrame with columns like "video.front_cam", "state.joint_pos", etc.
        step_index: The anchor step index to sample around
        modality_configs: Configuration for each modality with delta_indices and keys
        embodiment_tag: Embodiment identifier
        num_chunks: Maximum number of chunks to sample (default 16)

    Returns:
        VLAStepData with aligned video/state/action chunks

    Example:
        With num_chunks=16 and typical configs:
        - Video delta_indices=[0, 6, 12, 18, 24, 30, 36, 42] -> 8 frames/chunk, 129 total (8n+1)
        - Action delta_indices=list(range(48)) -> 48 frames/chunk, 768 total
        - State delta_indices=[0] -> 1 frame/chunk, 16 total
    """
    # step_index may arrive as numpy.int64 (from sharded step arrays); polars
    # scalar indexing rejects numpy ints, so coerce at the boundary.
    step_index = int(step_index)
    trajectory_length = episode_data.height
    step_data: dict[str, dict[str, Any]] = {}

    # =========================================================================
    # Step 1: Infer chunk_length from action modality's delta_indices
    # =========================================================================
    video_deltas = modality_configs["video"].delta_indices
    video_deltas_diff = np.diff(video_deltas)
    assert np.all(video_deltas_diff == video_deltas_diff[0]), "Video deltas must be constant"
    stride = video_deltas_diff[0]
    chunk_length = (
        len(video_deltas) * stride
    )  # NOTE(FY): for examples, if video_deltas=[0, 6, 12, 18, 24, 30, 36, 42], then chunk_length=8*6=48

    # =========================================================================
    # Step 2: Compute initial chunk anchors via bidirectional expansion
    # =========================================================================
    # Mask indices with different language
    language_keys = modality_configs["language"].modality_keys
    sampling_mask = np.ones(trajectory_length, dtype=bool)
    for key in language_keys:
        col = episode_data[f"language.{key}"]
        ref_value = col[step_index]
        sampling_mask &= (col == ref_value).to_numpy()
    anchor_indices = _compute_chunk_anchors(
        step_index=step_index,
        sampling_mask=sampling_mask,
        chunk_length=chunk_length,
        max_chunks=num_chunks,
    )
    assert (
        len(anchor_indices) > 0
    ), f"No valid chunks found for step_index={step_index}, trajectory_length={trajectory_length}, chunk_length={chunk_length}"

    # =========================================================================
    # Step 3: Sample each modality using its delta_indices pattern per chunk
    # =========================================================================
    for modality, config in modality_configs.items():
        step_data[modality] = {}

        # Build full index list by applying delta_indices at each anchor
        indices = []
        for anchor in anchor_indices:
            for delta in config.delta_indices:
                idx = anchor + delta
                # Clamp to valid range
                idx = max(0, min(idx, trajectory_length - 1))
                indices.append(idx)

        # Special case: video needs 8n+1 format (add one extra frame at the end)
        if modality == "video":
            indices.append(indices[-1] + stride)  # Add one extra frame at the end

        # Extract data for each key in the modality
        for key in config.modality_keys:
            col_name = f"{modality}.{key}"
            if col_name not in episode_data.columns:
                if modality == "rl_info":
                    # BC data doesn't have rl_info, skip silently
                    continue
                raise ValueError(f"{col_name} not found in episode data columns")

            gathered = episode_data[col_name].gather(indices)

            if modality in ["state", "action", "rl_info"]:
                # pl.Array column → (n, width) ndarray.
                step_data[modality][key] = gathered.to_numpy().astype(np.float32)
            else:
                # Keep as lists for other modalities (video, language)
                step_data[modality][key] = gathered.to_list()

    # =========================================================================
    # Step 4: Parse extracted data into VLAStepData structure
    # =========================================================================
    video_data = step_data.get("video", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    rl_info_data = step_data.get("rl_info", None) if "rl_info" in step_data else None

    # Get text from language data (deprecated but still supported)
    text = ""
    if language_data:
        first_key = list(language_data.keys())[0]
        if language_data[first_key]:
            text = language_data[first_key][0]

    vla_step_data = VLAStepData(
        images=video_data,
        states=state_data,
        actions=action_data,
        languages=language_data,
        text=text,
        rl_info=rl_info_data,
        embodiment=embodiment_tag,
    )
    return vla_step_data


def _compute_chunk_anchors(
    step_index: int,
    sampling_mask: np.ndarray,
    chunk_length: int,
    max_chunks: int,
) -> list[int]:
    """
    Compute chunk anchor indices via bidirectional expansion from step_index.

    Each chunk spans [anchor: anchor + chunk_length]. We expand outward
    from step_index in chunk_length steps, alternating backward and forward.

    Args:
        step_index: Starting anchor index
        sampling_mask: Mask of valid indices
        chunk_length: Number of frames per chunk
        max_chunks: Maximum number of chunks to return

    Returns:
        Sorted list of anchor indices
    """
    anchors: list[int] = []

    def can_add_anchor(anchor: int) -> bool:
        """Check if anchor allows a full chunk within trajectory bounds.
        e.g. if chunk_length=48, anchor=0, then anchor+chunk_length=48 needs to be less than trajectory_length, so we can get the 8n+1=49th frame.
        """
        if anchor < 0 or anchor + chunk_length >= len(sampling_mask):
            return False
        return sampling_mask[anchor]

    # Start with the anchor at step_index
    if can_add_anchor(step_index):
        anchors.append(step_index)

    # Expand bidirectionally in chunk_length steps
    step = 1
    back_done, fwd_done = False, False

    while len(anchors) < max_chunks and (not back_done or not fwd_done):
        # Backward expansion
        if not back_done:
            back_anchor = step_index - chunk_length * step
            if back_anchor < 0:
                back_done = True
            elif not sampling_mask[back_anchor]:
                back_done = True
            elif can_add_anchor(back_anchor):
                anchors.append(back_anchor)

        if len(anchors) >= max_chunks:
            break

        # Forward expansion
        if not fwd_done:
            fwd_anchor = step_index + chunk_length * step
            if fwd_anchor >= len(sampling_mask):
                fwd_done = True
            elif not sampling_mask[fwd_anchor]:
                fwd_done = True
            elif can_add_anchor(fwd_anchor):
                anchors.append(fwd_anchor)

        step += 1

    # Sort for consistent ordering
    return sorted(anchors)


class ShardedSingleStepChunkDataset(ShardedSingleStepDataset):
    """
    Single-step dataset with chunk-based temporal sampling for DreamZero-style training.

    This dataset extends ShardedSingleStepDataset to sample multi-chunk temporal windows
    instead of simple delta-indexed frames. The key differences are:

    1. **Chunk expansion**: Samples multiple consecutive chunks (up to num_chunks)
       by expanding bidirectionally from the sampled step index.

    2. **Cross-modality alignment**: All modalities (video, action, state) share the
       same chunk anchors, ensuring temporal alignment.

    3. **Per-modality patterns**: Each modality uses its own delta_indices to define
       the within-chunk sampling pattern:
       - Video: e.g., [0, 6, 12, 18, 24, 30, 36, 42] for 8 frames at stride 6
       - Action: e.g., list(range(48)) for 48 consecutive frames
       - State: e.g., [0] for just the chunk anchor

    4. **Video kn+1 format**: Automatically ensures video frames follow kn+1 format
       required by video VAE encoding.

    Args:
        num_chunks: Maximum number of chunks to sample per datapoint (default 16)
        **kwargs: All other arguments passed to ShardedSingleStepDataset

    Example:
        >>> dataset = ShardedSingleStepChunkDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(
        ...             delta_indices=[0, 6, 12, 18, 24, 30, 36, 42],
        ...             modality_keys=["front_cam"]
        ...         ),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(48)),
        ...             modality_keys=["joint_positions"]
        ...         ),
        ...         "state": ModalityConfig(
        ...             delta_indices=[0],
        ...             modality_keys=["joint_positions"]
        ...         ),
        ...         "language": ModalityConfig(
        ...             delta_indices=[0],
        ...             modality_keys=["instruction"]
        ...         ),
        ...     },
        ...     num_chunks=16,
        ...     shard_size=1024,
        ... )
    """

    def __init__(
        self,
        *args,
        num_chunks: int = 4,
        **kwargs,
    ):
        """Initialize chunk-based dataset with num_chunks parameter."""
        # Set num_chunks before super().__init__ because sharding calls
        # get_effective_episode_length() which needs self.num_chunks.
        self.num_chunks = num_chunks
        super().__init__(*args, **kwargs)

    def get_step_data(
        self,
        episode_data: data_types.EpisodeData,
        step_index: int,
        step_data_id: data_types.VLAStepDataId | None = None,
    ) -> VLAStepData:
        """Extract chunk-based VLAStepData from episode data.

        Unlike the parent class which uses simple delta_indices, this method
        expands bidirectionally to sample multiple aligned chunks across modalities.
        """
        vla_step_data = extract_chunk_data(
            episode_data=episode_data,
            step_index=step_index,
            modality_configs=self.modality_configs,
            embodiment_tag=self.embodiment_tag,
            num_chunks=self.num_chunks,
        )
        if step_data_id is not None:
            vla_step_data.metadata[data_types.VLAStepDataId.STEP_DATA_ID_KEY] = step_data_id
        return vla_step_data

    def get_effective_episode_length(self, episode_index: int) -> int:
        """
        Get the effective episode length accounting for chunk-based sampling.

        Restricts valid step_index range so that bidirectional expansion from
        any valid index can always find exactly num_chunks chunks. A step_index
        in [0, effective_length) guarantees num_chunks forward-reachable anchors
        at [step_index, step_index+C, ..., step_index+(N-1)*C] where C=chunk_length.

        This ensures fixed tensor shapes every step, which is required for
        torch.compile(dynamic=False) to avoid recompilation.
        """
        original_length = self.episode_loader.get_episode_length(episode_index)
        action_deltas = self.modality_configs["action"].delta_indices
        chunk_length = max(action_deltas) - min(action_deltas) + 1
        # Need room for num_chunks chunks plus one trailing frame (8n+1 format)
        # Last valid anchor is at step_index + (num_chunks-1)*chunk_length,
        # and that anchor needs chunk_length frames ahead, so:
        # step_index + num_chunks * chunk_length < original_length
        return max(0, original_length - self.num_chunks * chunk_length)
