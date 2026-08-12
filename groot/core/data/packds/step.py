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
PackDSStep: Canonical dataclass for PackDS step data.

This module defines the core data types used for reading and writing
steps to PackDS datasets. All encoding/decoding logic for video,
state, and action data uses these types.

Types:
    PackDSStep: A single timestep (one row).
    PackDSStepBatch: A columnar batch of timesteps (multiple rows).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass(kw_only=True)
class PackDSStep:
    """Canonical data type for a single timestep in a PackDS dataset.

    This dataclass represents one row in a PackDS table, with support for:
    - Multiple video streams (JPEG encoded for storage, numpy for processing)
    - Multiple state vectors
    - Multiple action vectors (single-step per row)
    - Typed annotations (task, action_source, original_frame_index, etc.)

    Attributes:
        episode_id: String episode ID (required primary identifier, e.g., "ep_000001")
        episode_index: Optional compatibility-only integer episode index.
            Canonical PackDS storage and the base reader/writer APIs use
            ``episode_id`` instead.
        step_index: Step index within the episode
        has_valid_action: Whether this step has valid action data
        annotations: Dictionary of typed annotations. Common keys:
            - "task": str - Task description
            - "sub_task": str - Sub-task description
            - "action_source": str - Action source (e.g., "human", "model", "other")
            - "original_frame_index": int - Original frame index before filtering
        encoded_video: JPEG-encoded video frames (key -> bytes), stored in DB
        decoded_video: Decoded video frames (key -> (H, W, C) array), transient
        state: State vectors (key -> (dim,) array)
        action: Action vectors (key -> (dim,) array, single-step per row)

    Note:
        Action horizons (multiple timesteps) are gathered at read time by
        reading consecutive rows, not stored per-row.

    Examples:
        # Create a step for writing
        step = PackDSStep(
            episode_id="ep_000001",
            step_index=0,
            step_time_ns=0,
            original_step_time_ns=0,
            annotations={"task": "pick up the red cup", "action_source": "human"},
            decoded_video={"left_cam": frame_array},  # Will be encoded
            state={"joint_pos": np.array([0.1, 0.2, 0.3])},
            action={"joint_pos": np.array([0.2, 0.3, 0.4])},  # (dim=3)
        )

        # Encode video before writing
        from groot.core.data.packds.video import encode_steps, decode_steps
        encoded = encode_steps([step], encode_jpeg=True)
        ds.write_steps(encoded)

        # Read a step
        steps = ds.read_episode_steps("ep_000001")
        steps = decode_steps(steps)
        print(steps[0].decoded_video["left_cam"].shape)  # (H, W, 3)
    """

    episode_id: str
    episode_index: int | None = None
    step_index: int = 0
    step_time_ns: int
    original_step_time_ns: int
    has_valid_action: bool = True

    # Typed annotations (str, int, float, or bool values)
    annotations: dict[str, str | int | float | bool] = field(default_factory=dict)

    # Schema override for annotation columns. Keys are column names
    # (e.g., "annotation:action_source"), values are PyArrow types.
    # Declares that the column should exist with this type in the schema,
    # even if annotations doesn't have a value for the key.
    schema_override: dict[str, pa.DataType] = field(default_factory=dict)

    # Video data - encoded (JPEG bytes) or decoded (numpy arrays)
    encoded_video: dict[str, bytes] | None = None
    decoded_video: dict[str, np.ndarray] | None = None

    # Multi-frame video blob data
    encoded_video_blob: Optional[dict[str, bytes | None]] = (
        None  # key -> video blob (None if not group leader)
    )
    video_blob_frame_index: Optional[dict[str, int]] = None  # key -> index within video blob

    # State vectors - key -> (dim,) array
    state: dict[str, np.ndarray] = field(default_factory=dict)

    # Action vectors - key -> (dim,) array (single-step per row)
    action: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        """Validate step data."""
        from groot.core.data.packds import utils

        utils.validate_episode_id_for_write(self.episode_id)

        # Ensure state arrays are numpy arrays
        for key, value in self.state.items():
            if not isinstance(value, np.ndarray):
                self.state[key] = np.asarray(value, dtype=np.float32)

        # Ensure action arrays are numpy arrays
        for key, value in self.action.items():
            if not isinstance(value, np.ndarray):
                self.action[key] = np.asarray(value, dtype=np.float32)

    @property
    def video_keys(self) -> list[str]:
        """Get all video keys (from encoded, decoded, or video blob)."""
        keys: set[str] = set()
        if self.encoded_video:
            keys = keys | set(self.encoded_video.keys())
        if self.decoded_video:
            keys = keys | set(self.decoded_video.keys())
        if self.encoded_video_blob:
            keys = keys | set(self.encoded_video_blob.keys())
        if self.video_blob_frame_index:
            keys = keys | set(self.video_blob_frame_index.keys())
        return list(sorted(keys))

    @property
    def state_keys(self) -> list[str]:
        """Get all state keys."""
        return sorted(self.state.keys())

    @property
    def action_keys(self) -> list[str]:
        """Get all action keys."""
        return sorted(self.action.keys())

    def concat_actions(self, keys: list[str] | None = None) -> np.ndarray:
        """Concatenate action arrays into a single 1D vector.

        Args:
            keys: Action keys in concatenation order. If None, uses sorted keys.

        Returns:
            1D float32 array of concatenated actions.
        """
        if keys is None:
            keys = self.action_keys
        return np.concatenate([self.action[k].flatten() for k in keys])


@dataclass
class PackDSStepBatch:
    """Columnar batch of timesteps from a PackDS dataset.

    This is the return type of ``from_table()`` and the internal
    representation used by ``read_episode_steps()``. Data is stored in
    columnar form for efficient bulk access. Individual ``PackDSStep``
    objects can be extracted via indexing or iteration.

    Attributes:
        episode_index: Optional ``(N,)`` int64 compatibility array. Canonical
            PackDS batches may leave this as ``None``.
        step_index: (N,) int64 array of step indices.
        step_time_ns: (N,) int64 array of step times in nanoseconds.
        original_step_time_ns: (N,) int64 array of original step times in nanoseconds.
        has_valid_action: (N,) bool array.
        episode_id: List of N string episode IDs.
        video: Dict mapping video key -> list of N JPEG byte strings.
        state: Dict mapping state key -> (N, *shape) float32 ndarray.
        action: Dict mapping action key -> (N, *shape) float32 ndarray.
        annotations: Dict mapping annotation key -> list of N values.
    """

    episode_index: np.ndarray | None = None
    step_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    step_time_ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    original_step_time_ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    has_valid_action: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    episode_id: list[str] = field(default_factory=list)

    video: dict[str, list[bytes]] = field(default_factory=dict)
    state: dict[str, np.ndarray] = field(default_factory=dict)
    action: dict[str, np.ndarray] = field(default_factory=dict)
    annotations: dict[str, list] = field(default_factory=dict)
    schema_override: dict[str, pa.DataType] = field(default_factory=dict)

    # Multi-frame video blob data (optional)
    video_blob: dict[str, list[bytes | None]] = field(default_factory=dict)
    video_blob_frame_index: dict[str, np.ndarray] = field(default_factory=dict)

    # Decoded video frames (populated by decode_video() / decode_steps())
    decoded_video: dict[str, list[np.ndarray | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from groot.core.data.packds import utils

        for episode_id in self.episode_id:
            utils.validate_episode_id_for_write(episode_id)

    def __len__(self) -> int:
        return len(self.step_index)

    def __getitem__(self, i: int) -> PackDSStep:
        """Extract a single PackDSStep from the batch."""
        # Populate decoded_video from batch if available
        dec_video = None
        if self.decoded_video:
            dec_video = {}
            for k, v in self.decoded_video.items():
                if i < len(v) and v[i] is not None:
                    dec_video[k] = v[i]
            if not dec_video:
                dec_video = None

        return PackDSStep(
            episode_index=(None if self.episode_index is None else int(self.episode_index[i])),
            step_index=int(self.step_index[i]),
            step_time_ns=int(self.step_time_ns[i]),
            original_step_time_ns=int(self.original_step_time_ns[i]),
            has_valid_action=bool(self.has_valid_action[i]),
            episode_id=self.episode_id[i],
            annotations={k: v[i] for k, v in self.annotations.items() if v[i] is not None},
            encoded_video={k: v[i] for k, v in self.video.items()} if self.video else None,
            decoded_video=dec_video,
            encoded_video_blob=(
                {k: v[i] for k, v in self.video_blob.items()} if self.video_blob else None
            ),
            video_blob_frame_index=(
                {k: int(v[i]) for k, v in self.video_blob_frame_index.items()}
                if self.video_blob_frame_index
                else None
            ),
            state={k: v[i] for k, v in self.state.items()},
            action={k: v[i] for k, v in self.action.items()},
        )

    def __iter__(self) -> Iterator[PackDSStep]:
        for i in range(len(self)):
            yield self[i]

    def encode_video(
        self,
        *,
        encode_jpeg: bool = False,
        encode_video_blob: bool = True,
        group_size: int | None = None,
        crf: int | None = None,
        jpeg_quality: int = 95,
        codec: str = "h264",
    ) -> "PackDSStepBatch":
        """Return a new batch with encoded video.

        Converts steps to a list, encodes, then rebuilds a batch.
        For batch-level encoding, this delegates to
        :func:`~groot.core.data.packds.video.encode_steps`.

        Args:
            encode_jpeg: If True, encode as per-frame JPEG.
            encode_video_blob: If True, encode as multi-frame video blobs.
            group_size: Number of frames per video blob group.
            crf: CRF quality parameter.
            jpeg_quality: JPEG quality (0-100).
            codec: Video codec (h264 or av1).

        Returns:
            A new PackDSStepBatch with encoded video populated.
        """
        from groot.core.data.packds import constants
        from groot.core.data.packds import video as video_module

        if group_size is None:
            group_size = constants.DEFAULT_VIDEO_BLOB_GROUP_SIZE
        if crf is None:
            crf = constants.DEFAULT_VIDEO_BLOB_CRF

        steps = list(self)
        encoded_steps = video_module.encode_steps(
            steps,
            encode_jpeg=encode_jpeg,
            encode_video_blob=encode_video_blob,
            group_size=group_size,
            crf=crf,
            jpeg_quality=jpeg_quality,
            codec=codec,
        )
        from groot.core.data.packds.utils import from_table, into_table

        table = into_table(encoded_steps)
        return from_table(table)

    def decode_video(self) -> "PackDSStepBatch":
        """Return a new batch with decoded video.

        Decodes encoded_video (JPEG) or encoded_video_blob (H.264).
        JPEG is preferred when both exist for a key.

        Returns:
            A new PackDSStepBatch with decoded_video populated.
        """
        from groot.core.data.packds import video as video_module

        new_batch = dataclasses.replace(self)
        # Create new dict for decoded_video so original is not mutated
        new_batch.decoded_video = dict(self.decoded_video)
        video_module._decode_batch_video(new_batch)
        return new_batch
