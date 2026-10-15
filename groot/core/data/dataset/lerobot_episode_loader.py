#!/usr/bin/env python  # noqa: EXE001
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LeRobot Dataset Loader.

A simplified, clean implementation for loading LeRobot datasets with video support.
This module provides the core functionality for loading episodes from LeRobot format datasets,
handling metadata parsing, video decoding, and data preprocessing for VLA training.

The LeRobotEpisodeLoader serves as the foundation for higher-level dataset classes,
providing episode-level data access with support for multi-modal data including:
- Video frames from multiple camera views
- Proprioceptive state information
- Action sequences
- Language instructions/annotations

Returns messages with VLAStepData as defined in types.py.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import time
from typing import Any, cast

import botocore.exceptions
import numpy as np
import polars as pl
import pyarrow as pa
import structlog

from groot.core.data import types as data_types
from groot.core.data.dataset.tactile_modality_meta import derive_tactile_modality_meta
from groot.core.data.hydrator import client_protocol, exceptions, service
from groot.core.data.hydrator import types as hydrator_types
from groot.core.utils.distributed import worker_logging
from groot.core.utils.io import video as video_utils
from groot.core.utils.state_action import initial_actions

logger = logging.getLogger(__name__)

# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"

ALLOWED_MODALITIES = ["video", "state", "action", "language", "rl_info", "tactile", "deform"]
# Modalities decoded from video files. "deform" (per-fingertip deform maps) is decoded like
# "video" but kept a separate modality so it never enters the Wan-VAE video path.
VIDEO_DECODED_MODALITIES = ["video", "deform"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
    "tactile": "observation.state",  # per-finger F6 slices of the state vector
}


def resolve_language_entry(x, tasks_map: dict) -> str:
    """Resolve a single parquet language cell to its task string.

    Two on-disk formats are supported:
    - String stored directly in the parquet (arrives as a 0-d numpy array of str)
    - Integer task index that must be looked up in ``tasks_map``
    """
    if isinstance(x, np.ndarray) and isinstance(x.item(), str):
        return x.item()
    return tasks_map[x]


def _rec_defaultdict() -> defaultdict:
    """Factory that creates an infinitely nestable defaultdict."""
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree):
    """Recursively turn a (nested) defaultdict into a regular dict."""
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


def _resolve_dataset_path(
    dataset_path_str: str,
    hydrator: client_protocol.HydratorClientProtocol | None,
) -> Path | None:
    """Resolve dataset path to a local Path or None for remote (S3).

    Hydrator.is_remote_path normalizes internally (e.g. s3:/ -> s3://).
    Returns Path for local dir (validated); None for remote. Raises ValueError if
    remote path but no hydrator, or FileNotFoundError if local path does not exist.
    """
    if service.Hydrator.is_remote_path(dataset_path_str):
        if hydrator is None:
            raise ValueError(
                "S3 dataset_path requires a Hydrator; pass hydrator=... for remote paths."
            )
        return None
    path = Path(dataset_path_str)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return path


class LeRobotEpisodeLoader:
    """Episode-level data loader for LeRobot format datasets.

    This class handles the loading and preprocessing of individual episodes from LeRobot datasets.
    It manages metadata parsing, video decoding, and data extraction across multiple modalities
    (video, state, action, language) while maintaining compatibility with the VLA training pipeline.

    Key responsibilities:
    - Parse LeRobot metadata files (info.json, episodes.jsonl, etc.)
    - Load and decode video data using configurable backends
    - Extract and process multi-modal data according to modality configurations
    - Provide dataset statistics for normalization
    - Handle initial action loading for policy initialization

    Args:
        dataset_path: Dataset root as a string (local path or s3://...). For local paths
            the loader uses Path internally for file operations; for S3, the string is used as-is.
        modality_configs: Dictionary mapping modality names to ModalityConfig objects
                         that specify temporal sampling and data keys to load
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend
        resample_on_invalid_frames: If True, clamp invalid frame indices and retry; if False, raise.
        data_filter: Optional filter of the data.
        hydrator: Optional hydrator client for remote (S3) file access.

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=["joint_velocities"])
        ...     }
        ... )
        >>> episode_data = loader[0]  # Load first episode as DataFrame
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict[str, data_types.ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        resample_on_invalid_frames: bool = False,
        data_filter: data_types.OnlineDataFilter | None = None,
        hydrator: client_protocol.HydratorClientProtocol | None = None,
    ) -> None:
        self._dataset_path_str = str(dataset_path).strip()
        self._hydrator = hydrator
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.resample_on_invalid_frames = resample_on_invalid_frames
        self.filter = data_filter

        self._path: Path | None = _resolve_dataset_path(self._dataset_path_str, hydrator)
        self._worker_log = worker_logging.get_unified_logger()

        # Load metadata files and parse dataset structure
        self._load_metadata()

        # Set up modality configs after metadata is loaded
        self.modality_configs = self._parse_and_validate_modality_configs(modality_configs)
        # Tactile datasets keep F6 under "state" and the deform videos under "video" in
        # modality.json; derive the requested tactile/deform groups from those.
        derive_tactile_modality_meta(
            self.modality_meta, self.modality_configs, dataset_path=self._dataset_path_str
        )

        # Compute effective episode lengths accounting for action horizon
        self.episode_lengths = self.get_episode_lengths()

    @property
    def dataset_path(self) -> Path | str:
        """Dataset root: Path for local (downstream ops), str for S3."""
        if self._path is not None:
            return self._path
        return self._dataset_path_str

    def _load_json_path(self, path: str | Path) -> dict:
        """Load a single JSON object from path (local or S3). Uses hydrator.hydrate() when hydrator is set."""
        path_str = str(path)
        if self._hydrator is not None:
            return cast(dict, self._hydrator.hydrate(path_str)[0])
        with open(path_str, encoding="utf-8") as f:
            return json.load(f)

    def _load_jsonl_path(self, path: str | Path) -> list[dict]:
        """Load JSONL from path (local or S3). Uses hydrator.hydrate() when hydrator is set."""
        path_str = str(path)
        if self._hydrator is not None:
            return cast(list[dict], self._hydrator.hydrate(path_str))
        with open(path_str, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _resolve_path(self, relative_path: str) -> str | Path:
        """Resolve relative path; with hydrator returns joined string; local returns Path."""
        if self._hydrator is not None:
            return service.Hydrator.join_path(self._dataset_path_str, relative_path)
        assert self._path is not None
        return self._path / relative_path

    def _load_metadata(self) -> None:
        """Load all metadata files including dataset statistics.

        Parses the standard LeRobot metadata structure:
        - info.json: Dataset configuration and file patterns
        - episodes.jsonl: Per-episode metadata (length, timestamps, etc.)
        - tasks.jsonl: Task descriptions and mappings
        - modality.json: Modality structure and data layout
        - stats.json: Dataset statistics for normalization
        """
        meta_prefix = f"{LEROBOT_META_DIR_NAME}/"

        # Load dataset configuration (local or S3 via Arrow + json.loads)
        self.info_meta = self._load_json_path(
            self._resolve_path(f"{meta_prefix}{LEROBOT_INFO_FILENAME}")
        )

        # Load episode metadata (one episode per line)
        self.episodes_metadata = self._load_jsonl_path(
            self._resolve_path(f"{meta_prefix}{LEROBOT_EPISODES_FILENAME}")
        )

        # Load task descriptions and create mapping
        tasks_data = self._load_jsonl_path(
            self._resolve_path(f"{meta_prefix}{LEROBOT_TASKS_FILENAME}")
        )
        self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        # Load modality (same JSON shape for local and S3)
        self.modality_meta = self._load_json_path(
            self._resolve_path(f"{meta_prefix}{LEROBOT_MODALITY_FILENAME}"),
        )

        # Load dataset statistics for normalization
        stats_path = self._resolve_path(f"{meta_prefix}{LEROBOT_STATS_FILE_NAME}")
        try:
            self.stats = self._load_json_path(stats_path)
        except (FileNotFoundError, OSError):
            raise FileNotFoundError(
                f"Stats file not found for {self._dataset_path_str}, "
                "please use generate_stats to generate it"
            ) from None

        relative_stats_path = self._resolve_path(f"{meta_prefix}{LEROBOT_RELATIVE_STATS_FILE_NAME}")
        try:
            self.stats["relative_action"] = self._load_json_path(relative_stats_path)
        except (
            FileNotFoundError,
            OSError,
            json.JSONDecodeError,
            ValueError,
            botocore.exceptions.ClientError,
        ):
            # relative_stats.json is optional; S3 NoSuchKey (ClientError) is the remote
            # equivalent of FileNotFoundError and should be silently skipped too.
            pass

        # Extract key configuration parameters
        self.feature_config = self.info_meta.get("features", {})
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta.get("fps", 30)

    def get_episode_lengths(self):
        """Compute original episode lengths.

        Returns:
            List of original episode lengths
        """
        episode_lengths = []
        for ep_meta in self.episodes_metadata:
            episode_lengths.append(int(ep_meta["length"]))
        return episode_lengths

    def get_episode_length(self, idx: int) -> int:
        """Get the length of a specific episode."""
        return self.episode_lengths[idx]

    def _parse_and_validate_modality_configs(
        self,
        modality_configs: dict[str, data_types.ModalityConfig],
    ) -> dict[str, data_types.ModalityConfig]:
        """Parse and validate modality configurations, filling in defaults where needed.

        For missing modality configs, creates default configurations:
        - video: All available camera views with single timestep
        - state: All available state keys with single timestep
        - action: All available action keys with 16-step horizon
        - language: Must be explicitly configured if needed

        Args:
            modality_configs: User-provided modality configurations

        Returns:
            Complete and validated modality configurations

        Raises:
            ValueError: If invalid modalities are specified
        """
        # Validate all modality configurations
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")

        return modality_configs

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _language_expr(self, original_key: str, alias: str, raw_pl: pl.DataFrame) -> pl.Expr:
        """Build a polars expression that resolves a language column to strings.

        On-disk formats supported:
        - ``pl.String`` → passthrough rename.
        - ``pl.List(pl.String)`` / ``pl.Array(pl.String, 1)`` → unwrap singleton
          cell to a plain string. Some legacy LeRobot parquets store the
          annotation as a one-element string list.
        - Integer task index → look up in ``self.tasks_map`` via ``replace_strict``.
        """
        col_dtype = raw_pl[original_key].dtype
        if col_dtype == pl.String:
            return pl.col(original_key).alias(alias)
        if col_dtype.base_type() in (pl.List, pl.Array) and col_dtype.inner == pl.String:
            return pl.col(original_key).cast(pl.List(pl.String)).list.first().alias(alias)
        return (
            pl.col(original_key).replace_strict(self.tasks_map, return_dtype=pl.String).alias(alias)
        )

    def _load_parquet_data(self, episode_index: int) -> pl.DataFrame:
        """Load and process parquet data for a specific episode.

        Builds the typed episode DataFrame in a single ``select`` over polars
        expressions: language strings, per-joint-group state/action ``pl.Array``
        columns, and rl_info passthrough. No Python-level row loops.

        Raises:
            FileNotFoundError: If the parquet file for this episode does not exist
        """
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        if self._hydrator is not None:
            full = service.Hydrator.join_path(self._dataset_path_str, parquet_filename)
            try:
                result = self._hydrator.hydrate(full)
            except (FileNotFoundError, OSError, exceptions.HydratorFileNotFoundError) as e:
                raise FileNotFoundError(f"Parquet not found: {full}") from e
            assert isinstance(result, (pa.Table, pl.DataFrame)), (  # noqa: UP038
                f"hydrate(.parquet) must return pyarrow.Table or polars.DataFrame, "
                f"got {type(result).__name__}"
            )
            raw_pl = pl.from_arrow(result) if isinstance(result, pa.Table) else result
        else:
            assert self._path is not None
            raw_pl = pl.read_parquet(self._path / parquet_filename)

        exprs: list[pl.Expr] = []

        # Language: index → string lookup, or passthrough if already string.
        if "language" in self.modality_configs:
            annotation_meta = self.modality_meta.get("annotation", {})
            for key in self.modality_configs["language"].modality_keys:
                if not key.startswith("annotation."):
                    raise ValueError(
                        f"Language key '{key}' must start with 'annotation.' prefix. "
                        f"Dataset: {self._dataset_path_str}"
                    )
                subkey = key.removeprefix("annotation.")
                if subkey not in annotation_meta:
                    raise KeyError(
                        f"Language key '{subkey}' (from '{key}') not found in "
                        f"modality.json 'annotation' section. "
                        f"Available keys: {list(annotation_meta.keys())}. "
                        f"Dataset: {self._dataset_path_str}"
                    )
                original_key = annotation_meta[subkey].get("original_key", key)
                if original_key not in raw_pl.columns:
                    raise KeyError(
                        f"Parquet column '{original_key}' (for language key '{key}') "
                        f"not found. Available columns: {raw_pl.columns}. "
                        f"Dataset: {self._dataset_path_str}"
                    )
                exprs.append(self._language_expr(original_key, f"language.{key}", raw_pl))

        # State / action / tactile: slice the source vector into per-joint-group pl.Array
        # columns (tactile = per-finger F6 slices of the state vector). Cast through pl.List
        # so this works whether the source is List or Array.
        for modality_type in ("state", "action", "tactile"):
            if modality_type not in self.modality_configs:
                continue
            modality_info = self.modality_meta.get(modality_type, {})
            for group_name in self.modality_configs[modality_type].modality_keys:
                if group_name not in modality_info:
                    logger.warning(
                        "Warning: Joint group '%s' not found in %s modality. Available groups: %s",
                        group_name,
                        modality_type,
                        list(modality_info.keys()),
                    )
                    continue
                gi = modality_info[group_name]
                start_idx, end_idx = gi["start"], gi["end"]
                width = end_idx - start_idx
                original_key = gi.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                exprs.append(
                    pl.col(original_key)
                    .cast(pl.List(pl.Float32))
                    .list.slice(start_idx, width)
                    .cast(pl.Array(pl.Float32, width))
                    .alias(f"{modality_type}.{group_name}")
                )

        # rl_info: passthrough, but coerce List → fixed-width Array so downstream
        # gather/.to_numpy() returns a 2D float32 ndarray (same contract as state/action).
        if "rl_info" in self.modality_configs:
            for key in self.modality_configs["rl_info"].modality_keys:
                rl_col = f"rl_info.{key}"
                if rl_col not in raw_pl.columns:
                    continue
                rl_dtype = raw_pl[rl_col].dtype
                if rl_dtype.base_type() == pl.List and raw_pl.height > 0:
                    width = len(raw_pl[rl_col][0])
                    exprs.append(
                        pl.col(rl_col)
                        .cast(pl.List(pl.Float32))
                        .cast(pl.Array(pl.Float32, width))
                        .alias(rl_col)
                    )
                else:
                    exprs.append(pl.col(rl_col))

        return raw_pl.select(exprs)

    def _load_video_data(
        self, episode_index: int, indices: np.ndarray, modality: str = "video"
    ) -> dict[str, np.ndarray]:
        """Load video data for all configured camera views at specified indices.

        Uses the configured video backend to decode video frames at the exact indices
        needed for the episode, supporting multiple camera views simultaneously.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of indices to extract frames at
            modality: A video-decoded modality ("video" or "deform")

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames.
            Deform streams retain only contiguous uint8 channel-zero maps.

        Raises:
            FileNotFoundError: If a video file for this episode does not exist
        """
        video_data = {}

        if not self.video_path_pattern or modality not in self.modality_configs:
            return video_data

        chunk_idx = episode_index // self.chunk_size
        image_keys = self.modality_configs[modality].modality_keys

        for image_key in image_keys:
            # Resolve the original key used in video file naming
            original_key = self._resolve_video_original_key(image_key, modality)
            assert (
                original_key in self.feature_config
            ), f"Original key {original_key} not found in feature config"

            # Construct video file path using pattern
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx, video_key=original_key, episode_index=episode_index
            )
            if self._hydrator is not None:
                full = service.Hydrator.join_path(self._dataset_path_str, video_filename)
                try:
                    video_path = self._hydrator.get_file(full)
                except exceptions.HydratorFileNotFoundError:
                    raise FileNotFoundError(f"Video not found: {full}") from None
                if video_path is None:
                    raise FileNotFoundError(f"Video not found: {full}")
                path_for_decode = str(video_path)
            else:
                assert self._path is not None
                path_for_decode = str(self._path / video_filename)

            # Decode video frames at specified timestamps
            frames = video_utils.get_frames_by_indices(
                path_for_decode,
                indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs or {},
                resample_on_invalid_frames=self.resample_on_invalid_frames,
            )
            if modality == "deform":
                # Compact each stream before decoding the next: a channel view alone
                # keeps the entire RGB episode alive through the Polars frame views.
                frames = np.ascontiguousarray(frames[..., 0] if frames.ndim == 4 else frames)
            video_data[image_key] = frames

        return video_data

    def _resolve_video_original_key(self, image_key: str, modality: str = "video") -> str:
        """Use the canonical feature name when modality.json omits a video alias."""
        canonical_key = f"observation.images.{image_key}"
        image_meta = self.modality_meta.get(modality, {}).get(image_key, {})
        return image_meta.get("original_key", canonical_key)

    def _get_video_paths_for_episodes(
        self, episode_indices: list[int], modality: str = "video"
    ) -> list[str]:
        """Return all video file paths (remote or local) needed for the given episodes, deduplicated.

        One path per configured camera view per episode. Use for prefetching via
        hydrator.register_files().
        """
        if not self.video_path_pattern or modality not in self.modality_configs:
            return []
        paths_set: set[str] = set()
        image_keys = self.modality_configs[modality].modality_keys
        for ep_idx in episode_indices:
            chunk_idx = ep_idx // self.chunk_size
            for image_key in image_keys:
                original_key = self._resolve_video_original_key(image_key, modality)
                if original_key not in self.feature_config:
                    continue
                video_filename = self.video_path_pattern.format(
                    episode_chunk=chunk_idx, video_key=original_key, episode_index=ep_idx
                )
                if self._hydrator is not None:
                    paths_set.add(
                        service.Hydrator.join_path(self._dataset_path_str, video_filename)
                    )
                else:
                    assert self._path is not None
                    paths_set.add(str(self._path / video_filename))
        return list(paths_set)

    def register_video_files_for_episodes(self, episode_indices: list[int]) -> None:
        """Prefetch video files for the given episodes via hydrator (async download).

        No-op if no hydrator or no video paths. Call before loading episodes so downloads
        can overlap with other work.
        """
        if self._hydrator is None:
            return
        paths = [
            path
            for modality in VIDEO_DECODED_MODALITIES
            for path in self._get_video_paths_for_episodes(episode_indices, modality)
        ]
        self._worker_log.info(
            "[REGISTER_VIDEO] start n_episodes=%s n_paths=%s",
            len(episode_indices),
            len(paths),
        )
        if paths:
            self._hydrator.register_files(paths, modality=hydrator_types.Modality.VIDEO)
        self._worker_log.info("[REGISTER_VIDEO] done")

    def release_video_files_for_episodes(self, episode_indices: list[int]) -> None:
        """Release video files for the given episodes, decrementing their ref_count.

        Call after all episodes in the shard have been loaded to allow LRU eviction.
        No-op if no hydrator or no video paths.
        """
        if self._hydrator is None:
            return
        paths = [
            path
            for modality in VIDEO_DECODED_MODALITIES
            for path in self._get_video_paths_for_episodes(episode_indices, modality)
        ]
        if paths:
            self._hydrator.release_files(paths)

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Extract dataset statistics for normalization from loaded metadata.

        Constructs a nested dictionary containing statistics (mean, std, min, max, q01, q99)
        for each joint group in state and action modalities. These statistics are used
        by processors for data normalization during training.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}
        """
        mapping = {"state": "observation.state", "action": "action"}
        dataset_statistics = _rec_defaultdict()

        for modality, default_stats_key in mapping.items():  # state, action
            for joint_key in self.modality_configs[modality].modality_keys:
                # Determine which statistics key to use
                if self.modality_meta[modality][joint_key].get("original_key", None) is not None:
                    stats_key = self.modality_meta[modality][joint_key]["original_key"]
                else:
                    stats_key = default_stats_key

                # Extract the relevant slice of statistics
                start_idx, end_idx = (
                    self.modality_meta[modality][joint_key]["start"],
                    self.modality_meta[modality][joint_key]["end"],
                )
                for stat_type in self.stats[stats_key]:  # mean, std, min, max, q01, q99
                    dataset_statistics[modality][joint_key][stat_type] = self.stats[stats_key][
                        stat_type
                    ][start_idx:end_idx]
        stats = _to_plain_dict(dataset_statistics)
        # Directly add relative action stats
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def __getitem__(self, idx: int) -> data_types.EpisodeData:
        """Load complete episode data as a polars DataFrame.

        Combines parquet data loading and video decoding to create a typed
        polars DataFrame with ``pl.Array`` columns for state/action,
        ``pl.Object`` columns for video frames, and ``pl.String`` for language.

        Raises:
            IndexError: If episode index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        getitem_t0 = time.perf_counter()

        t0 = time.time()
        episode_df = self._load_parquet_data(episode_id)
        parquet_ms = round((time.time() - t0) * 1000, 2)

        actual_length = min(episode_df.height, nominal_length)
        episode_df = episode_df.head(actual_length)

        t0 = time.time()
        video_data = self._load_video_data(episode_id, np.arange(actual_length))
        video_ms = round((time.time() - t0) * 1000, 2)

        for key, frames in video_data.items():
            assert len(frames) == actual_length, (
                f"Video data for {key} has length {len(frames)} "
                f"but episode has {actual_length} frames"
            )
            episode_df = episode_df.with_columns(
                pl.Series(f"video.{key}", list(frames), dtype=pl.Object)
            )

        # Deform maps decode like video but land in their own "deform.*" columns.
        if "deform" in self.modality_configs:
            t0 = time.time()
            deform_data = self._load_video_data(
                episode_id, np.arange(actual_length), modality="deform"
            )
            video_ms += round((time.time() - t0) * 1000, 2)
            for key, frames in deform_data.items():
                assert len(frames) == actual_length, (
                    f"Deform data for {key} has length {len(frames)} "
                    f"but episode has {actual_length} frames"
                )
                episode_df = episode_df.with_columns(
                    pl.Series(f"deform.{key}", list(frames), dtype=pl.Object)
                )

        if self.filter is not None:
            filter_meta = {**episode_meta, "modality_configs": self.modality_configs}
            episode_df = self.filter(episode_df, filter_meta)

        structlog.get_logger().debug(
            "episode_loader_getitem_completed",
            episode_id=int(episode_id),
            num_frames=int(actual_length),
            parquet_ms=parquet_ms,
            video_ms=video_ms,
            duration_ms=round((time.perf_counter() - getitem_t0) * 1000, 2),
        )
        return episode_df

    def get_num_cameras(self) -> int:
        """Return the number of camera views configured for loading."""
        if "video" not in self.modality_configs:
            return 0
        return len(self.modality_configs["video"].modality_keys)

    def get_initial_actions(self):
        """Load initial actions for policy initialization if available.

        Returns:
            List containing initial action dictionaries, or empty list if not available
        """
        initial_actions_rel = f"{LEROBOT_META_DIR_NAME}/{initial_actions.INITIAL_ACTIONS_FILENAME}"
        full = self._resolve_path(initial_actions_rel)
        if self._hydrator is not None:
            try:
                path = self._hydrator.get_file(str(full))
            except (exceptions.HydratorException, FileNotFoundError, OSError):
                path = None  # optional file
        else:
            p = full if isinstance(full, Path) else Path(full)
            path = str(p) if p.exists() else None
        if path is not None:
            return initial_actions.load_initial_actions(
                Path(path) if isinstance(path, str) else path
            )
        return []
