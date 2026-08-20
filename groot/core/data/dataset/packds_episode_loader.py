# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Episode-level data loader for PackDS format datasets.

Drop-in replacement for LeRobotEpisodeLoader that reads from PackDS datasets.
"""

from collections.abc import Sequence
import logging
import os
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

import numpy as np
import polars as pl

from groot.core.data import types as data_types
from groot.core.data.packds.video import decode_steps

logger = logging.getLogger(__name__)

ALLOWED_MODALITIES = ["video", "state", "action", "language", "rl_info"]
# Kept as a module-level seam for tests and downstream overrides without
# importing the optional DeltaCAT backend in ordinary PackDS builds.
PackDSStepsReader: Any | None = None


def _build_episode_dataframe_from_packds_steps(
    steps: list[Any],
    modality_configs: dict[str, data_types.ModalityConfig],
) -> data_types.EpisodeData:
    data: dict[str, list] = {}

    # State, action, video — read directly from step fields
    step_fields = {"state": "state", "action": "action", "video": "decoded_video"}
    for modality, field in step_fields.items():
        if modality not in modality_configs:
            continue
        for key in modality_configs[modality].modality_keys:
            data[f"{modality}.{key}"] = [getattr(step, field)[key] for step in steps]

    # Language — all language keys are stored as per-step annotations in PackDS
    if "language" in modality_configs:
        for lang_key in modality_configs["language"].modality_keys:
            ann_key = lang_key.removeprefix("annotation.")
            data[f"language.{lang_key}"] = [step.annotations.get(ann_key, "") for step in steps]

    return data_types.build_episode_data(data)


def _split_deltacat_loader_uri(dataset_path: str) -> tuple[str, dict[str, str]]:
    parsed = urlparse(dataset_path)
    if parsed.scheme != "dc":
        return dataset_path, {}
    query = {
        key: values[-1]
        for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
        if values
    }
    table_uri = urlunparse(parsed._replace(query="", fragment=""))
    return table_uri, query


def _env_or_query(
    query: dict[str, str],
    key: str,
    env_var: str,
) -> str | None:
    value = query.get(key)
    if value is not None:
        return value or None
    return os.environ.get(env_var) or None


def _int_env_or_query(
    query: dict[str, str],
    key: str,
    env_var: str,
    default: int,
) -> int:
    value = query.get(key)
    if value is None:
        value = os.environ.get(env_var)
    if value is None or value == "":
        return default
    return int(value)


def _step_index_upper_bound(sql: str | None) -> int | None:
    """Return the selected row count for common `step_index < N` smoke filters."""
    operator_and_value = _step_index_bound_operator_and_value(sql)
    if operator_and_value is None:
        return None
    operator, value = operator_and_value
    if operator == "lt":
        return value
    return value + 1


def _step_index_bound_operator_and_value(sql: str | None) -> tuple[str, int] | None:
    if sql is None:
        return None
    query = sql.strip()
    if query.lower().startswith("where "):
        query = query[6:].strip()
    match = re.fullmatch(r"step_index\s*<\s*(\d+)", query, flags=re.IGNORECASE)
    if match:
        return "lt", int(match.group(1))
    match = re.fullmatch(r"step_index\s*<=\s*(\d+)", query, flags=re.IGNORECASE)
    if match:
        return "lte", int(match.group(1))
    return None


def _modality_keys(
    modality_configs: dict[str, data_types.ModalityConfig],
    modality: str,
) -> list[str]:
    config = modality_configs.get(modality)
    return list(config.modality_keys) if config is not None else []


def _source_key(key: str, key_remap: dict[str, str] | None) -> str:
    return (key_remap or {}).get(key, key)


def _annotation_column(key: str) -> str:
    if key.startswith("annotation:"):
        return key
    if key.startswith("annotation."):
        return f"annotation:{key.removeprefix('annotation.')}"
    return f"annotation:{key}"


def _deltacat_step_columns(
    modality_configs: dict[str, data_types.ModalityConfig],
    key_remap: dict[str, str] | None = None,
) -> tuple[str, ...]:
    columns = ["episode_id", "step_index"]
    for key in _modality_keys(modality_configs, "state"):
        columns.append(f"state:{_source_key(key, key_remap)}")
    for key in _modality_keys(modality_configs, "action"):
        columns.append(f"action:{_source_key(key, key_remap)}")
    for lang_key in _modality_keys(modality_configs, "language"):
        columns.append(_annotation_column(_source_key(lang_key, key_remap)))
    return tuple(dict.fromkeys(columns))


def _requires_deltacat_statistics(
    modality_configs: dict[str, data_types.ModalityConfig],
) -> bool:
    """Return True when configured state/action keys need normalization stats."""
    for modality in ("state", "action"):
        config = modality_configs.get(modality)
        if config is not None and config.modality_keys:
            return True
    return False


def _row_array(row: dict[str, Any], column: str) -> np.ndarray:
    if column not in row or row[column] is None:
        raise KeyError(f"DeltaCAT PackDS row is missing required column {column!r}")
    return np.asarray(row[column], dtype=np.float32)


def _task_name_from_episode_id(episode_id: str) -> str | None:
    prefix = episode_id.split("__episode_", maxsplit=1)[0]
    if not prefix:
        return None
    task_name = prefix.rsplit("__", maxsplit=1)[-1]
    return task_name.replace("_", " ") or None


def _language_fallback_from_episode_meta(episode_meta: dict[str, Any]) -> str:
    if episode_meta.get("task_name"):
        return str(episode_meta["task_name"])
    tasks = episode_meta.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if task:
                return str(task)
    elif tasks:
        return str(tasks)
    trajectory_id = episode_meta.get("trajectory_id") or episode_meta.get("episode_id")
    if trajectory_id:
        return _task_name_from_episode_id(str(trajectory_id)) or ""
    return ""


def _build_episode_dataframe_from_deltacat_steps(
    steps: list[Any],
    modality_configs: dict[str, data_types.ModalityConfig],
    *,
    key_remap: dict[str, str] | None = None,
    language_fallback: str = "",
) -> data_types.EpisodeData:
    data: dict[str, list] = {}

    if "video" in modality_configs:
        for key in modality_configs["video"].modality_keys:
            data[f"video.{key}"] = [
                step.decoded_video[_source_key(key, key_remap)] for step in steps
            ]

    if "state" in modality_configs:
        for key in modality_configs["state"].modality_keys:
            column = f"state:{_source_key(key, key_remap)}"
            data[f"state.{key}"] = [_row_array(step.row, column) for step in steps]

    if "action" in modality_configs:
        for key in modality_configs["action"].modality_keys:
            column = f"action:{_source_key(key, key_remap)}"
            data[f"action.{key}"] = [_row_array(step.row, column) for step in steps]

    if "language" in modality_configs:
        for lang_key in modality_configs["language"].modality_keys:
            column = _annotation_column(_source_key(lang_key, key_remap))
            data[f"language.{lang_key}"] = [
                step.row.get(column) or language_fallback for step in steps
            ]

    return data_types.build_episode_data(data)


class PackDSEpisodeLoader:
    """Episode-level data loader for PackDS format datasets.

    Drop-in replacement for LeRobotEpisodeLoader. Reads episodes from a PackDS
    dataset and returns them as DataFrames with the same column structure.

    Args:
        dataset_path: Path to PackDS dataset directory.
        modality_configs: Dictionary mapping modality names to ModalityConfig objects.
        video_backend: Ignored (PackDS handles video decoding internally).
        video_backend_kwargs: Ignored.
        resample_on_invalid_frames: Ignored.
        data_filter: Optional filter applied to episode DataFrames.
        hydrator: Ignored (PackDS handles storage via LanceDB).
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict[str, data_types.ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        resample_on_invalid_frames: bool = False,
        data_filter: data_types.OnlineDataFilter | None = None,
        hydrator: Any | None = None,
    ) -> None:
        del video_backend, video_backend_kwargs, resample_on_invalid_frames, hydrator

        self._dataset_path_str = str(dataset_path).strip()
        self.filter = data_filter

        from groot.core.data.packds import PackDS

        self._packds = PackDS(self._dataset_path_str, mode="r")
        self._packds.__enter__()

        self.episodes_metadata = self._packds.get_episodes_metadata()
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")
        self.modality_configs = modality_configs
        self.episode_lengths = [int(ep["length"]) for ep in self.episodes_metadata]

        self._timing_read_cumulative = 0.0
        self._timing_decode_cumulative = 0.0

    @property
    def dataset_path(self) -> str:
        return self._dataset_path_str

    def __len__(self) -> int:
        return len(self.episodes_metadata)

    def get_episode_lengths(self) -> list[int]:
        return list(self.episode_lengths)

    def get_episode_length(self, idx: int) -> int:
        return self.episode_lengths[idx]

    def get_num_cameras(self) -> int:
        if "video" not in self.modality_configs:
            return 0
        return len(self.modality_configs["video"].modality_keys)

    def __getitem__(self, idx: int) -> data_types.EpisodeData:
        """Load complete episode data as a DataFrame.

        Returns DataFrame with columns matching LeRobotEpisodeLoader output:
        video.{key}, state.{key}, action.{key}, language.{key}.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]

        t0 = time.time()
        steps = self._packds.read_episode_steps(str(episode_meta["trajectory_id"]))
        self._timing_read_cumulative += time.time() - t0

        t0 = time.time()
        steps = decode_steps(steps)
        self._timing_decode_cumulative += time.time() - t0

        episode_df = _build_episode_dataframe_from_packds_steps(steps, self.modality_configs)

        if self.filter is not None:
            filter_meta = {**episode_meta, "modality_configs": self.modality_configs}
            episode_df = self.filter(episode_df, filter_meta)

        return episode_df

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Extract dataset statistics for normalization.

        Returns nested dict: {modality: {joint_group: {stat_type: values}}}
        """
        raw_stats = self._packds.stats
        result: dict[str, dict[str, Any]] = {}

        mapping = {"state": "observation", "action": "action"}
        for modality, prefix in mapping.items():
            if modality not in self.modality_configs:
                continue
            result[modality] = {}
            for joint_key in self.modality_configs[modality].modality_keys:
                stats_key = f"{prefix}.{joint_key}"
                if stats_key in raw_stats:
                    result[modality][joint_key] = raw_stats[stats_key]

        relative_stats = self._packds.relative_stats
        if relative_stats is not None:
            result["relative_action"] = relative_stats

        return result

    def register_video_files_for_episodes(self, episode_indices: list[int]) -> None:
        pass

    def release_video_files_for_episodes(self, episode_indices: list[int]) -> None:
        pass

    def get_initial_actions(self) -> list:
        return []

    def __del__(self):
        if hasattr(self, "_packds"):
            self._packds.__exit__(None, None, None)


class DeltaCatPackDSEpisodeLoader:
    """Episode-level loader for SQL-filtered DeltaCAT PackDS steps tables."""

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict[str, data_types.ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        resample_on_invalid_frames: bool = False,
        data_filter: data_types.OnlineDataFilter | None = None,
        hydrator: Any | None = None,
        _steps_reader_override: Any | None = None,
        episode_query: str | None = None,
        steps_query: str | None = None,
        as_of_timestamp: int | None = None,
        key_remap: dict[str, str] | None = None,
        cameras: Sequence[str] | None = None,
        seed: int | None = None,
        max_blob_cache_bytes: int = 256 * 1024 * 1024,
        max_decoded_cache_bytes: int = 512 * 1024 * 1024,
        decode_workers: int | None = None,
    ) -> None:
        del video_backend, video_backend_kwargs, resample_on_invalid_frames, hydrator

        self._dataset_path_str = str(dataset_path).strip()
        self.filter = data_filter
        table_uri, query = _split_deltacat_loader_uri(self._dataset_path_str)
        self._dataset_statistics: dict[str, Any] | None = None
        self.key_remap = {str(key): str(value) for key, value in (key_remap or {}).items()}
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")
        self.modality_configs = modality_configs

        steps_reader = _steps_reader_override
        if steps_reader is None:
            reader_type = PackDSStepsReader
            if reader_type is None:
                raise RuntimeError(
                    "DeltaCAT PackDS datasets are not included in this package build."
                )

            resolved_cameras = (
                tuple(cameras)
                if cameras is not None
                else tuple(
                    _source_key(key, self.key_remap)
                    for key in _modality_keys(modality_configs, "video")
                )
            )
            timeout_s = int(query["timeout_s"]) if query.get("timeout_s") else 120
            resolved_as_of_timestamp = as_of_timestamp
            if resolved_as_of_timestamp is None:
                raw_as_of = query.get("as_of_timestamp") or query.get("as_of")
                if raw_as_of:
                    resolved_as_of_timestamp = int(raw_as_of)
            resolved_seed = seed
            if resolved_seed is None and query.get("seed"):
                resolved_seed = int(query["seed"])
            resolved_episode_query = episode_query
            if resolved_episode_query is None:
                resolved_episode_query = _env_or_query(
                    query,
                    "episode_query",
                    "DELTACAT_PACKDS_EPISODE_QUERY",
                )
            resolved_steps_query = steps_query
            if resolved_steps_query is None:
                resolved_steps_query = _env_or_query(
                    query,
                    "steps_query",
                    "DELTACAT_PACKDS_STEPS_QUERY",
                )
            resolved_decode_workers = (
                decode_workers
                if decode_workers is not None
                else _int_env_or_query(
                    query,
                    "decode_workers",
                    "DELTACAT_PACKDS_DECODE_WORKERS",
                    8,
                )
            )
            resolved_max_blob_cache_bytes = _int_env_or_query(
                query,
                "max_blob_cache_bytes",
                "DELTACAT_PACKDS_MAX_BLOB_CACHE_BYTES",
                max_blob_cache_bytes,
            )
            resolved_max_decoded_cache_bytes = _int_env_or_query(
                query,
                "max_decoded_cache_bytes",
                "DELTACAT_PACKDS_MAX_DECODED_CACHE_BYTES",
                max_decoded_cache_bytes,
            )
            steps_reader = reader_type(
                table_uri,
                episode_query=resolved_episode_query,
                steps_query=resolved_steps_query,
                as_of_timestamp=resolved_as_of_timestamp,
                seed=resolved_seed,
                episodes_table_uri=query.get("episodes_table_uri") or None,
                cameras=resolved_cameras or None,
                step_columns=_deltacat_step_columns(modality_configs, self.key_remap),
                key_remap=self.key_remap,
                server_url=query.get("server_url") or None,
                timeout_s=timeout_s,
                max_blob_cache_bytes=resolved_max_blob_cache_bytes,
                max_decoded_cache_bytes=resolved_max_decoded_cache_bytes,
                decode_workers=resolved_decode_workers,
            )
        self._reader = steps_reader

        self.episodes_metadata = self._load_episodes_metadata()
        self.episode_lengths = [int(ep["length"]) for ep in self.episodes_metadata]
        self._timing_read_cumulative = 0.0
        self._timing_decode_cumulative = 0.0

    @property
    def dataset_path(self) -> str:
        return self._dataset_path_str

    def __len__(self) -> int:
        return len(self.episodes_metadata)

    def _load_episodes_metadata(self) -> list[dict[str, Any]]:
        get_episodes_metadata = getattr(self._reader, "get_episodes_metadata", None)
        if callable(get_episodes_metadata):
            rows = get_episodes_metadata()
        else:
            lf = self._reader.episodes_lazyframe()
            schema = lf.collect_schema()
            if "episode_id" not in schema:
                raise ValueError("episode_query must preserve an 'episode_id' column.")
            if "length" not in schema:
                raise ValueError("episode_query must preserve a 'length' column for training.")

            columns = ["episode_id", "length"]
            for optional_column in ("task_name", "tasks", "incomplete"):
                if optional_column in schema:
                    columns.append(optional_column)
            episodes_lf = lf.select(columns)
            if "incomplete" in columns:
                episodes_lf = episodes_lf.filter(pl.col("incomplete").fill_null(False).not_())
            rows = episodes_lf.collect().to_dicts()

        steps_query = getattr(self._reader, "steps_query", None)
        step_index_limit = _step_index_upper_bound(steps_query)
        step_counts: dict[str, int] | None = None
        count_filtered_steps = getattr(self._reader, "get_step_counts_by_episode", None)
        if steps_query and step_index_limit is None and callable(count_filtered_steps):
            step_counts = count_filtered_steps(
                [str(row.get("episode_id") or row["trajectory_id"]) for row in rows]
            )

        episodes = []
        for row in rows:
            episode_id = str(row.get("episode_id") or row["trajectory_id"])
            if step_index_limit is not None:
                length = min(int(row["length"]), step_index_limit)
            elif step_counts is not None:
                length = step_counts.get(episode_id, 0)
            else:
                length = int(row["length"])
            if length <= 0:
                continue
            tasks = []
            if row.get("task_name"):
                tasks = [str(row["task_name"])]
            elif isinstance(row.get("tasks"), list):
                tasks = [str(task) for task in row["tasks"] if task]
            elif row.get("tasks"):
                tasks = [str(row["tasks"])]
            episodes.append(
                {
                    **row,
                    "trajectory_id": episode_id,
                    "length": length,
                    "tasks": tasks,
                    "sub_tasks": [],
                }
            )
        return episodes

    def get_episode_lengths(self) -> list[int]:
        return list(self.episode_lengths)

    def get_episode_length(self, idx: int) -> int:
        return self.episode_lengths[idx]

    def get_num_cameras(self) -> int:
        if "video" not in self.modality_configs:
            return 0
        return len(self.modality_configs["video"].modality_keys)

    def preload_episodes(self, episode_indices: Sequence[int]) -> None:
        preload_episode_rows = getattr(self._reader, "preload_episode_rows", None)
        if not callable(preload_episode_rows):
            return
        episode_ids = []
        for idx in dict.fromkeys(int(index) for index in episode_indices):
            if idx < 0 or idx >= len(self.episodes_metadata):
                continue
            episode_ids.append(str(self.episodes_metadata[idx]["trajectory_id"]))
        try:
            preload_episode_rows(episode_ids)
        except Exception:  # noqa: BLE001
            # Preloading is an optimization that groups the catalog plan/read
            # for upcoming shard episodes. Fall back to the normal per-episode
            # read path so transient catalog failures can still retry there.
            logger.warning(
                "DeltaCAT episode row preload failed; falling back to per-episode reads",
                exc_info=True,
            )

    def __getitem__(self, idx: int) -> data_types.EpisodeData:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        t0 = time.time()
        steps = self._reader.get_steps_for_episode(str(episode_meta["trajectory_id"]))
        read_and_decode_s = time.time() - t0
        self._timing_read_cumulative += read_and_decode_s
        self._timing_decode_cumulative += 0.0

        episode_df = _build_episode_dataframe_from_deltacat_steps(
            steps,
            self.modality_configs,
            key_remap=self.key_remap,
            language_fallback=_language_fallback_from_episode_meta(episode_meta),
        )
        if self.filter is not None:
            filter_meta = {**episode_meta, "modality_configs": self.modality_configs}
            episode_df = self.filter(episode_df, filter_meta)

        return episode_df

    def get_episode_window(self, idx: int, start_idx: int, max_len: int) -> data_types.EpisodeData:
        """Load only a contiguous ``[start_idx, start_idx+max_len)`` row window of
        an episode, decoding ONLY the window's video (not the whole episode).

        Result is identical to ``self[idx].slice(start_idx, max_len)`` but avoids
        decoding the full episode -- the fix for trajectory (N2-TTT) training,
        where the prior whole-episode-then-slice path decoded ~4200 steps to use
        256 (~14x waste, ~21 min/shard). Callers must NOT also slice the result.
        """  # noqa: D205
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        t0 = time.time()
        steps = self._reader.get_steps_for_episode_window(
            str(episode_meta["trajectory_id"]), int(start_idx), int(max_len)
        )
        self._timing_read_cumulative += time.time() - t0

        episode_df = _build_episode_dataframe_from_deltacat_steps(
            steps,
            self.modality_configs,
            key_remap=self.key_remap,
            language_fallback=_language_fallback_from_episode_meta(episode_meta),
        )
        if self.filter is not None:
            filter_meta = {**episode_meta, "modality_configs": self.modality_configs}
            episode_df = self.filter(episode_df, filter_meta)

        return episode_df

    def get_dataset_statistics(self) -> dict[str, Any]:
        if self._dataset_statistics is not None:
            return self._dataset_statistics
        if not _requires_deltacat_statistics(self.modality_configs):
            self._dataset_statistics = {}
            return self._dataset_statistics
        raise RuntimeError(
            "DeltaCAT PackDS statistics are computed by DatasetFactory at training start. "
            "Build this loader through DatasetFactory so sampled q01/q99 statistics can be "
            "computed from the effective training distribution before batches are read."
        )

    def set_dataset_statistics(self, statistics: dict[str, Any]) -> None:
        self._dataset_statistics = statistics

    def register_video_files_for_episodes(self, episode_indices: list[int]) -> None:
        self.preload_episodes(episode_indices)

    def release_video_files_for_episodes(self, episode_indices: list[int]) -> None:
        pass

    def get_initial_actions(self) -> list:
        return []
