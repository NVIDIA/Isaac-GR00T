# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Legacy episode-index compatibility wrapper for PackDS.

This module keeps integer episode selection and synthetic ``episode_index``
values out of the canonical PackDS reader/writer classes. The base
``PackDSReader`` / ``PackDSLanceDB`` APIs operate purely on ``episode_id``.
``PackDS`` remains as a compatibility wrapper that translates between the old
integer-based API and the canonical episode-id based storage layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from groot.core.data.packds.lancedb_dataset import PackDSLanceDB
from groot.core.data.packds.reader import PackDSReader
from groot.core.data.packds import utils

_COMPAT_EPISODE_ORDER_META_KEY = "packds:compat_episode_id_order"


class _EpisodeIndexCompatMixin:
    _episode_id_order: list[str]
    _episode_id_to_index: dict[str, int]
    _episode_index_to_id: dict[int, str]

    def _refresh_episode_index_cache(self) -> None:
        order = self._read_schema_meta_json(_COMPAT_EPISODE_ORDER_META_KEY)
        if order is None and self.schema is not None and "episode_index" in self.schema.names:
            table = self._read_table(columns=["episode_id", "episode_index"])
            pairs = sorted(
                zip(
                    table.column("episode_index").to_pylist(),
                    table.column("episode_id").to_pylist(),
                    strict=True,
                )
            )
            order = [str(episode_id) for _, episode_id in pairs]
        # ``count_rows()`` is the *step* count, not the episode count, so
        # comparing it against ``len(order)`` would mark the cache stale on
        # essentially every multi-step dataset.  Compare against the
        # canonical episode count instead.
        cache_stale = (
            self.steps_table is not None
            and order is not None
            and len(order) < self.num_episodes
        )
        if order is None or cache_stale:
            # Metadata missing or stale — fall back to discovery.
            #
            # LanceDB-backed reader: ``self.steps_table`` is set; the
            # parent ``get_episode_ids`` returns the right list (using
            # the sibling ``episodes.lance`` companion or scanning steps).
            #
            # DeltaCAT-backed reader: ``steps_table`` is None but the
            # source is still resolvable; ``get_episode_ids`` reads
            # from the application ``_episodes`` companion.
            #
            # Pure write mode (no source yet, no steps_table): no episodes
            # exist to enumerate; return empty.
            if self.steps_table is not None or self._is_deltacat_resolved():
                order = self.get_episode_ids()
            elif order is None:
                order = []
        self._set_episode_id_order([str(episode_id) for episode_id in order])

    def _is_deltacat_resolved(self) -> bool:
        """True iff the resolved source is a DeltaCAT ``dc://`` URI."""
        resolved = getattr(self, "_resolved_source", None)
        return resolved is not None and resolved.source_key == "deltacat"

    def _set_episode_id_order(self, order: list[str]) -> None:
        deduped: list[str] = []
        seen: set[str] = set()
        for episode_id in order:
            if episode_id not in seen:
                deduped.append(episode_id)
                seen.add(episode_id)
        self._episode_id_order = deduped
        self._episode_id_to_index = {
            episode_id: index for index, episode_id in enumerate(self._episode_id_order)
        }
        self._episode_index_to_id = {
            index: episode_id for index, episode_id in enumerate(self._episode_id_order)
        }

    def _sync_episode_order(self) -> None:
        self._episode_id_order = [episode_id for _, episode_id in sorted(self._episode_index_to_id.items())]

    def _build_schema_metadata(self) -> dict[str, str]:
        meta = super()._build_schema_metadata()
        # Cache the episode order JSON — only recompute when episodes change.
        num_eps = len(self._episode_index_to_id)
        cached_num = getattr(self, "_cached_episode_order_count", -1)
        if num_eps != cached_num:
            self._sync_episode_order()
            self._cached_episode_order_json = json.dumps(self._episode_id_order)
            self._cached_episode_order_count = num_eps
        meta[_COMPAT_EPISODE_ORDER_META_KEY] = self._cached_episode_order_json
        return meta

    def _ensure_known_episode(self, episode_id: str, episode_index: int | None) -> None:
        episode_id = str(episode_id)
        if episode_index is None:
            if episode_id not in self._episode_id_to_index:
                next_index = max(self._episode_index_to_id) + 1 if self._episode_index_to_id else 0
                self._episode_id_to_index[episode_id] = next_index
                self._episode_index_to_id[next_index] = episode_id
                # Don't _sync_episode_order here — defer to finalize time
            return

        if episode_index < 0:
            raise ValueError(f"episode_index must be >= 0, got {episode_index}")

        if episode_id in self._episode_id_to_index:
            existing = self._episode_id_to_index[episode_id]
            if existing != episode_index:
                raise ValueError(
                    f"Inconsistent episode identifiers: episode_id {episode_id!r} was previously "
                    f"mapped to index {existing}, but now provided with index {episode_index}"
                )
            return

        if episode_index in self._episode_index_to_id:
            existing_id = self._episode_index_to_id[episode_index]
            if existing_id != episode_id:
                raise ValueError(
                    f"Inconsistent episode identifiers: episode_index {episode_index} was previously "
                    f"mapped to id {existing_id!r}, but now provided with id {episode_id!r}"
                )
            return

        self._episode_id_to_index[episode_id] = episode_index
        self._episode_index_to_id[episode_index] = episode_id
        # Don't _sync_episode_order here — defer to finalize time

    def _observe_step_sequence(self, steps: list[Any]) -> None:
        for step in steps:
            self._ensure_known_episode(step.episode_id, step.episode_index)
            step.episode_index = self._episode_id_to_index[step.episode_id]

    def _observe_batch(self, batch) -> None:
        for i, episode_id in enumerate(batch.episode_id):
            episode_index = None if batch.episode_index is None else int(batch.episode_index[i])
            self._ensure_known_episode(str(episode_id), episode_index)

    def _observe_table(self, table: pa.Table) -> pa.Table:
        if "episode_id" not in table.column_names:
            return table
        import pyarrow.compute as pc

        # Vectorized: extract unique (episode_id, episode_index) pairs
        # instead of iterating every row.
        has_index = "episode_index" in table.column_names
        unique_ids = pc.unique(table.column("episode_id")).to_pylist()
        if has_index:
            # Build a mapping from the first occurrence of each episode_id
            id_col = table.column("episode_id")
            idx_col = table.column("episode_index")
            seen: set[str] = set()
            for i in range(table.num_rows):
                eid = str(id_col[i].as_py())
                if eid not in seen:
                    seen.add(eid)
                    self._ensure_known_episode(eid, int(idx_col[i].as_py()))
                    if len(seen) == len(unique_ids):
                        break
            table = table.drop("episode_index")
        else:
            for eid in unique_ids:
                self._ensure_known_episode(str(eid), None)
        return table

    def _episode_id_from_selector(self, episode: int | str) -> str:
        if isinstance(episode, str):
            return utils.validate_episode_id_for_write(episode)
        if int(episode) not in self._episode_index_to_id:
            raise IndexError(f"episode_index out of range: {episode}")
        return self._episode_index_to_id[int(episode)]

    def _inject_episode_indices(self, batch):
        missing_ids = [
            str(episode_id)
            for episode_id in batch.episode_id
            if str(episode_id) not in self._episode_id_to_index
        ]
        if missing_ids:
            current_order = list(self._episode_id_order)
            for episode_id in self.get_episode_ids():
                episode_id = str(episode_id)
                if episode_id not in current_order:
                    current_order.append(episode_id)
            self._set_episode_id_order(current_order)

        batch.episode_index = np.array(
            [self._episode_id_to_index[str(episode_id)] for episode_id in batch.episode_id],
            dtype=np.int64,
        )
        return batch

    def get_episode_indices(self) -> list[int]:
        return sorted(self._episode_index_to_id)

    def get_episodes_metadata(self) -> list[dict[str, Any]]:
        episodes = super().get_episodes_metadata()
        for episode in episodes:
            episode_id = str(episode["trajectory_id"])
            episode["episode_index"] = self._episode_id_to_index[episode_id]
        episodes.sort(key=lambda episode: int(episode["episode_index"]))
        return episodes

    def read_episode_steps(self, episode: int | str | list[int] | list[str]):
        """Read episode steps by index or ID.

        Args:
            episode: A single episode index/ID, or a list of indices/IDs
                for batched reading.

        Returns:
            For a single selector: ``list[PackDSStep]`` sorted by step_index.
            For a list: ``dict[key, list[PackDSStep]]`` keyed by the same
                type passed in (int or str).
        """
        if isinstance(episode, list):
            return self._read_episode_steps_batched_compat(episode)
        episode_id = self._episode_id_from_selector(episode)
        steps = super().read_episode_steps(episode_id)
        episode_index = self._episode_id_to_index[episode_id]
        for step in steps:
            step.episode_index = episode_index
        return steps

    def _read_episode_steps_batched_compat(
        self, episodes: list[int] | list[str],
    ) -> dict:
        """Batched episode read with episode-index compatibility."""
        if not episodes:
            return {}
        # Resolve to episode IDs, preserving the original key type
        id_to_keys: dict[str, list] = {}
        for ep in episodes:
            eid = self._episode_id_from_selector(ep)
            id_to_keys.setdefault(eid, []).append(ep)
        unique_ids = list(id_to_keys.keys())
        batched = super().read_episode_steps(unique_ids)

        result: dict = {}
        for eid, steps in batched.items():
            episode_index = self._episode_id_to_index[eid]
            for step in steps:
                step.episode_index = episode_index
            # Map back to original key type (int or str)
            for key in id_to_keys[eid]:
                result[key] = steps
        return result

    def export_view(
        self,
        *,
        view: str,
        path: str | Path,
        encoding: Literal["auto", "av1", "h264"] = "auto",
        episode: int | str,
    ) -> Path:
        episode_id = self._episode_id_from_selector(episode)
        return super().export_view(view=view, path=path, encoding=encoding, episode_id=episode_id)


class PackDS(_EpisodeIndexCompatMixin, PackDSLanceDB):
    """Backward-compatible PackDS API with synthetic episode indices."""

    def __init__(self, *args, **kwargs):
        validate = kwargs.pop("validate", True)
        self._episode_id_order = []
        self._episode_id_to_index = {}
        self._episode_index_to_id = {}
        super().__init__(*args, validate=False, **kwargs)
        self._refresh_episode_index_cache()
        if validate and self._mode in ("r", "a"):
            is_valid, issues = self.validate()
            if not is_valid:
                issues_str = "\n  - ".join(issues)
                raise ValueError(f"Dataset validation failed for {self.path}:\n  - {issues_str}")

    def write_steps(self, data: utils.IntoSteps) -> None:
        if isinstance(data, list):
            if not data:
                return
            self._observe_step_sequence(data)
            super().write_steps(data)
            return

        from groot.core.data.packds import step

        if isinstance(data, step.PackDSStepBatch):
            if len(data) == 0:
                return
            self._observe_batch(data)
            data = step.PackDSStepBatch(
                episode_index=None,
                step_index=data.step_index,
                step_time_ns=data.step_time_ns,
                original_step_time_ns=data.original_step_time_ns,
                has_valid_action=data.has_valid_action,
                episode_id=data.episode_id,
                video=data.video,
                state=data.state,
                action=data.action,
                annotations=data.annotations,
                schema_override=data.schema_override,
                video_blob=data.video_blob,
                video_blob_frame_index=data.video_blob_frame_index,
                decoded_video=data.decoded_video,
            )
            super().write_steps(data)
            return

        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data], schema=data.schema)

        if isinstance(data, pa.Table):
            super().write_steps(self._observe_table(data))
            return

        super().write_steps(data)


__all__ = ["PackDS"]
