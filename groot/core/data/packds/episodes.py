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
"""Shared helpers for computing the PackDS episodes table.

Both the LanceDB and DeltaCAT writers materialize an aggregate
"episodes" table from per-episode aggregate state.  This module owns
the aggregation core so neither backend can drift from the canonical
schema in ``constants.EPISODES_SCHEMA``.

Layering:

- :class:`EpisodeInfo` — incremental per-episode aggregator.  ``update``
  folds a chunk's contributions into running scalars; ``to_row`` emits
  the final row dict.
- :func:`fold_episodes` — constant-memory streaming primitive.  Takes
  an iterable of step batches (one materialised table, a Lance
  scanner's ``to_batches()``, etc.) and returns one fresh
  :class:`EpisodeInfo` per distinct ``episode_id`` seen.
- :func:`episode_infos_to_table` — assembles a list of finalized
  ``EpisodeInfo`` objects into an ``EPISODES_SCHEMA`` Arrow table.
- :func:`compute_episodes_from_steps_table` — thin convenience wrapper
  around the two for callers that already have a materialised steps
  subset and want a typed table back (the DeltaCAT per-episode
  finalize path).

The LanceDB writer keeps a ``dict[episode_id, EpisodeInfo]`` populated
by ``write_steps`` (no read-back on the happy path); recovery and the
DeltaCAT finalize path both go through :func:`fold_episodes`.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from groot.core.data.packds import constants

logger = logging.getLogger(__name__)


class EpisodeInfo:
    """Incremental per-episode aggregate.

    Fold ``pa.Table`` / ``pa.RecordBatch`` chunks (rows for
    ``episode_id`` only) into the running state via :meth:`update`,
    then emit a row dict via :meth:`to_row`.  Caller is responsible
    for ensuring each chunk only contains rows for ``self.episode_id``.

    Per-instance memory: ~150 bytes plus one optional task-name
    string.  The writer holds one of these per *open* episode_id;
    eager ``finalize_episodes`` calls drop them as soon as the row
    lands in ``episodes.lance``.
    """

    __slots__ = (
        "episode_id",
        "_count",
        "_time_min",
        "_time_max",
        "_valid_sum",
        "_task_at_min",
        "_task_first",
        "_has_blob",
    )

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._count: int = 0
        self._time_min: int | None = None
        self._time_max: int | None = None
        self._valid_sum: int = 0
        # Earliest ``step_time_ns`` at which a non-null
        # ``annotation:task`` was observed, plus the task itself.
        self._task_at_min: int | None = None
        self._task_first: str | None = None
        self._has_blob: bool = False

    def update(
        self,
        chunk: pa.Table | pa.RecordBatch,
        *,
        task_col: str = "annotation:task",
        blob_idx_col: str | None = None,
    ) -> None:
        """Fold *chunk* into the running aggregate.

        *chunk* must contain only rows for ``self.episode_id``.  Any
        of ``task_col`` / ``blob_idx_col`` columns absent from
        *chunk* are silently skipped — the recovery path may project
        only what it needs.
        """
        if isinstance(chunk, pa.RecordBatch):
            table = pa.Table.from_batches([chunk])
        else:
            table = chunk
        if table.num_rows == 0:
            return

        self._count += table.num_rows
        self._valid_sum += int(
            pc.sum(table.column("has_valid_action")).as_py() or 0
        )
        chunk_min = pc.min(table.column("step_time_ns")).as_py()
        chunk_max = pc.max(table.column("step_time_ns")).as_py()
        if chunk_min is not None:
            self._time_min = (
                chunk_min if self._time_min is None else min(self._time_min, chunk_min)
            )
        if chunk_max is not None:
            self._time_max = (
                chunk_max if self._time_max is None else max(self._time_max, chunk_max)
            )

        # Task-name extraction.  Filter to non-null tasks first so the
        # min-by-time scan only touches relevant rows.
        if task_col in table.column_names:
            task_view = table.select(["step_time_ns", task_col])
            task_view = task_view.filter(pc.is_valid(task_view.column(task_col)))
            if task_view.num_rows > 0:
                # Earliest non-null task in this chunk.
                idx = pc.min_max(task_view.column("step_time_ns")).as_py()
                if idx is not None and idx["min"] is not None:
                    chunk_task_at_min = idx["min"]
                    # Find the row whose step_time_ns matches that min.
                    matches = task_view.filter(
                        pc.equal(task_view.column("step_time_ns"), chunk_task_at_min)
                    )
                    if matches.num_rows > 0:
                        chunk_task = matches.column(task_col)[0].as_py()
                        if (
                            self._task_at_min is None
                            or chunk_task_at_min < self._task_at_min
                        ):
                            self._task_at_min = chunk_task_at_min
                            self._task_first = chunk_task

        # Has-blob: any non-null blob_idx_col → flip the flag.
        if blob_idx_col is not None and blob_idx_col in table.column_names:
            valid = pc.is_valid(table.column(blob_idx_col))
            if pc.any(valid).as_py():
                self._has_blob = True

    def to_row(
        self,
        *,
        has_jpeg_schema: bool,
        provenance: dict[str, Any] | None,
        embodiment: str | None,
        now: dt.datetime,
        incomplete: bool = False,
    ) -> dict[str, Any]:
        """Return one ``EPISODES_SCHEMA``-shaped row as a column→value dict.

        ``has_jpeg_schema`` is dataset-level (depends on the steps
        schema, not on the rows seen here).  Provenance / embodiment
        flow in unchanged.  ``now`` is the ``modified_time`` value;
        callers should pass a single timestamp for the whole batch
        being finalized so rows in one merge_insert share it.

        ``incomplete`` lets callers emit a mid-write row that readers
        will filter out via :func:`_filter_complete_episodes`.  The
        dataset class uses ``incomplete=True`` for stub/refresh rows
        emitted during ingest and ``incomplete=False`` for finalized
        rows.
        """
        time_min = self._time_min if self._time_min is not None else 0
        time_max = self._time_max if self._time_max is not None else 0
        prov = provenance or {}
        row: dict[str, Any] = {
            "episode_id": self.episode_id,
            "length": self._count,
            "duration_ns": time_max - time_min,
            "num_valid_action_steps": self._valid_sum,
            "task_name": self._task_first,
            "has_video_blobs": self._has_blob,
            "has_jpegs": has_jpeg_schema,
            "start_time": None,
            "modified_time": now,
            "incomplete": incomplete,
            "vendor": None,
            "operator": None,
            "robot": None,
            "embodiment": embodiment,
            "version": None,
        }
        for field_name in constants.EPISODES_PROVENANCE_FIELDS:
            row[field_name] = prov.get(field_name)
        return row


def episode_infos_to_table(
    infos: list[EpisodeInfo],
    *,
    has_jpeg_schema: bool,
    provenance: dict[str, Any] | None = None,
    embodiment: str | None = None,
    now: dt.datetime | None = None,
    incomplete: bool = False,
) -> pa.Table:
    """Assemble an ``EPISODES_SCHEMA``-shaped ``pa.Table`` from finalized infos.

    All rows share ``modified_time`` (=*now*), ``has_jpegs``,
    provenance / embodiment, and the ``incomplete`` flag.  Per-episode
    fields come from each ``EpisodeInfo``'s running state.
    """
    if not infos:
        return constants.EPISODES_SCHEMA.empty_table()
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    n = len(infos)
    rows = [
        info.to_row(
            has_jpeg_schema=has_jpeg_schema,
            provenance=provenance,
            embodiment=embodiment,
            now=now,
            incomplete=incomplete,
        )
        for info in infos
    ]
    columns: dict[str, pa.Array] = {}
    for field in constants.EPISODES_SCHEMA:
        values = [row[field.name] for row in rows]
        columns[field.name] = pa.array(values, type=field.type)
    assert len(columns["episode_id"]) == n
    return pa.table(columns, schema=constants.EPISODES_SCHEMA)


def fold_episodes(
    batches: Iterable[pa.Table | pa.RecordBatch],
    *,
    blob_idx_col: str | None = None,
) -> list[EpisodeInfo]:
    """Fold step batches into per-episode :class:`EpisodeInfo` aggregates.

    Constant memory: caller controls iteration (``[table]`` for an
    in-RAM subset, ``scanner.to_batches()`` to stream).  Rows for the
    same eid may straddle multiple batches.  ``blob_idx_col`` is
    sniffed once on the first non-empty batch when *None*.
    """
    infos: dict[str, EpisodeInfo] = {}
    sniffed = False
    for batch in batches:
        if isinstance(batch, pa.RecordBatch):
            batch = pa.Table.from_batches([batch])
        if batch.num_rows == 0:
            continue
        if not sniffed and blob_idx_col is None:
            blob_idx_cols = [
                n for n in batch.column_names
                if n.startswith(f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:")
            ]
            blob_idx_col = blob_idx_cols[0] if blob_idx_cols else None
            sniffed = True
        unique = pc.unique(batch.column("episode_id")).to_pylist()
        if len(unique) == 1:
            eid_str = str(unique[0])
            info = infos.setdefault(eid_str, EpisodeInfo(eid_str))
            info.update(batch, blob_idx_col=blob_idx_col)
            continue
        for eid in unique:
            eid_str = str(eid)
            slice_ = batch.filter(pc.equal(batch.column("episode_id"), eid))
            info = infos.setdefault(eid_str, EpisodeInfo(eid_str))
            info.update(slice_, blob_idx_col=blob_idx_col)
    return list(infos.values())


def compute_episodes_from_steps_table(
    steps: pa.Table,
    steps_schema: pa.Schema,
    *,
    provenance: dict[str, Any] | None = None,
    embodiment: str | None = None,
) -> pa.Table:
    """Build an ``EPISODES_SCHEMA`` table from a materialised steps subset.

    Convenience wrapper around :func:`fold_episodes` +
    :func:`episode_infos_to_table` for callers that already have the
    subset in RAM (the DeltaCAT per-episode finalize path).  Streaming
    callers should compose the two directly.
    """
    infos = fold_episodes([steps])
    if not infos:
        return constants.EPISODES_SCHEMA.empty_table()
    return episode_infos_to_table(
        infos,
        has_jpeg_schema=_has_jpeg_columns(steps_schema),
        provenance=provenance,
        embodiment=embodiment,
    )


def _has_jpeg_columns(steps_schema: pa.Schema) -> bool:
    # Dataset-level fact — a steps subset that happens to contain only
    # blob-encoded videos can't answer it, so the schema is the source.
    return any(
        n.startswith(f"{constants.VIDEO_PREFIX}:")
        and not n.startswith(f"{constants.VIDEO_BLOB_PREFIX}:")
        and not n.startswith(f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:")
        for n in steps_schema.names
    )


def merge_user_metadata(
    new_episodes: pa.Table,
    existing_dirty: pa.Table,
) -> pa.Table:
    """Carry forward user-provided metadata from *existing_dirty* into *new_episodes*.

    Columns like ``vendor`` cannot be recomputed from steps; the
    upsert path reads them off the existing episode row before writing.
    """
    user_cols = ["start_time", "vendor", "operator", "robot", "embodiment", "version"]
    existing_by_eid: dict[str, int] = {}
    for i, eid in enumerate(existing_dirty.column("episode_id").to_pylist()):
        existing_by_eid[eid] = i

    arrays = {}
    for col_name in new_episodes.schema.names:
        if col_name in user_cols and col_name in existing_dirty.schema.names:
            new_col = new_episodes.column(col_name)
            existing_col = existing_dirty.column(col_name)
            vals = new_col.to_pylist()
            new_eids = new_episodes.column("episode_id").to_pylist()
            for j, eid in enumerate(new_eids):
                if eid in existing_by_eid:
                    existing_val = existing_col[existing_by_eid[eid]].as_py()
                    if existing_val is not None:
                        vals[j] = existing_val
            arrays[col_name] = pa.array(vals, type=new_episodes.schema.field(col_name).type)
        else:
            arrays[col_name] = new_episodes.column(col_name)
    return pa.table(arrays, schema=new_episodes.schema)


def episodes_aggregate_columns(steps_schema: pa.Schema) -> list[str]:
    """Return the steps columns needed to compute the episodes aggregate.

    Used by writers to project away heavy columns (video blobs, state,
    action) before reading the steps subset back from storage. The list
    includes ``step_index`` so DeltaCAT readers that sort the post-read
    table by composite merge keys ``(episode_id, step_index)`` find the
    column they expect — it adds negligible cost to the read.
    """
    cols = ["episode_id", "step_index", "step_time_ns", "has_valid_action"]
    if "annotation:task" in steps_schema.names:
        cols.append("annotation:task")
    cols.extend(
        n for n in steps_schema.names
        if n.startswith(f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:")
    )
    return cols
