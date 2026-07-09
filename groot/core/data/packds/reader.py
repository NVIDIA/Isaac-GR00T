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
"""Backend-agnostic read-only PackDS reader built on Arrow datasets."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads

from groot.core.data.packds import constants, utils
from groot.core.data.packds.backends import TABLE_BACKENDS, ResolvedTable
from groot.core.data.packds.sources import DeltaCatSourceResolver, resolve_source


def _backfill_video_blobs(table: pa.Table) -> pa.Table:
    """Forward-fill null video blob values in a step_index-sorted table.

    Video blob encoding stores the compressed blob only on the first row
    of each group ("leader", frame_index=0).  Remaining rows in the group
    have null.  This function copies the leader blob to all rows in its
    group so every row is self-contained.

    The eager duplication is intentional for the ``read_episodes()``
    Arrow-table API. Callers often filter or reorder the returned table
    before converting it to a batch or decoding video. In those flows the
    original leader row may be dropped or move away from ``i -
    frame_index``, so lazy leader lookup would stop working. Forward-fill
    keeps each row independently decodable even after downstream table
    filtering, while still allowing batch decoders to cache repeated blob
    payloads shared by rows from the same group.

    Operates on ``pa.Table`` columns directly — no Python per-row loop.
    Returns a new table with the blob columns replaced (or the same table
    if there are no blob columns).
    """
    blob_cols = [
        name for name in table.column_names
        if name.startswith(constants.VIDEO_BLOB_PREFIX + ":")
    ]
    if not blob_cols:
        return table

    replacements: dict[str, pa.Array] = {}
    for col_name in blob_cols:
        col = table.column(col_name)
        if not pc.any(pc.is_null(col)).as_py():
            continue  # no nulls to fill

        # Forward-fill: propagate each non-null value to subsequent nulls.
        # Arrow doesn't have a native fill_forward for binary columns, so
        # we build the filled array in one pass over the pylist.
        values = col.to_pylist()
        current: bytes | None = None
        for i, v in enumerate(values):
            if v is not None:
                current = v
            elif current is not None:
                values[i] = current
        replacements[col_name] = pa.array(values, type=col.type)

    if not replacements:
        return table

    # Replace columns in the table
    arrays = []
    for name in table.column_names:
        if name in replacements:
            arrays.append(replacements[name])
        else:
            arrays.append(table.column(name))
    return pa.table(dict(zip(table.column_names, arrays, strict=True)), schema=table.schema)


logger = logging.getLogger(__name__)


def _filter_complete_episodes(ep_table: pa.Table) -> pa.Table:
    """Drop rows whose ``incomplete`` column is ``True``.

    Used by every reader code path that surfaces episode metadata so
    the DeltaCAT replace-episode flow's mid-rewrite ``incomplete=True``
    rows are invisible to consumers.  ``NULL`` is treated as ``False``
    for backward-compat with older ``_episodes`` companions that
    predate the column.

    No-op when the column is absent (returns *ep_table* unchanged).
    """
    if "incomplete" not in ep_table.column_names:
        return ep_table
    incomplete = ep_table.column("incomplete")
    # ``fill_null(False)`` coerces NULL → False; ``invert`` then keeps
    # everything that isn't True.
    keep_mask = pc.invert(pc.fill_null(incomplete, False))
    return ep_table.filter(keep_mask)


class PackDSReader:
    """Canonical read-only PackDS API.

    Reads PackDS datasets using Arrow dataset abstractions. The constructor
    discovers a readable table representation inside a PackDS directory and
    normalizes it to a ``pyarrow.dataset.Dataset`` where possible.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        validate: bool = True,
        storage_options: dict[str, str] | None = None,
        prefer: list[Literal["lance", "parquet"]] | None = None,
        deltacat_client: Any | None = None,
        deltacat_plan_kwargs: Mapping[str, Any] | None = None,
    ):
        self._init_reader_state(
            path,
            storage_options=storage_options,
            prefer=prefer,
            deltacat_client=deltacat_client,
            deltacat_plan_kwargs=deltacat_plan_kwargs,
        )
        self._open()
        if validate:
            is_valid, issues = self.validate()
            if not is_valid:
                issues_str = "\n  - ".join(issues)
                raise ValueError(f"Dataset validation failed for {self.path}:\n  - {issues_str}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["schema"] = None
        state["_source"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def _init_reader_state(
        self,
        path: Path | str,
        *,
        storage_options: dict[str, str] | None = None,
        prefer: list[Literal["lance", "parquet"]] | None = None,
        deltacat_client: Any | None = None,
        deltacat_plan_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = str(Path(path)) if not utils.is_remote(str(path)) else str(path)
        # Retain the explicit client (if any) so the merge-on-read fetch
        # in ``_source_dataset`` doesn't re-derive a different client from
        # env vars / registry — important for tests that supply a fake
        # client and for callers using a non-default catalog server.
        self._explicit_deltacat_client = deltacat_client
        self._resolved_source = resolve_source(
            self.path,
            storage_options=storage_options,
            deltacat_client=deltacat_client,
            deltacat_plan_kwargs=deltacat_plan_kwargs,
        )
        self._storage_options = self._resolved_source.storage_options
        self._prefer = tuple(prefer) if prefer is not None else None
        self._resolved_table = None
        self._resolved_episodes_table: ResolvedTable | None = None
        self._resolved_episodes_table_done = False
        self._source = None
        self.schema = None
        self._closed = False
        self._cached_num_episodes = None

        self.jpeg_quality = constants.DEFAULT_JPEG_QUALITY
        self.video_blob_crf = constants.DEFAULT_VIDEO_BLOB_CRF
        self.video_blob_group_size = constants.DEFAULT_VIDEO_BLOB_GROUP_SIZE
        self.video_blob_codec = constants.DEFAULT_VIDEO_BLOB_CODEC
        self.steps_per_second = constants.DEFAULT_STEPS_PER_SECOND
        self.source_format = "packds"
        self.source_paths = [self._default_source_parent(self._resolved_source.packds_uri)]
        self.processing = None

    @staticmethod
    def _default_source_parent(path: str) -> str:
        if utils.is_remote(path):
            return path.rsplit("/", 1)[0] if "/" in path else path
        return str(Path(path).parent)

    @property
    def resolved_source(self) -> str:
        return self._resolved_source.source_key

    @property
    def resolved_source_uri(self) -> str:
        return self._resolved_source.packds_uri

    @property
    def resolved_backend(self) -> str | None:
        return None if self._resolved_table is None else self._resolved_table.backend_key

    @property
    def resolved_format(self) -> str | None:
        return None if self._resolved_table is None else self._resolved_table.format_name

    @property
    def resolved_uri(self) -> str | None:
        return None if self._resolved_table is None else self._resolved_table.uri

    def _preferred_formats(self) -> tuple[str, ...]:
        preferred = self._prefer or ()
        invalid = [format_name for format_name in preferred if format_name not in constants.TABLE_FORMATS]
        if invalid:
            raise ValueError(f"Unsupported formats in prefer=: {invalid}")
        return preferred

    def _select_table(self, candidates: list[ResolvedTable]) -> ResolvedTable:
        if not candidates:
            raise FileNotFoundError(f"No readable PackDS table found under {self.path}")
        if len(candidates) == 1:
            return candidates[0]

        available_formats = sorted({candidate.format_name for candidate in candidates})
        if self._prefer is None:
            if len(available_formats) == 1:
                return candidates[0]
            raise ValueError(
                "Multiple PackDS table formats found under "
                f"{self.path}: {available_formats}. Pass prefer=[...] to choose one."
            )

        for format_name in self._preferred_formats():
            matches = [candidate for candidate in candidates if candidate.format_name == format_name]
            if matches:
                return matches[0]

        if len(available_formats) == 1:
            return candidates[0]

        raise ValueError(
            f"No preferred format found under {self.path}. "
            f"Available formats: {available_formats}; prefer={list(self._preferred_formats())}."
        )

    def _open_source(self, resolved: ResolvedTable) -> pads.Dataset:
        backend = TABLE_BACKENDS[resolved.backend_key]
        return backend.open_dataset(resolved, storage_options=self._storage_options)

    def _resolve_table(self) -> ResolvedTable:
        if self._resolved_table is not None:
            return self._resolved_table

        # DeltaCAT sources don't enumerate Lance directories — the
        # authoritative read path is the catalog's merge-on-read view,
        # not ``plan.packds_uri`` (which only points at the latest delta).
        # We synthesize a sentinel ResolvedTable so the rest of the reader
        # plumbing keeps working; ``_open_source`` / ``_source_dataset``
        # branch on this below.
        if self._resolved_source.source_key == DeltaCatSourceResolver.source_key:
            self._resolved_table = ResolvedTable(
                table_name=constants.TABLE_NAME,
                backend_key="deltacat",
                format_name=constants.TABLE_FORMAT_LANCE,
                uri=self._resolved_source.packds_uri,
            )
            return self._resolved_table

        candidates: list[ResolvedTable] = []
        for backend in TABLE_BACKENDS.values():
            for candidate in backend.discover(
                self._resolved_source.packds_uri,
                table_name=constants.TABLE_NAME,
            ):
                try:
                    self._open_source(candidate)
                except (FileNotFoundError, NotImplementedError, OSError, ValueError):
                    continue
                candidates.append(candidate)

        self._resolved_table = self._select_table(candidates)
        return self._resolved_table

    def _resolve_episodes_table(self) -> ResolvedTable | None:
        """Discover the episodes table.  Returns ``None`` if not found.

        DeltaCAT sources skip sibling-directory discovery — the episodes
        companion is a separate DeltaCAT table reached via
        :meth:`_read_episodes_table` instead.

        Resolution is cached on the first call (``_resolved_episodes_table``)
        with ``_resolved_episodes_table_done`` distinguishing
        "not yet looked" from "looked and found nothing".
        """
        if self._resolved_episodes_table_done:
            return self._resolved_episodes_table
        self._resolved_episodes_table_done = True
        if self._resolved_source.source_key == DeltaCatSourceResolver.source_key:
            return None
        for backend in TABLE_BACKENDS.values():
            for candidate in backend.discover(
                self._resolved_source.packds_uri,
                table_name=constants.EPISODES_TABLE_NAME,
            ):
                try:
                    self._open_source(candidate)
                except (FileNotFoundError, NotImplementedError, OSError, ValueError):
                    continue
                self._resolved_episodes_table = candidate
                return candidate
        return None

    @staticmethod
    def _is_missing_table_or_projection_error(
        exc: BaseException,
        columns: list[str] | None,
    ) -> bool:
        """Return True for expected companion-missing compatibility cases."""
        if isinstance(exc, FileNotFoundError):
            return True
        msg = str(exc).lower()
        if any(marker in msg for marker in ("not found", "does not exist", "no such file")):
            return True
        return columns is not None and (
            "no field named" in msg or "fieldnotfound" in msg
        )

    def _read_episodes_table(
        self,
        columns: list[str] | None = None,
    ) -> pa.Table | None:
        """Read the episodes table, returning ``None`` for absent/old companions."""
        try:
            if self._resolved_source.source_key == DeltaCatSourceResolver.source_key:
                return DeltaCatSourceResolver.read_companion_table(
                    self._resolved_source,
                    "episodes",
                    columns=columns,
                    deltacat_client=self._explicit_deltacat_client,
                )
            resolved = self._resolve_episodes_table()
            if resolved is None:
                return None
            ds = self._open_source(resolved)
            return ds.to_table(columns=columns)
        except (FileNotFoundError, OSError, ValueError) as exc:
            if self._is_missing_table_or_projection_error(exc, columns):
                return None
            raise

    def _read_episodes_table_with_task_column(self) -> pa.Table | None:
        """Read episodes-table rows including whichever task column exists.

        v5 stores the first temporal task as ``task_name`` (single string).
        Older datasets stored all unique tasks as ``tasks`` (list[string]).
        Try the new column first; fall back to the legacy one.  Returning
        ``None`` means the companion table is absent entirely.

        Also drops rows whose ``incomplete`` flag is ``True``: those are
        episodes mid-rewrite by the DeltaCAT replace-episode flow and
        their steps-table partition is in flight.  Rows whose
        ``incomplete`` column is missing or null pre-date the flag and
        are kept (treated as fully-ingested).
        """
        ep_table = self._read_episodes_table(
            columns=["episode_id", "length", "task_name", "incomplete"],
        )
        if ep_table is None:
            ep_table = self._read_episodes_table(
                columns=["episode_id", "length", "tasks", "incomplete"],
            )
        # Older datasets predate ``incomplete`` and ``task_name`` —
        # retry without each in turn so we keep working against
        # pre-v5 ``_episodes`` companions.
        if ep_table is None:
            ep_table = self._read_episodes_table(
                columns=["episode_id", "length", "task_name"],
            )
        if ep_table is None:
            ep_table = self._read_episodes_table(
                columns=["episode_id", "length", "tasks"],
            )
        if ep_table is None:
            return None
        return _filter_complete_episodes(ep_table)

    @staticmethod
    def _episodes_row_tasks(ep_table: pa.Table, row: int) -> list[str]:
        """Extract the legacy ``tasks: list[str]`` view from one episodes row."""
        if "task_name" in ep_table.column_names:
            value = ep_table.column("task_name")[row].as_py()
            return [value] if value else []
        if "tasks" in ep_table.column_names:
            tasks_val = ep_table.column("tasks")[row].as_py()
            return sorted(t for t in tasks_val if t) if tasks_val else []
        return []

    def _source_dataset(self) -> pads.Dataset:
        if self._source is not None:
            return self._source

        # Always resolve the table first so ``resolved_format`` /
        # ``resolved_backend`` reflect the chosen backend, even on the
        # DeltaCAT path which doesn't go through ``_open_source``.
        resolved = self._resolve_table()

        if self._resolved_source.source_key == DeltaCatSourceResolver.source_key:
            # Materialize the steps table once. Used for indexed reads
            # (``read_raw_indices``), schema discovery, and any filter
            # we can't translate to a DeltaCAT JSON-AST predicate.
            #
            # Episode-scoped reads bypass this path entirely via the
            # ``_deltacat_filtered_read`` shortcut so the heavy
            # materialization only happens when truly needed.
            self._source = self._read_deltacat_steps_dataset()
            return self._source

        self._source = self._open_source(resolved)
        return self._source

    def _validate_read_columns(self, columns: list[str] | None) -> list[str] | None:
        """Fail before backend reads if a requested step column is absent."""
        if columns is None:
            return None
        requested = list(columns)
        if self.schema is None:
            return requested
        schema_names = set(self.schema.names)
        missing = [col for col in dict.fromkeys(requested) if col not in schema_names]
        if missing:
            raise ValueError(
                f"Requested PackDS column(s) not present: {missing}. "
                f"Available columns: {self.schema.names}"
            )
        return requested

    def _read_deltacat_steps_dataset(
        self,
        *,
        filter_predicate: dict[str, Any] | None = None,
        columns: list[str] | None = None,
    ) -> pads.Dataset:
        """Materialize a steps-table view for a dc:// source via ``cat.read``.

        When *filter_predicate* is supplied, the read is partition-pruned
        server-side; the result is *not* cached on ``self._source`` since
        it's a partial view.
        """
        catalog = self._resolved_source.metadata.get("catalog")
        namespace = self._resolved_source.metadata.get("namespace")
        table = self._resolved_source.metadata.get("table")
        if not namespace or not table:
            raise RuntimeError(
                f"DeltaCAT source for {self.path!r} is missing namespace/"
                f"table metadata; cannot materialize steps table."
            )
        client = self._explicit_deltacat_client
        if client is None:
            client = DeltaCatSourceResolver._client_from_env(catalog)  # noqa: SLF001
        if client is None:
            raise RuntimeError(
                f"No DeltaCAT client available for {self.path!r}: set "
                f"DELTACAT_SERVER_URL or register the {catalog!r} catalog "
                f"via register_deltacat_catalog(...)."
            )
        read_columns = self._validate_read_columns(columns)
        kwargs: dict[str, Any] = {
            "namespace": namespace,
            "table": table,
            "read_as": "pyarrow",
        }
        if read_columns is not None:
            kwargs["columns"] = read_columns
        if filter_predicate is not None:
            kwargs["filter_predicate"] = filter_predicate
        else:
            # Unfiltered PackDS reads hit the v5 read-scope guard by
            # default. The reader needs the full materialization for
            # schema discovery, ``num_steps``, ``read_raw_indices``, and
            # the ``validate()`` row sample — all of which are intended
            # uses, just expensive at scale. Per-episode access via
            # ``read_episodes`` continues to push its filter through and
            # never hits this branch.
            kwargs["allow_full_packds_scan"] = True
        steps = client.catalog.read(**kwargs)
        return pads.dataset(steps)

    def _read_table(
        self,
        *,
        columns: list[str] | None = None,
        filter_expr: pc.Expression | None = None,
        indices: list[int] | None = None,
    ) -> pa.Table:
        source = self._source_dataset()
        read_columns = self._validate_read_columns(columns)
        if indices is not None:
            table = source.take(indices, columns=read_columns)
            if filter_expr is not None:
                table = table.filter(filter_expr)
            return table
        return source.to_table(columns=read_columns, filter=filter_expr)

    def _read_table_for_episode_ids(
        self,
        episode_ids: list[str],
        *,
        columns: list[str] | None,
    ) -> pa.Table:
        """Episode-id-scoped read with DeltaCAT partition pruning.

        For DeltaCAT sources, pushes ``episode_id`` through to
        ``cat.read(filter_predicate=...)`` so only the target episode
        partitions are materialized. For local/remote PackDS sources,
        falls through to ``_read_table`` with an in-memory filter.
        """
        if not episode_ids:
            return self._read_table(filter_expr=None, columns=columns).slice(0, 0)
        if self._resolved_source.source_key == DeltaCatSourceResolver.source_key:
            predicate: dict[str, Any]
            if len(episode_ids) == 1:
                predicate = {"eq": ["episode_id", episode_ids[0]]}
            else:
                predicate = {"in": ["episode_id", list(episode_ids)]}
            ds = self._read_deltacat_steps_dataset(
                filter_predicate=predicate, columns=columns,
            )
            return ds.to_table(columns=columns)
        if len(episode_ids) == 1:
            expr = pc.field("episode_id") == pc.scalar(episode_ids[0])
        else:
            expr = pc.field("episode_id").isin(episode_ids)
        return self._read_table(filter_expr=expr, columns=columns)

    def _count_rows(self) -> int:
        return self._source_dataset().count_rows()

    def _open(self) -> None:
        local_path = None
        if not utils.is_remote(self._resolved_source.packds_uri):
            local_path = Path(self._resolved_source.packds_uri)
        if local_path is not None and not local_path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {self._resolved_source.packds_uri}")
        self.schema = self._source_dataset().schema
        self._read_config_from_schema()

    def _read_schema_meta(self, key: str) -> str | None:
        if self.schema is None or self.schema.metadata is None:
            return None
        raw = self.schema.metadata.get(key.encode("utf-8"))
        return None if raw is None else raw.decode("utf-8")

    def _read_schema_meta_json(self, key: str) -> Any:
        raw = self._read_schema_meta(key)
        return None if raw is None else json.loads(raw)

    def _read_config_from_schema(self) -> None:
        sps = self._read_schema_meta("packds:steps_per_second")
        if sps is not None:
            self.steps_per_second = float(sps)
        jq = self._read_schema_meta("packds:jpeg_quality")
        if jq is not None:
            self.jpeg_quality = int(jq)
        sf = self._read_schema_meta("packds:source_format")
        if sf is not None:
            self.source_format = sf
        sp = self._read_schema_meta_json("packds:source_paths")
        if sp is not None:
            self.source_paths = sp
        blob_crf = self._read_schema_meta("packds:video_blob_crf")
        if blob_crf is not None:
            self.video_blob_crf = int(blob_crf)
        blob_group_size = self._read_schema_meta("packds:video_blob_group_size")
        if blob_group_size is not None:
            self.video_blob_group_size = int(blob_group_size)
        blob_codec = self._read_schema_meta("packds:video_blob_codec")
        if blob_codec is not None:
            self.video_blob_codec = blob_codec
        proc = self._read_schema_meta_json("packds:processing")
        if proc is not None:
            self.processing = proc

    def _get_columns(self) -> list[str]:
        if self.schema is None:
            raise RuntimeError("Dataset not open for reading")
        return self.schema.names

    @property
    def stats(self) -> dict[str, Any]:
        result = self._read_schema_meta_json("packds:stats")
        return result if result is not None else {}

    @property
    def relative_stats(self) -> dict[str, Any] | None:
        return self._read_schema_meta_json("packds:relative_stats")

    @property
    def num_steps(self) -> int:
        return self._count_rows()

    @property
    def num_episodes(self) -> int:
        if self._cached_num_episodes is None:
            self._cached_num_episodes = len(self.get_episode_ids())
        return self._cached_num_episodes

    @property
    def video_keys(self) -> list[str]:
        keys: set[str] = set()
        for col in self._get_columns():
            if col.startswith(constants.VIDEO_PREFIX + ":") or col.startswith(
                constants.VIDEO_BLOB_PREFIX + ":"
            ):
                keys.add(utils.column_to_key(col)[1])
        return sorted(keys)

    @property
    def video_blob_keys(self) -> list[str]:
        return sorted(
            utils.column_to_key(col)[1]
            for col in self._get_columns()
            if col.startswith(constants.VIDEO_BLOB_PREFIX + ":")
        )

    @property
    def state_keys(self) -> list[str]:
        return sorted(
            utils.column_to_key(col)[1]
            for col in self._get_columns()
            if col.startswith(constants.STATE_PREFIX + ":")
        )

    @property
    def action_keys(self) -> list[str]:
        return sorted(
            utils.column_to_key(col)[1]
            for col in self._get_columns()
            if col.startswith(constants.ACTION_PREFIX + ":")
        )

    def get_episode_ids(self) -> list[str]:
        # Project ``incomplete`` so :func:`_filter_complete_episodes`
        # can hide mid-rewrite rows from the DeltaCAT replace-episode
        # flow.  Older companions that predate the column fall back
        # to the bare-``episode_id`` projection (no filter applied).
        ep_table = self._read_episodes_table(
            columns=["episode_id", "incomplete"],
        )
        if ep_table is None:
            ep_table = self._read_episodes_table(columns=["episode_id"])
        if ep_table is not None:
            ep_table = _filter_complete_episodes(ep_table)
            return sorted(ep_table.column("episode_id").to_pylist())
        # Fallback: scan the steps table
        table = self._read_table(columns=["episode_id"])
        return sorted(pc.unique(table.column("episode_id")).to_pylist())

    def _drop_incomplete_episode_ids(self, episode_ids: list[str]) -> list[str]:
        """Filter *episode_ids* down to those not marked ``incomplete=True``.

        Reads the ``_episodes`` companion's ``incomplete`` column for the
        requested ids only.  Returns *episode_ids* unchanged when the
        companion is absent or predates the ``incomplete`` column —
        keeps backward compatibility with older datasets.
        """
        if not episode_ids:
            return episode_ids
        ep_table = self._read_episodes_table(
            columns=["episode_id", "incomplete"],
        )
        if ep_table is None or "incomplete" not in ep_table.column_names:
            return list(episode_ids)
        # Restrict to rows whose ``episode_id`` is in the requested set,
        # then apply the same incomplete-mask helper.
        requested_arr = pa.array([str(eid) for eid in episode_ids], type=pa.string())
        keep = ep_table.filter(pc.is_in(ep_table.column("episode_id"), requested_arr))
        keep = _filter_complete_episodes(keep)
        complete_ids = set(keep.column("episode_id").to_pylist())
        return [eid for eid in episode_ids if str(eid) in complete_ids]

    def get_episode_lengths(self, episode_ids: list[str]) -> dict[str, int]:
        """Return the number of steps in each requested episode.

        Fast path reads the ``length`` column from the ``_episodes``
        companion table; fallback counts rows per ``episode_id`` in the
        steps table for older datasets that lack the companion. Episodes
        absent from the dataset (or mid-rewrite via the DeltaCAT
        replace-episode flow) are omitted from the returned mapping.

        Args:
            episode_ids: Episode IDs to look up.

        Returns:
            Dict mapping each found episode_id to its step count.
        """
        if not episode_ids:
            return {}

        requested = set(episode_ids)

        ep_table = self._read_episodes_table(
            columns=["episode_id", "length", "incomplete"],
        )
        if ep_table is None:
            ep_table = self._read_episodes_table(columns=["episode_id", "length"])
        if ep_table is not None:
            ep_table = _filter_complete_episodes(ep_table)
            ids = ep_table.column("episode_id").to_pylist()
            lengths = ep_table.column("length").to_pylist()
            return {
                str(eid): int(length)
                for eid, length in zip(ids, lengths, strict=True)
                if str(eid) in requested
            }

        # Fallback: count rows per episode_id in the steps table.
        table = self._read_table_for_episode_ids(
            list(requested), columns=["episode_id"],
        )
        counts: dict[str, int] = {}
        for eid in table.column("episode_id").to_pylist():
            eid_str = str(eid)
            counts[eid_str] = counts.get(eid_str, 0) + 1
        return counts

    def filter_table(self, expr: pc.Expression, columns: list[str] | None = None) -> pa.Table:
        return self._read_table(columns=columns, filter_expr=expr)

    def get_episodes_metadata(self) -> list[dict[str, Any]]:
        # Fast path: read from episodes table if available.  v5 schema
        # exposes the first temporal task as ``task_name`` (single
        # string); older datasets exposed all unique tasks as ``tasks``
        # (list[string]).  Read whichever column exists.
        ep_table = self._read_episodes_table_with_task_column()
        all_columns = self._get_columns()
        has_sub_task = "annotation:sub_task" in all_columns

        if ep_table is not None and not has_sub_task:
            # Episodes table has everything we need — no steps scan required
            episodes = []
            for i in range(ep_table.num_rows):
                tasks_val = self._episodes_row_tasks(ep_table, i)
                episodes.append({
                    "trajectory_id": str(ep_table.column("episode_id")[i].as_py()),
                    "length": int(ep_table.column("length")[i].as_py()),
                    "tasks": tasks_val,
                    "sub_tasks": [],
                })
            return episodes

        if ep_table is not None and has_sub_task:
            # Episodes table provides length/tasks; still need steps for sub_tasks
            ep_meta = {}
            for i in range(ep_table.num_rows):
                eid = str(ep_table.column("episode_id")[i].as_py())
                tasks_val = self._episodes_row_tasks(ep_table, i)
                ep_meta[eid] = {
                    "length": int(ep_table.column("length")[i].as_py()),
                    "tasks": tasks_val,
                }
            # Read only sub_task columns from steps table
            sub_table = self._read_table(
                columns=["episode_id", "step_index", "annotation:sub_task"],
            )
            sub_table = sub_table.sort_by([
                ("episode_id", "ascending"),
                ("step_index", "ascending"),
            ])
            ep_id_list = sub_table.column("episode_id").to_pylist()
            step_index_list = sub_table.column("step_index").to_pylist()
            sub_task_list = sub_table.column("annotation:sub_task").to_pylist()

            groups: dict[str, list[int]] = {}
            for i, eid in enumerate(ep_id_list):
                groups.setdefault(str(eid), []).append(i)

            episodes = []
            for ep_id in sorted(ep_meta):
                meta = ep_meta[ep_id]
                sub_tasks = self._extract_sub_tasks(
                    groups.get(ep_id, []), step_index_list, sub_task_list,
                )
                episodes.append({
                    "trajectory_id": ep_id,
                    "length": meta["length"],
                    "tasks": meta["tasks"],
                    "sub_tasks": sub_tasks,
                })
            return episodes

        # Fallback: full steps table scan (no episodes table).  Task annotations
        # are optional in PackDS; don't request columns that are absent from the
        # schema.  Public projection APIs still fail loudly for missing columns.
        has_task = "annotation:task" in all_columns
        columns = ["episode_id", "step_index"]
        if has_task:
            columns.append("annotation:task")
        if has_sub_task:
            columns.append("annotation:sub_task")

        table = self._read_table(columns=columns)
        if table.num_rows == 0:
            return []

        table = table.sort_by(
            [
                ("episode_id", "ascending"),
                ("step_index", "ascending"),
            ]
        )
        ep_id_list = table.column("episode_id").to_pylist()
        step_index_list = table.column("step_index").to_pylist()
        if has_task:
            task_list = table.column("annotation:task").to_pylist()
        else:
            task_list = [None] * table.num_rows
        if has_sub_task:
            sub_task_list = table.column("annotation:sub_task").to_pylist()
        else:
            sub_task_list = None

        groups = {}
        for i, eid in enumerate(ep_id_list):
            groups.setdefault(str(eid), []).append(i)

        episodes = []
        for ep_id, row_indices in groups.items():
            length = len(row_indices)
            tasks = sorted({task_list[i] for i in row_indices if task_list[i]})
            sub_tasks = (
                self._extract_sub_tasks(row_indices, step_index_list, sub_task_list)
                if has_sub_task else []
            )
            episodes.append(
                {
                    "trajectory_id": str(ep_id),
                    "length": length,
                    "tasks": tasks,
                    "sub_tasks": sub_tasks,
                }
            )
        return episodes

    @staticmethod
    def _extract_sub_tasks(
        row_indices: list[int],
        step_index_list: list[int],
        sub_task_list: list[str | None],
    ) -> list[dict[str, Any]]:
        """Extract sub_task segments from step-level annotations."""
        sub_tasks: list[dict[str, Any]] = []
        step_indices = [step_index_list[i] for i in row_indices]
        sub_task_vals = [sub_task_list[i] for i in row_indices]
        current_text: str | None = None
        seg_start = 0
        for i, (sidx, val) in enumerate(zip(step_indices, sub_task_vals, strict=True)):
            text = val if val and str(val).strip() else None
            if text != current_text:
                if current_text is not None:
                    sub_tasks.append(
                        {
                            "start": int(seg_start),
                            "end": int(step_indices[i - 1]) + 1,
                            "text": current_text,
                        }
                    )
                current_text = text
                seg_start = int(sidx)
        if current_text is not None:
            sub_tasks.append(
                {
                    "start": int(seg_start),
                    "end": int(step_indices[-1]) + 1,
                    "text": current_text,
                }
            )
        return sub_tasks

    def read_raw_indices(self, indices: list[int], columns: list[str] | None = None) -> pa.Table:
        """Read rows by absolute table indices and return an Arrow table.

        This is the low-level random-access read used by the training
        dataloader (:class:`PackSingleStepDataset`) for shuffled
        minibatch construction, and by :mod:`stats` for streaming
        statistics over sequential chunks.

        Prefer :meth:`read_episodes` when you want complete episodes —
        it is faster (one ``isin`` filter vs. per-row takes) and returns
        tables sorted by ``step_index`` with video blobs already
        backfilled.

        Core columns (``episode_id``, ``step_index``, ``step_time_ns``,
        ``original_step_time_ns``, ``has_valid_action``) are always
        included regardless of *columns*.  When a ``video_blob:*``
        column is requested its matching ``video_blob_frame_index:*``
        column is auto-included, **and** any sampled row whose blob
        leader is *not* in the index set has its leader fetched
        separately and copied into the row's blob column — so every
        returned row has a self-contained blob payload (matching
        :meth:`read_episodes`'s eager-backfill contract).

        Callers that need a :class:`PackDSStepBatch` can wrap the
        result::

            batch = from_table(ds.read_raw_indices(indices))

        Args:
            indices: Absolute row indices into the underlying table.
            columns: Optional column subset for projection.

        Returns:
            ``pa.Table`` with the requested rows and columns.
        """
        if not indices:
            return self._read_table(indices=[], columns=columns)
        if columns is not None:
            core = {
                "step_index",
                "step_time_ns",
                "original_step_time_ns",
                "has_valid_action",
                "episode_id",
            }
            columns = list(core | set(columns))
            for col in list(columns):
                if col.startswith(constants.VIDEO_BLOB_PREFIX + ":"):
                    fi_col = f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:{col.split(':', 1)[1]}"
                    if fi_col not in columns:
                        columns.append(fi_col)
        table = self._read_table(indices=indices, columns=columns)
        return self._backfill_random_access_blob_leaders(table, indices)

    def _backfill_random_access_blob_leaders(
        self, table: pa.Table, indices: list[int],
    ) -> pa.Table:
        """Backfill video-blob columns in a random-access read.

        For every row whose ``video_blob:<key>`` is null and whose
        ``video_blob_frame_index:<key>`` is > 0, the leader row sits
        at absolute table index ``indices[r] - frame_index``.  When the
        sampled index set didn't include that leader, the row's blob is
        unrecoverable from within the returned table — :func:`from_table`
        + :func:`_decode_batch_video` would silently produce missing
        frames.  Fetch any such leaders here and splice their blob
        bytes onto the affected rows so every returned row carries its
        own decodable blob.
        """
        blob_cols = [
            name for name in table.column_names
            if name.startswith(constants.VIDEO_BLOB_PREFIX + ":")
        ]
        if not blob_cols:
            return table

        replacements: dict[str, pa.Array] = {}
        for blob_col in blob_cols:
            key = blob_col.split(":", 1)[1]
            fi_col = f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:{key}"
            if fi_col not in table.column_names:
                continue
            blob_values = table.column(blob_col).to_pylist()
            fi_values = table.column(fi_col).to_pylist()

            # leader_abs_idx -> list of result-row indices that need it.
            needed: dict[int, list[int]] = {}
            for r, (blob, fi) in enumerate(zip(blob_values, fi_values, strict=True)):
                if blob is not None or fi is None or fi == 0:
                    continue
                leader_abs = int(indices[r]) - int(fi)
                needed.setdefault(leader_abs, []).append(r)
            if not needed:
                continue

            # One extra ``take`` per blob column with missing leaders.
            leader_idxs = list(needed.keys())
            leaders = self._read_table(indices=leader_idxs, columns=[blob_col])
            leader_blobs = leaders.column(blob_col).to_pylist()
            for li, leader_abs in enumerate(leader_idxs):
                payload = leader_blobs[li]
                if payload is None:
                    # Leader row genuinely has no blob — leave the
                    # downstream null intact so decode surfaces the gap.
                    continue
                for r in needed[leader_abs]:
                    blob_values[r] = payload
            replacements[blob_col] = pa.array(blob_values, type=table.column(blob_col).type)

        if not replacements:
            return table
        arrays = [
            replacements.get(name, table.column(name))
            for name in table.column_names
        ]
        return pa.table(
            dict(zip(table.column_names, arrays, strict=True)),
            schema=table.schema,
        )

    def read_episodes(
        self,
        episode_ids: list[str],
        *,
        columns: list[str] | None = None,
    ) -> dict[str, pa.Table]:
        """Read multiple episodes as Arrow tables.

        Uses a single ``isin`` filter + column projection for efficient
        batched I/O (especially over S3).  Each returned table is sorted
        by ``step_index``.  Video blob columns are backfilled so every
        row contains the full blob (not just leader rows).

        The backfill is deliberate: unlike ``read_batch()``, this API is
        designed to hand back Arrow tables that callers may slice, filter,
        or regroup before decoding. Making every row carry its blob keeps
        the table robust to those downstream transforms instead of relying
        on the original leader row remaining present and in-position.

        Args:
            episode_ids: Episode IDs to read.
            columns: Optional column subset.  Core columns (episode_id,
                step_index, step_time_ns, etc.) are always included.
                If a ``video_blob:*`` column is requested, its matching
                ``video_blob_frame_index:*`` column is auto-included.

        Returns:
            Dict mapping each requested episode_id to its ``pa.Table``.
        """
        if not episode_ids:
            return {}

        # Skip ids that the ``_episodes`` companion marks ``incomplete=True``
        # — they're mid-rewrite via the DeltaCAT replace-episode flow and
        # exposing partial step rows would violate the documented atomic
        # replace contract.  Older companions that lack the column behave
        # the same as today (no rows skipped).
        episode_ids = self._drop_incomplete_episode_ids(episode_ids)
        if not episode_ids:
            return {}

        if columns is not None:
            core = {"episode_id", "step_index", "step_time_ns",
                    "original_step_time_ns", "has_valid_action"}
            col_set = set(columns) | core
            # Auto-include frame_index columns for requested blob columns
            for col in list(col_set):
                if col.startswith(constants.VIDEO_BLOB_PREFIX + ":"):
                    fi_col = f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:{col.split(':', 1)[1]}"
                    col_set.add(fi_col)
            columns = sorted(col_set)

        if len(episode_ids) == 1:
            eid = episode_ids[0]
            table = self._read_table_for_episode_ids(
                [eid], columns=columns,
            ).sort_by("step_index")
            return {eid: _backfill_video_blobs(table)}

        table = self._read_table_for_episode_ids(episode_ids, columns=columns)

        # Split by episode_id
        result: dict[str, pa.Table] = {}
        ep_col = table.column("episode_id")
        for eid in episode_ids:
            mask = pc.equal(ep_col, eid)
            ep_table = table.filter(mask).sort_by("step_index")
            if ep_table.num_rows > 0:
                result[eid] = _backfill_video_blobs(ep_table)
        return result

    def read_episode_steps(
        self,
        episode_id: str | list[str],
    ) -> list | dict[str, list]:
        """Read episode steps by episode ID.

        .. deprecated::
            Use :meth:`read_episodes` + :func:`~groot.core.data.packds.from_table`
            instead::

                table = ds.read_episodes(["ep_000"])["ep_000"]
                batch = from_table(table)

        Thin wrapper around :meth:`read_episodes` that converts Arrow
        tables to ``list[PackDSStep]``.

        Args:
            episode_id: A single episode ID string, or a list of episode
                IDs for batched reading.

        Returns:
            For a single ID: ``list[PackDSStep]`` sorted by step_index.
            For a list: ``dict[str, list[PackDSStep]]`` keyed by episode ID.
        """
        if isinstance(episode_id, list):
            unique_ids = list(dict.fromkeys(episode_id))
            tables = self.read_episodes(unique_ids)
            return {
                eid: list(utils.from_table(table))
                for eid, table in tables.items()
            }
        episode_id = utils.validate_episode_id_for_write(episode_id)
        tables = self.read_episodes([episode_id])
        if episode_id not in tables:
            return []
        return list(utils.from_table(tables[episode_id]))

    def export_view(
        self,
        *,
        view: str,
        path: str | Path,
        encoding: Literal["auto", "av1", "h264"] = "auto",
        episode_id: str,
    ) -> Path:
        from groot.core.data.packds import video

        if view not in self.video_keys:
            raise ValueError(f"View {view!r} not found. Available views: {self.video_keys}")

        out = Path(path)
        if not out.suffix:
            out = out.with_suffix(".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        columns = self._get_columns()
        vb_col = utils.key_to_column(constants.VIDEO_BLOB_PREFIX, view)
        jpeg_col = utils.key_to_column(constants.VIDEO_PREFIX, view)
        has_blobs = vb_col in columns
        has_jpeg = jpeg_col in columns

        episode_id = utils.validate_episode_id_for_write(episode_id)
        expr = pc.field("episode_id") == pc.scalar(episode_id)

        if has_blobs:
            vb_fi_col = utils.key_to_column(constants.VIDEO_BLOB_FRAME_INDEX_PREFIX, view)
            table = self._read_table(columns=[vb_col, vb_fi_col, "step_index"], filter_expr=expr)
            si = table.column("step_index").to_pylist()
            fi = table.column(vb_fi_col).to_pylist()
            blobs = table.column(vb_col).to_pylist()
            order = sorted(range(len(si)), key=lambda i: si[i])
            ordered_blobs = [blobs[i] for i in order if fi[i] == 0 and blobs[i] is not None]
            if not ordered_blobs:
                raise RuntimeError(f"No video blobs for view {view!r} episode {episode_id!r}")
            blob_codecs = [video.probe_video_blob_codec(blob) for blob in ordered_blobs]
            if encoding == "auto":
                unique_blob_codecs = {codec for codec in blob_codecs if codec is not None}
                encoding = next(iter(unique_blob_codecs)) if len(unique_blob_codecs) == 1 else "h264"
            video.export_video_blobs(
                ordered_blobs,
                out,
                stored_codecs=blob_codecs,
                target_codec=encoding,
                fps=self.steps_per_second,
            )
        elif has_jpeg:
            if encoding == "auto":
                encoding = "h264"
            table = self._read_table(columns=[jpeg_col, "step_index"], filter_expr=expr)
            si = table.column("step_index").to_pylist()
            jpegs = table.column(jpeg_col).to_pylist()
            order = sorted(range(len(si)), key=lambda i: si[i])
            ordered_jpegs = [jpegs[i] for i in order if jpegs[i] is not None]
            if not ordered_jpegs:
                raise RuntimeError(f"No JPEG frames for view {view!r} episode {episode_id!r}")
            video.export_jpeg_frames(
                ordered_jpegs,
                out,
                target_codec=encoding,
                fps=self.steps_per_second,
            )
        else:
            raise RuntimeError(f"No video data for view {view!r}")

        logger.info("Exported %s episode %s → %s (%s)", view, episode_id, out, encoding)
        return out

    def validate(
        self,
        check_all_rows: bool = False,
        sample_size: int = constants.VALIDATE_DEFAULT_SAMPLE_SIZE,
    ) -> tuple[bool, list[str]]:
        issues: list[str] = []
        num_rows = self.num_steps
        if num_rows == 0:
            return False, ["Table has 0 rows"]
        # Stats are optional metadata — not having them is fine for
        # DeltaCAT-backed datasets where metadata lives in table properties.
        id_table = self._read_table(columns=["episode_id", "step_index"])
        ep_ids = id_table.column("episode_id").to_pylist()
        step_indices = id_table.column("step_index").to_pylist()
        seen = set()
        for eid, si in zip(ep_ids, step_indices, strict=True):
            pair = (eid, si)
            if pair in seen:
                issues.append(f"Duplicate (episode_id, step_index) pair: {pair}")
                break
            seen.add(pair)
        if check_all_rows:
            indices_to_check = list(range(num_rows))
        else:
            indices_to_check = sorted(
                np.random.default_rng()
                .choice(num_rows, size=min(sample_size, num_rows), replace=False)
                .tolist()
            )
        # Validate that rows are readable using only the core required columns.
        # We don't attempt to decode all state/action/video columns here —
        # extra columns with different shapes (e.g. from mixed writes to a
        # DeltaCAT table) would cause spurious reshape errors.
        core_columns = [c for c in constants.REQUIRED_COLUMNS if c in self.schema.names]
        try:
            for start in range(0, len(indices_to_check), constants.VALIDATE_BATCH_SIZE):
                chunk = indices_to_check[start : start + constants.VALIDATE_BATCH_SIZE]
                table = self._read_table(indices=chunk, columns=core_columns)
                if table.num_rows != len(chunk):
                    issues.append("read returned unexpected row count")
        except Exception as e:  # noqa: BLE001
            issues.append(f"read failed: {e}")
        return len(issues) == 0, issues

    def close(self) -> None:
        self._source = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()
