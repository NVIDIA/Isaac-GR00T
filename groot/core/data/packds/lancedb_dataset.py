# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LanceDB-backed PackDS implementation.

This module provides the writable LanceDB backend for PackDS. The canonical
read-only API lives in :mod:`groot.core.data.packds.reader`.

API:
    - PackDS(path, mode="r") - Open or create a dataset
    - write_steps(data: IntoSteps) - Write steps
    - read_episode_steps(episode_id: str) - Read all steps for an episode
    - get_episode_ids() -> list[str] - Get unique episode IDs
    - video_keys / state_keys / action_keys - Typed key accessors (properties)
    - close() - Flush writes, optimize, save stats (auto-called by context manager)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
import time
from typing import Any, Literal

import lance
import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import Self

from groot.core.data.packds import constants, utils
from groot.core.data.packds.episodes import (
    EpisodeInfo,
    _has_jpeg_columns,
    episode_infos_to_table,
    episodes_aggregate_columns,
    fold_episodes,
    merge_user_metadata,
)
from groot.core.data.packds.lance_write_session import LanceWriteSession, create_lance_write_session
from groot.core.data.packds.reader import PackDSReader
from groot.core.data.packds.write_session_wrappers import (
    BatchingWriteSession,
    ThreadPoolWriteSession,
)
from groot.core.data.packds.writer_types import PackDSWriteConfig

logger = logging.getLogger(__name__)


def _build_episodes_from_steps_via_disk_read(
    steps_ds: lance.LanceDataset,
    dirty_episode_ids: set[str],
    steps_schema: pa.Schema,
    *,
    embodiment: str | None = None,
) -> pa.Table:
    """Recovery-only helper: read *dirty_episode_ids* from disk and aggregate.

    Used by :func:`repair_episodes_table` when there is no in-process
    writer state to draw on.  The happy-path bulk finalize on
    ``PackDSLanceDB`` no longer goes through this code path — the
    writer keeps a per-episode :class:`EpisodeInfo` populated by
    every ``write_steps`` call and emits rows from that state with
    no read.

    Streams batches from the Lance scanner so memory stays bounded by
    one batch regardless of how many dirty episodes need rebuilding.
    """
    read_cols = episodes_aggregate_columns(steps_schema)
    dirty_arr = pa.array(list(dirty_episode_ids), type=pa.string())
    scanner = steps_ds.scanner(
        columns=read_cols,
        filter=pc.field("episode_id").isin(dirty_arr),
    )
    infos = fold_episodes(scanner.to_batches())
    if not infos:
        return constants.EPISODES_SCHEMA.empty_table()
    return episode_infos_to_table(
        infos,
        has_jpeg_schema=_has_jpeg_columns(steps_schema),
        embodiment=embodiment,
    )


_merge_user_metadata = merge_user_metadata


def _is_missing_lance_dataset_error(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in ("not found", "does not exist", "no such file"))


def _is_lance_already_exists_error(exc: BaseException) -> bool:
    return "already exists" in str(exc).lower()


def repair_episodes_table(packds_path: str, storage_options: dict[str, str] | None = None) -> int:
    """Verify and repair the episodes table after concurrent writes.

    Compares episode IDs in the steps table against the episodes table and
    rebuilds any rows that are missing (e.g., due to merge_insert commit
    failures during concurrent finalization).

    Call this after all concurrent writers have finished — typically from
    a post-processing step like ``lance_postprocess.verify_and_recover``.

    Returns the number of episodes that were repaired (0 if none missing).
    """
    from groot.core.data.packds.sources import resolve_source

    source = resolve_source(packds_path, storage_options=storage_options)
    base_uri = source.packds_uri.rstrip("/")
    opts = source.storage_options or {}

    steps_uri = base_uri + f"/{constants.TABLE_NAME}.{constants.TABLE_FORMAT_LANCE}"
    ep_uri = base_uri + f"/{constants.EPISODES_TABLE_NAME}.{constants.TABLE_FORMAT_LANCE}"

    # Read episode IDs from steps table
    steps_ds = lance.dataset(steps_uri, storage_options=opts)
    steps_eids = set(pc.unique(
        steps_ds.to_table(columns=["episode_id"]).column("episode_id"),
    ).to_pylist())

    if not steps_eids:
        return 0

    # Read episode IDs from episodes table (may not exist)
    ep_eids: set[str] = set()
    ep_exists = False
    try:
        ep_ds = lance.dataset(ep_uri, storage_options=opts)
        ep_eids = set(ep_ds.to_table(columns=["episode_id"]).column("episode_id").to_pylist())
        ep_exists = True
    except (FileNotFoundError, OSError, ValueError) as exc:
        if not _is_missing_lance_dataset_error(exc):
            raise

    missing = steps_eids - ep_eids
    if not missing:
        return 0

    logger.info(
        "repair_episodes_table: %d/%d episodes missing from episodes table, rebuilding",
        len(missing), len(steps_eids),
    )

    new_episodes = _build_episodes_from_steps_via_disk_read(
        steps_ds, missing, steps_ds.schema,
    )

    if not ep_exists:
        lance.write_dataset(new_episodes, ep_uri, mode="create", storage_options=opts)
        ep_ds = lance.dataset(ep_uri, storage_options=opts)
        ep_ds.create_scalar_index("episode_id", index_type="BTREE", replace=True)
    else:
        ep_ds = lance.dataset(ep_uri, storage_options=opts)
        for attempt in range(1, constants.COMMIT_MAX_RETRIES + 1):
            try:
                ep_ds = lance.dataset(ep_uri, storage_options=opts)
                (
                    ep_ds.merge_insert("episode_id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(new_episodes)
                )
                break
            except OSError:
                if attempt >= constants.COMMIT_MAX_RETRIES:
                    raise
                time.sleep(constants.COMMIT_RETRY_SLEEP_S * attempt)

    logger.info("repair_episodes_table: rebuilt %d episode(s)", len(missing))
    return len(missing)


class PackDSLanceDB(PackDSReader):
    """Unified interface for PackDS datasets.

    Provides read, write, and validation operations. The validate() method
    performs comprehensive integrity checks; other methods assume the dataset
    is valid for simplicity.

    All non-derivable metadata is stored in the LanceDB schema metadata
    with the ``packds:`` prefix. No external meta/ directory is used.

    Args:
        path: Dataset directory path (local or ``s3://`` URI).
        mode: "r" for read, "w" for write (create new), "a" for append.
        min_batch_size: Minimum number of steps to accumulate before writing
            a fragment. Writes from ``list[PackDSStep]`` are buffered in a
            background thread until this threshold is reached. ``pa.Table``
            and ``RecordBatch`` inputs are written immediately regardless.
            Buffered rows are only flushed when this threshold is reached or
            when :meth:`flush` / :meth:`close` is called.
        jpeg_quality: JPEG quality for video encoding (1-100).
        encode_jpeg: If True, encode video as per-frame JPEG.
        encode_video_blob: If True, encode video as multi-frame blobs.
        video_blob_crf: CRF quality parameter for video blob encoding.
        video_blob_group_size: Number of frames per video blob group.
        video_blob_codec: Dataset-level blob codec metadata — "h264" or "av1".
        write_video_blob_codec: Optional write-time blob codec override. When set,
            newly written blobs use this codec even if existing dataset metadata
            advertises a different codec.
        steps_per_second: Control frequency in Hz (write mode).
        source_format: Origin format (e.g., "xdof", "lerobot", "packds").
        source_paths: Original data paths.
        processing: Conversion processing params.
        embodiment_tag: Robot embodiment tag for relative stats (write mode).
        validate: If True (default), validate dataset integrity when opening
            in read or append mode. Set to False to skip validation.
        overwrite: If True, delete existing dataset at path when opening in
            write mode. If False (default), raise FileExistsError.
        storage_options: Optional S3/remote storage options dict for
            lancedb.connect(). For S3 paths, credentials are auto-loaded
            from gear if not provided.

    Raises:
        ValueError: If an invalid mode is provided.

    Example:
        # Writing with PackDSStep
        with PackDS("/path/to/output.packds", mode="w") as ds:
            ds.write_steps(batch_of_steps)

        # Reading with PackDSStep
        with PackDS("/path/to/dataset.packds") as ds:
            episode_steps = ds.read_episode_steps("ep_000000")
            step = episode_steps[0]
            print(ds.video_keys, ds.state_keys, ds.action_keys)

        # Validation
        ds = PackDS("/path/to/dataset.packds")
        is_valid, issues = ds.validate(check_all_rows=True)
    """

    def __init__(
        self,
        path: Path | str,
        mode: Literal["r", "w", "a"] = "r",
        min_batch_size: int = constants.DEFAULT_WRITE_BATCH_SIZE,
        jpeg_quality: int = constants.DEFAULT_JPEG_QUALITY,
        encode_jpeg: bool = False,
        encode_video_blob: bool = True,
        video_blob_crf: int = constants.DEFAULT_VIDEO_BLOB_CRF,
        video_blob_group_size: int = constants.DEFAULT_VIDEO_BLOB_GROUP_SIZE,
        video_blob_codec: Literal["h264", "av1"] = constants.DEFAULT_VIDEO_BLOB_CODEC,
        write_video_blob_codec: Literal["h264", "av1"] | None = None,
        steps_per_second: float = constants.DEFAULT_STEPS_PER_SECOND,
        source_format: str = "packds",
        source_paths: list[str] | None = None,
        processing: dict[str, Any] | None = None,
        embodiment_tag: str | None = None,
        validate: bool = True,
        overwrite: bool = False,
        storage_options: dict[str, str] | None = None,
        prefer: list[Literal["lance", "parquet"]] | None = None,
        finalize_on_close: bool = True,
        compute_stats: bool = False,
        s3_upload_threshold_bytes: int = constants.S3_UPLOAD_THRESHOLD_BYTES,
        s3_upload_workers: int = constants.S3_UPLOAD_WORKERS,
    ):
        self._init_reader_state(path, storage_options=storage_options, prefer=prefer)
        self.write_config = PackDSWriteConfig(
            encode_jpeg=encode_jpeg,
            encode_video_blob=encode_video_blob,
            embodiment_tag=embodiment_tag,
            min_batch_size=min_batch_size,
            finalize_on_close=finalize_on_close,
            compute_stats=compute_stats,
            s3_upload_threshold_bytes=s3_upload_threshold_bytes,
            s3_upload_workers=s3_upload_workers,
        )
        self.jpeg_quality = jpeg_quality
        self.video_blob_crf = video_blob_crf
        self.video_blob_group_size = video_blob_group_size
        self.video_blob_codec = video_blob_codec
        self._explicit_write_video_blob_codec = write_video_blob_codec is not None
        self._write_video_blob_codec = write_video_blob_codec or video_blob_codec
        self.steps_per_second = steps_per_second
        self.source_format = source_format
        self.source_paths = source_paths or (
            [str(path)] if utils.is_remote(str(path)) else [str(Path(path).parent)]
        )
        self.processing = processing

        self._read_db: lancedb.DBConnection | None = None
        self._steps_table: lancedb.table.Table | None = None

        self._mode = mode
        self._overwrite = overwrite
        self._write_session: BatchingWriteSession | ThreadPoolWriteSession | None = None
        self._inner_write_session: LanceWriteSession | None = None

        # Dataset-class bookkeeping (not on the session protocol)
        self._schema_hint: pa.Schema | None = None
        self._written_rows: int = 0
        self._cached_schema_metadata: dict[str, str] | None = None
        # ``_dirty_episode_ids`` shrinks as ``finalize_episode`` lands rows in
        # the companion; ``_all_episode_ids`` keeps the full session-wide view
        # so ``num_episodes`` doesn't decrement during incremental finalize.
        self._dirty_episode_ids: set[str] = set()
        self._all_episode_ids: set[str] = set()
        # Per-episode running aggregate state.  Every ``write_steps``
        # call fans rows out by ``episode_id`` and folds each slice
        # into the matching ``EpisodeInfo``, so ``finalize_episodes``
        # can emit aggregate rows without re-reading from the steps
        # table.  Keys are dropped from this dict as their rows are
        # committed to ``episodes.lance``.
        self._episode_infos: dict[str, EpisodeInfo] = {}
        # Eids whose in-memory ``EpisodeInfo`` does NOT cover all
        # on-disk steps for that episode — finalize must recover from
        # disk instead of trusting the partial in-memory state.
        # Populated when:
        #   - we open in append mode (the eid may have prior on-disk
        #     steps from an earlier writer session), OR
        #   - the eid was finalized earlier in this session, popped,
        #     and then ``write_steps`` allocated a fresh
        #     ``EpisodeInfo`` for it (which only sees the post-
        #     finalize delta).
        self._partial_episode_infos: set[str] = set()
        # Episodes-table-side scalar-index bookkeeping.  Lance scans
        # benefit from a btree index on ``episode_id``; we build it
        # lazily on the first recovery-path read so common-case
        # ingest (which never reads back) doesn't pay the build
        # cost.
        self._episode_id_index_built: bool = False
        # Schema-level fact: does the steps schema declare any JPEG
        # video columns?  Captured once on first ``write_steps`` so
        # ``EpisodeInfo.to_row`` can broadcast it onto every row.
        self._has_jpeg_schema: bool = False
        # Steps-table column name for the video-blob frame index, if
        # the schema has one.  Used to set ``has_video_blobs`` per
        # episode.
        self._blob_idx_col: str | None = None

        self._open()

        # Validate on open for existing datasets
        if validate and self._mode in ("r", "a"):
            is_valid, issues = self.validate()
            if not is_valid:
                issues_str = "\n  - ".join(issues)
                raise ValueError(f"Dataset validation failed for {self.path}:\n  - {issues_str}")

    def __getstate__(self):
        """Exclude unpicklable LanceDB connections and thread state for pickling."""
        state = self.__dict__.copy()
        state["_db"] = None
        state["steps_table"] = None
        state["schema"] = None
        state["_read_db"] = None
        state["_steps_table"] = None
        state["_schema"] = None
        state["_source"] = None
        state["_write_session"] = None
        state["_inner_write_session"] = None
        state["_schema_hint"] = None
        # Per-session working state — meaningless across processes.
        state["_episode_infos"] = {}
        state["_partial_episode_infos"] = set()
        state["_episode_id_index_built"] = False
        return state

    def __setstate__(self, state):
        """Restore from pickle — dataset class rebuilds sessions."""
        state = dict(state)
        state.pop("_db", None)
        state.pop("steps_table", None)
        state.pop("schema", None)
        self.__dict__.update(state)
        self._open()

    @property
    def _db(self) -> lancedb.DBConnection | None:
        inner = getattr(self, "_inner_write_session", None)
        if inner is not None and getattr(self, "_mode", None) in ("w", "a"):
            return inner._db  # noqa: SLF001
        return self._read_db

    @_db.setter
    def _db(self, value: lancedb.DBConnection | None) -> None:
        inner = getattr(self, "_inner_write_session", None)
        if inner is not None and getattr(self, "_mode", None) in ("w", "a"):
            inner._db = value  # noqa: SLF001
        else:
            self._read_db = value

    @property
    def steps_table(self) -> lancedb.table.Table | None:
        inner = getattr(self, "_inner_write_session", None)
        if inner is not None and getattr(self, "_mode", None) in ("w", "a"):
            return inner.steps_table
        return self._steps_table

    @steps_table.setter
    def steps_table(self, value: lancedb.table.Table | None) -> None:
        inner = getattr(self, "_inner_write_session", None)
        if inner is not None and getattr(self, "_mode", None) in ("w", "a"):
            inner.steps_table = value
        else:
            self._steps_table = value

    @property
    def schema(self) -> pa.Schema | None:
        if getattr(self, "_mode", None) in ("w", "a"):
            return getattr(self, "_schema_hint", None)
        return getattr(self, "_schema", None)

    @schema.setter
    def schema(self, value: pa.Schema | None) -> None:
        if getattr(self, "_mode", None) in ("w", "a"):
            self._schema_hint = value
        else:
            self._schema = value

    def _open(self) -> None:
        """Open the dataset based on mode."""
        if self._mode == "r":
            # DeltaCAT-resolved sources don't have a Lance root we can
            # ``lancedb.connect`` to in v5 (``packds_uri`` is no longer
            # an authoritative plan field). Defer to the PackDSReader
            # base implementation, which routes reads through
            # ``cat.read(...)`` instead.
            if (
                hasattr(self, "_resolved_source")
                and self._resolved_source is not None
                and self._resolved_source.source_key == "deltacat"
            ):
                super()._open()
                self._sync_write_config_from_schema()
                return
            # Local / direct-remote sources: open Lance directly.
            connect_path = self.path
            if hasattr(self, "_resolved_source") and self._resolved_source is not None:
                connect_path = self._resolved_source.packds_uri
            local_path = None if utils.is_remote(connect_path) else Path(connect_path)
            if local_path is not None and not local_path.exists():
                raise FileNotFoundError(f"Dataset does not exist: {connect_path}")
            connect_kwargs: dict[str, Any] = {}
            if self._storage_options is not None:
                connect_kwargs["storage_options"] = self._storage_options
            self._db = lancedb.connect(connect_path, **connect_kwargs)
            assert self._db is not None
            self.steps_table = self._db.open_table(constants.TABLE_NAME)
            self.schema = self.steps_table.to_lance().schema
            self._read_config_from_schema()
            self._sync_write_config_from_schema()
            return

        if self._mode not in ("w", "a"):
            raise ValueError(f"Invalid mode: {self._mode}")

        inner = create_lance_write_session(
            path=self.path,
            mode=self._mode,
            overwrite=self._overwrite,
            storage_options=self._storage_options,
            upload_workers=self.write_config.s3_upload_workers,
            upload_threshold_bytes=self.write_config.s3_upload_threshold_bytes,
        )
        self._inner_write_session = inner
        if utils.is_remote(self.path):
            # Remote: ThreadPool overlaps fragment creation with S3 upload.
            # Batching outside ThreadPool so small writes are cheap list appends.
            session = ThreadPoolWriteSession(inner, max_workers=1)
            session = BatchingWriteSession(session, min_batch_size=self.write_config.min_batch_size)
        else:
            # Local: DirectLanceWriteSession.write_batch uses lance.write_dataset
            # directly — no need for ThreadPool (each write is already fast and
            # atomic). Just batch to reduce the number of Lance commits.
            session = BatchingWriteSession(inner, min_batch_size=self.write_config.min_batch_size)
        self._write_session = session
        # Seed schema hint from inner session (set during construction for append)
        self._schema_hint = inner._schema  # noqa: SLF001
        if self._mode == "a":
            self._read_config_from_schema()
            if not self._explicit_write_video_blob_codec:
                self._write_video_blob_codec = self.video_blob_codec
            self._sync_write_config_from_schema()

    def _sync_write_config_from_schema(self) -> None:
        if self.schema is None:
            return
        column_names = set(self.schema.names)
        self.write_config.encode_jpeg = any(
            name.startswith(constants.VIDEO_PREFIX + ":") for name in column_names
        )
        self.write_config.encode_video_blob = any(
            name.startswith(constants.VIDEO_BLOB_PREFIX + ":") for name in column_names
        )

    # --- Schema metadata helpers ---

    def _build_schema_metadata(self) -> dict[str, str]:
        """Build schema metadata dict from current config.

        Cached — the metadata values are fixed at construction time.
        """
        if self._cached_schema_metadata is not None:
            return self._cached_schema_metadata
        meta: dict[str, str] = {
            "packds:version": str(constants.PACKDS_FORMAT_VERSION),
            "packds:steps_per_second": str(self.steps_per_second),
            "packds:jpeg_quality": str(self.jpeg_quality),
            "packds:video_blob_crf": str(self.video_blob_crf),
            "packds:video_blob_group_size": str(self.video_blob_group_size),
            "packds:video_blob_codec": self.video_blob_codec,
            "packds:source_format": self.source_format,
            "packds:source_paths": json.dumps(self.source_paths),
        }
        if self.processing is not None:
            meta["packds:processing"] = json.dumps(self.processing)
        self._cached_schema_metadata = meta
        return meta


    # --- Computed properties ---

    @property
    def num_episodes(self) -> int:
        """Number of episodes in dataset.

        Reflects the same incomplete-episode masking contract as
        :meth:`get_episode_ids` and :meth:`read_episodes`: rows in the
        ``_episodes`` companion with ``incomplete=True`` (mid-rewrite
        via DeltaCAT replace-episode) are excluded.  Older companions
        without the column count every row, matching the no-filter
        fallback in :func:`_filter_complete_episodes`.
        """
        if self._mode in ("w", "a") and self._all_episode_ids:
            return len(self._all_episode_ids)
        if self._cached_num_episodes is not None:
            return self._cached_num_episodes
        # Fast path: count rows in the sibling episodes table when it
        # has been discovered as a Lance/Parquet directory beside the
        # steps Lance dataset (LanceDB-backed sources only).
        ep_resolved = self._resolve_episodes_table()
        if ep_resolved is not None:
            ds = self._open_source(ep_resolved)
            if "incomplete" in ds.schema.names:
                # ``count_rows`` accepts a filter expression; restrict to
                # the rows that survive ``_filter_complete_episodes``.
                # NULL ``incomplete`` is treated as ``False`` (complete)
                # for backward-compat.
                filt = (pc.field("incomplete").is_null()) | (
                    pc.field("incomplete") == False  # noqa: E712
                )
                result = ds.count_rows(filter=filt)
            else:
                result = ds.count_rows()
            self._cached_num_episodes = result
            return result
        # LanceDB-backed: scan the local steps table directly.
        if self.steps_table is not None:
            result = len(self.get_episode_ids())
            self._cached_num_episodes = result
            return result
        # DeltaCAT-backed: defer to the reader, which fetches the
        # episode list from the application ``_episodes`` companion
        # via ``cat.read``.
        result = len(self.get_episode_ids())
        self._cached_num_episodes = result
        return result

    # --- Read Operations ---

    def _drain_writes(self) -> None:
        """Internal sync barrier: drain pending writes through the session.

        After this returns, every ``write_steps`` call made before it has
        landed in the on-disk Lance dataset.  Does *not* touch the
        episodes companion; use :meth:`flush` for the public sync that
        also refreshes companion ``length`` rows.
        """
        if self._write_session is not None:
            self._write_session.flush()

    def flush(self) -> None:
        """Drain pending writes and refresh companion ``length`` rows.

        After this returns:
          - every ``write_steps`` call made before it has landed in
            ``steps.lance``, and
          - every dirty episode has an ``_episodes`` row with
            ``length`` matching the on-disk step count and
            ``incomplete=True``.

        The companion-refresh half is what callers/tooling rely on to
        observe in-progress writes — without it the companion shows
        the stub ``length=0`` until ``finalize_episode`` lands.
        Internal sync points that don't need the refresh (notably
        :meth:`finalize_episodes`, which will overwrite the row with
        ``incomplete=False`` immediately afterwards) use
        :meth:`_drain_writes` directly to skip the wasted commit.
        """
        self._drain_writes()
        self._refresh_dirty_companion_lengths()

    def _refresh_dirty_companion_lengths(self) -> None:
        """Emit ``incomplete=True`` companion rows for every dirty episode.

        Reads in-memory :class:`EpisodeInfo` state — caller must have
        already drained pending writes through :meth:`_drain_writes` so
        the in-memory aggregate matches the on-disk row count for each
        dirty id.  Used by :meth:`flush`; not called from
        :meth:`finalize_episodes` (which emits the finalized
        ``incomplete=False`` row instead).
        """
        if not self._dirty_episode_ids or self._inner_write_session is None:
            return
        infos = [
            self._episode_infos[eid]
            for eid in sorted(self._dirty_episode_ids)
            if eid in self._episode_infos
        ]
        if not infos:
            return
        new_episodes = episode_infos_to_table(
            infos,
            has_jpeg_schema=self._has_jpeg_schema,
            provenance=None,
            embodiment=self.write_config.embodiment_tag,
            incomplete=True,
        )
        self._commit_episode_rows(new_episodes)

    def _commit_episode_stubs(self, episode_ids: list[str]) -> None:
        """Insert ``incomplete=True, length=0`` placeholder rows for *episode_ids*.

        Called from :meth:`write_steps` the first time each ``episode_id``
        is seen in this session.  The stub row makes the episode visible
        to crash-recovery tooling (which lists ``incomplete=True`` rows
        as "in flight") and ensures the companion is never silently
        ahead of the steps table: every episode that has steps on disk
        also has a companion row, just one that readers filter out
        until :meth:`finalize_episode` clears the flag.
        """
        if not episode_ids or self._inner_write_session is None:
            return
        now = dt.datetime.now(dt.timezone.utc)
        rows: list[dict[str, Any]] = []
        for eid in episode_ids:
            row: dict[str, Any] = {name: None for name in constants.EPISODES_SCHEMA.names}
            row["episode_id"] = eid
            row["length"] = 0
            row["has_jpegs"] = self._has_jpeg_schema
            row["embodiment"] = self.write_config.embodiment_tag
            row["modified_time"] = now
            row["incomplete"] = True
            rows.append(row)
        columns: dict[str, pa.Array] = {}
        for field in constants.EPISODES_SCHEMA:
            columns[field.name] = pa.array(
                [row[field.name] for row in rows], type=field.type
            )
        stubs = pa.table(columns, schema=constants.EPISODES_SCHEMA)
        self._commit_episode_rows(stubs)

    def _finalize(self) -> None:
        """Finalize the dataset.

        Flushes all pending writes, optimizes storage, computes stats,
        and writes all metadata to schema metadata.

        This method is intended for write mode after recording an episode.
        It is automatically called by close() when in write/append mode.
        """
        if self._mode not in ("w", "a"):
            raise RuntimeError("Cannot finalize read-only dataset")
        if utils.is_remote(self.path):
            # Finalize reads back column data to compute stats and
            # compacts fragments — both are O(dataset_size) and
            # dominated by per-request latency against S3/SwiftStack.
            # Bulk-uploading a locally-finalized dataset is
            # consistently faster.  See
            # docs/error_handling_philosophy.md — fail at the entry
            # point, not 10 000 fragment-reads later.
            raise RuntimeError(
                f"Cannot finalize a remote PackDS dataset ({self.path!r}). "
                f"Finalize-on-remote is consistently slower than staging "
                f"locally and bulk-uploading.\n"
                f"Either:\n"
                f"  - Set finalize_on_close=False to skip finalization "
                f"(the episodes companion is still written per-episode), "
                f"or\n"
                f"  - Write to a local path, finalize there, then bulk-"
                f"upload via groot.core.data.packds.staged_upload."
            )

        logger.info("finalize: flushing pending writes for %s", self.path)
        flush_start = time.perf_counter()
        # ``_drain_writes`` (not ``flush``) — the companion rows for
        # dirty ids will be rewritten with ``incomplete=False`` by
        # ``_write_episodes_table`` below, so the public flush's
        # ``incomplete=True`` refresh would be wasted commits.
        self._drain_writes()
        logger.info(
            "finalize: flush complete for %s in %.1fs",
            self.path,
            time.perf_counter() - flush_start,
        )

        if self._write_session is None:
            raise RuntimeError("Write session is not initialized")

        # Stats may need to be (re-)computed even when no rows were
        # written through this writer instance — e.g. the conversion
        # coordinator re-opens the dataset in append mode purely to
        # finalize after Ray workers wrote the actual rows.  Detect
        # that case so the early-return below doesn't skip stats.
        needs_stats = (
            self.write_config.compute_stats
            and not self._read_schema_meta_json("packds:stats")
        )
        if self._written_rows == 0 and not needs_stats:
            return

        schema_meta = self._build_schema_metadata()

        # Stats computation is expensive (reads entire dataset back into memory).
        # Only compute when explicitly requested via compute_stats_on_close=True.
        if self.write_config.compute_stats:
            stats_start = time.perf_counter()
            if needs_stats:
                logger.info("finalize: computing stats for %s", self.path)
                utils._compute_and_set_stats_to_meta(  # noqa: SLF001
                    self,
                    schema_meta,
                    embodiment_tag=self.write_config.embodiment_tag,
                )
                logger.info(
                    "finalize: stats computation complete for %s in %.1fs",
                    self.path,
                    time.perf_counter() - stats_start,
                )
            elif self.write_config.embodiment_tag:
                rel_stats = self._read_schema_meta_json("packds:relative_stats")
                if not rel_stats:
                    logger.info("finalize: computing relative stats for %s", self.path)
                    utils._compute_and_set_stats_to_meta(  # noqa: SLF001
                        self,
                        schema_meta,
                        embodiment_tag=self.write_config.embodiment_tag,
                    )
                    logger.info(
                        "finalize: relative stats computation complete for %s in %.1fs",
                        self.path,
                        time.perf_counter() - stats_start,
                    )

        if self._written_rows == 0:
            # Stats-only finalize: no new rows to commit, but stats
            # were just computed.  Persist the updated schema metadata
            # directly on the existing Lance dataset (the session's
            # ``finalize`` would no-op because nothing was written
            # through the wrappers).
            logger.info(
                "finalize: writing stats-only schema metadata for %s", self.path,
            )
            steps_uri = (
                self.path.rstrip("/")
                + f"/{constants.TABLE_NAME}.{constants.TABLE_FORMAT_LANCE}"
            )
            ds = lance.dataset(steps_uri, storage_options=self._storage_options)
            ds.update_schema_metadata(schema_meta)
            return

        logger.info("finalize: finalizing session for %s", self.path)
        finalize_start = time.perf_counter()
        self._write_session.finalize(schema_meta)
        logger.info(
            "finalize: session finalized for %s in %.1fs",
            self.path,
            time.perf_counter() - finalize_start,
        )

        # Build and write episodes table from dirty episode IDs.  Done
        # *before* any post-finalize optimize pass so that a kill /
        # OOM during compaction still leaves a fully-functional
        # dataset (steps + episodes companion) on disk.
        self._write_episodes_table()

        # Post-episodes optimize pass.  Local backend compacts
        # fragments here; staged-remote is a no-op (uploads already
        # finished in ``_finalize_storage``).
        if self._inner_write_session is not None:
            self._inner_write_session.post_episodes_optimize()

    def _episodes_table_uri(self) -> str | None:
        """Return the published URI for the episodes companion.

        Local writers publish directly under the PackDS root.  Staged remote
        writers keep ``working_path`` as a local temp directory, so use the
        session's published URI helpers to address the remote S3 table.
        """
        if self._inner_write_session is None:
            return None
        return self._inner_write_session.published_table_uri(constants.EPISODES_TABLE_NAME)

    def _commit_episode_rows(self, new_episodes: pa.Table) -> None:
        """Merge-insert *new_episodes* into ``episodes.lance``.

        Creates the dataset on the first call, otherwise upserts via
        ``merge_insert("episode_id")``.  User-provided metadata
        already on existing episode rows is preserved.  Caller is
        responsible for ensuring all writes to the steps table have
        been flushed if those rows feed into the aggregate.
        """
        assert self._inner_write_session is not None
        if new_episodes.num_rows == 0:
            return

        connect_kwargs = self._inner_write_session.published_connect_kwargs()
        ep_uri = self._episodes_table_uri()
        assert ep_uri is not None

        ep_ds = None
        try:
            ep_ds = lance.dataset(ep_uri, **connect_kwargs)
        except (FileNotFoundError, OSError, ValueError) as exc:
            if not _is_missing_lance_dataset_error(exc):
                raise

        if ep_ds is None:
            try:
                lance.write_dataset(new_episodes, ep_uri, mode="create", **connect_kwargs)
                ep_ds = lance.dataset(ep_uri, **connect_kwargs)
                ep_ds.create_scalar_index("episode_id", index_type="BTREE", replace=True)
                return
            except (OSError, ValueError) as exc:
                if not _is_lance_already_exists_error(exc):
                    raise

        # Preserve user-provided metadata from existing rows before upsert.
        ep_ds = lance.dataset(ep_uri, **connect_kwargs)
        dirty_arr = new_episodes.column("episode_id")
        existing_dirty = ep_ds.to_table(
            filter=pc.field("episode_id").isin(dirty_arr),
        )
        if existing_dirty.num_rows > 0:
            new_episodes = _merge_user_metadata(new_episodes, existing_dirty)

        for attempt in range(1, constants.COMMIT_MAX_RETRIES + 1):
            try:
                ep_ds = lance.dataset(ep_uri, **connect_kwargs)
                (
                    ep_ds.merge_insert("episode_id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(new_episodes)
                )
                return
            except OSError as exc:
                if attempt >= constants.COMMIT_MAX_RETRIES:
                    raise
                if attempt == constants.COMMIT_WARN_RETRIES:
                    logger.warning(
                        "Episodes merge_insert has failed %d times, still retrying: %s",
                        attempt, exc,
                    )
                time.sleep(constants.COMMIT_RETRY_SLEEP_S * attempt)

    def _ensure_episode_id_index(self) -> None:
        """Lazily build the steps-table ``episode_id`` scalar index.

        Used by the recovery path so single-eid (or small IN-list)
        filter reads on the steps table are point lookups instead of
        full scans.  When the index already exists (e.g. the inner
        session's ``finalize`` built it), this is a no-op — re-running
        ``create_scalar_index`` or ``optimize_indices`` here would
        bump the dataset version and race the subsequent
        ``optimize_local_storage`` Rewrite transaction in close-time
        flow.  Lance's index can serve filter reads against newer
        fragments via a scan fallback, so a slightly-stale index is
        fine for the small recovery reads we do.
        """
        assert self._inner_write_session is not None
        if self._episode_id_index_built:
            return
        steps_ds = self._inner_write_session.published_table_dataset(constants.TABLE_NAME)
        already_built = any(
            "episode_id" in idx.get("fields", [])
            for idx in steps_ds.list_indices()
        )
        if not already_built:
            logger.info(
                "finalize: building episode_id scalar index for recovery reads",
            )
            steps_ds.create_scalar_index("episode_id", index_type="BTREE")
        self._episode_id_index_built = True

    def _recover_episode_infos(self, eids: list[str]) -> list[EpisodeInfo]:
        """Aggregate steps for *eids* from disk into one :class:`EpisodeInfo` each.

        Used both by the routine append-mode finalize path (the
        in-memory aggregate is flagged ``partial`` so the steps table
        is the authoritative source) and as a fallback when the
        in-memory aggregate was never populated (e.g.
        ``finalize_episode`` called on a re-opened existing dataset
        without an intervening ``write_steps``).  Streams scanner
        batches into :func:`fold_episodes` so memory stays bounded by
        one batch regardless of how many eids are aggregated.
        """
        if not eids:
            return []
        assert self._inner_write_session is not None
        self._ensure_episode_id_index()
        steps_ds = self._inner_write_session.published_table_dataset(constants.TABLE_NAME)

        read_cols = episodes_aggregate_columns(steps_ds.schema)
        eid_arr = pa.array(eids, type=pa.string())
        scanner = steps_ds.scanner(
            columns=read_cols,
            filter=pc.field("episode_id").isin(eid_arr),
        )
        return fold_episodes(scanner.to_batches())

    def _collect_infos_for_finalize(
        self, eids: list[str],
    ) -> list[EpisodeInfo]:
        """Return one ``EpisodeInfo`` per eid, re-aggregating from disk when needed.

        Two cases route through the disk re-aggregation path:
          - eids in ``_partial_episode_infos`` (every fresh
            ``EpisodeInfo`` allocated under ``mode="a"``, or any eid
            whose info was popped by an earlier finalize and then
            re-allocated by a later ``write_steps``) — the in-memory
            aggregate covers only this writer's contribution, not
            prior on-disk steps;
          - eids with no in-memory entry at all (``finalize_episode``
            called on a re-opened existing dataset).
        Both paths produce the same result: read the matching steps
        from ``steps.lance`` and fold them into a fresh aggregate.
        """
        infos: list[EpisodeInfo] = []
        from_disk: list[str] = []
        for eid in eids:
            info = self._episode_infos.pop(eid, None)
            if eid in self._partial_episode_infos:
                from_disk.append(eid)
            elif info is not None:
                infos.append(info)
            else:
                from_disk.append(eid)
        if from_disk:
            # Routine in append mode — every ``write_steps`` under
            # ``mode="a"`` flags its eid as ``partial``, so finalize
            # always re-reads steps.lance.  Logged at INFO so bulk
            # finalize runs are visible in conversion logs; the
            # message describes the work (steps→episodes-row
            # aggregation) rather than implying error recovery.
            logger.info(
                "finalize: building %d episodes-table row(s) by aggregating "
                "steps from steps.lance",
                len(from_disk),
            )
            infos.extend(self._recover_episode_infos(from_disk))
        return infos

    def _write_episodes_table(self) -> None:
        """Bulk close-time finalize: drain any remaining dirty IDs in chunks.

        Called from ``_finalize`` after the steps-table session has
        been finalized.  Iterates ``_dirty_episode_ids`` in chunks
        (default 1000) and calls :meth:`finalize_episodes` per chunk
        — each chunk is one ``merge_insert`` into ``episodes.lance``.
        """
        if not self._dirty_episode_ids or self._inner_write_session is None:
            return
        chunk_size = constants.EPISODES_FINALIZE_CHUNK_SIZE
        remaining = sorted(self._dirty_episode_ids)
        total = len(remaining)
        bulk_start = time.perf_counter()
        logger.info(
            "finalize: bulk-finalizing %d remaining dirty episode(s) in chunks of %d",
            total, chunk_size,
        )
        last_log = bulk_start
        done = 0
        for start in range(0, total, chunk_size):
            chunk = remaining[start : start + chunk_size]
            self.finalize_episodes(chunk)
            done += len(chunk)
            now = time.perf_counter()
            if now - last_log >= 5.0 or done == total:
                logger.info(
                    "finalize: bulk %d/%d episode(s) committed (%.1fs elapsed)",
                    done, total, now - bulk_start,
                )
                last_log = now

    def delete_episodes(self, episode_ids: str | list[str]) -> None:
        """Delete *episode_ids* from both the steps and episodes-companion tables.

        Accepts a single ID or a list. Drains pending writes so the delete
        sees the latest state, then issues row-level deletes against
        ``steps.lance`` and ``episodes.lance``. Idempotent — missing
        rows / missing companion are both fine.
        """
        if self._mode not in ("w", "a"):
            raise RuntimeError("Cannot delete_episodes on a read-only dataset")
        if self._inner_write_session is None:
            return
        if isinstance(episode_ids, str):
            ids = [episode_ids]
        else:
            ids = list(episode_ids)
        if not ids:
            return
        # ``_drain_writes`` not ``flush``: we're about to delete these
        # eids from the companion anyway, so the public flush's
        # ``incomplete=True`` refresh would be immediately undone.
        self._drain_writes()

        connect_kwargs = self._inner_write_session.published_connect_kwargs()
        steps_uri = self._inner_write_session.published_table_uri(constants.TABLE_NAME)
        predicate = self._episode_id_predicate(ids)
        try:
            steps_ds = lance.dataset(steps_uri, **connect_kwargs)
        except (FileNotFoundError, OSError, ValueError) as exc:
            if not _is_missing_lance_dataset_error(exc):
                raise
        else:
            steps_ds.delete(predicate)

        ep_uri = self._episodes_table_uri()
        if ep_uri is not None:
            try:
                ep_ds = lance.dataset(ep_uri, **connect_kwargs)
            except (FileNotFoundError, OSError, ValueError) as exc:
                if not _is_missing_lance_dataset_error(exc):
                    raise
            else:
                ep_ds.delete(predicate)

        for eid in ids:
            self._dirty_episode_ids.discard(eid)
            self._all_episode_ids.discard(eid)
            self._episode_infos.pop(eid, None)
            self._partial_episode_infos.discard(eid)
        self._cached_num_episodes = None

    @staticmethod
    def _episode_id_predicate(episode_ids: list[str]) -> str:
        """Build a SQL predicate matching rows whose episode_id is in *episode_ids*."""
        literals = ", ".join(
            "'" + eid.replace("'", "''") + "'" for eid in episode_ids
        )
        return f"episode_id IN ({literals})"

    def finalize_episode(self, episode_id: str) -> None:
        """Per-episode upsert into the ``episodes.lance`` companion.

        Pops the in-memory :class:`EpisodeInfo` for *episode_id*, builds
        a one-row aggregate, and merge-inserts it.  No read of the
        steps table on the happy path.  Falls back to a recovery
        read (single-eid filter against an indexed scan) if the
        in-memory state is gone — e.g., when ``finalize_episode`` is
        called on a re-opened existing dataset.

        Call after writing the last step of an episode; presence in
        the companion is the reader-side "ingest finished" signal.
        Idempotent.

        For the bulk case where many episodes are ready at once,
        prefer :meth:`finalize_episodes` — it amortises the
        ``merge_insert`` commit cost across all rows in one call.
        """
        if self._mode not in ("w", "a"):
            raise RuntimeError("Cannot finalize_episode on a read-only dataset")
        if self._inner_write_session is None:
            return
        self.finalize_episodes([episode_id])

    def finalize_episodes(self, episode_ids: list[str]) -> None:
        """Batched per-episode upsert.  One ``merge_insert`` per call.

        Pops the in-memory :class:`EpisodeInfo` for each id (or
        recovers it from the steps table when missing), assembles a
        single ``EPISODES_SCHEMA`` table, and commits via one
        ``merge_insert`` against ``episodes.lance``.  Use this from
        the convert pipeline (or any caller writing many episodes
        in a row) to avoid 1-Lance-commit-per-episode overhead.

        Idempotent on already-finalized ids — they're simply
        re-emitted with the same aggregate values.
        """
        if self._mode not in ("w", "a"):
            raise RuntimeError("Cannot finalize_episodes on a read-only dataset")
        if self._inner_write_session is None:
            return
        if not episode_ids:
            return
        # ``_drain_writes`` not ``flush``: the ``incomplete=False`` row
        # we're about to commit replaces any ``incomplete=True`` stub
        # already in the companion, so the public flush's refresh
        # would be a wasted commit on the same rows.
        self._drain_writes()
        infos = self._collect_infos_for_finalize(list(episode_ids))
        if not infos:
            for eid in episode_ids:
                self._dirty_episode_ids.discard(eid)
            return
        new_episodes = episode_infos_to_table(
            infos,
            has_jpeg_schema=self._has_jpeg_schema,
            # Provenance per-episode lives in the steps-table schema
            # metadata for the LanceDB backend; the episodes
            # companion's provenance columns are populated by the
            # DeltaCAT path only.  Leave them null here.
            provenance=None,
            embodiment=self.write_config.embodiment_tag,
        )
        self._commit_episode_rows(new_episodes)
        for eid in episode_ids:
            self._dirty_episode_ids.discard(eid)
            self._partial_episode_infos.discard(eid)
        self._cached_num_episodes = None

    def close(self) -> None:
        """Finalize and close the dataset.

        When in write/append mode, automatically calls _finalize() to flush
        pending writes, optimize storage, and save stats to schema metadata.

        When ``finalize_on_close=False`` we still promote every dirty
        episode's companion row to ``incomplete=False`` so the dataset
        is reader-visible — just without the stats compute / fragment
        optimize that ``_finalize`` adds on top.

        All errors propagate — per docs/error_handling_philosophy.md,
        incorrect behavior must be impossible to miss.
        """
        if self._closed:
            return

        try:
            if self._mode in ("w", "a"):
                if self.write_config.finalize_on_close:
                    self._finalize()
                elif self._dirty_episode_ids:
                    # ``finalize_episodes`` drains pending writes and
                    # commits the final ``incomplete=False`` rows.  An
                    # explicit ``flush()`` would do the same drain plus
                    # an ``incomplete=True`` refresh that we'd overwrite
                    # one line later, so go straight to the finalize.
                    self.finalize_episodes(list(self._dirty_episode_ids))
                else:
                    self._drain_writes()
        finally:
            if self._write_session is not None:
                self._write_session.close()
                self._write_session = None
            self._inner_write_session = None
            self._db = None
            self.steps_table = None
            self.schema = None
            self._source = None
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()

    def write_steps(self, data: utils.IntoSteps) -> None:
        """Write steps to the dataset."""
        if self._mode not in ("w", "a"):
            raise RuntimeError("Dataset not open for writing")
        if self._write_session is None:
            raise RuntimeError("Write session is not initialized")

        # Fast path: pa.Table and pa.RecordBatch skip into_table entirely
        if isinstance(data, pa.Table):
            table = data
        elif isinstance(data, pa.RecordBatch):
            table = pa.Table.from_batches([data], schema=data.schema)
        else:
            table = utils.into_table(
                data,
                schema=self._schema_hint,
                episode_id_validator=utils.validate_episode_id_for_write,
                encode_jpeg=self.write_config.encode_jpeg,
                encode_video_blob=self.write_config.encode_video_blob,
                group_size=self.video_blob_group_size,
                crf=self.video_blob_crf,
                jpeg_quality=self.jpeg_quality,
                codec=self._write_video_blob_codec,
                schema_metadata=self._build_schema_metadata(),
            )
        if table.num_rows > 0:
            if self._schema_hint is None:
                self._schema_hint = table.schema
                # Cache schema-level facts so ``EpisodeInfo.to_row``
                # doesn't have to recompute them per episode.
                self._has_jpeg_schema = any(
                    n.startswith(f"{constants.VIDEO_PREFIX}:")
                    and not n.startswith(f"{constants.VIDEO_BLOB_PREFIX}:")
                    and not n.startswith(
                        f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:"
                    )
                    for n in table.schema.names
                )
                blob_idx_cols = [
                    n for n in table.schema.names
                    if n.startswith(f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:")
                ]
                self._blob_idx_col = blob_idx_cols[0] if blob_idx_cols else None
            self._written_rows += table.num_rows
            # Fan out by ``episode_id`` so each slice flows through
            # the matching :class:`EpisodeInfo`.  Single-episode
            # writes (the common case for the convert pipeline) skip
            # the slicing call.
            unique_eids = pc.unique(table.column("episode_id")).to_pylist()
            self._update_episode_infos(table, unique_eids)
            new_eids: list[str] = []
            for eid in unique_eids:
                eid_str = str(eid)
                if eid_str not in self._all_episode_ids:
                    new_eids.append(eid_str)
                self._dirty_episode_ids.add(eid_str)
                self._all_episode_ids.add(eid_str)
            # Stub-first ordering: commit ``incomplete=True`` rows for
            # any newly-seen episode_ids BEFORE handing the step batch
            # to the write session.  A crash between stub-commit and
            # step-write leaves a stub the reader will filter out; the
            # inverse order would leave orphan steps with no companion
            # row.
            if new_eids:
                self._commit_episode_stubs(new_eids)
        self._write_session.write_batch(table)

    def _update_episode_infos(
        self,
        table: pa.Table,
        unique_eids: list[str] | None = None,
    ) -> None:
        """Fold *table* rows into per-episode :class:`EpisodeInfo` state.

        Splits *table* by ``episode_id`` and calls
        :meth:`EpisodeInfo.update` for each slice.  When *table* has
        a single distinct ``episode_id`` the slice is the table
        itself — the common ingest case avoids any filter cost.
        """
        if unique_eids is None:
            unique_eids = pc.unique(table.column("episode_id")).to_pylist()
        # A *fresh* allocation needs the partial flag whenever prior
        # on-disk steps may exist for this eid — either we opened in
        # append mode, or this is a re-write after an earlier
        # finalize popped the info (signalled by ``eid_str`` already
        # being in ``_all_episode_ids``).
        appending = self._mode == "a"
        if len(unique_eids) == 1:
            (eid,) = unique_eids
            eid_str = str(eid)
            info = self._episode_infos.get(eid_str)
            if info is None:
                info = EpisodeInfo(eid_str)
                self._episode_infos[eid_str] = info
                if appending or eid_str in self._all_episode_ids:
                    self._partial_episode_infos.add(eid_str)
            info.update(table, blob_idx_col=self._blob_idx_col)
            return
        for eid in unique_eids:
            eid_str = str(eid)
            slice_ = table.filter(
                pc.equal(table.column("episode_id"), eid)
            )
            info = self._episode_infos.get(eid_str)
            if info is None:
                info = EpisodeInfo(eid_str)
                self._episode_infos[eid_str] = info
                if appending or eid_str in self._all_episode_ids:
                    self._partial_episode_infos.add(eid_str)
            info.update(slice_, blob_idx_col=self._blob_idx_col)


