# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Literal

import lance
import lancedb
import pyarrow as pa

from groot.core.data.packds import constants, utils
from groot.core.data.packds.s5cmd_runner import S5cmdRunner

logger = logging.getLogger(__name__)

_RETRYABLE_COMMIT_SUBSTRINGS = (
    "incompatible transaction",
    "commit conflict",
    "version conflict",
    "too many concurrent writers",
)


def is_retryable_commit_error(exc: OSError) -> bool:
    """Return True if *exc* indicates a Lance commit that definitely did NOT land.

    Only version/transaction conflicts are safe to retry.  Any other ``OSError``
    is ambiguous — the commit may have succeeded on the storage backend — so
    retrying would risk creating duplicate rows.
    """
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_COMMIT_SUBSTRINGS)


def _frag_file_paths(frags: list[Any]) -> set[str]:
    """Extract data-file paths from Lance fragment metadata objects."""
    paths: set[str] = set()
    for frag in frags:
        for f in frag.files:
            paths.add(f.path)
    return paths


def _fragments_committed(
    uri: str, frags: list[Any], storage_options: dict[str, str]
) -> bool:
    """Return True if every data file in *frags* is already committed at *uri*.

    Used after an ambiguous commit error to decide whether the commit actually
    landed (safe to return) or definitely did not (safe to retry).
    """
    target = _frag_file_paths(frags)
    if not target:
        return True
    ds = lance.dataset(uri, storage_options=storage_options)
    for frag in ds.get_fragments():
        for df in frag.data_files():
            target.discard(df.path)
            if not target:
                return True
    return not target


class _StreamingUploader:
    """Uploads local Lance files to S3 in background via ``s5cmd``.

    Per-upload debug logging is enabled by default at INFO level so a
    convert/ingest run shows when each batch is queued, picked up by
    a worker, and finished — together with the in-flight count.  Use
    these lines to diagnose throughput stalls (e.g. queue draining
    below ``num_workers`` because the writer is the bottleneck, or
    single uploads dominating because one fragment is huge).
    """

    def __init__(
        self,
        *,
        local_base: str,
        remote_base: str,
        storage_options: dict[str, str],
        schema: pa.Schema,
        num_workers: int = constants.S3_UPLOAD_WORKERS,
        upload_threshold_bytes: int = constants.S3_UPLOAD_THRESHOLD_BYTES,
    ) -> None:
        self._local_base = Path(local_base)
        self._remote_base = remote_base
        self._opts = {**storage_options, "conditional_put": "etag"}
        self._schema = schema
        self._remote_dataset_created = False
        self._uploaded: set[str] = set()
        self._pending_files: list[Path] = []
        self._pending_frags: list[Any] = []
        self._pending_bytes = 0
        self._num_workers = num_workers
        self._upload_threshold_bytes = upload_threshold_bytes
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self._futures: list[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        # Held by *one* worker at a time during the actual Lance
        # commit.  Workers that finish uploading append their frags to
        # ``_frags_to_commit`` and attempt a non-blocking acquire of
        # this lock; whoever wins drains the entire queue in a single
        # batched commit.  Others return immediately (their frag is
        # already queued and will land in the next batch).  This keeps
        # the upload pool fully utilised even though Lance commits
        # themselves remain serialised.
        self._commit_lock = threading.Lock()
        self._frags_to_commit: list[Any] = []
        # Diagnostic counters.  ``_in_flight`` only goes up in the
        # worker thread (between queue-pop and task-return); reading
        # it from the producer side gives a reasonable estimate of
        # current concurrency.
        self._in_flight = 0
        self._submitted_count = 0
        self._completed_count = 0
        self._total_bytes_uploaded = 0
        self._commits_done = 0
        self._frags_committed = 0
        # Lazily constructed on first ``_s5cmd_upload`` so we don't
        # build env vars before they're needed; cached afterwards
        # because env/endpoint don't change for the lifetime of this
        # uploader.
        self._uploader: S5cmdRunner | None = None

    def enqueue_fragment(self, frag: Any, new_files: list[Path]) -> None:
        with self._lock:
            total_size = sum(path.stat().st_size for path in new_files)
            for path in new_files:
                self._uploaded.add(str(path))

            if total_size >= self._upload_threshold_bytes:
                files = [*list(self._pending_files), *list(new_files)]
                frags = [*list(self._pending_frags), frag]
                self._pending_files.clear()
                self._pending_frags.clear()
                self._pending_bytes = 0
                self._submit_upload(files, frags)
                return

            self._pending_files.extend(new_files)
            self._pending_frags.append(frag)
            self._pending_bytes += total_size
            if self._pending_bytes >= self._upload_threshold_bytes:
                files = list(self._pending_files)
                frags = list(self._pending_frags)
                self._pending_files.clear()
                self._pending_frags.clear()
                self._pending_bytes = 0
                self._submit_upload(files, frags)

    def flush_pending(self) -> None:
        with self._lock:
            if not self._pending_files:
                return
            files = list(self._pending_files)
            frags = list(self._pending_frags)
            self._pending_files.clear()
            self._pending_frags.clear()
            self._pending_bytes = 0
            self._submit_upload(files, frags)

    def upload_remaining(self) -> None:
        with self._lock:
            remaining: list[Path] = []
            for path in self._local_base.rglob("*"):
                key = str(path)
                if path.is_file() and key not in self._uploaded:
                    self._uploaded.add(key)
                    remaining.append(path)
            if remaining:
                self._submit_upload(remaining, [])

    def wait(self) -> None:
        with self._lock:
            futures = list(self._futures)
            self._futures.clear()
        for future in concurrent.futures.as_completed(futures):
            future.result()
        # Final drain: a worker may have queued its frag and lost the
        # non-blocking ``_commit_lock`` race, so its frag is still
        # uncommitted.  Take the lock blocking now and commit anything
        # left.
        self._drain_commit_queue(blocking=True)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)

    def _submit_upload(self, files: list[Path], frags: list[Any]) -> None:
        """Submit one upload+commit task to the pool, with diagnostics.

        Caller already holds ``self._lock``.  Uses the lock-protected
        counters to log queue depth at submit time so the producer
        side can see whether the worker pool is keeping up.
        """
        self._submitted_count += 1
        upload_id = self._submitted_count
        total_bytes = sum(path.stat().st_size for path in files)
        queue_depth = len(self._futures) - self._completed_count
        logger.debug(
            "upload #%d queued: %d file(s), %.1f MiB | "
            "in_flight≈%d queued≈%d workers=%d",
            upload_id, len(files), total_bytes / (1024 * 1024),
            self._in_flight, max(queue_depth, 0), self._num_workers,
        )
        submitted_at = time.perf_counter()
        future = self._pool.submit(
            self._upload_and_maybe_commit,
            upload_id, files, frags, total_bytes, submitted_at,
        )
        self._futures.append(future)

    def _upload_and_maybe_commit(
        self,
        upload_id: int,
        files: list[Path],
        frags: list[Any],
        total_bytes: int,
        submitted_at: float,
    ) -> None:
        with self._lock:
            self._in_flight += 1
            in_flight_now = self._in_flight
        wait_s = time.perf_counter() - submitted_at
        logger.debug(
            "upload #%d starting: waited %.2fs in queue, in_flight=%d",
            upload_id, wait_s, in_flight_now,
        )
        try:
            up_start = time.perf_counter()
            self._s5cmd_upload(files)
            up_elapsed = time.perf_counter() - up_start
            mb = total_bytes / (1024 * 1024)
            mb_per_s = mb / up_elapsed if up_elapsed > 0 else 0.0
            logger.debug(
                "upload #%d uploaded: %d file(s), %.1f MiB in %.2fs (%.1f MiB/s)",
                upload_id, len(files), mb, up_elapsed, mb_per_s,
            )
            if frags:
                with self._lock:
                    self._frags_to_commit.extend(frags)
                # Try to drain the commit queue.  Non-blocking: if
                # another worker holds ``_commit_lock``, it will pick
                # up our frag in its next iteration.  Worst case the
                # final ``wait()`` does a blocking drain.
                self._drain_commit_queue(blocking=False, upload_id=upload_id)
        finally:
            with self._lock:
                self._in_flight -= 1
                self._completed_count += 1
                self._total_bytes_uploaded += total_bytes
                in_flight_after = self._in_flight
                completed = self._completed_count
                total_mb = self._total_bytes_uploaded / (1024 * 1024)
            logger.debug(
                "upload #%d done: in_flight=%d, completed=%d, total uploaded %.1f MiB",
                upload_id, in_flight_after, completed, total_mb,
            )

    def _ensure_remote_dataset(self) -> None:
        if self._remote_dataset_created:
            return
        empty = pa.table(
            {field.name: pa.array([], type=field.type) for field in self._schema},
            schema=self._schema,
        )
        try:
            lance.write_dataset(
                empty,
                self._remote_base,
                mode="create",
                storage_options=self._opts,
            )
        except OSError as exc:
            if "already exists" not in str(exc).lower():
                raise
        self._remote_dataset_created = True

    def _drain_commit_queue(
        self, *, blocking: bool, upload_id: int | None = None,
    ) -> None:
        """Single-leader drain of ``_frags_to_commit``.

        Whichever worker holds ``_commit_lock`` runs the loop: pop
        every queued frag, commit them in one Lance call, repeat
        until the queue is empty.  Other workers calling concurrently
        with ``blocking=False`` short-circuit immediately — their
        frag is already queued and the running leader will pick it
        up in its next iteration.
        """
        if not self._commit_lock.acquire(blocking=blocking):
            return
        try:
            while True:
                with self._lock:
                    pending = self._frags_to_commit
                    self._frags_to_commit = []
                if not pending:
                    return
                commit_start = time.perf_counter()
                self._commit_remote(pending)
                elapsed = time.perf_counter() - commit_start
                with self._lock:
                    self._commits_done += 1
                    self._frags_committed += len(pending)
                    commits_done = self._commits_done
                    frags_committed = self._frags_committed
                tag = f"#{upload_id}" if upload_id is not None else "(final drain)"
                logger.debug(
                    "upload %s committed %d frag(s) in %.2fs "
                    "[total commits=%d, total frags committed=%d]",
                    tag, len(pending), elapsed, commits_done, frags_committed,
                )
        finally:
            self._commit_lock.release()

    def _commit_remote(self, frags: list[Any]) -> None:
        """Single Lance commit for *frags*.  Caller holds ``_commit_lock``."""
        import logging

        self._ensure_remote_dataset()
        for attempt in range(1, constants.COMMIT_MAX_RETRIES + 1):
            try:
                ds = lance.dataset(self._remote_base, storage_options=self._opts)
                lance.LanceDataset.commit(
                    self._remote_base,
                    lance.LanceOperation.Append(frags),
                    read_version=ds.version,
                    max_retries=constants.COMMIT_INNER_MAX_RETRIES,
                    storage_options=self._opts,
                )
                return
            except OSError as exc:
                if not is_retryable_commit_error(exc) and _fragments_committed(
                    self._remote_base, frags, self._opts
                ):
                    return  # Commit succeeded despite the error
                if attempt >= constants.COMMIT_MAX_RETRIES:
                    raise
                if attempt == constants.COMMIT_WARN_RETRIES:
                    logging.getLogger(__name__).warning(
                        "Remote commit has failed %d times, still retrying",
                        attempt,
                    )
                time.sleep(constants.COMMIT_RETRY_SLEEP_S * attempt)

    def _s5cmd_upload(self, files: list[Path]) -> None:
        """Upload *files* via :class:`S5cmdRunner`.

        Per ``docs/error_handling_philosophy.md`` the uploader raises
        loudly on any partial-upload condition (subprocess error,
        watchdog kill, or per-op error from ``s5cmd --json``).  We
        deliberately do not catch here: the failure must propagate
        up through the worker future to ``wait()`` so the writer
        halts before committing Lance fragment metadata that points
        at missing S3 objects.
        """
        ops: list[tuple[str | Path, str]] = [
            (path, self._remote_base.rstrip("/") + "/" + str(path.relative_to(self._local_base)))
            for path in files
        ]
        # Local-side stat for the size-scaled watchdog; we know the
        # exact byte count for an upload because every src is a
        # local file we just wrote.
        total_bytes = 0
        for path in files:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
        env = utils.storage_options_to_s3_env(self._opts)
        endpoint = self._opts.get("aws_endpoint", "")
        if self._uploader is None:
            self._uploader = S5cmdRunner(endpoint=endpoint, env=env)
        self._uploader.cp(
            ops,
            total_bytes=total_bytes,
            expected_count=len(ops),
            log_label="s5cmd upload",
        )


class BaseLanceWriteSession:
    """Base class for sessions that write to a working Lance dataset.

    Subclasses decide whether the working dataset is the final destination
    (direct/local writes) or a staged local copy that is later published to a
    remote destination.

    Each session writes to a single named table (default: "steps").  To write
    multiple tables (e.g., steps + episodes), create one session per table.
    """
    def __init__(
        self,
        *,
        path: str,
        mode: Literal["w", "a"],
        overwrite: bool,
        storage_options: dict[str, str] | None,
        table_name: str = constants.TABLE_NAME,
    ) -> None:
        self.path = path
        self.mode = mode
        self.overwrite = overwrite
        self.storage_options = storage_options
        self.table_name = table_name

        self.working_path = path
        self.working_storage_options = storage_options
        self._db: lancedb.DBConnection | None = None
        # The lancedb table handle for the table being written.
        # Named steps_table for historical reasons; holds whichever table
        # this session targets (steps, episodes, etc.).
        self.steps_table: lancedb.table.Table | None = None
        self._schema: pa.Schema | None = None

        self._first_write_done = False
        self._pending_fragments: list[Any] = []
        self._fragment_lock = threading.Lock()

    @staticmethod
    def _join_table_uri(base_path: str, table_dir: str) -> str:
        if utils.is_remote(base_path):
            return base_path.rstrip("/") + "/" + table_dir
        return str(Path(base_path) / table_dir)

    @staticmethod
    def _table_dir_for(table_name: str) -> str:
        return f"{table_name}.{constants.TABLE_FORMAT_LANCE}"

    def _connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.working_storage_options is not None:
            kwargs["storage_options"] = self.working_storage_options
        return kwargs

    def working_table_uri(self, table_name: str | None = None) -> str:
        return self._join_table_uri(
            self.working_path,
            self._table_dir_for(table_name or self.table_name),
        )

    def working_table_dataset(self, table_name: str | None = None) -> lance.LanceDataset:
        return lance.dataset(self.working_table_uri(table_name), **self._connect_kwargs())

    def published_table_uri(self, table_name: str | None = None) -> str:
        return self.working_table_uri(table_name)

    def published_connect_kwargs(self) -> dict[str, Any]:
        return self._connect_kwargs()

    def published_table_dataset(self, table_name: str | None = None) -> lance.LanceDataset:
        return lance.dataset(
            self.published_table_uri(table_name),
            **self.published_connect_kwargs(),
        )

    @staticmethod
    def align_table_to_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
        arrays = []
        for field in target_schema:
            if field.name in table.column_names:
                arrays.append(table.column(field.name).cast(field.type))
            else:
                arrays.append(pa.nulls(table.num_rows, type=field.type))
        return pa.table(arrays, schema=target_schema)

    # -- WriteSession interface ---------------------------------------------

    def write_batch(self, data: pa.Table | pa.RecordBatch) -> None:
        if data.num_rows == 0:
            return
        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data], schema=data.schema)
        self._write_deferred(data)

    def flush(self) -> None:
        """Commit pending fragments and wait for any background uploads."""
        self._commit_deferred_fragments()
        self._flush_remote_uploads()

    def finalize(self, metadata: dict[str, str]) -> None:
        """Build indices, write metadata, and publish storage."""
        if not self._first_write_done:
            return

        # Scalar indices — only for columns present in this table's schema
        ds = self.working_table_dataset()
        schema_names = set(ds.schema.names)
        index_cols = [c for c in constants.INDEX_COLUMNS if c in schema_names]
        for i, col in enumerate(index_cols, 1):
            idx_start = time.perf_counter()
            logger.info(
                "finalize: building scalar index %d/%d on %r",
                i, len(index_cols), col,
            )
            ds.create_scalar_index(col, index_type="BTREE", replace=True)
            logger.info(
                "finalize: scalar index on %r built in %.1fs",
                col, time.perf_counter() - idx_start,
            )

        # Schema metadata
        meta_start = time.perf_counter()
        logger.info("finalize: writing schema metadata (%d keys)", len(metadata))
        ds = self.working_table_dataset()
        ds.update_schema_metadata(metadata)
        assert self._db is not None
        self.steps_table = self._db.open_table(self.table_name)
        self._schema = self.steps_table.to_lance().schema
        logger.info(
            "finalize: schema metadata written in %.1fs",
            time.perf_counter() - meta_start,
        )

        # Publish (subclass hook)
        publish_start = time.perf_counter()
        logger.info("finalize: publishing dataset")
        self._finalize_storage()
        logger.info(
            "finalize: dataset published in %.1fs",
            time.perf_counter() - publish_start,
        )

    def close(self) -> None:
        self.steps_table = None
        self._db = None

    # -- internal -----------------------------------------------------------

    def _commit_deferred_fragments(self) -> None:
        if not self._pending_fragments:
            return

        commit_start = time.perf_counter()
        n_fragments = len(self._pending_fragments)
        logger.info(
            "flush: committing %d deferred fragment(s)", n_fragments,
        )

        uri = self.working_table_uri()
        kwargs = self._connect_kwargs()

        if self.mode == "a" and self.steps_table is not None:
            op = lance.LanceOperation.Append(self._pending_fragments)
            for attempt in range(1, constants.COMMIT_MAX_RETRIES + 1):
                try:
                    ds = self.working_table_dataset()
                    lance.LanceDataset.commit(uri, op, read_version=ds.version, **kwargs)
                    break
                except OSError as exc:
                    if not is_retryable_commit_error(exc) and _fragments_committed(
                        uri,
                        self._pending_fragments,
                        kwargs.get("storage_options", {}),
                    ):
                        break  # Commit succeeded despite the error
                    if attempt >= constants.COMMIT_MAX_RETRIES:
                        raise
                    if attempt == constants.COMMIT_WARN_RETRIES:
                        logger.warning(
                            "Append commit has failed %d times, still retrying",
                            attempt,
                        )
                    time.sleep(constants.COMMIT_RETRY_SLEEP_S)
        else:
            op = lance.LanceOperation.Overwrite(self._schema, self._pending_fragments)
            lance.LanceDataset.commit(uri, op, **kwargs)

        assert self._db is not None
        self.steps_table = self._db.open_table(self.table_name)
        self._schema = self.steps_table.to_lance().schema
        self._pending_fragments = []
        logger.info(
            "flush: committed %d fragment(s) in %.1fs",
            n_fragments, time.perf_counter() - commit_start,
        )

    def _flush_remote_uploads(self) -> None:
        pass

    def _finalize_storage(self) -> None:
        raise NotImplementedError

    def _write_deferred(self, data: pa.Table) -> None:
        """Create a Lance fragment from *data* and append to pending list.

        Thread-safe for DirectLanceWriteSession (local writes) after the
        first write establishes the schema.  For StagedRemoteLanceWriteSession,
        the glob-based new-file detection requires single-threaded calls.
        """
        from lance.fragment import LanceFragment

        if not self._first_write_done:
            if self._schema is None:
                self._schema = data.schema
            self._first_write_done = True
        elif data.schema != self._schema:
            new_fields = [
                data.schema.field(name)
                for name in data.schema.names
                if name not in self._schema.names
            ]
            if new_fields:
                self._schema = pa.schema(list(self._schema) + new_fields, metadata=self._schema.metadata)
            data = self.align_table_to_schema(data, self._schema)

        uri = self.working_table_uri()
        frag = LanceFragment.create(uri, data, mode="create", **self._connect_kwargs())
        with self._fragment_lock:
            self._pending_fragments.append(frag)

        self._handle_new_fragment(frag, uri=uri)

    def _handle_new_fragment(self, frag: Any, *, uri: str) -> None:
        pass

    def optimize_local_storage(self) -> None:
        """Compact all Lance fragments in one pass.

        Logged separately because ``LanceTable.optimize`` is the
        heaviest step in ``finalize`` for large local datasets — it
        rewrites every fragment and can use enough RAM on multi-GiB
        datasets that a system OOM killer may step in (SIGTERM →
        exit 143 from the shell).  Callers run this *after* the
        ``_episodes`` companion is upserted so an interrupted
        compaction still leaves a fully-functional dataset behind.
        """
        if self.steps_table is None:
            return
        import datetime as dt

        opt_start = time.perf_counter()
        logger.info("finalize: optimizing local Lance storage (compacting fragments)")
        self.steps_table.optimize(cleanup_older_than=dt.timedelta(0))
        logger.info(
            "finalize: local Lance storage optimized in %.1fs",
            time.perf_counter() - opt_start,
        )

    def post_episodes_optimize(self) -> None:
        """Hook that runs *after* the episodes companion is built.

        Subclasses override; the default is a no-op so backends that
        don't have a local-compaction step (e.g. staged-remote, which
        already finished publishing in ``_finalize_storage``) don't
        need to implement anything.
        """


class DirectLanceWriteSession(BaseLanceWriteSession):
    """Local Lance write session using ``lance.write_dataset`` directly.

    Unlike the fragment-based approach in ``BaseLanceWriteSession``, this
    uses ``lance.write_dataset(mode='append')`` for each batched write.
    This avoids the ``LanceFragment.create`` overhead that scales poorly
    with dataset size (each fragment creation interacts with the existing
    version history).
    """

    def __init__(
        self,
        *,
        path: str,
        mode: Literal["w", "a"],
        overwrite: bool,
        storage_options: dict[str, str] | None,
        table_name: str = constants.TABLE_NAME,
    ) -> None:
        super().__init__(
            path=path,
            mode=mode,
            overwrite=overwrite,
            storage_options=storage_options,
            table_name=table_name,
        )
        local_path = Path(self.working_path)
        connect_kwargs = self._connect_kwargs()

        if self.mode == "a":
            if not local_path.exists():
                # Create the dataset so concurrent append workers can proceed.
                local_path.mkdir(parents=True, exist_ok=True)
                self._db = lancedb.connect(self.working_path, **connect_kwargs)
                return  # schema will be set by first write
            self._db = lancedb.connect(self.working_path, **connect_kwargs)
            try:
                self.steps_table = self._db.open_table(self.table_name)
            except ValueError:
                # Directory exists but no table yet (bootstrapped but empty).
                return
            self._schema = self.steps_table.to_lance().schema
            self._first_write_done = True
            return

        if self.table_name == constants.TABLE_NAME:
            # Default table (steps): clean the entire dataset directory for
            # backward compatibility — the caller expects a clean slate.
            if local_path.exists() or local_path.is_symlink():
                if not self.overwrite:
                    raise FileExistsError(
                        f"Output path already exists: {local_path}. "
                        "Pass overwrite=True to delete and recreate it."
                    )
                if local_path.is_symlink():
                    local_path.unlink()
                else:
                    shutil.rmtree(local_path)
        else:
            # Non-default table (e.g., episodes): only remove the table's
            # own directory to avoid destroying sibling tables.
            table_path = Path(self.working_table_uri())
            if table_path.exists():
                if not self.overwrite:
                    raise FileExistsError(
                        f"Table already exists: {table_path}. "
                        "Pass overwrite=True to delete and recreate it."
                    )
                shutil.rmtree(table_path)
        local_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.working_path, **connect_kwargs)

    def write_batch(self, data: pa.Table | pa.RecordBatch) -> None:
        """Write via lance.write_dataset — faster than fragment-create for local."""
        if data.num_rows == 0:
            return
        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data], schema=data.schema)

        # Schema evolution (same as base class)
        if not self._first_write_done:
            if self._schema is None:
                self._schema = data.schema
            self._first_write_done = True
        elif data.schema != self._schema:
            new_fields = [
                data.schema.field(name)
                for name in data.schema.names
                if name not in self._schema.names
            ]
            if new_fields:
                self._schema = pa.schema(
                    list(self._schema) + new_fields,
                    metadata=self._schema.metadata,
                )
            data = self.align_table_to_schema(data, self._schema)

        uri = self.working_table_uri()
        if (Path(uri) / "_versions").exists():
            lance.write_dataset(data, uri, mode="append")
        else:
            lance.write_dataset(data, uri, mode="create")

    def flush(self) -> None:
        """No-op — each write_batch is immediately committed."""
        # Reconnect LanceDB to pick up new data written via lance.write_dataset
        if self._db is not None and self._first_write_done:
            try:
                self.steps_table = self._db.open_table(self.table_name)
                self._schema = self.steps_table.to_lance().schema
            except ValueError:
                pass  # Table doesn't exist yet

    def _finalize_storage(self) -> None:
        # No-op for the local backend: data is already at its final
        # location after the index + schema-meta steps run.  The
        # ``optimize`` compaction pass moved to ``post_episodes_optimize``
        # so it runs *after* the episodes companion is committed,
        # ensuring a kill or OOM mid-compaction still leaves a fully
        # functional dataset behind.
        pass

    def post_episodes_optimize(self) -> None:
        self.optimize_local_storage()


class StagedRemoteLanceWriteSession(BaseLanceWriteSession):
    def _remote_table_uri(self, table_name: str | None = None) -> str:
        return self._join_table_uri(
            self._remote_path,
            self._table_dir_for(table_name or self.table_name),
        )

    def _local_working_table_exists(self) -> bool:
        return (Path(super().working_table_uri()) / "_versions").exists()

    def __init__(
        self,
        *,
        path: str,
        mode: Literal["w", "a"],
        overwrite: bool,
        storage_options: dict[str, str] | None,
        table_name: str = constants.TABLE_NAME,
        upload_workers: int = constants.S3_UPLOAD_WORKERS,
        upload_threshold_bytes: int = constants.S3_UPLOAD_THRESHOLD_BYTES,
    ) -> None:
        super().__init__(
            path=path,
            mode=mode,
            overwrite=overwrite,
            storage_options=storage_options,
            table_name=table_name,
        )
        self._remote_path = path
        self._remote_storage_options = storage_options
        self._streaming_uploader: _StreamingUploader | None = None
        self._upload_workers = upload_workers
        self._upload_threshold_bytes = upload_threshold_bytes
        self._staging_dir = tempfile.mkdtemp(prefix="packds_staging_")
        ds_name = path.rstrip("/").rsplit("/", 1)[-1]
        self.working_path = str(Path(self._staging_dir) / ds_name)
        self.working_storage_options = None

        if mode == "w":
            remote_connect_kwargs: dict[str, Any] = {}
            if storage_options is not None:
                remote_connect_kwargs["storage_options"] = storage_options
            remote_db = lancedb.connect(path, **remote_connect_kwargs)
            existing_tables = set(remote_db.list_tables().tables)
            tables_to_reset = [self.table_name]
            if self.table_name == constants.TABLE_NAME:
                tables_to_reset.append(constants.EPISODES_TABLE_NAME)
            existing_targets = [name for name in tables_to_reset if name in existing_tables]
            if existing_targets:
                if not overwrite:
                    raise FileExistsError(
                        f"Remote dataset already exists: {path}. "
                        "Pass overwrite=True to delete and recreate it."
                    )
                for table in existing_targets:
                    remote_db.drop_table(table)

        # Open local staging
        local_path = Path(self.working_path)
        local_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.working_path)

        if mode == "a":
            try:
                remote_db = lancedb.connect(self._remote_path, **self._remote_connect_kwargs())
                remote_table = remote_db.open_table(self.table_name)
                self._schema = remote_table.to_lance().schema
                self._first_write_done = True
            except (FileNotFoundError, OSError, ValueError):
                pass  # Dataset doesn't exist yet; schema set by first write

    def _remote_connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._remote_storage_options is not None:
            kwargs["storage_options"] = self._remote_storage_options
        return kwargs

    def working_table_dataset(self, table_name: str | None = None) -> lance.LanceDataset:
        if (table_name is None or table_name == self.table_name) and self._local_working_table_exists():
            return super().working_table_dataset()
        return lance.dataset(self._remote_table_uri(table_name), **self._remote_connect_kwargs())

    def published_table_uri(self, table_name: str | None = None) -> str:
        return self._remote_table_uri(table_name)

    def published_connect_kwargs(self) -> dict[str, Any]:
        return self._remote_connect_kwargs()

    def finalize(self, metadata: dict[str, str]) -> None:
        """Build indices, write metadata, then finish all uploads."""
        if not self._first_write_done:
            return

        # Scalar indices — only for columns present in this table's schema
        ds = self.working_table_dataset()
        schema_names = set(ds.schema.names)
        index_cols = [c for c in constants.INDEX_COLUMNS if c in schema_names]
        for i, col in enumerate(index_cols, 1):
            idx_start = time.perf_counter()
            logger.info(
                "finalize: building scalar index %d/%d on %r",
                i, len(index_cols), col,
            )
            ds.create_scalar_index(col, index_type="BTREE", replace=True)
            logger.info(
                "finalize: scalar index on %r built in %.1fs",
                col, time.perf_counter() - idx_start,
            )

        # Schema metadata — handle both local staging and remote-only cases
        meta_start = time.perf_counter()
        logger.info("finalize: writing schema metadata (%d keys)", len(metadata))
        ds = self.working_table_dataset()
        ds.update_schema_metadata(metadata)
        if self._local_working_table_exists():
            assert self._db is not None
            self.steps_table = self._db.open_table(self.table_name)
            self._schema = self.steps_table.to_lance().schema
        else:
            self.steps_table = None
            self._schema = ds.schema
        logger.info(
            "finalize: schema metadata written in %.1fs",
            time.perf_counter() - meta_start,
        )

        # Publish: finish all uploads
        publish_start = time.perf_counter()
        logger.info("finalize: publishing dataset (uploading remaining files)")
        self._finalize_storage()
        logger.info(
            "finalize: dataset published in %.1fs",
            time.perf_counter() - publish_start,
        )

    def _flush_remote_uploads(self) -> None:
        if self._streaming_uploader is None:
            return
        up_start = time.perf_counter()
        logger.info("flush: waiting on staged S3 uploads")
        # Heartbeat: a long flush() looks identical to a hang from
        # the convert-script log perspective (no episode-progress
        # lines for many seconds).  Emit periodic state so the user
        # can tell uploads are still draining.
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(constants.S3_UPLOAD_FLUSH_HEARTBEAT_S):
                u = self._streaming_uploader
                if u is None:
                    return
                with u._lock:  # noqa: SLF001 - intentional peek into uploader counters
                    submitted = u._submitted_count
                    completed = u._completed_count
                    in_flight = u._in_flight
                    pending_commit = len(u._frags_to_commit)
                    total_mb = u._total_bytes_uploaded / (1024 * 1024)
                    commits_done = u._commits_done
                logger.info(
                    "flush heartbeat: %.0fs elapsed | "
                    "submitted=%d completed=%d in_flight=%d "
                    "pending_commit_frags=%d commits_done=%d "
                    "uploaded=%.1f MiB",
                    time.perf_counter() - up_start,
                    submitted, completed, in_flight,
                    pending_commit, commits_done, total_mb,
                )

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        try:
            self._streaming_uploader.flush_pending()
            self._streaming_uploader.wait()
        finally:
            stop.set()
            hb.join(timeout=1.0)
        logger.info(
            "flush: staged S3 uploads complete in %.1fs",
            time.perf_counter() - up_start,
        )

    def _finalize_storage(self) -> None:
        if self._streaming_uploader is None:
            return
        wait_start = time.perf_counter()
        logger.info("finalize: draining S3 upload queue")
        self._streaming_uploader.wait()
        logger.info("finalize: uploading any remaining staged files")
        self._streaming_uploader.upload_remaining()
        self._streaming_uploader.wait()
        logger.info(
            "finalize: S3 upload queue drained in %.1fs",
            time.perf_counter() - wait_start,
        )

    def close(self) -> None:
        if self._streaming_uploader is not None:
            self._streaming_uploader.shutdown()
            self._streaming_uploader = None
        super().close()
        shutil.rmtree(self._staging_dir, ignore_errors=True)

    def _handle_new_fragment(self, frag: Any, *, uri: str) -> None:
        new_files = [Path(uri) / "data" / df.path for df in frag.data_files()]
        if not new_files:
            return
        if self._streaming_uploader is None:
            self._streaming_uploader = _StreamingUploader(
                local_base=uri,
                remote_base=self._remote_table_uri(),
                storage_options=self._remote_storage_options or {},
                schema=self._schema,
                num_workers=self._upload_workers,
                upload_threshold_bytes=self._upload_threshold_bytes,
            )
        self._streaming_uploader.enqueue_fragment(frag, new_files)


LanceWriteSession = BaseLanceWriteSession


def create_lance_write_session(
    *,
    path: str,
    mode: Literal["w", "a"],
    overwrite: bool,
    storage_options: dict[str, str] | None,
    table_name: str = constants.TABLE_NAME,
    upload_workers: int = constants.S3_UPLOAD_WORKERS,
    upload_threshold_bytes: int = constants.S3_UPLOAD_THRESHOLD_BYTES,
) -> BaseLanceWriteSession:
    if utils.is_remote(path):
        return StagedRemoteLanceWriteSession(
            path=path,
            mode=mode,
            overwrite=overwrite,
            storage_options=storage_options,
            table_name=table_name,
            upload_workers=upload_workers,
            upload_threshold_bytes=upload_threshold_bytes,
        )
    return DirectLanceWriteSession(
        path=path,
        mode=mode,
        overwrite=overwrite,
        storage_options=storage_options,
        table_name=table_name,
    )
