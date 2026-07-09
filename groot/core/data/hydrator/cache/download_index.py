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

"""Shared DuckDB download index for cross-rank file discovery on shared Lustre.

Scope: the index is **per training job**. Each job gets its own DuckDB
shards under the per-job hydrator cache subdir, so concurrent jobs never
see each other's rows and a fresh job re-discovers files from S3. Within
a single job the index is shared across all ranks/workers.

DB file routing is delegated to a ``PartitionStrategy``: writes for a URI
go to ``sync_{strategy.db_key(uri)}.duckdb`` and reads use the same key.
Default = ``HashPartitionStrategy`` (sha1-bucketed keys, e.g.
``sync_42.duckdb``) for uniform per-shard write contention regardless of
filename distribution. Callers needing the legacy one-file-per-prefix
layout pass an ``AlphanumericPrefixStrategy`` explicitly.

Each row is the terminal outcome of a download attempt: either success
(``local_path`` set, ``error_message`` is None) or failure (``local_path``
is None, ``error_message`` set). Non-owners poll ``lookup()`` to see when
an owner has settled the row and can route around terminal failures.

Concurrency is handled entirely by DuckDB's own file locking, which works
both within a node (multiple processes) and across nodes on shared Lustre.
fcntl.flock is NOT used because Lustre requires the ``flock`` mount option
for cross-node flock semantics, and it is off by default.  Instead:
- _write() retries up to 20 times when DuckDB rejects a concurrent open.
- lookup() catches DuckDB lock errors and returns None so get_file() retries.
"""

from dataclasses import dataclass
from pathlib import Path
import threading
import time

from groot.core.data.hydrator.cache.partition_strategy import (
    HashPartitionStrategy,
    PartitionStrategy,
)
from groot.core.utils.distributed import worker_logging

log = worker_logging.get_unified_logger()


@dataclass(frozen=True)
class DownloadRecord:
    """Terminal outcome of a download attempt published to the cross-rank index.

    Exactly one of ``local_path`` or ``error_message`` is set: success rows
    carry the local path; failure rows carry the error message so non-owners
    can surface the same exception instead of polling forever.

    Used as the row shape for both ``batch_insert`` (input) and
    ``lookup`` (output) — a single concept end-to-end.
    """

    s3_uri: str
    local_path: str | None
    error_message: str | None


class DownloadIndex:
    """Sharded DuckDB index of (s3_uri → DownloadRecord) entries.

    DB file routing is decided by ``partition_strategy.db_key(uri)``:
    one DuckDB file per distinct key. ``batch_insert`` groups records by
    key and writes each group to its file; ``lookup`` opens only the one
    file for that URI's key.
    """

    def __init__(
        self,
        db_dir: str,
        partition_strategy: PartitionStrategy | None = None,
    ) -> None:
        self.db_dir = db_dir
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        # Default = HashPartitionStrategy → uniform per-shard write
        # contention regardless of filename distribution. Callers that
        # need the legacy one-file-per-filename-prefix layout pass an
        # AlphanumericPrefixStrategy explicitly.
        self.partition_strategy = partition_strategy or HashPartitionStrategy()
        # Per-key in-process locks: serialise same-process read/write opens.
        self._rw_locks: dict[str, threading.Lock] = {}
        self._locks_mu = threading.Lock()
        log.info(
            "[DownloadIndex] init db_dir=%s partition_strategy=%s",
            db_dir,
            type(self.partition_strategy).__name__,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _db_path(self, key: str) -> str:
        path = str(Path(self.db_dir) / f"sync_{key}.duckdb")
        # debug-level so a real download workload (thousands of file
        # lookups) doesn't drown the log; enable with LOGURU_LEVEL=DEBUG.
        log.debug("[DownloadIndex] _db_path key=%s -> %s", key, path)
        return path

    def _rw_lock(self, key: str) -> threading.Lock:
        with self._locks_mu:
            if key not in self._rw_locks:
                self._rw_locks[key] = threading.Lock()
            return self._rw_locks[key]

    def _write(self, key: str, fn, max_retries: int = 20) -> None:
        """Execute fn(con) inside a DuckDB read-write connection with retries.

        DuckDB allows only one read-write connection at a time.  When another
        process (same or different node) holds the file lock, duckdb.connect()
        raises an IO Error.  We retry up to max_retries times with a short
        back-off so that concurrent writers and readers eventually succeed.
        """
        import duckdb

        db_path = self._db_path(key)
        rw_lock = self._rw_lock(key)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(max_retries):
            try:
                with rw_lock:
                    con = duckdb.connect(db_path)
                    try:
                        con.execute(
                            "CREATE TABLE IF NOT EXISTS synced_files "
                            "(s3_uri TEXT PRIMARY KEY, local_path TEXT, "
                            "error_message TEXT)"
                        )
                        # Migrate pre-existing tables created with the older
                        # schema (`local_path TEXT NOT NULL`, no error_message).
                        # Both ALTERs are idempotent on already-migrated tables.
                        con.execute(
                            "ALTER TABLE synced_files "
                            "ADD COLUMN IF NOT EXISTS error_message TEXT"
                        )
                        con.execute(
                            "ALTER TABLE synced_files "
                            "ALTER COLUMN local_path DROP NOT NULL"
                        )
                        fn(con)
                        con.commit()
                    finally:
                        con.close()
                return
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_insert(self, records: list[DownloadRecord]) -> None:
        """Insert DownloadRecord rows grouped by ``partition_strategy.db_key``.

        ON CONFLICT DO NOTHING — once a URI's terminal outcome is recorded,
        subsequent inserts for the same URI are no-ops. Owners publish at
        most one row per URI (success or failure).
        """
        if not records:
            return
        by_key: dict[str, list[DownloadRecord]] = {}
        for r in records:
            by_key.setdefault(self.partition_strategy.db_key(r.s3_uri), []).append(r)

        for key, key_records in by_key.items():
            params = [(r.s3_uri, r.local_path, r.error_message) for r in key_records]
            self._write(
                key,
                lambda con, p=params: con.executemany(
                    "INSERT INTO synced_files (s3_uri, local_path, error_message) "
                    "VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                    p,
                ),
            )

    def lookup(self, s3_uri: str) -> DownloadRecord | None:
        """Return the DownloadRecord for s3_uri, or None if not present / DB busy.

        When a writer holds the DuckDB file lock (same or different node),
        duckdb.connect(read_only=True) raises an IO Error.  We catch it and
        return None so the caller retries.
        """
        import duckdb

        key = self.partition_strategy.db_key(s3_uri)
        db_path = self._db_path(key)
        if not Path(db_path).exists():
            return None
        rw_lock = self._rw_lock(key)
        if not rw_lock.acquire(blocking=False):
            # Another thread in this process is writing; skip this poll cycle.
            return None
        try:
            con = duckdb.connect(db_path, read_only=True)
            row = con.execute(
                "SELECT local_path, error_message FROM synced_files "
                "WHERE s3_uri = ?",
                [s3_uri],
            ).fetchone()
            con.close()
            if row is None:
                return None
            return DownloadRecord(s3_uri=s3_uri, local_path=row[0], error_message=row[1])
        except Exception:  # noqa: BLE001
            # DuckDB lock held by another process (same or cross-node writer),
            # or table migration in progress. Return None silently; caller retries.
            return None
        finally:
            rw_lock.release()
