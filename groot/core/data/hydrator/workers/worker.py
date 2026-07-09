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

"""Xenna worker class for storage I/O (download and upload)."""

from pathlib import Path
import time
from typing import Any

from groot.core.utils.distributed import worker_logging
from groot.core.data.hydrator import backends

logger = worker_logging.get_unified_logger()

# (kind, path, content, profile)  — profile=None means use the default credentials_profile
# from backend_config; a non-None profile overrides credentials_profile for that task.
_Task = tuple[str, str, bytes | Path | None, str | None]
_Result = tuple[str, str, Any, str | None]

# Flush intermediate DuckDB inserts after this many pairs so get_file() pollers
# see entries well before the full prefix sync completes.
_DUCKDB_FLUSH_EVERY = 100


class StorageIOWorker:
    """Xenna worker: process one storage task (download or upload). Returns result tuple, never raises.

    Backends are created lazily, one per credentials profile.  The default backend
    (profile=None) uses whatever ``credentials_profile`` is in *backend_config*; tasks
    with an explicit profile get a backend whose ``credentials_profile`` is overridden.
    """

    def __init__(
        self,
        backend_type: str,
        backend_config: dict,
        cache_dir: str,  # Local directory for downloading; backend uses it as working_dir
        retry_attempts: int,
        retry_delay: float,
        download_index_db_dir: str | None = None,
    ) -> None:
        self.backend_type = backend_type
        self.backend_config = backend_config
        self.cache_dir = cache_dir
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        # Lazily-created backends keyed by profile (None = default from backend_config).
        self._backends: dict[str | None, backends.StorageBackend] = {}
        if download_index_db_dir:
            from groot.core.data.hydrator.cache.download_index import DownloadIndex
            self._download_index = DownloadIndex(download_index_db_dir)
            logger.info(
                "[StorageIOWorker] __init__: DownloadIndex at %s", download_index_db_dir
            )
        else:
            self._download_index = None
        self._total_duckdb_inserts = 0

    def _get_backend(self, profile: str | None = None) -> backends.StorageBackend:
        """Return (creating if needed) the backend for *profile*."""
        if profile not in self._backends:
            config = self.backend_config
            if profile is not None:
                config = {**self.backend_config, "credentials_profile": profile}
            logger.info(
                "[StorageIOWorker] creating backend type=%s profile=%r", self.backend_type, profile
            )
            self._backends[profile] = backends.create_backend(
                self.backend_type,
                working_dir=self.cache_dir,
                backend_config=config,
            )
        return self._backends[profile]

    def process(self, task: _Task) -> _Result:
        """Process one task. Returns (kind, path, value, error). Never raises."""
        kind, remote_path, content, profile = task
        if kind == "download":
            return self._download(remote_path, profile)
        if kind == "upload":
            if content is None:
                return ("upload", remote_path, None, "upload task missing content")
            return self._upload(remote_path, content, profile)
        if kind == "sync_dir":
            local_dir = content.decode("utf-8") if content else ""
            return self._sync_dir(remote_path, local_dir, profile)
        return (kind, remote_path, None, f"Unknown task kind: {kind!r}")

    def _download(self, remote_path: str, profile: str | None) -> _Result:
        last_exception = None
        backend = self._get_backend(profile)
        for attempt in range(self.retry_attempts):
            try:
                local_path = backend.download_file(remote_path)
                return ("download", remote_path, local_path, None)
            except Exception as e:  # noqa: BLE001
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2**attempt))
        return (
            "download",
            remote_path,
            None,
            f"Failed after {self.retry_attempts} attempts: {last_exception}",
        )

    def _sync_dir(self, s3_path: str, local_dir: str, profile: str | None = None) -> _Result:
        from groot.core.data.hydrator.cache.download_index import DownloadRecord

        last_exception: Exception | None = None
        for attempt in range(self.retry_attempts):
            buf: list[DownloadRecord] = []
            flush_count = 0

            def _on_pair(s3_uri: str, local_path: str, _buf: list = buf) -> None:
                nonlocal flush_count
                _buf.append(DownloadRecord(s3_uri=s3_uri, local_path=local_path, error_message=None))
                if self._download_index is not None and len(_buf) >= _DUCKDB_FLUSH_EVERY:
                    self._download_index.batch_insert(_buf[:])
                    flush_count += 1
                    self._total_duckdb_inserts += len(_buf)
                    logger.info(
                        "[StorageIOWorker] DuckDB mid-sync flush #%d: +%d rows"
                        " (total_inserts=%d) s3_path=%r",
                        flush_count,
                        len(_buf),
                        self._total_duckdb_inserts,
                        s3_path,
                    )
                    _buf.clear()

            try:
                pairs = self._get_backend(profile).sync_directory(s3_path, local_dir, on_pair=_on_pair)
                # flush remainder not yet flushed
                if buf and self._download_index is not None:
                    self._download_index.batch_insert(buf)
                    self._total_duckdb_inserts += len(buf)
                    logger.info(
                        "[StorageIOWorker] DuckDB final flush: +%d rows"
                        " (total_inserts=%d mid_sync_flushes=%d) s3_path=%r",
                        len(buf),
                        self._total_duckdb_inserts,
                        flush_count,
                        s3_path,
                    )
                total_pairs = len(pairs) if pairs else 0
                logger.info(
                    "[StorageIOWorker] sync_dir done: s3_path=%r pairs=%d duckdb_inserts=%d",
                    s3_path,
                    total_pairs,
                    self._total_duckdb_inserts,
                )
                return ("sync_dir", s3_path, pairs, None)
            except Exception as e:  # noqa: BLE001
                last_exception = e
                stderr = getattr(e, "stderr", None) or ""
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        "[StorageIOWorker] s5cmd sync failed (attempt %d/%d), retrying in %.1fs"
                        " s3_path=%r error=%s stderr=%s",
                        attempt + 1,
                        self.retry_attempts,
                        delay,
                        s3_path,
                        e,
                        stderr.strip()[-500:] if stderr else "(none)",
                    )
                    time.sleep(delay)

        stderr = getattr(last_exception, "stderr", None) or ""
        return (
            "sync_dir",
            s3_path,
            None,
            f"Failed after {self.retry_attempts} attempts: {last_exception}"
            + (f"\nstderr: {stderr.strip()[-500:]}" if stderr else ""),
        )

    def _upload(
        self, remote_path: str, content: bytes | Path, profile: str | None
    ) -> _Result:
        last_exception = None
        backend = self._get_backend(profile)
        for attempt in range(self.retry_attempts):
            try:
                backend.upload_file(remote_path, content)
                return ("upload", remote_path, None, None)
            except Exception as e:  # noqa: BLE001
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2**attempt))
        return (
            "upload",
            remote_path,
            None,
            f"Failed after {self.retry_attempts} attempts: {last_exception}",
        )
