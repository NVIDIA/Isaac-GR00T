# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Cache manager for local file storage and LRU eviction.

IMPORTANT: This is an INTERNAL component of the Hydrator.
Users should NEVER import or use CacheManager directly.
All cache operations are managed through the service layer:
- hydrator.register_files() → internally manages cache
- hydrator.get_file() → internally queries cache
- hydrator.release_files() → internally updates cache
"""

import atexit
import ctypes
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from groot.core.data.hydrator import backends, types, workers
from groot.core.data.hydrator import modality as hydrator_modality
from groot.core.data.hydrator.cache import eviction_policy as cache_eviction_policy
from groot.core.data.hydrator.cache.dir_sync_job import DirSyncJob
from groot.core.data.hydrator.cache.download_index import DownloadIndex, DownloadRecord
from groot.core.data.hydrator.cache.partition_strategy import (
    AlphanumericPrefixStrategy,
    HashPartitionStrategy,
    IdentityStrategy,
    PartitionStrategy,
)
from groot.core.data.hydrator.utils import io as _io_utils
from groot.core.utils.distributed import worker_logging

log = worker_logging.get_unified_logger()

# The shared index's directory, a sibling of the per-worker caches under the job
# directory. Shutdown names it too, to tell the shared index apart from a worker's own
# cache when it decides what it may remove.
DOWNLOAD_INDEX_DIRNAME = "download_index"

# Prevents GC of ctypes callbacks pending on the main thread (Py_AddPendingCall
# stores only a raw C pointer; if the Python wrapper is collected the pointer
# becomes dangling).  Each entry is removed once the callback fires.
_pending_signal_cfuncs: list = []
_pending_signal_lock = threading.Lock()


class CacheManager:
    """Manages local cache storage with LRU eviction.

    Responsibilities:
    - Track file states and metadata
    - Manage local storage paths
    - Trigger downloads for cacheable files
    - Implement LRU eviction when size limits are reached
    - Distinguish between active files (currently in use) and cached files (eligible for eviction)

    The CacheManager owns the WorkerPool (read + write) and ensures files
    are available locally when needed; uploads are submitted to the same pool.

    Args:
        cache_root: Root directory for cache storage
        max_cache_size_gb: Maximum cache size in GB
        eviction_buffer: Evict until size < max_size * buffer (0.9 = 90%)
        eviction_policy: Eviction policy to use (default: LRUEvictionPolicy)
                       Can be LRUEvictionPolicy, FIFOEvictionPolicy, or SizeBasedEvictionPolicy.
        backend: Storage backend instance (if None, worker pool will create one)
        backend_type: Backend type for workers (e.g., "boto3")
        backend_config: Backend-specific configuration (e.g., transfer settings)
        max_workers: Number of parallel download workers
        download_timeout: Timeout for individual downloads
        retry_attempts: Number of retry attempts for failed downloads
        retry_delay: Initial delay between retries
        runtime_env: Optional Ray runtime_env for worker pool
            (e.g. ``{"py_executable": sys.executable}`` in tests).
        rank: This process's rank in the distributed group (0-indexed).
        world_size: Total number of ranks (1 = no sharding).
        partition_strategy: Decides per-file ownership and DuckDB shard
            routing under download_index. ``None`` selects
            ``IdentityStrategy()`` when ``enable_download_index=True``;
            ignored otherwise. ``IdentityStrategy`` is the safe default for
            VLA-style sharded dataloaders where each rank only registers
            its own shard — every registrant downloads and dedup is
            best-effort via cross-rank adoption. Callers whose registrant
            set is guaranteed to include the hash-picked owner for every
            URI (e.g. eager full-set registration) can opt into
            ``HashPartitionStrategy()`` for true 1/world_size dedup.
    """

    def __init__(
        self,
        cache_root: str,
        max_cache_size_gb: float,
        eviction_buffer: float = 0.9,
        eviction_policy: cache_eviction_policy.EvictionPolicy | None = None,
        backend: backends.StorageBackend | None = None,
        backend_type: str = "boto3",
        backend_config: dict | None = None,
        max_workers: int = 4,
        download_timeout: float = 300.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        runtime_env: dict | None = None,
        enable_download_index: bool = False,
        rank: int = 0,
        world_size: int = 1,
        partition_strategy: PartitionStrategy | None = None,
    ):
        self.cache_root = Path(cache_root)
        # cache_root = {base_cache_dir}/{job_name}/worker_{N}; go up two levels
        self._sweep_stale_caches(str(self.cache_root.parent.parent))
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self._prev_sigterm: object = None
        self._prev_sigint: object = None
        self.eviction_buffer = eviction_buffer
        self.eviction_policy = eviction_policy or cache_eviction_policy.LRUEvictionPolicy()

        # File metadata tracking
        self.files: dict[str, types.FileMetadata] = {}

        # Statistics
        self.cache_hits = 0
        self.downloads = 0
        self.evictions = 0
        self.download_errors = 0

        # Thread safety
        self._lock = threading.RLock()

        # Create cache directory
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Distributed identity + partition strategy. The strategy is only
        # consumed when download_index is enabled (it routes ownership and
        # DuckDB shards); we still store it unconditionally so callers can
        # introspect.
        self.rank = rank
        self.world_size = world_size
        if partition_strategy is None and enable_download_index:
            partition_strategy = IdentityStrategy()
        self.partition_strategy = partition_strategy
        # Under download_index, adopters across ranks reference the
        # owner's worker dir. Evicting from any rank invalidates other
        # ranks' local_paths. LRU is therefore disabled until cross-worker
        # ref-counting lands. Bookkeeping flag for the one-shot
        # over-quota warning emitted from evict_if_needed.
        self._over_quota_warning_logged = False

        # Worker pool for download and upload (owned by cache)
        # Shared DuckDB index for cross-rank file discovery.
        # Lives at {job_dir}/download_index — sibling of the per-worker dirs.
        if enable_download_index:
            download_index_db_dir = str(self.cache_root.parent / DOWNLOAD_INDEX_DIRNAME)
            self.download_index: DownloadIndex | None = DownloadIndex(
                download_index_db_dir, partition_strategy=self.partition_strategy
            )
        else:
            download_index_db_dir = None
            self.download_index = None

        self.workers = workers.WorkerPool(
            backend=backend,  # type: ignore[arg-type]
            backend_type=backend_type,
            backend_config=backend_config or {},
            cache_dir=str(self.cache_root),
            max_workers=max_workers,
            download_timeout=download_timeout,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            runtime_env=runtime_env,
            download_index_db_dir=download_index_db_dir,
        )
        self.workers.on_download_complete = self._on_download_complete
        self.workers.on_download_error = self._on_download_error
        self.workers.on_upload_complete = self._on_upload_complete
        self.workers.on_upload_error = self._on_upload_error

        # Tracks in-progress distributed directory syncs keyed by normalized s3_base
        self._dir_sync_jobs: dict[str, DirSyncJob] = {}

        # Backward compatibility: code may still reference .downloader
        self.downloader = self.workers

        # Service state
        self._is_started = False

        # Register crash cleanup: atexit covers normal exits and unhandled
        # exceptions; signal handlers cover SIGTERM/SIGINT.
        # signal.signal() requires the main thread — use Py_AddPendingCall to
        # post from Ray actor threads.
        atexit.register(self._crash_cleanup)
        if threading.current_thread() is threading.main_thread():
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
            self._prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)
            log.info(
                "[HydratorCache] registered SIGTERM/SIGINT handlers cache_root=%s",
                self.cache_root,
            )
        else:
            self._schedule_signal_handlers_on_main_thread()
            log.info(
                "[HydratorCache] scheduled SIGTERM/SIGINT registration on main thread"
                " cache_root=%s",
                self.cache_root,
            )

    def get_download_queue_size(self) -> int:
        """Number of download/upload tasks still in the worker queue (not yet handed to Ray)."""
        return self.workers.get_pending_task_count()

    def start(self) -> None:
        """Start the cache manager and its download worker pool."""
        with self._lock:
            if self._is_started:
                return
            self.workers.start()
            self._is_started = True

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the cache manager.

        Args:
            wait: If True, wait for active downloads/uploads to complete
        """
        with self._lock:
            if not self._is_started:
                return
            self.workers.shutdown(wait=wait)
            self._is_started = False

    def register_file(
        self,
        remote_path: str,
        modality: str,
        access_mode: hydrator_modality.AccessMode | None = None,
        profile: str | None = None,
    ) -> types.FileMetadata:
        """Register a file for tracking and trigger download if needed.

        Registering a file increments ref_count, marking it as active (ref_count > 0).
        Files with ref_count = 0 are eligible for eviction.

        When a ``download_index`` is configured, register_file consults it before
        enqueueing: if another rank already published the file, this rank adopts
        the local_path (state=READY, no download). If the URI is not yet
        published, ``partition_strategy.is_owner`` decides whether to enqueue
        the download or sit in ``PENDING`` until the owner publishes (the
        get_file poll loop adopts on hit).

        Args:
            remote_path: Remote file path
            modality: File modality type (string, will be converted to Modality enum)
            access_mode: How the file should be accessed (DOWNLOAD, DIRECT_READ, STREAMING)
            profile: Credentials profile to use for download (None = default pool)

        Returns:
            FileMetadata instance
        """
        with self._lock:
            # Check if file already exists in cache
            if remote_path in self.files:
                metadata = self.files[remote_path]

                # If file is already READY, it's a cache hit!
                if metadata.state == types.FileState.READY:
                    self.cache_hits += 1
                    metadata.ref_count += 1
                    metadata.last_accessed_at = time.time()
                    metadata.access_count += 1
                    return metadata

                # File exists but not ready yet (PENDING/DOWNLOADING/ERROR)
                # Just update the ref count
                metadata.ref_count += 1
                return metadata

            modality_enum = self._parse_modality(modality)

            # Cross-rank adoption: if another rank already published this URI
            # (success or terminal failure), register the file at the
            # corresponding terminal state and skip the enqueue entirely.
            record = self._try_adopt_from_index(remote_path)
            if record is not None:
                if record.local_path is not None:
                    metadata = types.FileMetadata(
                        remote_path=remote_path,
                        local_path=record.local_path,
                        state=types.FileState.READY,
                        size_bytes=0,
                        modality=modality_enum,
                        access_mode=access_mode,
                        last_accessed_at=time.time(),
                        access_count=1,
                        ref_count=1,
                        download_start_time=None,
                        error_message=None,
                    )
                    self.files[remote_path] = metadata
                    self.cache_hits += 1
                    return metadata
                # Terminal failure published by another rank — surface it
                # immediately rather than enqueueing a doomed retry.
                metadata = types.FileMetadata(
                    remote_path=remote_path,
                    local_path=None,
                    state=types.FileState.ERROR,
                    size_bytes=0,
                    modality=modality_enum,
                    access_mode=access_mode,
                    last_accessed_at=time.time(),
                    access_count=1,
                    ref_count=1,
                    download_start_time=None,
                    error_message=record.error_message,
                )
                self.files[remote_path] = metadata
                self.download_errors += 1
                return metadata

            # Not adopted: create a PENDING entry. Decide whether to enqueue
            # based on partition ownership when the download_index is on.
            metadata = types.FileMetadata(
                remote_path=remote_path,
                local_path=None,  # Will be set by _on_download_complete callback
                state=types.FileState.PENDING,
                size_bytes=0,  # Will be updated after download
                modality=modality_enum,
                access_mode=access_mode,
                last_accessed_at=time.time(),
                access_count=1,
                ref_count=1,
                download_start_time=None,
                error_message=None,
            )
            self.files[remote_path] = metadata

            # Ownership decision under download_index. The strategy's
            # is_owner is a static deterministic predicate; under
            # HashPartitionStrategy it assumes every rank that knows
            # about the URI registers it (otherwise the hash-picked
            # owner may not be a registrant → pollers deadlock). Under
            # IdentityStrategy it always returns True, so every
            # registrant enqueues and dedup falls back to best-effort
            # via _try_adopt_from_index for late registrants.
            #
            # Future stronger dedup (deferred): replace this is_owner
            # call with an atomic DownloadIndex.try_claim that races
            # via INSERT … ON CONFLICT DO NOTHING RETURNING. First
            # registrant wins regardless of strategy; concurrent
            # registrants all see in-flight and skip enqueue. Requires
            # UPSERT on completion + crash-recovery for stranded
            # claims — out of scope for the current iteration.
            should_enqueue = True
            if self.download_index is not None and self.partition_strategy is not None:
                should_enqueue = self.partition_strategy.is_owner(
                    remote_path, self.rank, self.world_size
                )
            if should_enqueue:
                # Profile travels with the task to the worker.
                self.workers.enqueue_download(remote_path, profile=profile)
            return metadata

    def _try_adopt_from_index(self, remote_path: str) -> DownloadRecord | None:
        """Check the cross-rank download_index for ``remote_path``.

        Returns a DownloadRecord iff the index has a *terminal, actionable*
        outcome for this URI:
        - success: ``local_path`` set AND the path exists on local
          filesystem (cross-rank readers see the owner's worker dir via
          shared Lustre).
        - error: ``error_message`` set (terminal failure published by
          another rank's ``_on_download_error``).

        Returns None when there is no record, or when ``local_path`` is
        set but the file isn't yet visible on this node (caller should
        keep waiting). Caller dispatches on which field is set.
        """
        if self.download_index is None:
            return None
        record = self.download_index.lookup(remote_path)
        if record is None:
            return None
        if record.local_path is not None and Path(record.local_path).exists():
            return record
        if record.error_message is not None:
            return record
        return None

    def adopt_if_published(self, remote_path: str) -> bool:
        """Try to move an existing PENDING entry to its terminal state via the cross-rank index.

        Returns True iff the entry was adopted (state moved to READY on
        success or ERROR on terminal failure). Used by the get_file poll
        loop on non-owner ranks: while the owner downloads, this rank's
        metadata sits in PENDING; once the owner publishes (success or
        error), the next poll tick adopts.

        Safe to call regardless of state — no-op if the entry is already
        READY/ERROR or if the index has no actionable record yet.
        """
        with self._lock:
            metadata = self.files.get(remote_path)
            if metadata is None or metadata.state != types.FileState.PENDING:
                return False
            record = self._try_adopt_from_index(remote_path)
            if record is None:
                return False
            metadata.last_accessed_at = time.time()
            if record.local_path is not None:
                metadata.local_path = record.local_path
                metadata.state = types.FileState.READY
                self.cache_hits += 1
            else:
                metadata.error_message = record.error_message
                metadata.state = types.FileState.ERROR
                self.download_errors += 1
            return True

    def register_directory_sync(
        self,
        s3_path: str,
        local_dir: str,
        rank: int = 0,
        world_size: int = 1,
        partition: str = "prefix",
        profile: str | None = None,
    ) -> None:
        """Async: enqueue this rank's slice of a directory transfer.

        Returns immediately; tasks run in the background on the WorkerPool.
        Workers stream file pairs directly into DuckDB (download_index) as
        they complete — get_file() pollers see entries within seconds.

        Two ways to cut the directory into slices:

        ``"prefix"`` gives each rank a set of filename-first-character globs, and lays the files
        out under one subdirectory per prefix. One listing is never taken, so nothing is known
        about the directory before the transfer starts -- but balance is only as good as the
        filenames. A directory whose large files share a first character (``model-*.safetensors``,
        ``shard-*.parquet``) puts all of them on a single rank while the others idle.

        ``"hash"`` lists the directory first and assigns each object by a hash of its URI, which
        balances regardless of how the files are named, and mirrors the remote layout locally
        rather than grouping by prefix. The cost is the listing, and that the object set is fixed
        when the slice is computed.

        Ranks must agree on ``partition``: each covers only its own slice, and the union is the
        directory only if they were cut the same way.

        Args:
            s3_path: S3 prefix URI (e.g. s3://bucket/prefix)
            local_dir: Local directory to sync files into
            rank: This process's rank (default 0)
            world_size: Total number of ranks (default 1 = no sharding)
            partition: ``"prefix"`` (default) or ``"hash"``
            profile: Credentials profile for listing and transfer (None = default pool)
        """
        if self.download_index is None:
            raise RuntimeError(
                "register_directory_sync() requires enable_download_index=True. "
                "Directory sync relies on DuckDB to stream file locations across "
                "ranks and processes. Re-create CacheManager with enable_download_index=True."
            )
        if partition not in ("prefix", "hash"):
            raise ValueError(f"unknown partition {partition!r}; expected 'prefix' or 'hash'")
        s3_base = _io_utils.normalize_path(s3_path).rstrip("/")
        # Recorded before any transfer is enqueued, and for the whole directory rather than this
        # rank's slice: it is what get_file() polls against, and every rank has to wait on objects
        # its neighbours own.
        with self._lock:
            self._dir_sync_jobs[s3_base] = DirSyncJob(
                s3_base=s3_base,
                local_dir=local_dir,
            )

        if partition == "hash":
            strategy = HashPartitionStrategy()
            pairs = [
                (uri, os.path.join(local_dir, uri[len(s3_base) + 1 :]))
                for uri in _io_utils.glob_paths(s3_base, "*", profile=profile)
                if strategy.is_owner(uri, rank, world_size)
            ]
            log.info(
                "[CacheManager] directory sync s3_base=%s rank=%d/%d partition=hash owned=%d",
                s3_base,
                rank,
                world_size,
                len(pairs),
            )
            if pairs:
                self.workers.enqueue_sync_files(s3_base, pairs, profile=profile)
            return

        strategy = AlphanumericPrefixStrategy()
        for prefix in strategy.all_prefixes():
            if not strategy.is_owner(prefix, rank, world_size):
                continue
            self.workers.enqueue_sync_dir(
                f"{s3_base}/{prefix}*", os.path.join(local_dir, prefix), profile=profile
            )

    def get_covering_dir_sync(self, remote_path: str) -> "DirSyncJob | None":
        """Return the DirSyncJob whose s3_base is a prefix of remote_path, or None.

        Used by get_file() to decide whether to wait before falling back to
        per-file download.

        Args:
            remote_path: Normalized S3 URI to check

        Returns:
            DirSyncJob if remote_path is covered by a registered sync, else None.
        """
        for s3_base, job in self._dir_sync_jobs.items():
            if remote_path.startswith(s3_base + "/"):
                return job
        return None

    def submit_upload(
        self,
        remote_path: str,
        content: bytes | Path,
        profile: str | None = None,
    ) -> None:
        """Submit an upload task to the worker pool and wait for completion.

        Used by the service layer (e.g. Hydrator.put_file) to write content to S3.

        Args:
            remote_path: S3 URI (e.g. s3://bucket/key/meta/stats.json)
            content: Bytes for in-memory content, or
                ``Path`` to stream a file directly from local disk via the
                backend's native streaming upload (boto3 ``upload_file``).
            profile: Credentials profile to use (None = default pool)

        Raises:
            RuntimeError: If upload fails or times out
        """
        size_bytes = content.stat().st_size if isinstance(content, Path) else len(content)
        started = time.monotonic()
        log.info(
            "[Hydrator] upload started size_bytes=%d size_mib=%.2f destination=%s",
            size_bytes,
            size_bytes / (1024**2),
            remote_path,
        )
        self.workers.enqueue_upload(remote_path, content, profile=profile)
        if not self.workers.wait_for_upload(remote_path):
            raise RuntimeError(f"Upload timed out: {remote_path}")
        log.info(
            "[Hydrator] upload completed size_bytes=%d size_mib=%.2f "
            "destination=%s elapsed_seconds=%.3f",
            size_bytes,
            size_bytes / (1024**2),
            remote_path,
            time.monotonic() - started,
        )

    def submit_upload_async(
        self,
        remote_path: str,
        content: bytes | Path,
        profile: str | None = None,
    ) -> None:
        """Submit an upload task without blocking on completion.

        Creates a FileMetadata entry in ``files`` with state=PENDING and
        enqueues the upload on the worker pool. State transitions to READY
        (success) or ERROR (failure) via _on_upload_complete /
        _on_upload_error callbacks fired by the WorkerPool consumer.

        Args:
            remote_path: S3 URI (e.g. s3://bucket/key/data/...parquet)
            content: Bytes for in-memory content, or
                ``Path`` to stream a file directly from local disk. Path
                must remain valid until the upload finishes.
            profile: Credentials profile to use (None = default pool)
        """
        size_bytes = content.stat().st_size if isinstance(content, Path) else len(content)
        with self._lock:
            # local_path stays None (no on-disk staging for uploads) and
            # ref_count stays 0 (uploads aren't held by a reader).
            self.files[remote_path] = types.FileMetadata(
                remote_path=remote_path,
                local_path=None,
                state=types.FileState.PENDING,
                size_bytes=size_bytes,
                modality=types.Modality.UNKNOWN,
                access_mode=None,
                last_accessed_at=time.time(),
                access_count=0,
                ref_count=0,
                download_start_time=None,
                error_message=None,
            )
        self.workers.enqueue_upload(remote_path, content, profile=profile)

    def _parse_modality(self, modality: str) -> types.Modality:
        """Parse modality string to Modality enum.

        Args:
            modality: Modality string (e.g., "video", "parquet", "auto")

        Returns:
            Modality enum value
        """
        modality_lower = modality.lower()

        if modality_lower in ("video", "mp4", "avi", "mov", "mkv", "webm"):
            return types.Modality.VIDEO
        elif modality_lower in ("parquet", "pq"):
            return types.Modality.PARQUET
        else:
            return types.Modality.UNKNOWN

    def get_file_metadata(self, remote_path: str) -> types.FileMetadata | None:
        """Get metadata for a file.

        Args:
            remote_path: Remote file path

        Returns:
            FileMetadata if file is registered, None otherwise
        """
        with self._lock:
            return self.files.get(remote_path)

    def update_file_state(
        self,
        remote_path: str,
        state: types.FileState,
        error_message: str | None = None,
    ) -> None:
        """Update the state of a file in the cache.

        Args:
            remote_path: Remote file path
            state: New state for the file
            error_message: Optional error message if state is ERROR
        """
        with self._lock:
            if remote_path not in self.files:
                return

            metadata = self.files[remote_path]
            old_state = metadata.state
            metadata.state = state

            if error_message:
                metadata.error_message = error_message

            # Update timestamp when transitioning to READY
            if state == types.FileState.READY and old_state != types.FileState.READY:
                metadata.last_accessed_at = time.time()

    def mark_accessed(self, remote_path: str) -> None:
        """Mark file as accessed (for LRU tracking).

        Updates access timestamp and counter for the file. This is useful for
        tracking file usage patterns and improving LRU eviction decisions.

        Note: get_file() already calls this automatically, so explicit calls
        are only needed for special tracking scenarios.

        Args:
            remote_path: Remote file path

        Example:
            # Access tracking is automatic when using get_file()
            local_path = cache.get_file("s3://bucket/video.mp4")

            # Manual tracking (rare, only if needed)
            cache.mark_accessed("s3://bucket/video.mp4")
        """
        with self._lock:
            if remote_path not in self.files:
                # File not registered, nothing to update
                return

            metadata = self.files[remote_path]

            # Update access tracking
            metadata.last_accessed_at = time.time()
            metadata.access_count += 1

    def release_files(self, remote_paths: list[str]) -> None:
        """Release files, indicating they are no longer actively needed by caller.

        This decrements the reference count for each file. Files with ref_count=0
        become eligible for LRU eviction.

        This is generic and works for any caller (shards, batches, sessions, etc.).

        Args:
            remote_paths: List of remote paths to release

        Example:
            # Caller registers files
            cache.register_file("s3://bucket/video1.mp4", "video")
            cache.register_file("s3://bucket/video2.mp4", "video")

            # Use files...

            # When done, release them
            cache.release_files(["s3://bucket/video1.mp4", "s3://bucket/video2.mp4"])
        """
        with self._lock:
            for remote_path in remote_paths:
                if remote_path not in self.files:
                    # File not registered, skip silently
                    continue

                metadata = self.files[remote_path]

                # Decrement ref_count (but not below 0)
                # Files with ref_count=0 are eligible for LRU eviction
                if metadata.ref_count > 0:
                    metadata.ref_count -= 1

    def drop_entry(self, remote_path: str) -> bool:
        """Remove a file's tracking entry from the cache state.

        Companion to ``Hydrator.purge_files``: the service layer deletes
        the file on disk; this method clears the corresponding metadata
        so subsequent ``register_file`` calls treat the path as fresh
        (re-download), and so ``cleanup`` doesn't try to ``unlink`` an
        already-gone file.

        Caller must hold no other reference to the file's metadata
        (i.e. ``release_files`` should have driven ``ref_count`` to 0
        first), and must have already deleted ``metadata.local_path``
        on disk if appropriate.

        Args:
            remote_path: Remote file path to drop from tracking.

        Returns:
            True if an entry was removed, False if the path was not
            registered (no-op, not an error).
        """
        with self._lock:
            if remote_path not in self.files:
                return False
            del self.files[remote_path]
            return True

    def evict_if_needed(self) -> int:
        """Evict files if cache size exceeds limit.

        Uses the configured eviction policy (default: LRU) on inactive files only (ref_count=0).
        Active files (ref_count > 0, currently in use) are never evicted.

        Algorithm:
        1. Calculate current total cache size
        2. If size > max_cache_size_bytes:
           - Target size = max_cache_size_bytes * eviction_buffer (90% by default)
           - Use eviction_policy to select files to evict (sorted order)
           - Evict files one by one until size < target_size
           - Update evictions counter
        3. Never evict active files (ref_count > 0) - policy ensures this

        Thread-safe - uses locking to prevent concurrent modification.

        Returns:
            Number of files evicted

        Raises:
            RuntimeError: If unable to free enough space (all files are active)

        Example:
            # Called automatically after downloads complete
            evicted_count = cache.evict_if_needed()
            if evicted_count > 0:
                print(f"Evicted {evicted_count} files to free disk space")
        """
        with self._lock:
            # Cache-quota accounting covers on-disk download entries only.
            # Upload-state entries (submit_upload_async) carry the upload's
            # size_bytes but local_path=None and never occupy local disk;
            # including them inflates total_size and, worse, makes them
            # eligible for the unconditional del below — which silently
            # drops the state that the upload-polling caller is waiting on.
            on_disk_files = {
                remote_path: metadata
                for remote_path, metadata in self.files.items()
                if metadata.local_path is not None
            }
            total_size = sum(metadata.size_bytes for metadata in on_disk_files.values())

            # Check if eviction is needed
            if total_size <= self.max_cache_size_bytes:
                return 0  # No eviction needed

            # Cross-worker references from adopt_if_published mean any
            # rank's eviction can dangle other ranks' local_paths. Skip
            # eviction entirely when the cross-rank index is on; warn
            # once the first time we cross the configured quota so
            # operators know to size max_cache_size_gb appropriately.
            if self.download_index is not None:
                if not self._over_quota_warning_logged:
                    log.warning(
                        "[HydratorCache] cache size %.2f GB exceeds quota %.2f GB"
                        " — eviction disabled under enable_download_index=True"
                        " (cross-worker references not yet ref-counted)."
                        " Increase max_cache_size_gb.",
                        total_size / (1024**3),
                        self.max_cache_size_bytes / (1024**3),
                    )
                    self._over_quota_warning_logged = True
                return 0

            # Calculate target size (with buffer)
            target_size = int(self.max_cache_size_bytes * self.eviction_buffer)

            # Use eviction policy to select files to evict (sorted by policy's strategy)
            # Policy returns only inactive files (ref_count=0) in eviction order
            files_to_evict = self.eviction_policy.select_files_to_evict(on_disk_files)

            # Check if we can free enough space
            if not files_to_evict:
                # All files are active - cannot evict.
                # In practice, release_files() is never called by callers, so ref_count
                # never reaches 0 and this branch is always hit when the cache is full.
                raise RuntimeError(
                    f"Cache size ({total_size / (1024**3):.2f} GB) exceeds limit "
                    f"({self.max_cache_size_bytes / (1024**3):.2f} GB), but all files "
                    f"are currently active (ref_count > 0). Cannot evict any files. "
                    f"Consider increasing max_cache_size_gb or reducing concurrent file usage."
                )

            # Evict files until we're under target size
            evicted_count = 0
            for remote_path, metadata in files_to_evict:
                # Check if we've freed enough space
                if total_size < target_size:
                    break

                # Delete file from disk
                if metadata.local_path:
                    try:
                        if os.path.exists(metadata.local_path):
                            file_size = metadata.size_bytes
                            os.remove(metadata.local_path)

                            # Try to remove parent directory if empty
                            parent_dir = os.path.dirname(metadata.local_path)
                            try:
                                if parent_dir != str(self.cache_root) and os.path.isdir(parent_dir):
                                    os.rmdir(parent_dir)  # Only succeeds if empty
                            except OSError:
                                # Directory not empty or other error - ignore
                                pass

                            # Update total size
                            total_size -= file_size
                            evicted_count += 1

                    except Exception as e:  # noqa: BLE001
                        log.warning("Failed to evict %s: %s", metadata.local_path, e)
                        continue

                # Remove from tracking
                del self.files[remote_path]

            # Update evictions counter
            self.evictions += evicted_count

            return evicted_count

    def get_stats(self) -> types.CacheStats:
        """Get cache statistics.

        Returns comprehensive metrics about cache state, performance, and resource usage.
        Thread-safe - can be called concurrently.

        Returns:
            CacheStats instance with current statistics

        Example:
            stats = cache.get_stats()
            print(f"Cache size: {stats.total_size_gb:.2f} GB")
            print(f"Cache hits: {stats.cache_hits}")
            print(f"Active files: {stats.active_files}")
        """
        with self._lock:
            total_size_bytes = 0
            total_files = 0
            active_files = 0
            cached_files = 0
            pending_downloads = 0
            downloading_files = 0

            for metadata in self.files.values():
                total_files += 1
                total_size_bytes += metadata.size_bytes

                # Count by activity (ref_count)
                if metadata.ref_count > 0:
                    active_files += 1
                else:
                    cached_files += 1

                # Count by state
                if metadata.state == types.FileState.PENDING:
                    pending_downloads += 1
                elif metadata.state == types.FileState.DOWNLOADING:
                    downloading_files += 1

            return types.CacheStats(
                total_size_gb=total_size_bytes / (1024**3),
                total_files=total_files,
                active_files=active_files,
                cached_files=cached_files,
                cache_hits=self.cache_hits,
                downloads=self.downloads,
                evictions=self.evictions,
                download_errors=self.download_errors,
                pending_downloads=pending_downloads,
                downloading_files=downloading_files,
            )

    def cleanup(self, remove_all: bool = False) -> None:
        """Clean up cache by removing files from disk.

        This method removes cached files and cleans up the cache state.
        - If remove_all=True: Deletes ALL files (active and inactive)
        - If remove_all=False: Only deletes INACTIVE files (ref_count=0)

        Active files (ref_count > 0) are protected when remove_all=False.

        Thread-safe - uses locking to prevent concurrent modification.

        Args:
            remove_all: If True, remove all cached files regardless of ref_count.
                       If False, only remove inactive files (ref_count=0).

        Example:
            # Remove only inactive files (safe - doesn't affect active files)
            cache.cleanup(remove_all=False)

            # Remove everything (typically on shutdown)
            cache.cleanup(remove_all=True)
        """
        with self._lock:
            files_to_remove = []  # Files with local_path to remove from disk

            for remote_path, metadata in list(self.files.items()):
                # Determine if we should remove this file
                should_remove = remove_all or (metadata.ref_count == 0)

                if should_remove:
                    # Add to disk removal list if it has a local_path
                    if metadata.local_path:
                        files_to_remove.append((remote_path, metadata.local_path))

                    # Remove from tracking immediately (works because we iterate over list())
                    del self.files[remote_path]
                    self.evictions += 1

            # Remove files from disk
            for _remote_path, local_path in files_to_remove:
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        log.debug("Removed file: %s", local_path)
                        # Try to remove parent directory if empty
                        parent_dir = os.path.dirname(local_path)
                        try:
                            if parent_dir != str(self.cache_root) and os.path.isdir(parent_dir):
                                os.rmdir(parent_dir)  # Only succeeds if empty
                        except OSError:
                            # Directory not empty or other error - ignore
                            pass

                except Exception as e:  # noqa: BLE001
                    log.warning("Failed to remove %s: %s", local_path, e)
                    continue

            # If remove_all, also try to clean up the entire cache root
            if remove_all and os.path.exists(self.cache_root):
                try:
                    # Remove any remaining empty directories
                    for root, dirs, _files in os.walk(self.cache_root, topdown=False):
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            try:
                                os.rmdir(dir_path)  # Only succeeds if empty
                            except OSError:
                                pass
                except Exception as e:  # noqa: BLE001
                    log.warning("Failed to clean cache root %s: %s", self.cache_root, e)

    def _crash_cleanup(self) -> None:
        """Shut down workers and delete all cached files. Called on crash/exit."""
        log.info("[HydratorCache] crash cleanup started cache_root=%s", self.cache_root)
        self.shutdown(wait=False)
        self.cleanup(remove_all=True)
        log.info("[HydratorCache] crash cleanup done cache_root=%s", self.cache_root)

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Clean up cache and re-raise signal so the process exits normally."""
        sig_name = signal.Signals(signum).name
        log.info("[HydratorCache] received %s, running crash cleanup", sig_name)
        self._crash_cleanup()
        prev = self._prev_sigterm if signum == signal.SIGTERM else self._prev_sigint
        if callable(prev):
            prev(signum, frame)
        else:
            signal.raise_signal(signum)

    def _schedule_signal_handlers_on_main_thread(self) -> None:
        """Post signal handler registration to the main thread via Py_AddPendingCall.

        Py_AddPendingCall schedules a C-level callback to run on the main
        thread at the next interpreter bytecode check — the standard CPython
        mechanism for posting work from any thread to the main thread.
        """
        _PendingCallType = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

        def _register(_arg: ctypes.c_void_p) -> int:
            try:
                self._prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
                self._prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)
            except Exception:  # noqa: BLE001, S110
                pass
            with _pending_signal_lock:
                _pending_signal_cfuncs.remove(cfunc)
            return 0

        cfunc = _PendingCallType(_register)
        with _pending_signal_lock:
            _pending_signal_cfuncs.append(cfunc)
        ctypes.pythonapi.Py_AddPendingCall(cfunc, None)

    def _sweep_stale_caches(self, base_cache_dir: str) -> None:
        """Delete cached files left by previous crashed runs (SIGKILL/OOM).

        On cluster runs, base_cache_dir is ``{root}/{workflow_id}/{ray_job_id}``.
        Each run gets a fresh ray_job_id, so stale data from a crashed run lives
        in sibling directories under ``{workflow_id}/``.  We delete those siblings
        while leaving the current run's directory untouched.

        The sweep is skipped when WORKFLOW_ID is not embedded in the path, which
        means the cache dir is not scoped to this workflow and may be shared.
        """
        workflow_id = os.environ.get("WORKFLOW_ID", "")
        if not workflow_id or workflow_id not in base_cache_dir:
            return
        workflow_dir = os.path.dirname(base_cache_dir)
        current_job_dir = os.path.basename(base_cache_dir)
        if not os.path.exists(workflow_dir):
            return
        for entry in os.scandir(workflow_dir):
            if entry.is_dir() and entry.name != current_job_dir:
                log.info("[HydratorCache] sweeping stale cache from previous run: %s", entry.path)
                shutil.rmtree(entry.path, ignore_errors=True)

    def _on_download_complete(self, remote_path: str, local_path: str) -> None:
        """Callback when a download completes successfully (internal).

        Args:
            remote_path: Remote file path that finished downloading
            local_path: Local file path where backend saved the file (source of truth)
        """
        with self._lock:
            self.downloads += 1
            in_files = remote_path in self.files

            # Update local_path from backend (backend is source of truth)
            if in_files:
                metadata = self.files[remote_path]
                # Set the actual local path that backend used
                metadata.local_path = local_path

        self.update_file_state(remote_path, types.FileState.READY)
        if self.download_index is not None and local_path is not None:
            self.download_index.batch_insert(
                [DownloadRecord(s3_uri=remote_path, local_path=local_path, error_message=None)]
            )
        log.debug(
            "[HydratorCache] download READY remote=%s local=%s in_files=%s",
            remote_path,
            local_path,
            in_files,
        )

    def _on_download_error(self, remote_path: str, error: Exception) -> None:
        """Callback when a download fails after all retries (internal).

        Publishes the terminal failure to the cross-rank download_index so
        non-owner ranks polling for this URI can surface the same error
        instead of waiting for a successful publish that will never come.

        Args:
            remote_path: Remote file path
            error: The exception that caused the failure
        """
        error_message = str(error)
        with self._lock:
            self.download_errors += 1

        self.update_file_state(remote_path, types.FileState.ERROR, error_message=error_message)
        if self.download_index is not None:
            self.download_index.batch_insert(
                [
                    DownloadRecord(
                        s3_uri=remote_path,
                        local_path=None,
                        error_message=error_message,
                    )
                ]
            )

    def _on_upload_complete(self, remote_path: str) -> None:
        """Callback when an async upload completes successfully (internal).

        Sync ``submit_upload`` callers don't register a ``files`` entry, so
        ``update_file_state`` is a silent no-op for them — only async
        ``submit_upload_async`` callers get a state transition.
        """
        self.update_file_state(remote_path, types.FileState.READY)

    def _on_upload_error(self, remote_path: str, error: Exception) -> None:
        """Callback when an async upload fails after all retries (internal)."""
        self.update_file_state(remote_path, types.FileState.ERROR, error_message=str(error))
