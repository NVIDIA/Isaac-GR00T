# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main service implementation for Data Hydrator."""

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import pyarrow as pa

try:
    import ray as _ray

    _RAY_AVAILABLE = True
except ImportError:
    _ray = None
    _RAY_AVAILABLE = False

from groot.core.data.hydrator import backends, cache, exceptions, types
from groot.core.data.hydrator import client as _hydrator_client
from groot.core.data.hydrator import config as hydrator_config
from groot.core.data.hydrator import data_source_resolver as _dsr
from groot.core.data.hydrator import modality as hydrator_modality
from groot.core.data.hydrator.utils import io
from groot.core.utils.distributed import worker_logging

log = worker_logging.get_unified_logger()

# Osmo pool → cache root mapping. GB300 / SkyPilot DGXC clusters don't expose
# a pool name through osmo, so they're discovered by the filesystem-probe
# fallback below.
_GB200_POOLS = {
    "groot-gb200-01",
    "groot-gb200-02",
    "groot-gb200-ci-01",
    "groot-gb200-ci-02",
    "groot-gb200-nvlink-02",
}
_GB200_CACHE_ROOT = "/mnt/aws-lfs-02/shared/hydrator_cache"
_GB300_CACHE_ROOT = "/mnt/aws-lfs-03/shared/hydrator_cache"
_H100_CACHE_ROOT = "/mnt/amlfs-07/shared/hydrator_cache"

# Probe order for filesystem-based detection (sky / DGXC / unknown osmo pool).
# Checked top-down; first existing directory wins.
_LUSTRE_PROBE_ORDER = (
    ("/mnt/aws-lfs-03/shared", _GB300_CACHE_ROOT),
    ("/mnt/aws-lfs-02/shared", _GB200_CACHE_ROOT),
    ("/mnt/amlfs-07/shared", _H100_CACHE_ROOT),
)


def _default_max_workers() -> int:
    """Return num_nodes * 24, or 24 if Ray is not initialized."""
    try:
        if _RAY_AVAILABLE and _ray is not None and _ray.is_initialized():
            num_nodes = len([n for n in _ray.nodes() if n.get("Alive")])
            if num_nodes > 0:
                return num_nodes * 24
    except Exception:  # noqa: BLE001, S110
        pass
    return 24


def _default_tmp_root() -> str:
    """Return a sensible default cache root based on the current cluster.

    Resolution order:

    1. Osmo pool lookup (``WORKFLOW_ID`` set): select the cache root associated
       with the workflow's pool.
    2. Filesystem-mount probe: if osmo lookup didn't apply (no WORKFLOW_ID, or
       it failed), pick the first mount in ``_LUSTRE_PROBE_ORDER`` that exists
       on disk. This covers SkyPilot / DGXC pods (GB300 → aws-lfs-03,
       GB200-sky → aws-lfs-02) which don't advertise a pool name via osmo.
    3. Fallback: ``<cwd>/hydrate_cache`` when no Lustre mount is found.

    Paths are namespaced by ``<workflow_or_task_id>/<ray-job-id>`` to keep
    concurrent jobs isolated.
    """
    workflow_id = os.environ.get("WORKFLOW_ID")
    ray_job_id = os.environ.get("RAY_JOB_ID", "no_ray_job_id")

    if workflow_id:
        try:
            result = subprocess.run(
                [
                    "osmo",
                    "task",
                    "list",
                    "--workflow-id",
                    workflow_id,
                    "-a",
                    "-t",
                    "json",
                    "--count",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            tasks = json.loads(result.stdout).get("tasks", [])
            pool = tasks[0].get("pool") if tasks else None
            if pool in _GB200_POOLS:
                return os.path.join(_GB200_CACHE_ROOT, workflow_id, ray_job_id)
            # Any other osmo pool (h100, l40, gb300, new pools) falls through to the
            # filesystem-mount probe below — it has its own existence check, so we never
            # return a path on a Lustre mount that doesn't exist on this node.
        except Exception:  # noqa: BLE001, S110
            pass

    # Sky / DGXC / unknown osmo pool: probe known Lustre mounts in priority order.
    # ``GROOT_RUN_ID`` is injected by both ``gear sky launch-job`` (per-launch) and
    # ``gear sky start-cluster`` (per-cluster), so it unifies the sky env contract.
    task_id = workflow_id or os.environ.get("GROOT_RUN_ID") or "no_task_id"
    for mount, root in _LUSTRE_PROBE_ORDER:
        if os.path.isdir(mount):
            return os.path.join(root, task_id, ray_job_id)

    return os.path.join(os.getcwd(), "hydrate_cache")


class Hydrator:
    """Main interface for data hydration from remote storage.

    This service provides:
    - Modality-aware file access (videos cached locally, parquet read directly from S3)
    - Asynchronous prefetching and download management
    - LRU cache with automatic eviction
    - Pluggable storage backends (S3, LustreFS, etc.)
    - Thread-safe API for DataLoader workers

    Usage:
        # Initialize hydrator
        config = HydratorConfig(max_workers=4, max_cache_size_gb=50.0)
        hydrator = Hydrator(job_name="train_job", worker_id=0, config=config)

        # With custom backend settings
        config = HydratorConfig(
            backend="boto3",
            backend_config={"transfer_config": {"multipart_threshold": 26214400}}
        )
        hydrator = Hydrator(job_name="train_job", worker_id=0, config=config)

        # Register files (downloads start immediately for videos)
        video_paths = ["s3://bucket/video1.mp4", "s3://bucket/video2.mp4"]
        hydrator.register_files(video_paths, modality=Modality.VIDEO)

        # Get file (blocks if not ready)
        local_path = hydrator.get_file("s3://bucket/video1.mp4")

        # Release files when done
        hydrator.release_files(video_paths)

        # Shutdown when done
        hydrator.shutdown()

    Args:
        job_name: Unique job identifier (e.g., Ray job ID)
        worker_id: DataLoader worker ID (int) for per-process cache isolation.
        config: Service configuration
    """

    @staticmethod
    def normalize_path(path: str) -> str:
        """Return canonical path string for use with hydrator.

        Fixes s3:/ to s3:// (str(Path('s3://...')) can become s3:/ on some OSes).
        Callers do not need to call this; join_path, get_file, glob, is_remote_path,
        etc. normalize at entry so path forms (e.g. s3:/ from str(Path('s3://...'))) stay internal.
        """
        return io.normalize_path(path)

    @staticmethod
    def is_remote_path(path: str) -> bool:
        """Return True if path is a remote (S3) path that hydrator should handle. Normalizes at entry."""
        return io.normalize_path(path).startswith("s3://")

    def resolve_to_local_root(self, path: str) -> str | None:
        """Resolve a dataset path to a local directory for file I/O.

        - Local path: returns path as-is if it is an existing directory, else None.
        - S3 path: returns None (cache does not expose one local root); callers use
          get_file/glob per file and thus still generate stats from local files after download.

        Args:
            path: Dataset path (local or S3 URI).

        Returns:
            Local directory path to use for stats generation, or None if caller must
            use get_file/glob per file (e.g. S3).
        """
        canonical = Hydrator.normalize_path(path)
        if not canonical.startswith("s3://"):
            return canonical if Path(canonical).is_dir() else None
        return None

    @staticmethod
    def resolve_path(path: str) -> str:
        """Resolve a local dataset path to the best available URI.

        Tries AWS S3 first, then PDX/Ceph, then returns the original path
        unchanged (LustreFS fallback).  Paths already starting with ``s3://``
        are returned unchanged.

        This is a thin wrapper around the default :class:`DataSourceResolver`
        so callers do not need to import the resolver directly.

        Args:
            path: Dataset root path (local or ``s3://``).

        Returns:
            Best available URI (``s3://`` if a remote copy exists, else the
            original local path).
        """
        uri, _profile = _dsr.default_resolver().resolve(Hydrator.normalize_path(path))
        return uri

    def hydrate(self, path: str) -> list[dict] | pa.Table:
        """Read a .json, .jsonl, or .parquet file and return its content."""
        profile = self._get_winning_profile(path)
        return io.consume(path, profile=profile)

    @staticmethod
    def join_path(base: str, relative: str) -> str:
        """Join a base path and a relative path. Paths are normalized at entry; callers may pass raw paths.

        For local paths (base does not start with "s3://"), uses Path to join
        and normalize. For S3 URIs, concatenates with a single "/" between
        and normalizes the result (collapses repeated slashes).

        Args:
            base: Base path (e.g. dataset root), local or s3://...
            relative: Relative path (e.g. "meta/info.json", "meta/stats.json")

        Returns:
            Joined path as string.

        Example:
            >>> Hydrator.join_path("/data/dataset", "meta/info.json")
            '/data/dataset/meta/info.json'
            >>> Hydrator.join_path("s3://bucket/key", "meta/stats.json")
            's3://bucket/key/meta/stats.json'
        """
        base = Hydrator.normalize_path(base)
        if base.startswith("s3://"):
            joined = base.rstrip("/") + "/" + relative.lstrip("/")
            return Hydrator.normalize_path(joined)
        return str(Path(base) / relative)

    @staticmethod
    def remote_file_exists(path: str) -> bool:
        """Check whether a remote (``s3://``) object exists, HEAD-only.

        Does not download the object body. Returns False on 404 / NoSuchKey;
        re-raises other S3 errors. Credentials profile is resolved per-URI via
        :meth:`DataSourceResolver.get_profile_for_uri` — which probes each
        configured tier to find one that can access the bucket — and passed
        explicitly to ``io.s3_object_exists``, matching the contract that
        every io read in the package passes an explicit profile.

        Local paths fall back to :meth:`pathlib.Path.exists` for convenience,
        so this is safe to call on a path of unknown locality.

        Args:
            path: Path or S3 URI to check.

        Returns:
            ``True`` if the object/file exists, ``False`` otherwise.
        """
        from groot.core.data.hydrator.utils import io

        profile = _dsr.default_resolver().get_profile_for_uri(path)
        return io.s3_object_exists(path, profile=profile)

    def glob(self, base_path: str, pattern: str) -> list[str]:
        """List paths matching pattern under base_path (local or S3). See utils.glob.glob_paths."""
        profile = self._get_winning_profile(base_path)
        return io.glob_paths(base_path, pattern, profile=profile)

    def __init__(
        self,
        job_name: str,
        worker_id: int,
        config: hydrator_config.HydratorConfig | None = None,
    ):
        self.job_name = job_name
        self.worker_id = worker_id
        self.config = (config or hydrator_config.HydratorConfig()).with_distributed_defaults()

        # Auto-resolve credentials profile: pick the first tier with working credentials.
        if self.config.credentials_profile is None:
            import dataclasses

            active_profile = _dsr.default_resolver()._get_active_profile()  # noqa: SLF001
            self.config = dataclasses.replace(
                self.config, credentials_profile=active_profile or "aws"
            )

        # Determine cache root directory.
        base_cache_dir = (
            self.config.tmp_root if self.config.tmp_root is not None else _default_tmp_root()
        )

        # Resolve max_workers: None means auto-detect from Ray cluster size.
        max_workers = (
            self.config.max_workers
            if self.config.max_workers is not None
            else _default_max_workers()
        )

        # Job directory groups all data for this (workflow, ray-job) pair so that
        # shutdown() can clean it up with a single rmtree.
        self._job_dir = os.path.join(base_cache_dir, self.job_name)

        # Each worker syncs s5cmd files into its own subdirectory.
        # Exposed as cache_root so callers can use it as the local sync target.
        self.cache_root = os.path.join(self._job_dir, f"worker_{worker_id}")
        cache_root = self.cache_root

        # Operator-friendly trace of where this rank's hydrator lives on
        # shared storage. Combined with the per-shard log lines emitted by
        # DownloadIndex this makes it easy to confirm cross-rank ownership
        # routing without having to inspect any DBs.
        log.info(
            "[Hydrator] init job=%s worker_id=%s tmp_root=%s job_dir=%s cache_root=%s",
            self.job_name,
            worker_id,
            base_cache_dir,
            self._job_dir,
            cache_root,
        )

        # Merge credentials into backend_config so they flow through to workers.
        backend_config = {
            **self.config.backend_config,
            "credentials_profile": self.config.credentials_profile,
            "credentials_file": self.config.credentials_file,
        }

        self.backend = backends.create_backend(
            backend_type=self.config.backend,
            working_dir=cache_root,
            backend_config=backend_config,
        )

        self.modality_router = hydrator_modality.ModalityRouter()

        self.cache_manager = cache.CacheManager(
            cache_root=cache_root,
            max_cache_size_gb=self.config.max_cache_size_gb,
            eviction_buffer=self.config.eviction_buffer,
            backend=self.backend,
            backend_type=self.config.backend,
            backend_config=backend_config,
            max_workers=max_workers,
            download_timeout=self.config.download_timeout,
            retry_attempts=self.config.retry_attempts,
            retry_delay=self.config.retry_delay,
            runtime_env=self.config.runtime_env,
            enable_download_index=self.config.enable_download_index,
            rank=self.config.rank,
            world_size=self.config.world_size,
            partition_strategy=self.config.download_partition_strategy,
        )

        self._active_files: list[str] = []
        self._lock = threading.RLock()
        self._is_started = False
        self._is_shutdown = False

        # IPC server for create_ipc_client() (DataLoader workers, etc.)
        self._ipc_manager: Any | None = None
        self._ipc_request_queue: Any | None = None
        self._ipc_response_dict: Any | None = None
        self._ipc_server_thread: threading.Thread | None = None

        self._start()

    def _start(self) -> None:
        """Start the service (internal)."""
        with self._lock:
            if self._is_started:
                return
            self.cache_manager.start()
            self.start_ipc_server()
            self._is_started = True

    def _get_winning_profile(self, path: str) -> str:
        """Return the credentials profile that should be used to access *path*.

        Checks ``DataSourceResolver``'s bucket→profile map first (populated
        by any successful :meth:`DataSourceResolver.resolve` call, whether via
        :meth:`resolve_path`, :meth:`register_files`, or direct resolver use).
        Falls back to ``self.config.credentials_profile`` for buckets not yet
        seen (e.g. bare S3 URIs passed without a prior resolve call).

        Args:
            path: S3 URI or local path.

        Returns:
            Credentials profile string.
        """
        winning = _dsr.default_resolver().get_profile_for_uri(path)
        return winning if winning is not None else self.config.credentials_profile

    def _ipc_server_loop(self) -> None:
        """Run in IPC server thread.

        Reads (request_id, method, args, kwargs) from queue, calls hydrator,
        and puts result in response_dict.
        """
        _IPC_ERROR = "_IPC_ERROR"
        request_queue = self._ipc_request_queue
        response_dict = self._ipc_response_dict
        assert request_queue is not None and response_dict is not None
        while True:
            item = request_queue.get()
            if item is None:
                break
            request_id, method, args, kwargs = item
            try:
                result = getattr(self, method)(*args, **kwargs)
                response_dict[request_id] = result
            except Exception as e:  # noqa: BLE001
                response_dict[request_id] = (
                    _IPC_ERROR,
                    type(e).__name__,
                    str(e),
                )

    def start_ipc_server(self) -> None:
        """Start the IPC server thread so forked processes (e.g. DataLoader workers) can use get_ipc_client().

        Creates a Manager, queue, and response dict; runs the server thread in this process.
        Call get_ipc_client() to obtain a client, and get_ipc_shutdown_handles() to attach to your
        dataset (or similar) for shutdown when done.
        """
        if self._ipc_server_thread is not None:
            return
        self._ipc_manager = multiprocessing.Manager()
        self._ipc_request_queue = self._ipc_manager.Queue()
        self._ipc_response_dict = self._ipc_manager.dict()
        self._ipc_server_thread = threading.Thread(
            target=self._ipc_server_loop,
            daemon=False,
        )
        self._ipc_server_thread.start()
        # Ensure server thread is running before return (e.g. from __init__)
        deadline = time.monotonic() + 5.0
        while not self._ipc_server_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.001)
        if not self._ipc_server_thread.is_alive():
            raise RuntimeError("IPC server thread failed to start")

    def get_ipc_client(self) -> Any:
        """Return a HydratorClient that talks to this Hydrator via IPC.

        The IPC server is started in the constructor.
        """
        if self._ipc_request_queue is None or self._ipc_response_dict is None:
            raise RuntimeError("start_ipc_server() must be called before get_ipc_client()")
        return _hydrator_client.HydratorClient(
            request_queue=self._ipc_request_queue,
            response_dict=self._ipc_response_dict,
            credentials_profile=self.config.credentials_profile,
        )

    def get_ipc_shutdown_handles(
        self,
    ) -> tuple[Any, Any, threading.Thread]:
        """Return (request_queue, response_dict, server_thread) for the caller.

        Use request_queue and response_dict to create HydratorClients in workers;
        use queue and thread for shutdown (put None on queue, join thread).
        """
        if (
            self._ipc_request_queue is None
            or self._ipc_response_dict is None
            or self._ipc_server_thread is None
        ):
            raise RuntimeError(
                "start_ipc_server() must be called before get_ipc_shutdown_handles()"
            )
        return (self._ipc_request_queue, self._ipc_response_dict, self._ipc_server_thread)

    def _stop_ipc_server(self, timeout: float = 30.0) -> None:
        """Stop the IPC server thread. Safe to call multiple times. Only used by shutdown()."""
        with self._lock:
            if self._ipc_server_thread is None:
                return
            queue = self._ipc_request_queue
            thread = self._ipc_server_thread
            if queue is not None:
                queue.put(None)
            if thread is not None:
                thread.join(timeout=timeout)
            self._ipc_manager = None
            self._ipc_request_queue = None
            self._ipc_response_dict = None
            self._ipc_server_thread = None

    def register_files(
        self,
        paths: list[str],
        modality: types.Modality = types.Modality.AUTO,
    ) -> None:
        """Register files for async prefetching (optional optimization).

        For cacheable files (videos), downloads start immediately in the order provided.
        For direct-read files (parquet), this is a no-op.

        NOTE: Calling register_files() is OPTIONAL. If you call get_file() without
        registering first, it will auto-register and download synchronously.
        register_files() is an optimization for async prefetching.

        Multiple calls to register_files() can be made to have multiple active file sets.
        Call `release_files()` explicitly when done to make files eligible for eviction.

        Args:
            paths: List of remote paths (S3, LustreFS, etc.)
            modality: Modality enum (VIDEO, PARQUET, AUTO for auto-detect)

        Raises:
            exceptions.HydratorException: If streaming mode is requested (not yet supported).
        """
        for raw_path in paths:
            path = Hydrator.normalize_path(raw_path)
            access_mode = self.modality_router.get_access_mode(
                path=path,
                modality=modality,
                backend_type=self.config.backend,
            )

            if access_mode == hydrator_modality.AccessMode.DOWNLOAD:
                profile = self._get_winning_profile(path)
                self.cache_manager.register_file(
                    remote_path=path,
                    modality=modality.value,
                    access_mode=access_mode,
                    profile=profile,
                )
            elif access_mode == hydrator_modality.AccessMode.DIRECT_READ:
                pass
            elif access_mode == hydrator_modality.AccessMode.STREAMING:
                raise exceptions.HydratorException(f"Streaming mode not yet supported for {path}")

        # NOTE: evict_if_needed() will raise RuntimeError if the cache is full, because
        # release_files() is never called in production — all files keep ref_count > 0
        # and are therefore ineligible for LRU eviction. Disk space is only reclaimed
        # at shutdown() via cleanup(remove_all=True). If this becomes a problem, wire up
        # release_files() at the shard/batch level so files can actually be evicted.
        self.cache_manager.evict_if_needed()

    def _poll_covering_sync(self, path: str, timeout_val: float) -> str | None:
        """Poll DuckDB until `path` appears in the covering directory sync or deadline expires.

        download_index is guaranteed non-None here: register_directory_sync() raises if it is None.
        """
        deadline = time.time() + timeout_val
        poll_log_interval = 30.0
        last_poll_log = time.time()
        assert self.cache_manager.download_index is not None
        while time.time() < deadline:
            record = self.cache_manager.download_index.lookup(path)
            if (
                record is not None
                and record.local_path is not None
                and Path(record.local_path).is_file()
            ):
                return record.local_path
            now = time.time()
            if now - last_poll_log >= poll_log_interval:
                worker_logging.get_unified_logger().info(
                    "[HYDRATOR] get_file DuckDB poll waiting path=%r remaining=%.0fs pid=%s",
                    path,
                    deadline - now,
                    os.getpid(),
                )
                last_poll_log = now
            time.sleep(0.5)
        return None

    def get_file(
        self,
        path: str,
        timeout: float | None = None,
    ) -> str | None:
        """Get local path to a file if it exists.

        Two usage patterns:
        1. Async (prefetch): Call register_files() first, then get_file() returns fast
        2. Sync (on-demand): Call get_file() directly, auto-registers if needed

        For local paths (no "s3://"): returns path if it exists, else None.
        For remote: returns local path after download.

        Raises:
            exceptions.HydratorFileNotFoundError: If the remote object does not exist (404/NoSuchKey)
                or the file remains in a non-ready state after wait (e.g. missing object).
            exceptions.HydratorTimeoutError: If download does not complete within timeout.
            exceptions.HydratorException: For other download failures (permission, etc.).

        Args:
            path: Path (local or S3, e.g. from Hydrator.join_path(base, relative))
            timeout: Max seconds to wait for download (None = config.download_timeout)

        Returns:
            Local path (or original path for DIRECT_READ), or None for missing local file.
        """
        path = Hydrator.normalize_path(path)
        if not path.startswith("s3://"):
            return path if Path(path).exists() else None

        # Fast path: already registered as READY (downloaded or s5cmd-synced)
        quick_meta = self.cache_manager.get_file_metadata(path)
        if quick_meta is not None and quick_meta.state == types.FileState.READY:
            return quick_meta.local_path

        # If a directory sync covers this path, poll until the file appears or raise on timeout.
        # Never fall back to per-file download when a covering sync exists — that would mask
        # sync failures and create confusing partial-download states.
        covering_job = self.cache_manager.get_covering_dir_sync(path)
        if covering_job is not None:
            timeout_val = timeout or self.config.download_timeout
            local_path = self._poll_covering_sync(path, timeout_val)
            if local_path is not None:
                return local_path
            raise exceptions.HydratorTimeoutError(
                f"File not found in directory sync after {timeout_val}s: {path}"
            )

        access_mode = self.modality_router.get_access_mode(
            path=path,
            modality=types.Modality.AUTO,
            backend_type=self.config.backend,
        )

        if access_mode == hydrator_modality.AccessMode.DIRECT_READ:
            return path

        if access_mode == hydrator_modality.AccessMode.DOWNLOAD:
            metadata = self.cache_manager.get_file_metadata(path)

            if metadata is None:
                self.register_files([path], modality=types.Modality.AUTO)
                metadata = self.cache_manager.get_file_metadata(path)
                if metadata is None:
                    raise exceptions.HydratorException(f"Failed to register file: {path}")

            if metadata.state == types.FileState.READY:
                if metadata.local_path is None:
                    raise exceptions.HydratorException(
                        f"File {path} is READY but has no local_path"
                    )
                return metadata.local_path

            if metadata.state == types.FileState.ERROR:
                if self._is_s3_not_found_error(metadata.error_message):
                    raise exceptions.HydratorFileNotFoundError(path)
                error_msg = metadata.error_message or "Unknown error"
                raise exceptions.HydratorException(f"Download failed for {path}: {error_msg}")

            if metadata.state in (types.FileState.PENDING, types.FileState.DOWNLOADING):
                timeout_val = timeout or self.config.download_timeout
                queue_size = self.cache_manager.get_download_queue_size()
                log.info(
                    "[HYDRATOR] get_file waiting for download path=%r state=%s"
                    " timeout=%ss queue_size=%s pid=%s"
                    " (0=task handed to Ray, >0=still in Python queue)",
                    path,
                    metadata.state.value,
                    timeout_val,
                    queue_size,
                    os.getpid(),
                )
                try:
                    success = self._wait_for_file(path, timeout=timeout_val)
                except Exception as e:
                    raise exceptions.HydratorException(f"Download failed for {path}: {e}") from e

                metadata = self.cache_manager.get_file_metadata(path)
                if metadata is None:
                    raise exceptions.HydratorException(
                        f"File metadata disappeared after wait: {path}"
                    )
                if metadata.state == types.FileState.ERROR:
                    if self._is_s3_not_found_error(metadata.error_message):
                        raise exceptions.HydratorFileNotFoundError(path)
                    error_msg = metadata.error_message or "Unknown error"
                    raise exceptions.HydratorException(f"Download failed for {path}: {error_msg}")
                if not success:
                    raise exceptions.HydratorTimeoutError(
                        f"Download for {path} did not complete within {timeout_val}s"
                    )
                if metadata.state != types.FileState.READY or metadata.local_path is None:
                    raise exceptions.HydratorException(
                        f"File {path} in unexpected state after wait: {metadata.state}"
                    )
                return metadata.local_path

            raise exceptions.HydratorException(f"File {path} in unknown state: {metadata.state}")

        if access_mode == hydrator_modality.AccessMode.STREAMING:
            raise exceptions.HydratorException(f"Streaming mode not yet supported for {path}")

        raise exceptions.HydratorException(f"Unknown access mode: {access_mode}")

    def _wait_for_file(self, remote_path: str, timeout: float | None = None) -> bool:
        """Wait until a registered file reaches READY or ERROR state. Used by get_file()."""
        timeout = timeout or 300.0
        poll_interval = 0.05
        log_interval = 10.0
        deadline = time.time() + timeout
        last_log = 0.0
        while time.time() < deadline:
            # Cross-rank adoption: a PENDING entry on a non-owner rank
            # transitions to READY here when the owner publishes.
            self.cache_manager.adopt_if_published(remote_path)

            metadata = self.cache_manager.get_file_metadata(remote_path)
            if metadata is None:
                time.sleep(poll_interval)
                continue
            if metadata.state == types.FileState.READY:
                return True
            if metadata.state == types.FileState.ERROR:
                return False
            now = time.time()
            if now - last_log >= log_interval:
                queue_size = self.cache_manager.get_download_queue_size()
                elapsed = now - (deadline - timeout)
                log.info(
                    "[HYDRATOR] get_file still waiting path=%r state=%s"
                    " elapsed=%.1fs queue_size=%s pid=%s"
                    " (0=with Ray, >0=in Python queue; same pid=same process)",
                    remote_path,
                    metadata.state.value,
                    elapsed,
                    queue_size,
                    os.getpid(),
                )
                last_log = now
            time.sleep(poll_interval)
        return False

    def _wait_all(self, timeout: float | None = None) -> None:
        """Wait until all registered files are no longer PENDING or DOWNLOADING. Used by wait_all()."""
        poll_interval = 0.1
        start = time.time()
        while True:
            stats = self.cache_manager.get_stats()
            pending = stats.pending_downloads + stats.downloading_files
            if pending == 0:
                return
            if timeout is not None and (time.time() - start) >= timeout:
                raise TimeoutError(
                    f"Not all downloads completed within {timeout}s (still {pending} in progress)"
                )
            time.sleep(poll_interval)

    @staticmethod
    def _is_s3_not_found_error(error_message: str | None) -> bool:
        """True if error indicates S3 object does not exist (404/NoSuchKey)."""
        if not error_message:
            return False
        msg = error_message.lower()
        return "not found" in msg or "nosuchkey" in msg or "404" in msg

    def register_directory(
        self,
        s3_path: str,
        local_dir: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Async: submit s5cmd sync tasks for this rank's prefix slice (default: all 65).

        Returns immediately. Workers stream synced file locations into the shared
        DuckDB download_index so get_file() pollers can find them within seconds.

        Args:
            s3_path: S3 prefix URI (e.g. s3://bucket/prefix)
            local_dir: Local directory to sync files into
            rank: This process's rank (default 0)
            world_size: Total number of ranks (default 1 = no sharding)
        """
        import logging

        logging.getLogger(__name__).info(
            "[Hydrator] register_directory: async start %s -> %s (rank=%d/%d)",
            s3_path,
            local_dir,
            rank,
            world_size,
        )
        self.cache_manager.register_directory_sync(s3_path, local_dir, rank, world_size)

    def list_directory(self, s3_path: str) -> list[str]:
        """List all object URIs under an S3 prefix (metadata only, no download).

        Use this to build a file manifest before starting an async sync so that
        training can begin immediately while files are being fetched in background.

        Args:
            s3_path: S3 prefix URI (e.g. s3://bucket/prefix/)

        Returns:
            List of full S3 URIs for all objects under the prefix
        """
        return io.glob_paths(s3_path, "*", profile=self.config.credentials_profile)

    def get_directory(self, s3_path: str, local_dir: str, pattern: str = "**/*.pkl") -> list[str]:
        """Sync an S3 directory and return matching local paths.

        Starts the directory sync, then blocks by calling get_file() for each
        file matching ``pattern`` under the S3 prefix.  Callers should prefer
        register_directory() + get_file() per-file for async behaviour.

        Args:
            s3_path: S3 URI prefix (e.g. ``s3://bucket/prefix/``).
            local_dir: Local directory to sync files into.
            pattern: Glob pattern for files to return (default ``**/*.pkl``).

        Returns:
            Sorted list of local file paths matching ``pattern``.
        """
        self.register_directory(s3_path, local_dir)
        remote_files = io.glob_paths(s3_path, pattern, profile=self.config.credentials_profile)
        local_paths = []
        for remote_path in remote_files:
            local_path = self.get_file(remote_path)
            if local_path is not None:
                local_paths.append(local_path)
        return sorted(local_paths)

    def put_file(self, remote_path: str, content: bytes | Path) -> None:
        """Write content to a remote path (upload). Submits to worker pool and waits.

        NOTE: put_file lives on Hydrator intentionally — it reuses the hydrator's
        worker pool and retry/timeout settings, and selects credentials per-URI
        (the profile that resolved the destination bucket wins, falling back to
        the default), avoiding the need for a separate S3 client in callers
        (e.g. stats.py).

        Args:
            remote_path: Remote URI (e.g. s3://bucket/key/meta/stats.json). Must start with s3://.
            content: Bytes for in-memory content, or
                ``Path`` to stream a local file directly via the backend's
                native streaming upload (boto3 ``upload_file``: multipart
                for >8 MB, no full-file buffering). Recommended for any
                file already on disk.

        Raises:
            exceptions.HydratorException: If remote_path is not an S3 URI or on upload failure/timeout.
        """
        if not remote_path.startswith("s3://"):
            raise exceptions.HydratorException(
                f"put_file only supports s3:// paths, got: {remote_path}"
            )
        remote_path = Hydrator.normalize_path(remote_path)
        profile = self._get_winning_profile(remote_path)
        self.cache_manager.submit_upload(remote_path, content, profile=profile)

    def put_file_async(self, remote_path: str, content: bytes | Path) -> None:
        """Submit an upload without waiting for it to complete.

        Counterpart to :meth:`put_file`. Useful when callers want to
        dispatch many uploads, then poll for completion via
        :meth:`get_file_state` — e.g. a bounded-window dispatch loop
        that keeps the worker pool saturated without blocking on each
        individual upload.

        Args:
            remote_path: Remote URI (must start with s3://).
            content: Bytes for in-memory content, or ``Path`` (streamed
                from disk; must remain valid until the upload finishes).

        Raises:
            exceptions.HydratorException: If remote_path is not an S3 URI.
        """
        if not remote_path.startswith("s3://"):
            raise exceptions.HydratorException(
                f"put_file_async only supports s3:// paths, got: {remote_path}"
            )
        remote_path = Hydrator.normalize_path(remote_path)
        profile = self._get_winning_profile(remote_path)
        self.cache_manager.submit_upload_async(remote_path, content, profile=profile)

    def release_files(self, paths: list[str] | None = None) -> None:
        """Release files, indicating they are no longer actively needed.

        Files are decremented in reference count and marked as eligible for LRU eviction.

        Args:
            paths: List of remote paths to release. If None, releases all currently active files.
        """
        if paths is None:
            paths = self._active_files

        if paths:
            normalized = [Hydrator.normalize_path(p) if p.startswith("s3://") else p for p in paths]
            self.cache_manager.release_files(normalized)
            if paths == self._active_files:
                self._active_files = []

    def purge_files(self, paths: list[str]) -> None:
        """Aggressively delete cached files from disk + tracking state.

        Companion to :meth:`release_files`, which only marks files
        eligible for lazy LRU eviction. ``purge_files`` deletes
        immediately, regardless of ``max_cache_size_gb`` headroom.
        Useful for callers with strong "this chunk's data will never
        be reused" knowledge — e.g. a per-chunk pipeline that wants
        to free Lustre as soon as a chunk's compute finishes, not at
        shutdown.

        DIRECT_READ paths are skipped silently (no local copy exists
        to delete). Paths that aren't currently registered are
        skipped silently (idempotent re-purge is a no-op).

        Args:
            paths: List of remote paths to purge.

        Raises:
            exceptions.HydratorException: If a path is currently
                ``DOWNLOADING`` (purging mid-download would race the
                worker pool); caller must ``wait_all`` or ``release``
                first.
        """
        for raw_path in paths:
            path = Hydrator.normalize_path(raw_path) if raw_path.startswith("s3://") else raw_path
            metadata = self.cache_manager.get_file_metadata(path)
            if metadata is None:
                continue  # not registered (e.g. DIRECT_READ) — nothing to do
            if metadata.state == types.FileState.DOWNLOADING:
                raise exceptions.HydratorException(f"Cannot purge in-flight download: {path}")
            local_path = metadata.local_path
            self.cache_manager.drop_entry(path)
            if local_path:
                try:
                    os.remove(local_path)
                except FileNotFoundError:
                    pass  # already gone — fine, tracking is what matters
                except OSError as exc:
                    log.warning("purge_files unlink failed for %s: %s", local_path, exc)

    def get_file_state(self, path: str) -> types.FileInfo | None:
        """Get file state and access mode information.

        Returns information about registered files only — both downloads
        (have ``access_mode``) and async uploads submitted via
        :meth:`put_file_async` (``access_mode`` is None). Returns None
        for unregistered paths.

        Args:
            path: Remote file path

        Returns:
            FileInfo with state and (optional) access_mode if registered, None otherwise
        """
        if path.startswith("s3://"):
            path = Hydrator.normalize_path(path)
        metadata = self.cache_manager.get_file_metadata(path)
        if metadata is None:
            return None
        return types.FileInfo(
            state=metadata.state,
            access_mode=metadata.access_mode,
            error_message=metadata.error_message,
        )

    def get_cache_stats(self) -> types.CacheStats:
        """Get cache statistics."""
        return self.cache_manager.get_stats()

    def wait_all(self, timeout: float | None = None) -> None:
        """Wait for all registered files to finish downloading.

        Args:
            timeout: Max time to wait (None = wait forever)
        """
        self._wait_all(timeout)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the service.

        NOTE: This is currently the only point where disk space is reclaimed.
        When cleanup_on_shutdown=True, the entire job directory (_job_dir) is removed
        with a single rmtree — covering both downloaded file cache and s5cmd-synced dirs.

        Args:
            wait: If True, wait for active downloads to complete
        """
        self._stop_ipc_server()
        with self._lock:
            if self._is_shutdown:
                return
            self.cache_manager.shutdown(wait=wait)
            if self.config.cleanup_on_shutdown:
                import shutil

                shutil.rmtree(self._job_dir, ignore_errors=True)
            self._is_shutdown = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        del exc_type, exc_val, exc_tb
        self.shutdown(wait=True)
        return False
