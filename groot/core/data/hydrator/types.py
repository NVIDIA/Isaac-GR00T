# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Type definitions and enums for the data serving layer."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from groot.core.data.hydrator.modality import AccessMode


class FileState(Enum):
    """State of a file in the cache.

    Used for both download and async-upload state machines (see
    Hydrator.put_file_async + cache_manager.submit_upload_async).

    Reality of the current WorkerPool: there is no "worker started
    transfer" signal — only a single completion result per task. So
    state actually transitions ``PENDING → READY`` (or ``→ ERROR``)
    with no observable in-progress phase. ``DOWNLOADING`` /
    ``UPLOADING`` are reserved for a future per-task progress hook
    and currently aren't written by production code; they remain in
    the enum for API symmetry and to give safety paths
    (e.g. ``service.purge_files`` refuses to purge entries in those
    states should they ever be set).
    """

    PENDING = "pending"  # Registered, not yet completed
    DOWNLOADING = "downloading"  # Reserved (see class docstring)
    UPLOADING = "uploading"  # Reserved (see class docstring)
    READY = "ready"  # Transfer complete; file available (or upload acked)
    ERROR = "error"  # Transfer failed after all retries

    def __str__(self) -> str:
        """Return the enum value as string."""
        return self.value


@dataclass
class FileInfo:
    """Public-facing file information.

    Returned by Hydrator.get_file_state() to provide file state and access mode.

    Attributes:
        state: Current state. In practice ``PENDING`` (just registered)
            or ``READY`` / ``ERROR`` (terminal); see ``FileState``
            docstring for why ``DOWNLOADING`` / ``UPLOADING`` are
            reserved-but-unused under the current WorkerPool.
        access_mode: How the file is accessed (DOWNLOAD, DIRECT_READ, STREAMING).
            None for entries created by put_file_async (uploads have no
            access_mode — the file is being written, not read).
        error_message: For ``state == ERROR``, the underlying failure
            string (the exception ``str()`` from the worker). ``None``
            otherwise. Surfaces the actual cause of failure to callers
            polling via ``get_file_state`` / ``get_file_states`` without
            forcing them to fish in ``cache_manager.files`` directly.
    """

    state: FileState
    access_mode: Optional["AccessMode"] = None
    error_message: str | None = None


class Modality(Enum):
    """Supported data modalities.

    Phase 1 (MVP): Only VIDEO and PARQUET are supported.
    AUTO will detect modality from file extension.
    Future phases can add more modalities as needed.
    """

    AUTO = "auto"  # Auto-detect from file extension
    VIDEO = "video"
    PARQUET = "parquet"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Return the enum value as string."""
        return self.value


@dataclass
class FileMetadata:
    """Metadata for a cached file.

    Attributes:
        remote_path: Original remote path (S3, LustreFS, etc.)
        local_path: Local cached file path
        state: Current state of the file
        size_bytes: File size in bytes
        modality: Data modality type
        access_mode: How the file should be accessed (DOWNLOAD, DIRECT_READ, STREAMING)
        last_accessed_at: Timestamp of last access (for LRU)
        access_count: Number of times file was accessed
        ref_count: Reference count (>0 means active, =0 means eligible for eviction)
        download_start_time: When download started (for timeout tracking)
        error_message: Error message if state is ERROR
    """

    remote_path: str
    local_path: str | None = None
    state: FileState = FileState.PENDING
    size_bytes: int = 0
    modality: Modality = Modality.UNKNOWN
    access_mode: Optional["AccessMode"] = None
    last_accessed_at: float = 0.0
    access_count: int = 0
    ref_count: int = 0
    download_start_time: float | None = None
    error_message: str | None = None

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        size_mb = self.size_bytes / (1024 * 1024) if self.size_bytes > 0 else 0
        status = f"state={self.state.value}, ref_count={self.ref_count}"

        if self.error_message:
            status += f", error='{self.error_message}'"

        return (
            f"FileMetadata(remote={self.remote_path}, "
            f"local={self.local_path}, "
            f"{status}, "
            f"size={size_mb:.2f}MB, "
            f"modality={self.modality.value}, "
            f"accesses={self.access_count})"
        )


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring.

    Attributes:
        total_size_gb: Total size of cached files in GB
        total_files: Total number of cached files
        active_files: Files currently in use (ref_count > 0, won't be evicted)
        cached_files: Files no longer in use (ref_count = 0, eligible for eviction)
        cache_hits: Number of times a file was reused from cache
        downloads: Number of files downloaded
        evictions: Number of files evicted due to size limits
        download_errors: Number of download failures
        pending_downloads: Number of files waiting to download
        downloading_files: Number of files currently downloading
    """

    total_size_gb: float = 0.0
    total_files: int = 0
    active_files: int = 0
    cached_files: int = 0
    cache_hits: int = 0
    downloads: int = 0
    evictions: int = 0
    download_errors: int = 0
    pending_downloads: int = 0
    downloading_files: int = 0

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        hit_rate = (
            f"{(self.cache_hits / (self.cache_hits + self.downloads) * 100):.1f}%"
            if (self.cache_hits + self.downloads) > 0
            else "N/A"
        )

        return (
            f"CacheStats(\n"
            f"  Files: {self.total_files} total ({self.active_files} active, "
            f"{self.cached_files} cached, {self.pending_downloads} pending, "
            f"{self.downloading_files} downloading)\n"
            f"  Size: {self.total_size_gb:.2f} GB\n"
            f"  Performance: {self.cache_hits} hits, {self.downloads} downloads "
            f"(hit rate: {hit_rate})\n"
            f"  Errors: {self.download_errors} failed, {self.evictions} evicted\n"
            f")"
        )
