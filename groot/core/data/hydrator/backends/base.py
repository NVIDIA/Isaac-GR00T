# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Abstract base class for storage backends.

IMPORTANT: This is an INTERNAL component of the AsyncDataServingService.
Users should NEVER import backend classes directly.
"""

import abc
from collections.abc import Callable
import hashlib
from pathlib import Path

OnPair = Callable[[str, str], None]
"""Callback fired by sync_directory for each (remote_uri, local_path) pair written."""


class StorageBackend(abc.ABC):
    """Abstract base class for remote storage backends.

    Supports different storage systems (S3, LustreFS, NFS, etc.) and
    different client libraries (boto3, s5cmd, aioboto3, etc.). Subclasses
    implement the concrete download/upload/list/sync operations; the base
    class provides the ``generate_local_path`` path-derivation contract.
    """

    def __init__(self, working_dir: str):
        """Initialize the backend.

        Args:
            working_dir: Local working directory root where files will be
                downloaded. Concrete backends may subdivide this further
                (e.g. per-prefix subdirectories) but ``generate_local_path``
                treats it as the root.
        """
        self.working_dir = working_dir

    @staticmethod
    def generate_local_path(working_dir: str, remote_path: str) -> str:
        """Generate local file path for a remote file using hash-based naming.

        This uses SHA256 hashing to avoid filesystem path length limits while
        preserving file extensions. This is the CONTRACT between backends and
        cache manager - all backends MUST use this method to generate local paths.

        Design rationale:
        - Hash-based naming avoids filesystem path length limits (e.g., 255 chars)
        - Preserves file extension for proper file type detection
        - Deterministic - same remote_path always maps to same local path
        - Collision-resistant with 16-char (64-bit) hash prefix

        Args:
            working_dir: Root working directory where files will be stored
            remote_path: Full remote URI (e.g., s3://bucket/key, /local/path/file)

        Returns:
            Full local file path as string

        Example:
            >>> StorageBackend.generate_local_path(
            ...     "/tmp/working_dir",
            ...     "s3://bucket/very/long/path/to/video.mp4"
            ... )
            '/tmp/working_dir/a1b2c3d4e5f6g7h8.mp4'
        """
        # Hash the remote path to get a unique but short identifier
        path_hash = hashlib.sha256(remote_path.encode()).hexdigest()[:16]

        # Preserve file extension
        remote_suffix = Path(remote_path).suffix

        # Generate local path: {working_dir}/{hash}{extension}
        local_path = Path(working_dir) / f"{path_hash}{remote_suffix}"

        return str(local_path)

    @abc.abstractmethod
    def download_file(
        self,
        remote_path: str,
    ) -> str:
        """Download a file from remote storage to cache directory.

        The backend controls the destination file structure within cache_dir.

        Args:
            remote_path: Remote URI (e.g., s3://bucket/key, /local/path/file)

        Returns:
            local_path: Full path to downloaded file

        """

    @abc.abstractmethod
    def get_file_size(
        self,
        remote_path: str,
        credentials: dict | None = None,
    ) -> int:
        """Get file size in bytes without downloading.

        Args:
            remote_path: Remote URI
            credentials: Optional credentials dict

        Returns:
            File size in bytes

        """

    @abc.abstractmethod
    def supports_direct_read(self, modality: str) -> bool:
        """Check if modality can be read directly from this backend.

        Args:
            modality: File modality type ("video", "parquet", etc.)

        Returns:
            True if files can be read directly without downloading
        """

    def upload_file(self, remote_path: str, content: bytes | Path) -> None:
        """Upload content to remote path. Optional; not all backends support write.

        Args:
            remote_path: Remote URI (e.g. s3://bucket/key)
            content: ``bytes`` for in-memory content; ``Path`` to stream
                from local disk (caller must keep the file alive until
                upload returns).

        Raises:
            NotImplementedError: If this backend does not support upload
        """
        raise NotImplementedError("This backend does not support upload")

    def list_directory(self, remote_path: str) -> list[str]:
        """List all object URIs under a remote prefix (metadata only, no download).

        Args:
            remote_path: Remote prefix URI (e.g. s3://bucket/prefix/)

        Returns:
            List of full remote URIs for all objects under the prefix

        Raises:
            NotImplementedError: If this backend does not support directory listing
        """
        raise NotImplementedError("This backend does not support directory listing")

    def sync_directory(
        self,
        remote_path: str,
        local_dir: str,
        on_pair: OnPair | None = None,
    ) -> list[tuple[str, str]]:
        """Sync all objects under a remote prefix to a local directory.

        Args:
            remote_path: Remote prefix URI (e.g. s3://bucket/prefix/)
            local_dir: Local directory to sync files into
            on_pair: Optional callback fired for each (remote_uri, local_path)
                pair as it is written, enabling streaming registration of files
                before the full sync completes.

        Returns:
            List of (remote_uri, local_path) pairs for every file successfully
            copied. Callers may treat this as authoritative.

        Raises:
            NotImplementedError: If this backend does not support directory sync
        """
        raise NotImplementedError("This backend does not support sync_directory")

    def sync_files(
        self,
        pairs: list[tuple[str, str]],
        on_pair: OnPair | None = None,
    ) -> list[tuple[str, str]]:
        """Copy an explicit list of (remote_uri, local_path) pairs.

        The counterpart to :meth:`sync_directory` for callers that need to choose the exact set of
        objects and where each one lands, rather than whatever a single prefix glob covers.

        Args:
            pairs: (remote_uri, local_path) to copy.
            on_pair: Optional callback fired for each pair as it is written.

        Returns:
            List of (remote_uri, local_path) pairs for every file successfully copied.

        Raises:
            NotImplementedError: If this backend does not support explicit-pair copying
        """
        raise NotImplementedError("This backend does not support sync_files")
