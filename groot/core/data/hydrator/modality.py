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

"""Modality-based routing for different file access patterns.

Determines the optimal way to access different file types:
- DOWNLOAD: Full download to local storage (e.g., videos - decoders need seekable files)
- DIRECT_READ: Read directly from remote without download (e.g., parquet via pandas + s3fs)
- STREAMING: Stream data without full download (future: stream videos, large files in chunks)
"""

from enum import Enum
from pathlib import Path

from groot.core.data.hydrator.types import Modality


class AccessMode(Enum):
    """How a file should be accessed from remote storage."""

    DOWNLOAD = "download"  # Download entire file to local storage first
    DIRECT_READ = "direct_read"  # Read directly from remote (no local copy)
    STREAMING = "streaming"  # Stream data in chunks (no full download)


class ModalityRouter:
    """Routes file access based on file type and storage backend.

    Different file types have different optimal I/O patterns:
    - Videos: Download to local (decoders need seekable file handles)
    - Parquet: Direct read from S3 (pandas + s3fs handles it efficiently)
    - Large files: Streaming (future - read in chunks without full download)
    """

    ACCESS_MODE_MAP = {
        ".mp4": AccessMode.DOWNLOAD,
        ".avi": AccessMode.DOWNLOAD,
        ".mov": AccessMode.DOWNLOAD,
        ".mkv": AccessMode.DOWNLOAD,
        ".webm": AccessMode.DOWNLOAD,
        ".parquet": AccessMode.DIRECT_READ,
        ".json": AccessMode.DIRECT_READ,
    }

    def get_access_mode(
        self,
        path: str,
        modality: Modality = Modality.AUTO,
        backend_type: str = "s3",
    ) -> AccessMode:
        """Determine optimal access mode for a file."""
        if modality != Modality.AUTO:
            if modality == Modality.VIDEO:
                return AccessMode.DOWNLOAD
            elif modality == Modality.PARQUET:
                return AccessMode.DIRECT_READ

        ext = Path(path).suffix.lower()
        if ext in self.ACCESS_MODE_MAP:
            access_mode = self.ACCESS_MODE_MAP[ext]
            if backend_type in ("lustre", "nfs", "local"):
                return AccessMode.DIRECT_READ
            return access_mode
        return AccessMode.DOWNLOAD

    def needs_download(
        self,
        path: str,
        modality: Modality = Modality.AUTO,
        backend_type: str = "s3",
    ) -> bool:
        """Check if file needs to be downloaded to local storage."""
        return self.get_access_mode(path, modality, backend_type) == AccessMode.DOWNLOAD
