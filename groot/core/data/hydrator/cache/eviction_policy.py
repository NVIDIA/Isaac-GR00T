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

"""Cache eviction policies for managing disk space.

This module provides pluggable eviction strategies that determine which files
to remove when the cache exceeds its size limit.

Supported Policies:
- LRU (Least Recently Used): Evict files that haven't been accessed recently (DEFAULT)
- FIFO (First In First Out): Evict oldest files first (by creation time)
- Size-based: Evict largest files first to free space quickly

All policies respect active files (ref_count > 0) and never evict them.
"""

from abc import ABC, abstractmethod

from groot.core.data.hydrator.types import FileMetadata


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def select_files_to_evict(
        self,
        files: dict[str, FileMetadata],
    ) -> list[tuple[str, FileMetadata]]:
        """Select and sort files for eviction. First item = evict first. Only inactive (ref_count=0)."""
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used (LRU) eviction policy."""

    def select_files_to_evict(
        self,
        files: dict[str, FileMetadata],
    ) -> list[tuple[str, FileMetadata]]:
        inactive_files = [
            (remote_path, metadata)
            for remote_path, metadata in files.items()
            if metadata.ref_count == 0
        ]
        inactive_files.sort(key=lambda x: x[1].last_accessed_at)
        return inactive_files


class FIFOEvictionPolicy(EvictionPolicy):
    """First In First Out (FIFO) eviction policy."""

    def select_files_to_evict(
        self,
        files: dict[str, FileMetadata],
    ) -> list[tuple[str, FileMetadata]]:
        inactive_files = [
            (remote_path, metadata)
            for remote_path, metadata in files.items()
            if metadata.ref_count == 0
        ]
        inactive_files.sort(
            key=lambda x: x[1].download_start_time if x[1].download_start_time else float("inf")
        )
        return inactive_files


class SizeBasedEvictionPolicy(EvictionPolicy):
    """Size-based eviction policy. Evict largest files first."""

    def select_files_to_evict(
        self,
        files: dict[str, FileMetadata],
    ) -> list[tuple[str, FileMetadata]]:
        inactive_files = [
            (remote_path, metadata)
            for remote_path, metadata in files.items()
            if metadata.ref_count == 0
        ]
        inactive_files.sort(key=lambda x: x[1].size_bytes, reverse=True)
        return inactive_files


DEFAULT_EVICTION_POLICY = LRUEvictionPolicy
