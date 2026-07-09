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

"""Data Hydrator.

A module for asynchronously hydrating multi-modal data from remote storage in distributed
training workflows. Handles modality-specific optimizations, storage backend abstraction,
and intelligent caching.

PUBLIC API:
-----------
This module exports ONLY the user-facing interface:
  - Hydrator: Main hydrator class (the only interface users need).
    Use ``Hydrator.resolve_path(path)`` to resolve local paths to best available URI.
  - HydratorConfig: Configuration dataclass
  - AccessMode: Enum for understanding access patterns (informational)
  - ModalityRouter: Router class (informational, mostly for advanced users)

INTERNAL COMPONENTS (Do NOT import these directly):
----------------------------------------------------
  - CacheManager: Internal cache implementation
  - WorkerPool: Internal read/write worker pool (download + upload)
  - StorageBackend: Internal storage abstraction
  - DataSourceResolver: Internal path resolution (AWS S3 → PDX → LustreFS)

Users should ONLY use Hydrator. All caching, downloading, storage,
and path resolution operations are managed internally.
"""

from groot.core.data.hydrator.backends import (
    S3Boto3Backend,
    StorageBackend,
    create_backend,
)
from groot.core.data.hydrator.config import HydratorConfig
from groot.core.data.hydrator.modality import AccessMode, ModalityRouter
from groot.core.data.hydrator.service import Hydrator

__all__ = [
    "AccessMode",
    "Hydrator",
    "HydratorConfig",
    "ModalityRouter",
    "S3Boto3Backend",
    "StorageBackend",
    "create_backend",
]
