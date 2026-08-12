# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
