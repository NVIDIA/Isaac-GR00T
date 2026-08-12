# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Worker layer for storage I/O (read and write).

IMPORTANT: This is an INTERNAL component. Use Hydrator and CacheManager only.

Usage: WorkerPool for CacheManager (start, enqueue_download, enqueue_upload, callbacks).
"""

from groot.core.data.hydrator.workers.pool import WorkerPool
from groot.core.data.hydrator.workers.worker import StorageIOWorker

__all__ = [
    "StorageIOWorker",
    "WorkerPool",
]
