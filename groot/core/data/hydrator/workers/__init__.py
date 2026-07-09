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
