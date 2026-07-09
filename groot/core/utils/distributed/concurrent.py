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
"""Shared utilities for process-pool parallelism (e.g. omni data stats, factory)."""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")


def run_with_process_pool(
    worker: Callable[..., T],
    max_workers: int,
    *arg_columns: list,
) -> list[T]:
    """Run worker in a process pool via executor.map.

    Each column in arg_columns must have the same length (number of tasks).
    worker is called as worker(arg_columns[0][i], arg_columns[1][i], ...) for each i.

    Args:
        worker: Callable that will be run in worker processes (must be picklable).
        max_workers: Maximum number of worker processes.
        *arg_columns: One list per worker argument; all must have len == num_tasks.

    Returns:
        List of return values from worker, in task order.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, *arg_columns))
