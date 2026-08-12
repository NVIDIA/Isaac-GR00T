# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
