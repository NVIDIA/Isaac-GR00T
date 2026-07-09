#!/usr/bin/env python3
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
"""
Ray Initialization Utilities - Smart Ray cluster initialization.

This module provides utilities to automatically detect existing Ray clusters
and initialize Ray connections appropriately.
"""

from collections.abc import Callable
import functools
import logging
import os
from typing import TypeVar

import ray

# Set up logging
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def with_coverage(func: F) -> F:
    """Decorator to enable coverage collection in Ray remote functions.

    When COVERAGE_PROCESS_START env var is set, this decorator wraps the
    function to start coverage before execution and save it after completion.

    Usage:
        @ray.remote
        @with_coverage
        def my_remote_function(...):
            ...

    Note: The decorator must be applied BEFORE @ray.remote (i.e., closer to
    the function definition) so that coverage wrapping happens inside the
    Ray worker.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coverage_config = os.environ.get("COVERAGE_PROCESS_START")
        if not coverage_config:
            return func(*args, **kwargs)

        import coverage

        cov = coverage.Coverage(config_file=coverage_config)
        cov.start()
        try:
            return func(*args, **kwargs)
        finally:
            cov.stop()
            cov.save()

    return wrapper  # type: ignore[return-value]


def start_local_ray_cluster(
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 8265,
    **kwargs,
) -> str:
    """
    Start a local Ray cluster.

    Args:
        num_cpus: Number of CPUs to allocate (None for auto-detect)
        num_gpus: Number of GPUs to allocate (None for auto-detect)
        dashboard_host: Dashboard host address
        dashboard_port: Dashboard port
        **kwargs: Additional arguments to pass to ray.init()

    Returns:
        str: Address of the started cluster
    """
    logger.info("Starting local Ray cluster...")

    # For local clusters, explicitly disable working_dir upload.
    # Workers run on the same machine and can access the same filesystem,
    # so they don't need uploaded files. Setting working_dir=None disables
    # Ray's auto-detection of git repos and pyproject.toml files.
    #
    # This approach:
    # 1. Avoids package size limits (no upload)
    # 2. Ensures workers have access to all installed dependencies
    # 3. Avoids issues with editable installs (gear_cli, etc.)
    runtime_env: dict = {"working_dir": None}  # Disable auto-detection

    # Propagate coverage config to Ray workers so @with_coverage works
    coverage_config = os.environ.get("COVERAGE_PROCESS_START")
    if coverage_config:
        runtime_env["env_vars"] = {"COVERAGE_PROCESS_START": coverage_config}

    init_kwargs = {
        "dashboard_host": dashboard_host,
        "dashboard_port": dashboard_port,
        "runtime_env": runtime_env,
        **kwargs,
    }

    # Add resource specifications if provided
    if num_cpus is not None:
        init_kwargs["num_cpus"] = num_cpus
    if num_gpus is not None:
        init_kwargs["num_gpus"] = num_gpus

    try:
        ray.init(**init_kwargs)
        cluster_address = ray.get_runtime_context().gcs_address
        logger.info(f"Started local Ray cluster at {cluster_address}")
        logger.info(f"Ray dashboard available at " f"http://{dashboard_host}:{dashboard_port}")
        return cluster_address
    except Exception as e:
        logger.error(f"Failed to start local Ray cluster: {e}")
        raise


def ray_init_maybe_new_cluster(
    local_cluster_config: dict | None = None,
) -> tuple[bool, str]:
    """
    Smart Ray initialization that always prefers existing clusters.

    Args:
        local_cluster_config: Configuration for local cluster if started

    Returns:
        Tuple[bool, str]: (is_new_cluster, cluster_address)
            - is_new_cluster: True if we started a new cluster,
              False if connected to existing
            - cluster_address: Address of the cluster we connected to
    """
    if local_cluster_config is None:
        local_cluster_config = {}

    # Check if Ray is already initialized in this process
    if ray.is_initialized():
        cluster_address = ray.get_runtime_context().gcs_address
        logger.info(f"Ray already initialized, connected to {cluster_address}")
        return False, cluster_address

    # Try ray.init(address='auto') to find any running cluster
    try:
        logger.info("Trying to connect to existing Ray cluster...")
        ray.init(address="auto")
        cluster_address = ray.get_runtime_context().gcs_address
        logger.info(f"Connected to existing Ray cluster: {cluster_address}")
        return False, cluster_address
    except Exception as e:
        logger.debug(f"No existing cluster found: {e}")

    # Start new local cluster
    logger.info("No existing cluster found, starting new local cluster...")
    cluster_address = start_local_ray_cluster(**local_cluster_config)
    return True, cluster_address


def find_large_files(directory: str, size_threshold_mb: float = 1.0) -> list[str]:
    """
    Find all files in a directory that exceed a given size threshold.

    Args:
        directory: Path to the directory to search
        size_threshold_mb: Size threshold in megabytes (default: 1.0 MB)

    Returns:
        List of relative file paths (relative to directory) that exceed the size threshold
    """
    import os
    from pathlib import Path

    large_files = []
    size_threshold_bytes = size_threshold_mb * 1024 * 1024

    directory_path = Path(directory).resolve()
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return large_files

    for root, dirs, files in os.walk(directory_path):
        # Filter out hidden directories (those starting with '.')
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            file_path = Path(root) / file
            try:
                if file_path.stat().st_size > size_threshold_bytes:
                    # Convert to relative path from the base directory
                    relative_path = file_path.relative_to(directory_path)
                    large_files.append(str(relative_path))
            except (OSError, PermissionError) as e:
                logger.debug(f"Could not access file {file_path}: {e}")

    return large_files
