#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: D205, D415, T201, TRY004

"""VLA Data Utilities

This module provides comprehensive utilities for VLA data operations including:
- S3 client creation and credential management (replacement for gear.data)
- Dataset path resolution using environment variables
- Unified interface for both cache and experiment systems

Key Features:
- Environment-aware dataset path resolution (env var: LFS_CACHE_PATH)
- S3 client creation without gear.data dependency
- Support for both relative and absolute dataset paths
- Configuration object manipulation for dataset paths
"""

import json
import os
from pathlib import Path
from typing import Any

import boto3

# Default credentials file path (equivalent to gear.data.CREDENTIALS_FILE)
CREDENTIALS_FILE = Path.home() / ".gear" / "data_credentials"

# LeRobot layout: if this file exists at path/meta/info.json, path is a single dataset root
LEROBOT_META_INFO = "meta/info.json"


# =============================================================================
# S3 Utilities (replacement for gear.data functionality)
# =============================================================================


def create_s3_client(credentials_file: str, profile: str = "default"):
    """Create S3 client from credentials file or environment variables.
    Prioritizes environment variables, then falls back to credentials file.

    Args:
        credentials_file: Path to credentials JSON file
                         (e.g., ~/.gear/data_credentials)
        profile: Profile name to use from credentials

    Returns:
        boto3 S3 client

    Raises:
        ValueError: If profile not found in credentials file or env vars incomplete
        FileNotFoundError: If credentials file doesn't exist and env vars not available
        json.JSONDecodeError: If credentials file is not valid JSON
    """
    # First, try to use environment variables (same as gear.common pattern)
    if (
        profile == "default"
        and os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
        and os.environ.get("AWS_DEFAULT_REGION")
    ):
        print("ENV VARS FOUND, USING THEM INSTEAD OF CREDENTIALS FROM FILE")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
        )
        return s3_client

    # Fall back to credentials file
    credentials_file = os.path.expanduser(credentials_file)

    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file not found at {credentials_file}")

    # Load credentials from file
    with open(credentials_file) as f:
        creds = json.load(f)

    if profile not in creds:
        available_profiles = list(creds.keys())
        raise ValueError(
            f"Profile '{profile}' not found in credentials file. "
            f"Available profiles: {available_profiles}"
        )

    profile_creds = creds[profile]

    # Validate required fields
    required_fields = ["aws_access_key_id", "aws_secret_access_key"]
    for field in required_fields:
        if field not in profile_creds:
            raise ValueError(f"Missing required field '{field}' in profile '{profile}'")

    # Create S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=profile_creds["aws_access_key_id"],
        aws_secret_access_key=profile_creds["aws_secret_access_key"],
        endpoint_url=profile_creds.get("endpoint_url"),
        region_name=profile_creds.get("region", "us-west-2"),
    )

    return s3_client


def load_s3_credentials(credentials_file: str, profile: str = "default") -> dict[str, Any]:
    """Load S3 credentials from file or environment variables.
    Prioritizes environment variables, then falls back to credentials file.

    Args:
        credentials_file: Path to credentials JSON file
        profile: Profile name to use from credentials

    Returns:
        Dictionary containing the profile credentials

    Raises:
        ValueError: If profile not found in credentials file or env vars incomplete
        FileNotFoundError: If credentials file doesn't exist and env vars not available
        json.JSONDecodeError: If credentials file is not valid JSON
    """
    # First, try to use environment variables (same as gear.common pattern)
    if (
        profile == "default"
        and os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
        and os.environ.get("AWS_DEFAULT_REGION")
    ):
        return {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "region": os.environ.get("AWS_DEFAULT_REGION"),
            "endpoint_url": os.environ.get("AWS_ENDPOINT_URL", ""),
        }

    # Fall back to credentials file
    credentials_file = os.path.expanduser(credentials_file)

    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file not found at {credentials_file}")

    with open(credentials_file) as f:
        creds = json.load(f)

    if profile not in creds:
        available_profiles = list(creds.keys())
        raise ValueError(
            f"Profile '{profile}' not found in credentials file. "
            f"Available profiles: {available_profiles}"
        )

    return creds[profile]


def get_s3_credentials_from_file(credentials_file: str, profile: str = "default") -> dict[str, str]:
    """Get S3 credentials from file (equivalent to gear.data.get_s3_credentials_from_file).

    Args:
        credentials_file: Path to credentials JSON file
        profile: Profile name to use from credentials

    Returns:
        Dictionary containing AWS credentials with standardized keys

    Raises:
        ValueError: If profile not found in credentials file
        FileNotFoundError: If credentials file doesn't exist
        json.JSONDecodeError: If credentials file is not valid JSON
    """
    profile_creds = load_s3_credentials(credentials_file, profile)

    # Return in the format expected by gear.data.get_s3_credentials_from_file
    return {
        "AWS_ACCESS_KEY_ID": profile_creds["aws_access_key_id"],
        "AWS_SECRET_ACCESS_KEY": profile_creds["aws_secret_access_key"],
        "AWS_DEFAULT_REGION": profile_creds.get("region", "us-west-2"),
        "AWS_ENDPOINT_URL": profile_creds.get("endpoint_url", ""),
    }


# =============================================================================
# Dataset Path Resolution Utilities
# =============================================================================


def get_datasets_base_path() -> Path:
    """Get the base path for datasets using environment variables.

    Returns:
        Path: Base directory where datasets are stored

    Environment Variables:
        LFS_CACHE_PATH: Path to the datasets directory (takes precedence)

    If the environment variable is unset, the repository's shared dataset root
    is used.
    """
    base_path = os.environ.get("LFS_CACHE_PATH") or "/mnt/aws-lfs-01/shared/datasets"
    return Path(base_path)


def resolve_dataset_path(dataset_path: str | Path) -> Path | str:
    """Resolve a dataset path to an absolute path.

    If the dataset_path is relative, it will be resolved relative to the
    datasets base path from get_datasets_base_path().
    If it's already absolute or an S3 URI (s3://...), it will be returned as-is.

    Args:
        dataset_path: Dataset path (relative, absolute, or S3 URI)

    Returns:
        Path or str: Absolute path, or S3 URI string (Path normalizes s3:// to s3:/ on some OSes).

    Examples:
        >>> resolve_dataset_path("/absolute/path/to/dataset")
        PosixPath('/absolute/path/to/dataset')

        >>> resolve_dataset_path("s3://bucket/key")
        's3://bucket/key'
    """
    path_str = str(dataset_path).strip()
    # URI schemes are not considered absolute by Path; do not prepend LFS_CACHE_PATH.
    if path_str.startswith(("s3://", "dc://")):
        return path_str
    dataset_path = Path(dataset_path)

    if dataset_path.is_absolute():
        return dataset_path
    else:
        base_path = get_datasets_base_path()
        return base_path / dataset_path


def expand_lerobot_dataset_paths(path: Path) -> list[Path]:
    """Normalize a dataset path so it always yields a list of LeRobot dataset roots.

    Handles two scenarios:

    1. **Single LeRobot folder**: The path is already a LeRobot dataset root
       (contains meta/info.json). Returns [path] unchanged.

    2. **Container of LeRobot folders**: The path is a directory that does *not*
       contain meta/info.json but has subdirectories that *do* (e.g. job7/part1,
       job7/part2, ...). Returns only those subdirs that have meta/info.json,
       so each is treated as a separate dataset. If no subdir is a valid
       LeRobot root, returns [path] (downstream will then fail with a clear
       error).

    Non-directory or non-existent paths are returned as [path] unchanged.

    Args:
        path: Resolved absolute path (file or directory).

    Returns:
        List of paths: one or more LeRobot dataset roots.
    """
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return [path]
    info_path = path / LEROBOT_META_INFO
    if info_path.exists():
        # Scenario 1: path is itself a LeRobot dataset root
        return [path]
    # Scenario 2: path may be a container; only include subdirs that are LeRobot roots
    subdirs = sorted(p for p in path.iterdir() if p.is_dir())
    lerobot_subdirs = [p for p in subdirs if (p / LEROBOT_META_INFO).exists()]
    if lerobot_subdirs:
        return lerobot_subdirs
    if subdirs:
        # Has subdirs but none contain meta/info.json; return as-is, let downstream fail
        return [path]
    return [path]


def resolve_all_dataset_paths(config) -> None:
    """Resolve all dataset paths in a configuration object to absolute paths.

    This function modifies the config object in-place, updating both:
    - config.data.dataset_path (legacy single dataset)
    - config.data.datasets[].dataset_path (new multi-dataset format)

    For string paths (robotics datasets), also expands container directories:
    if a path is a directory without meta/info.json but with subdirs that have
    it (e.g. job7/part1, job7/part2, ...), it is replaced by the list of those
    subdirs so the omni reader and stats generation see one path per LeRobot root.

    Args:
        config: Configuration object with dataset paths to resolve
    """
    # Handle multiple datasets format (new style)
    if hasattr(config.data, "datasets") and config.data.datasets:
        for dataset in config.data.datasets:
            resolved_paths = []
            for path in dataset.dataset_paths:
                if isinstance(path, str):
                    resolved = resolve_dataset_path(path)
                    resolved_str = str(resolved)
                    if resolved_str.startswith(("s3://", "dc://")) or resolved_str.endswith(
                        ".packds"
                    ):
                        # Remote URIs and PackDS roots aren't LeRobot trees;
                        # don't run them through expand_lerobot_dataset_paths, which
                        # Path-ifies URI schemes into local-looking paths.
                        resolved_paths.append(resolved_str)
                    else:
                        for expanded in expand_lerobot_dataset_paths(resolved):
                            resolved_paths.append(str(expanded))
                elif isinstance(path, dict):
                    # Handle dictionary format for vision_language datasets
                    resolved_dict = {}
                    for key, value in path.items():
                        if isinstance(value, str):
                            # Resolve relative paths to absolute paths using LFS_CACHE_PATH
                            resolved_dict[key] = str(resolve_dataset_path(value))
                        else:
                            resolved_dict[key] = value
                    resolved_paths.append(resolved_dict)
                else:
                    raise ValueError(f"Invalid dataset path: {path}")
            dataset.dataset_paths = resolved_paths


# =============================================================================
# Utility Functions for Cache and Experiment Integration
# =============================================================================


def extract_dataset_s3_paths(config) -> list[str]:
    """Extract S3 paths from configuration object.

    This utility function extracts dataset paths from either format:
    - Multiple datasets: config.data.datasets[].dataset_path
    - Single dataset: config.data.dataset_path

    Args:
        config: Configuration object

    Returns:
        List[str]: List of dataset S3 paths (relative paths)

    Note:
        This returns the original relative paths as they appear in the config,
        suitable for S3 operations. For local filesystem access, use
        resolve_all_dataset_paths() first.
    """
    s3_paths = []

    # Handle multiple datasets format (new style)
    if hasattr(config.data, "datasets") and config.data.datasets:
        for dataset in config.data.datasets:
            for dataset_path in dataset.dataset_paths:
                # Handle dictionary format for vision_language datasets
                if isinstance(dataset_path, dict):
                    # For vision_language datasets, extract the top-level directories to download
                    if "jsonl_path" in dataset_path:
                        # Extract top-level directory (e.g., "lmdb_jsonl" from nested path)
                        jsonl_path = dataset_path["jsonl_path"]
                        from pathlib import Path

                        first_component = Path(jsonl_path).parts[0]
                        if first_component and first_component not in s3_paths:
                            s3_paths.append(first_component)
                    if "data_root" in dataset_path:
                        s3_paths.append(dataset_path["data_root"])
                else:
                    # Handle string format (legacy robotics datasets)
                    s3_paths.append(dataset_path)
    # Handle single dataset format (legacy style)
    elif hasattr(config.data, "dataset_path") and config.data.dataset_path:
        s3_paths.append(config.data.dataset_path)

    return s3_paths
