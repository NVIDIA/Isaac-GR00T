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

"""Shared utilities for E2E test scripts (backend tests).

Provides S3 file sampling, file verification, and CLI helpers for E2E tests.
"""

import os
import random
import sys
from typing import Any

from groot.core.data.hydrator.utils import aws, io


def sample_s3_files(s3_dir: str, sample_size: int, verbose: bool = True) -> list[str]:
    """Sample files from an S3 directory.

    Args:
        s3_dir: S3 directory URI (e.g., s3://bucket/path/to/dir/)
        sample_size: Number of files to sample
        verbose: Whether to print progress messages

    Returns:
        List of S3 file URIs
    """
    if verbose:
        print(f"\nSampling {sample_size} files from {s3_dir}...")  # noqa: T201

    try:
        import boto3
    except ImportError:
        print("❌ boto3 is required for S3 directory sampling")  # noqa: T201
        print("   Install with: pip install boto3")  # noqa: T201
        sys.exit(1)

    bucket, prefix = io.parse_s3_uri(s3_dir)

    if prefix and not prefix.endswith("/"):
        prefix += "/"

    credentials = aws.get_boto3_client_config(profile="aws")
    s3_client = boto3.client("s3", **credentials)

    if verbose:
        print(f"  Listing objects in s3://{bucket}/{prefix}...")  # noqa: T201
    all_files = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if not key.endswith("/"):
                all_files.append(f"s3://{bucket}/{key}")

    if verbose:
        print(f"  Found {len(all_files)} file(s)")  # noqa: T201

    if len(all_files) == 0:
        if verbose:
            print(f"  ❌ No files found in {s3_dir}")  # noqa: T201
        return []

    if sample_size >= len(all_files):
        if verbose:
            print(f"  Using all {len(all_files)} file(s)")  # noqa: T201
        sampled = all_files
    else:
        sampled = random.sample(all_files, sample_size)
        if verbose:
            print(f"  Randomly sampled {sample_size} file(s)")  # noqa: T201

    if verbose:
        print("\n  Sampled files:")  # noqa: T201
        for i, file_uri in enumerate(sampled, 1):
            filename = os.path.basename(file_uri)
            print(f"    {i}. {filename}")  # noqa: T201

    return sampled


def add_s3_cli_args(parser, default_sample_size: int = 10) -> None:
    """Add common S3 file selection arguments to argument parser."""
    parser.add_argument(
        "--s3-file",
        action="append",
        dest="s3_files",
        help="S3 file URI to test (can be specified multiple times)",
    )
    parser.add_argument(
        "--s3-dir",
        type=str,
        help="S3 directory URI to sample files from (e.g., s3://bucket/path/)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=default_sample_size,
        help=f"Number of files to randomly sample from --s3-dir (default: {default_sample_size})",
    )


def print_usage_tip(
    script_name: str, supports_s3_dir: bool = True, supports_max_workers: bool = False
) -> None:
    """Print usage tip when no S3 files are provided."""
    print("\n" + "=" * 80)  # noqa: T201
    print("💡 TIP: Run with --s3-file to test with real S3 downloads")  # noqa: T201
    if supports_s3_dir:
        print("        Or use --s3-dir to sample files from a directory")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print("\nExamples:")  # noqa: T201
    print("  # Test specific files")  # noqa: T201
    print(f"  python -m groot.core.data.hydrator.e2e_tests.{script_name} \\")  # noqa: T201
    print("      --s3-file s3://your-bucket/video.mp4 \\")  # noqa: T201
    print("      --s3-file s3://your-bucket/data.parquet")  # noqa: T201

    if supports_s3_dir:
        print()  # noqa: T201
        print("  # Sample files from a directory")  # noqa: T201
        print(f"  python -m groot.core.data.hydrator.e2e_tests.{script_name} \\")  # noqa: T201
        print("      --s3-dir s3://your-bucket/videos/ \\")  # noqa: T201
        if supports_max_workers:
            print("      --sample-size 20 \\")  # noqa: T201
            print("      --max-workers 8")  # noqa: T201
        else:
            print("      --sample-size 20")  # noqa: T201


def _check_file_size(
    remote_path: str,
    local_path: str,
    expected_sizes: dict[str, int],
    verbose: bool,
) -> tuple[str, str] | None:
    """Check if file size matches expected size.

    Args:
        remote_path: S3 URI key for looking up expected size.
        local_path: Local file path to check.
        expected_sizes: Dict mapping remote_path to expected byte count.
        verbose: Whether to print size mismatch details.

    Returns:
        None if size matches, otherwise (remote_path, error_message) tuple
    """
    if not expected_sizes or remote_path not in expected_sizes:
        return None

    actual_size = os.path.getsize(local_path)
    expected_size = expected_sizes[remote_path]

    if actual_size != expected_size:
        if verbose:
            print(f"  ⚠️  {os.path.basename(local_path)}: Size mismatch")  # noqa: T201
            print(f"      Expected: {expected_size:,}, Got: {actual_size:,}")  # noqa: T201
        return (remote_path, f"Size mismatch: expected {expected_size}, got {actual_size}")

    return None


def _print_file_success(
    remote_path: str,
    local_path: str,
    expected_sizes: dict[str, int] | None,
    verbose: bool,
) -> None:
    """Print success message for verified file."""
    if not verbose:
        return

    file_size = os.path.getsize(local_path)

    fname = os.path.basename(local_path)
    if expected_sizes and remote_path and remote_path in expected_sizes:
        print(f"  ✓ {fname}: Size matches ({file_size:,} bytes)")  # noqa: T201
    else:
        display_path = remote_path if remote_path else os.path.basename(local_path)
        print(f"  ✓ {display_path}")  # noqa: T201
        print(f"    Local: {local_path}")  # noqa: T201
        print(f"    Size: {file_size / (1024*1024):.2f} MB")  # noqa: T201


def _verify_single_file(
    remote_path: str,
    local_path: str,
    expected_sizes: dict[str, int] | None,
    verbose: bool,
) -> tuple[str, str] | None:
    """Verify a single file exists and optionally check size.

    Args:
        remote_path: S3 URI for the file (used as key in expected_sizes).
        local_path: Local file path to verify.
        expected_sizes: Optional dict mapping remote_path to expected byte count.
        verbose: Whether to print verification details.

    Returns:
        None if verification passes, otherwise (path, error_message) tuple
    """
    if not os.path.exists(local_path):
        if verbose:
            print(f"  ❌ File not found: {local_path}")  # noqa: T201
        return (remote_path or local_path, f"File not found: {local_path}")

    if expected_sizes:
        size_error = _check_file_size(remote_path, local_path, expected_sizes, verbose)
        if size_error:
            return size_error

    _print_file_success(remote_path, local_path, expected_sizes, verbose)
    return None


def verify_files(
    paths: list[str] | None = None,
    file_tuples: list[tuple[str, str]] | None = None,
    cache_manager: Any | None = None,
    expected_sizes: dict[str, int] | None = None,
    verbose: bool = True,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Unified file verification for E2E tests.

    Supports:
    - Verify from cache metadata (paths + cache_manager)
    - Verify local paths (file_tuples, optional expected_sizes)

    Args:
        paths: List of remote paths (S3 URIs) - requires cache_manager
        file_tuples: List of (remote_path, local_path) for direct verification
        cache_manager: CacheManager instance - required if using paths
        expected_sizes: Optional dict mapping remote_path to expected size in bytes
        verbose: Whether to print verification details

    Returns:
        Tuple of (verified_count, failed_count, errors_list)
    """
    if verbose and (paths or file_tuples):
        print("\nVerifying files...")  # noqa: T201

    verified_count = 0
    failed_count = 0
    errors: list[tuple[str, str]] = []

    if paths and cache_manager is not None:
        for remote_path in paths:
            metadata = cache_manager.get_file_metadata(remote_path)
            if not metadata or not metadata.local_path:
                if verbose:
                    print(f"  ✗ No local path for {remote_path}")  # noqa: T201
                failed_count += 1
                errors.append((remote_path, "No local path in metadata"))
                continue
            error = _verify_single_file(remote_path, metadata.local_path, expected_sizes, verbose)
            if error:
                failed_count += 1
                errors.append(error)
            else:
                verified_count += 1
    elif file_tuples:
        for remote_path, local_path in file_tuples:
            error = _verify_single_file(remote_path, local_path, expected_sizes, verbose)
            if error:
                failed_count += 1
                errors.append(error)
            else:
                verified_count += 1

    return verified_count, failed_count, errors
