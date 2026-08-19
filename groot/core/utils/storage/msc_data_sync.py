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

# ruff: noqa: BLE001, D301, EXE001, TRY400

"""MSC Data Sync - Efficiently synchronize data between local and S3 using Multi-Storage Client.

A simple example to use it (AWS S3 CLI compatible syntax):

- Synchronize the data
```
python -m groot.vla.utils.msc_data_sync \\
    ~/Downloads/pdx_cache \\
    s3://example-bucket/path/to/dataset/ \\
    --num-workers 6 --credentials ~/.gear/data_credentials --profile default
```
"""

import argparse
import logging
import os
import sys

import multistorageclient as msc
import ray

from groot.core.utils.storage.msc import DataSyncConfigBuilder, parse_s3_uri

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_path(object_metadata, base_url):
    """Extract and normalize file path from ObjectMetadata based on the base URL.

    >>> class MockMetadata:
    ...     def __init__(self, key):
    ...         self.key = key

    >>> # Test MSC source path - removes protocol and first path component
    >>> mock_obj = MockMetadata("msc://source/2025-09-09-16-25-02-YAM-05/20250909T163044783657/action-left-pos.npy")
    >>> normalize_path(mock_obj, "msc://source/2025-09-09-16-25-02-YAM-05")
    '20250909T163044783657/action-left-pos.npy'

    >>> # Test MSC target path - removes protocol prefix
    >>> mock_obj = MockMetadata("msc://target/yam_lerobot_v3/xdof.a4_2025-10-03_14-01-48_v4_USA/data/chunk-000/episode_000000.parquet")
    >>> normalize_path(mock_obj, "msc://target/yam_lerobot_v3/xdof.a4_2025-10-03_14-01-48_v4_USA")
    'data/chunk-000/episode_000000.parquet'

    >>> # Test local path - removes base path prefix
    >>> mock_obj = MockMetadata("/tmp/xdof.a4_2025-10-03_14-01-48_v4_USA/data/chunk-000/episode_000000.parquet")
    >>> normalize_path(mock_obj, "/tmp/xdof.a4_2025-10-03_14-01-48_v4_USA")
    'data/chunk-000/episode_000000.parquet'

    >>> # Test path that doesn't match base_url
    >>> mock_obj = MockMetadata("/other/path/file.txt")
    >>> normalize_path(mock_obj, "/tmp/base")
    '/other/path/file.txt'
    """
    key = object_metadata.key

    # Remove base_url prefix if it matches
    if key.startswith(base_url):
        relative_path = key[len(base_url) :].lstrip("/")
        return relative_path

    # Handle MSC protocol prefixes when base_url doesn't include the full prefix
    if key.startswith("msc://source/") and not base_url.startswith("msc://source/"):
        relative_path = key[len("msc://source/") :]
        # Remove one more path component for source paths
        path_parts = relative_path.split("/")
        if len(path_parts) > 1:
            relative_path = "/".join(path_parts[1:])
        return relative_path
    elif key.startswith("msc://target/") and not base_url.startswith("msc://target/"):
        return key[len("msc://target/") :]

    return key


def validate_file_sets(source_url, target_url):
    """Validate that source and target have the same set of files.

    Returns:
        0: Success (files match)
        2: Failure (files don't match)
    """
    logger.info("Performing post-sync validation...")

    # List source files
    logger.info(f"Listing source files: {source_url}")
    source_files = msc.list(source_url)
    # Extract and normalize file paths from ObjectMetadata objects
    source_paths = [normalize_path(f, source_url) for f in source_files]
    source_set = set(source_paths)
    logger.info(f"Source file count: {len(source_set)}")

    # List destination files
    logger.info(f"Listing destination files: {target_url}")
    target_files = msc.list(target_url)
    # Extract and normalize file paths from ObjectMetadata objects
    target_paths = [normalize_path(f, target_url) for f in target_files]
    target_set = set(target_paths)
    logger.info(f"Destination file count: {len(target_set)}")

    # Check for zero files in either source or destination
    if len(source_set) == 0:
        logger.error(f"Error: Source ({source_url}) has 0 files")
        return 3
    if len(target_set) == 0:
        logger.error(f"Error: Destination ({target_url}) has 0 files after sync")
        return 3

    # Check if file sets match
    if source_set != target_set:
        logger.error(
            f"File set mismatch: source has {len(source_set)} files, destination has {len(target_set)} files"
        )

        # Log files only in source
        source_only = source_set - target_set
        if source_only:
            logger.error(f"Files only in source ({len(source_only)}): {sorted(source_only)}")

        # Log files only in destination
        target_only = target_set - source_set
        if target_only:
            logger.error(f"Files only in destination ({len(target_only)}): {sorted(target_only)}")

        return 2
    else:
        logger.info(
            f"Validation successful: both source and destination have the same {len(source_set)} files"
        )
        return 0


def main():
    """Main function for the MSC data uploader."""
    parser = argparse.ArgumentParser(
        description="MSC Data Sync",
        usage="%(prog)s <source> <destination> [options]",
    )
    parser.add_argument(
        "source",
        help="Source path to upload, can be a local path or an S3 path (e.g., /tmp/foo or s3://bucket/path)",
    )
    parser.add_argument(
        "destination",
        help="Destination path to upload, can be a local path or an S3 path (e.g., /tmp/foo or s3://bucket/path)",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of MSC workers")
    parser.add_argument(
        "--credentials", default="~/.gear/data_credentials", help="Path to S3 credentials file"
    )
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Check that source and destination have the same number of files after sync",
    )
    parser.add_argument(
        "--no-check",
        action="store_false",
        dest="check",
        help="Skip file count validation after sync",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_false",
        dest="enable_telemetry",
        help="Disable MSC telemetry",
    )
    parser.add_argument(
        "--ray-address", default=None, help="Ray cluster address (default: start local cluster)"
    )

    args = parser.parse_args()

    try:
        # Parse source and destination URLs
        if args.source.startswith("s3://"):
            source_url = args.source
        else:
            source_url = os.path.expanduser(args.source)

        if args.destination.startswith("s3://"):
            target_url = args.destination

        if args.destination.startswith("s3://"):
            target_url = args.destination
        else:
            target_url = os.path.expanduser(args.destination)

        # Expand paths properly
        credentials_file = os.path.expanduser(args.credentials)

        logger.info("MSC Upload configuration:")
        logger.info(f"  Source: {args.source}")
        logger.info(f"  Destination: {args.destination}")
        logger.info(f"  Workers: {args.num_workers}")
        logger.info(f"  Credentials: {credentials_file}")
        logger.info(f"  Profile: {args.profile}")
    except ValueError as e:
        logger.error(f"Invalid S3 URI: {e}")
        return 1

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        from groot.core.utils.distributed.ray_init import (
            ray_init_maybe_new_cluster,
        )

        logger.info("Initializing Ray cluster...")
        is_new, cluster_address = ray_init_maybe_new_cluster()
        logger.info(f"Ray cluster ready at: {cluster_address}")

    try:
        config_builder = DataSyncConfigBuilder(
            credentials_file,
            args.profile,
            use_rust_client=True,
            enable_telemetry=args.enable_telemetry,
        )

        # Create MSC configuration file
        if source_url.startswith("s3://"):
            bucket, prefix = parse_s3_uri(source_url)
            source_url = f"msc://source/{prefix}"
            config_builder.add_profile("source", bucket)

        if target_url.startswith("s3://"):
            bucket, prefix = parse_s3_uri(target_url)
            target_url = f"msc://target/{prefix}"
            config_builder.add_profile("target", bucket)

        # Export the MSC configuration file to the environment variable
        config_builder.build()

        # Configure number of processes
        if args.num_workers > 128:
            raise ValueError("Number of workers cannot be greater than 128")

        os.environ["MSC_NUM_PROCESSES"] = str(args.num_workers)
        os.environ["MSC_NUM_THREADS_PER_PROCESS"] = "16"

        # Start upload
        msc.sync(
            source_url=source_url,
            target_url=target_url,
            execution_mode=msc.types.ExecutionMode.RAY,
        )

        if not args.check:
            logger.info("Skipping post-sync validation due to --no-check flag")
            return 0

        validation_result = validate_file_sets(source_url, target_url)
        return validation_result
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
