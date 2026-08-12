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

import os
from pathlib import Path

# Disable OpenTelemetry SDK before importing MSC to avoid
# 'AutoProxy[opentelemetry.metrics.Counter]' object has no attribute 'add'
# crash during multiprocessing sync on clusters with broken MSAL module.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import multistorageclient as msc

from groot.core.utils.distributed.ray_init import (
    ray_init_maybe_new_cluster,
)
from groot.core.utils.storage.msc import DataSyncConfigBuilder
from groot.core.utils.storage.data_paths import extract_dataset_s3_paths, get_datasets_base_path


def maybe_download_cache_contents(cfg):
    # Resolve base cache dir from env or default constant using shared utility
    base_cache_dir = get_datasets_base_path()

    # Extract dataset paths from config
    s3_paths = extract_dataset_s3_paths(cfg)
    if not s3_paths:
        print("No dataset paths found in configuration")
        return

    # Prepare Ray cluster
    ray_init_maybe_new_cluster()

    credentials_path = str(Path.home() / ".gear" / "data_credentials")
    bucket = "GrootDatasets"
    profile = "default"
    num_workers = 8

    for s3_path in s3_paths:
        full_cache_path = str(base_cache_dir / s3_path)
        fill_cache(s3_path, credentials_path, full_cache_path, bucket, profile, num_workers)


def fill_cache(
    s3_path: str,
    credentials_path: str,
    cache_path: str,
    bucket: str,
    profile: str,
    num_workers: int,
    use_ray: bool = False,
):
    """
    Fill the cache with the contents of the S3 path using MSC.

    Args:
        s3_path: The S3 path to fill the cache with.
        credentials_path: The path to the credentials file.
        cache_path: The path to the cache.
        bucket: The bucket to fill the cache from.
        profile: The profile to use for the credentials.
        num_workers: The number of workers to use for the download.
        use_ray: If True, use Ray execution mode (requires Ray cluster).
                 If False, use local execution mode (default, more reliable).
    """
    config_builder = DataSyncConfigBuilder(credentials_path, profile)
    config_builder.add_profile("datasets", bucket)
    config_builder.build()

    os.environ["MSC_NUM_PROCESSES"] = str(num_workers)
    os.environ["MSC_NUM_THREADS_PER_PROCESS"] = "16"

    # Choose execution mode - LOCAL is more reliable and avoids Ray/multiprocessing auth issues
    execution_mode = msc.types.ExecutionMode.RAY if use_ray else msc.types.ExecutionMode.LOCAL

    if use_ray:
        # Initialize Ray
        print("\nInitializing Ray cluster...")
        ray_init_maybe_new_cluster()
        print("Ray cluster initialized.\n")

    # Start download
    msc.sync(
        source_url=f"msc://datasets/{s3_path}",
        target_url=cache_path,
        execution_mode=execution_mode,
    )


if __name__ == "__main__":
    home_dir = str(Path.home())
    s3_path = (
        "78dc_fixed_annotations/"
        "gr1_unified.gr1_unified.PnPPotatoToMicrowaveClose_"
        "GR1ArmsAndWaistFourierHands/"
    )
    credentials_path = str(Path(home_dir) / ".gear" / "data_credentials")
    base_cache_path = Path(home_dir) / "Downloads" / "pdx_cache"
    cache_path = str(base_cache_path / s3_path)
    bucket = "data"
    profile = "default"
    num_workers = 1
    fill_cache(
        s3_path,
        credentials_path,
        cache_path,
        bucket,
        profile,
        num_workers,
    )
