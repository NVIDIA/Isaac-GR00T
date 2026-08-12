# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""Provision an S3 bucket for a given local path on a target cluster pool.

Bucket naming:
  Path segments are joined with hyphens (underscores within segments become
  hyphens); leading slashes and empty segments are ignored. Suffix is
  ``-<region>-nvidia-gear``.

Example:
    path=/mnt/aws-lfs-02/GrootDatasets  pool=groot-gb200-02
    -> bucket: mnt-aws-lfs-02-grootdatasets-us-west-2-nvidia-gear

Usage:
  python provision_object_store.py --path /mnt/aws-lfs-02/GrootDatasets
                                   --pool groot-gb200-02
  python provision_object_store.py --path /mnt/aws-lfs-02/GrootDatasets
                                   --region us-west-2
"""

from __future__ import annotations

import argparse
import os
import sys

import boto3

from groot.core.data.hydrator import data_source_resolver
from groot.core.data.hydrator.migration._types import load_pool_to_region
from groot.core.data.hydrator.utils import aws as aws_utils


def create_bucket(s3_client, bucket_name: str, region: str) -> None:
    """Create the bucket in the given region."""
    kwargs: dict = {"Bucket": bucket_name}
    # us-east-1 must NOT include LocationConstraint (AWS quirk)
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3_client.create_bucket(**kwargs)


def _list_child_buckets(s3_client, local_path: str, region: str) -> list[str]:
    """Return names of buckets that were provisioned for paths *under* local_path.

    A child bucket has a name that starts with the path's bucket prefix + '-'
    and ends with '-<region>', but has more segments (i.e. is more specific).
    """
    prefix = "-".join(s.replace("_", "-") for s in local_path.split("/") if s).lower() + "-"
    suffix = "-" + region + "-nvidia-gear"
    try:
        response = s3_client.list_buckets()
    except Exception:  # noqa: BLE001
        return []
    return [
        b["Name"]
        for b in response.get("Buckets", [])
        if b["Name"].startswith(prefix)
        and b["Name"].endswith(suffix)
        and b["Name"] != prefix.rstrip("-") + suffix
    ]


def resolve_bucket(
    local_path: str,
    region: str,
    s3_client: boto3.client = None,
    prompt_fn: callable = input,
) -> tuple[str, str]:
    """Resolve a local path to an existing S3 bucket, with partial-match warnings.

    Resolution order:
      1. Exact bucket match -> return it immediately.
      2. Parent match (e.g. user gives /A/B/C/D, bucket exists for /A/B/C) ->
         reuse the closest parent bucket automatically (path stored as a prefix).
      3. Child match (e.g. user gives /A/B, buckets exist for /A/B/C) ->
         warn about more specific buckets and ask whether to proceed to provision.
      4. No match -> print provisioning instructions and call sys.exit(1).

    Args:
        local_path: Local folder path to resolve (e.g. /mnt/aws-lfs-02/GrootDatasets).
        region: AWS region to search in.
        s3_client: Optional pre-built boto3 S3 client (defaults to one built from
            gear credentials). Useful for testing.
        prompt_fn: Callable used for interactive prompts (defaults to built-in
            ``input``). Override in tests to avoid blocking on stdin.

    Returns:
        Tuple of ``(bucket_name, key_prefix)`` where *key_prefix* is the
        sub-path to use inside the bucket (empty string for an exact match,
        or the relative path from the provisioned root for a parent match).
        Example: path ``/mnt/data/foo/bar``, bucket provisioned for ``/mnt/data``
        -> ``("mnt-data-us-west-2-nvidia-gear", "foo/bar")``.
    """
    if s3_client is None:
        s3_client = aws_utils.make_s3_client(profile="aws", region=region)

    exact = data_source_resolver.path_to_bucket_name(local_path, region)
    print(f"[resolve_bucket] path='{local_path}' region='{region}'")  # noqa: T201
    print(f"[resolve_bucket] exact candidate: {exact}")  # noqa: T201

    # 1. Exact match
    if data_source_resolver.bucket_exists(s3_client, exact):
        return exact, ""

    # 2. Parent match: user path is more specific than a provisioned bucket
    ancestors = data_source_resolver.ancestor_paths(local_path)
    ancestor_buckets = [data_source_resolver.path_to_bucket_name(a, region) for a in ancestors]
    print(  # noqa: T201
        f"[resolve_bucket] checking {len(ancestors)} ancestor(s): {ancestor_buckets}"
    )
    parent_matches: list[tuple[str, str]] = []  # (ancestor_path, bucket_name)
    for ancestor in ancestors:
        candidate = data_source_resolver.path_to_bucket_name(ancestor, region)
        if data_source_resolver.bucket_exists(s3_client, candidate):
            parent_matches.append((ancestor, candidate))

    if parent_matches:
        closest_path, closest_bucket = parent_matches[0]
        key_prefix = os.path.relpath(local_path, closest_path)
        print(  # noqa: T201
            f"\nNo exact bucket found for '{local_path}'.\n"
            f"Reusing parent bucket '{closest_bucket}' (provisioned for '{closest_path}').\n"
            f"Sub-path '{key_prefix}' will be used as prefix inside the bucket."
        )
        return closest_bucket, key_prefix

    # 3. Child match: user path is less specific than provisioned buckets
    child_buckets = _list_child_buckets(s3_client, local_path, region)
    if child_buckets:
        print(f"\nNo exact bucket found for '{local_path}'.")  # noqa: T201
        print("Found more specific bucket(s) already provisioned under this path:")  # noqa: T201
        for bucket in child_buckets:
            print(f"  s3://{bucket}/")  # noqa: T201
        print(  # noqa: T201
            "\nProvisioning a parent bucket may cause organizational confusion.\n"
            "Consider using one of the existing specific buckets above instead."
        )
        answer = (
            prompt_fn("Provision a new bucket for this parent path anyway? [y/N] ").strip().lower()
        )
        if answer != "y":
            sys.exit(1)
        print(  # noqa: T201
            f"\nTo create the bucket, run:\n"
            f"  ./provision_object_store.sh --path {local_path} --region {region}"
        )
        sys.exit(1)

    # 4. No match
    print(  # noqa: T201
        f"\nNo bucket found for '{local_path}' (region={region}).\n"
        f"Provision one first:\n"
        f"  ./provision_object_store.sh --path {local_path} --region {region}"
    )
    sys.exit(1)


def provision(local_path: str, region: str) -> str:
    """Ensure a bucket for *local_path* exists in *region*. Returns bucket name."""
    bucket_name = data_source_resolver.path_to_bucket_name(local_path, region)
    s3 = aws_utils.make_s3_client(profile="aws", region=region)

    if data_source_resolver.bucket_exists(s3, bucket_name):
        print(f"Bucket already exists: {bucket_name}")  # noqa: T201
    else:
        create_bucket(s3, bucket_name, region)
        print(f"Created bucket: {bucket_name} (region={region})")  # noqa: T201

    return bucket_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision an S3 bucket for a local path on a cluster pool."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Local folder path (e.g. /mnt/aws-lfs-02/GrootDatasets)",
    )

    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--pool",
        help="Cluster pool name (e.g. groot-gb200-02). Region is looked up automatically.",
    )
    region_group.add_argument(
        "--region",
        help="AWS region (e.g. us-east-1). Use instead of --pool for explicit control.",
    )

    args = parser.parse_args()

    if args.pool:
        pool_to_region = load_pool_to_region()
        if args.pool not in pool_to_region:
            known = ", ".join(sorted(pool_to_region))
            print(  # noqa: T201
                f"Unknown pool '{args.pool}'. Known pools:\n  {known}",
                file=sys.stderr,
            )
            sys.exit(1)
        region = pool_to_region[args.pool]
        print(f"Pool '{args.pool}' -> region '{region}'")  # noqa: T201
    else:
        region = args.region

    bucket = provision(args.path, region)
    print(f"s3://{bucket}/")  # noqa: T201


if __name__ == "__main__":
    main()
