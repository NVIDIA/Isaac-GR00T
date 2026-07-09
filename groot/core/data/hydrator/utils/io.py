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

"""Shared I/O for local and S3: path normalization, glob, read file content, read parquet, consume."""

import json
from pathlib import Path
import re
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from groot.core.data.hydrator.utils import aws


def _resolve_profile() -> str:
    """Return the active credentials profile from DataSourceResolver, falling back to 'aws'."""
    from groot.core.data.hydrator import data_source_resolver as _dsr

    return _dsr.default_resolver()._get_active_profile() or "aws"  # noqa: SLF001


def resolve_s3_region_and_endpoint_style(
    bucket: str, profile: str
) -> tuple[str | None, Literal["regional", "profile"]]:
    """Region and endpoint style for boto3 S3 clients, aligned with DataSourceResolver.

    Endpoint style is taken from the tier that owns *bucket* so PDX-tier
    buckets keep their custom endpoint URL (style=``"profile"``) and AWS-tier
    buckets use the regional URL (style=``"regional"``). Falls back to the
    region-presence heuristic only when the resolver has no opinion.
    """
    from groot.core.data.hydrator import data_source_resolver as _dsr

    resolver = _dsr.default_resolver()
    region = aws.resolve_s3_region(bucket, profile)
    if region is None:
        resolver._ensure_loaded()  # noqa: SLF001
        tiers = resolver._tiers  # noqa: SLF001
        if tiers:
            region = tiers[0]._region  # noqa: SLF001
    endpoint_style = resolver.get_tier_endpoint_style_for_profile(profile)
    if endpoint_style is None:
        endpoint_style = "regional" if region else "profile"
    return region, endpoint_style


def normalize_path(path: str) -> str:
    """Return canonical path string for use with hydrator.

    - Strips whitespace.
    - Fixes s3:/ to s3:// (str(Path('s3://...')) can become s3:/ on some OSes).
    - For s3:// URIs: collapses repeated slashes and trims trailing slash.
    """
    s = path.strip()
    if s.startswith("s3:/") and not s.startswith("s3://"):
        s = "s3://" + s[4:]
    if s.startswith("s3://"):
        rest = s[5:]
        rest = re.sub(r"/+", "/", rest).rstrip("/")
        s = "s3://" + rest
    return s


def parse_s3_uri(s3_path: str) -> tuple[str, str]:
    """Parse S3 URI into bucket and key. Uses normalize_path for consistent parsing."""
    uri = normalize_path(s3_path)
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {s3_path}")
    parts = uri[5:].split("/", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


def glob_paths(base_path: str, pattern: str, profile: str | None = None) -> list[str]:
    """List paths matching pattern under base_path (local or S3).

    For local paths uses Path.glob(); for S3 lists objects under the
    prefix derived from pattern and returns full s3:// URIs matching the pattern suffix.

    Args:
        base_path: Base path (local or s3://...). Will be normalized.
        pattern: Glob pattern (e.g. "data/*/*.parquet").
        profile: Credentials profile to use for S3 listing. Defaults to the active profile.

    Returns:
        Sorted list of path strings (local paths or s3:// URIs).
    """
    base_path = normalize_path(base_path)
    if not base_path.startswith("s3://"):
        return sorted(str(p) for p in Path(base_path).glob(pattern))
    # S3: list objects under prefix
    bucket, base_key = parse_s3_uri(base_path)
    prefix = pattern.split("*")[0]
    if base_key:
        prefix = base_key.rstrip("/") + "/" + prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    _profile = profile or _resolve_profile()
    region, endpoint_style = resolve_s3_region_and_endpoint_style(bucket, _profile)
    s3 = aws.make_s3_client(profile=_profile, region=region, endpoint_style=endpoint_style)
    uris = []
    paginator = s3.get_paginator("list_objects_v2")
    suffix = pattern.split("*")[-1] if "*" in pattern else pattern
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not suffix or key.endswith(suffix):
                uris.append(f"s3://{bucket}/{key}")
    return sorted(uris)


def read_file_content(path: str, profile: str | None = None) -> str:
    """Read file as UTF-8 string. Supports local paths and S3 URIs. Path is normalized."""
    path = normalize_path(path)
    if path.startswith("s3://"):
        return aws.read_s3_file_as_text(path, profile=profile or _resolve_profile())
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_parquet_path(path: str, profile: str | None = None) -> pa.Table:
    """Read a parquet file from local path or S3 URI as a pyarrow Table."""
    path = normalize_path(path)
    if path.startswith("s3://"):
        bucket, key = parse_s3_uri(path)
        s3_fs = aws.get_pyarrow_s3_fs(profile=profile or _resolve_profile(), bucket=bucket)
        s3_path = f"{bucket}/{key}" if key else bucket
        return pq.read_table(s3_path, filesystem=s3_fs)
    return pq.read_table(path)


def consume(path: str, profile: str | None = None) -> list[dict] | pa.Table:
    """Read a .json, .jsonl, or .parquet file from local or S3 and return its contents.

    - .json: list of one dict
    - .jsonl: list of dicts
    - .parquet: pyarrow Table

    Raises ValueError if path does not end with .json, .jsonl, or .parquet.
    """
    path = normalize_path(path)
    if path.endswith(".jsonl"):
        content = read_file_content(path, profile=profile)
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    if path.endswith(".json"):
        content = read_file_content(path, profile=profile)
        return [json.loads(content)]
    if path.endswith(".parquet"):
        return read_parquet_path(path, profile=profile)
    raise ValueError(f"consume() only supports .json, .jsonl, and .parquet paths, got: {path!r}")


def s3_object_exists(path: str, profile: str | None = None) -> bool:
    """Return True if an S3 object exists at ``path`` (HEAD only; no body download).

    Local paths fall back to ``Path.exists``. When ``profile`` is omitted, the
    active credentials profile is resolved via ``_resolve_profile()`` (i.e.
    ``DataSourceResolver.default_resolver()._get_active_profile()`` with an
    ``"aws"`` fallback), matching the idiom in ``read_file_content``,
    ``read_parquet_path``, and ``glob_paths``.
    """
    path = normalize_path(path)
    if not path.startswith("s3://"):
        return Path(path).exists()

    bucket, key = parse_s3_uri(path)
    if not key:
        return False

    _profile = profile or _resolve_profile()
    region, endpoint_style = resolve_s3_region_and_endpoint_style(bucket, _profile)
    s3 = aws.make_s3_client(profile=_profile, region=region, endpoint_style=endpoint_style)

    import botocore.exceptions

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


# Backward-compat alias for callers that imported `hydrate_path`.
hydrate_path = consume
