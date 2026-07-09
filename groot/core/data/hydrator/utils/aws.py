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

"""AWS utilities for credential management and S3 operations.

Shared utilities for accessing real AWS S3 from the hydrator and callers.
Credentials are resolved in this order:
  1. GEAR_DATA_CREDENTIALS env var (base64 multi-profile JSON injected by osmo/gear)
  2. credentials_file on disk via :class:`groot.infra.job.data_credentials.DataCredentials`
     (default: ~/.gear/data_credentials)
Default profile is "aws".

Note: GEAR_DATA_CREDENTIALS is decoded directly here so that Ray worker actors work
correctly even when the cluster image has an older gear version that does not support
this env var.
"""

import base64
import json
import os
from typing import Literal

import boto3

from groot.core.utils.distributed import worker_logging
from groot.infra.job.data_credentials import DataCredentials

log = worker_logging.get_unified_logger()

_KNOWN_REGION_PREFIXES = ("us-", "eu-", "ap-", "sa-", "ca-", "af-", "me-")


def _extract_region_from_bucket(bucket: str) -> str | None:
    """Extract AWS region from a bucket name ending in -{region}-nvidia-gear.

    Example: ``mnt-aws-lfs-02-grootdatasets-us-west-2-nvidia-gear`` → ``us-west-2``
    """
    if not bucket.endswith("-nvidia-gear"):
        return None
    without_suffix = bucket[: -len("-nvidia-gear")]
    parts = without_suffix.split("-")
    for n in (3, 2):
        if len(parts) >= n:
            candidate = "-".join(parts[-n:])
            if any(candidate.startswith(p) for p in _KNOWN_REGION_PREFIXES):
                return candidate
    return None


def _get_s3_credentials(profile: str, credentials_file: str) -> dict:
    """Resolve S3 credentials, checking GEAR_DATA_CREDENTIALS env var first."""
    env_creds = os.environ.get("GEAR_DATA_CREDENTIALS")
    if not env_creds:
        log.debug(
            "_get_s3_credentials: GEAR_DATA_CREDENTIALS not set, falling back to file %s",
            credentials_file,
        )
    else:
        log.debug(
            "_get_s3_credentials: GEAR_DATA_CREDENTIALS is set (length=%d), attempting decode",
            len(env_creds),
        )
        try:
            all_creds = json.loads(base64.b64decode(env_creds).decode())
            available_profiles = list(all_creds.keys())
            log.debug(
                "_get_s3_credentials: decoded profiles=%s, requested profile=%r",
                available_profiles,
                profile,
            )
            if profile in all_creds:
                log.debug("_get_s3_credentials: found profile %r in GEAR_DATA_CREDENTIALS", profile)
                return all_creds[profile]
            else:
                log.warning(
                    "_get_s3_credentials: profile %r not found in GEAR_DATA_CREDENTIALS "
                    "(available: %s), falling back to file",
                    profile,
                    available_profiles,
                )
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(
                "_get_s3_credentials: failed to decode GEAR_DATA_CREDENTIALS: %s, falling back to file",
                e,
            )

    log.info("_get_s3_credentials: loading from file %s profile=%r", credentials_file, profile)
    return DataCredentials(credentials_file=credentials_file).get_s3_credentials(profile=profile)


def resolve_s3_region(
    bucket: str,
    profile: str,
    credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE,
) -> str | None:
    """Best-effort AWS region for accessing *bucket* with *profile*.

    Uses the ``-{region}-nvidia-gear`` bucket suffix when present, else the
    ``region`` field on the gear credentials profile. Returns ``None`` when
    neither source yields a region — callers fall back to the resolver's
    tier-0 region.
    """
    region = _extract_region_from_bucket(bucket)
    if region is not None:
        return region
    try:
        config = get_boto3_client_config(profile=profile, credentials_file=credentials_file)
    except (ValueError, FileNotFoundError):
        return None
    return config.get("region_name")


def get_boto3_client_config(
    profile: str,
    credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE,
) -> dict:
    """Build boto3 S3 client configuration from gear credentials.

    Returns dict suitable for boto3.client('s3', **config).
    """
    creds = _get_s3_credentials(profile=profile, credentials_file=credentials_file)

    client_config: dict = {}

    if creds.get("aws_access_key_id") and creds.get("aws_secret_access_key"):
        client_config["aws_access_key_id"] = creds["aws_access_key_id"]
        client_config["aws_secret_access_key"] = creds["aws_secret_access_key"]

    if creds.get("region"):
        client_config["region_name"] = creds["region"]

    endpoint_url = (creds.get("endpoint_url") or "").strip()
    if endpoint_url:
        client_config["endpoint_url"] = endpoint_url

    return client_config


def make_s3_client(
    profile: str,
    region: str | None = None,
    endpoint_style: Literal["regional", "profile"] = "regional",
    credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE,
):
    """Build a boto3 S3 client from gear credentials.

    Args:
        profile: Gear credentials profile (e.g. ``"aws"`` or ``"default"``).
        region: AWS region string (e.g. ``"us-west-2"``), or ``None``.
        endpoint_style: Controls how the S3 endpoint URL is chosen.

            - ``"regional"`` *(default)*: endpoint is set to
              ``https://s3.{region}.amazonaws.com`` when *region* is given.
              Correct for AWS S3 — ensures the client always hits the regional
              endpoint regardless of what is in the credential profile.
            - ``"profile"``: endpoint (if any) is taken from the credential
              profile as-is.  Correct for PDX/Ceph tiers that store a custom
              endpoint URL in ``~/.gear/data_credentials``.  *region* may still
              be provided for bucket-name derivation but does **not** influence
              the endpoint.
        credentials_file: Path to gear credentials file.

    Returns:
        Configured boto3 S3 client.
    """
    from botocore import config as botocore_config

    client_config = get_boto3_client_config(profile=profile, credentials_file=credentials_file)
    if region is not None:
        client_config["region_name"] = region
    if endpoint_style == "regional" and region is not None:
        client_config["endpoint_url"] = f"https://s3.{region}.amazonaws.com"
    if client_config.get("endpoint_url"):
        client_config.setdefault("config", botocore_config.Config(s3={"addressing_style": "path"}))
    return boto3.client("s3", **client_config)


def read_s3_file_as_text(
    s3_uri: str,
    profile: str,
    credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE,
) -> str:
    """Read an S3 object as UTF-8 text."""
    from groot.core.data.hydrator.utils import io

    bucket, key = io.parse_s3_uri(s3_uri)
    region, endpoint_style = io.resolve_s3_region_and_endpoint_style(bucket, profile)
    log.info(
        "read_s3_file_as_text: bucket=%s key=%s region=%s profile=%s endpoint_style=%s",
        bucket,
        key,
        region,
        profile,
        endpoint_style,
    )
    client = make_s3_client(
        profile=profile,
        region=region,
        endpoint_style=endpoint_style,
        credentials_file=credentials_file,
    )
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def get_pyarrow_s3_fs(
    profile: str,
    credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE,
    bucket: str | None = None,
):
    """Build a PyArrow S3 FileSystem using gear credentials.

    Use with pyarrow.parquet.read_table(path, filesystem=fs).
    Path should omit the scheme, e.g. "bucket/path/to/file.parquet".

    If *bucket* follows the ``-{region}-nvidia-gear`` naming scheme, the region
    and endpoint are derived from the bucket name, overriding the credentials
    default to avoid PermanentRedirect errors.
    """
    import pyarrow.fs as pafs

    creds = _get_s3_credentials(profile=profile, credentials_file=credentials_file)
    region = (bucket and _extract_region_from_bucket(bucket)) or creds.get("region")
    endpoint_url = (
        f"https://s3.{region}.amazonaws.com"
        if bucket and _extract_region_from_bucket(bucket)
        else (creds.get("endpoint_url") or "").strip() or None
    )
    return pafs.S3FileSystem(
        access_key=creds.get("aws_access_key_id"),
        secret_key=creds.get("aws_secret_access_key"),
        region=region,
        endpoint_override=endpoint_url,
    )


def get_s3_storage_options(
    profile: str,
    credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE,
) -> dict | None:
    """Return fsspec/pandas storage_options dict for S3 access (e.g. pd.read_parquet).

    Returns None if no credentials are found.
    """
    try:
        creds = _get_s3_credentials(profile=profile, credentials_file=credentials_file)
    except (ValueError, FileNotFoundError):
        return None

    opts: dict = {}
    if creds.get("aws_access_key_id"):
        opts["key"] = creds["aws_access_key_id"]
    if creds.get("aws_secret_access_key"):
        opts["secret"] = creds["aws_secret_access_key"]
    endpoint_url = (creds.get("endpoint_url") or "").strip()
    if endpoint_url:
        opts["client_kwargs"] = {"endpoint_url": endpoint_url}
    return opts or None
