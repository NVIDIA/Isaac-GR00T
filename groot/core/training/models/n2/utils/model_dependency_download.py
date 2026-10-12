# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Synchronous model-asset transport using Hydrator's S3 backend.

The Hydrator service creates Ray workers, and its directory sync requires s5cmd.
Standalone model loaders need neither: reuse its per-file boto3 transfer backend
and credential source instead. The resolver still owns inventory validation,
locking and atomic publication of a complete dependency directory.
"""

import os
from pathlib import Path
import tempfile
from typing import Any


def _client_config(profile: str, credentials_file: str | None) -> dict[str, Any]:
    """Preserve loader credential sources without changing Hydrator's defaults."""
    from groot.core.data.hydrator.utils import aws
    from groot.infra.job.data_credentials import DataCredentials

    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL")
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    credentials_file = credentials_file or DataCredentials.DEFAULT_CREDENTIALS_FILE

    def profile_credentials() -> dict[str, Any]:
        # The shared config builder currently omits session tokens; retain the
        # raw profile token without changing credentials for unrelated callers.
        return aws._get_s3_credentials(profile, credentials_file)  # noqa: SLF001

    if access_key and secret_key:
        credentials = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
        }
    elif os.environ.get("GEAR_DATA_CREDENTIALS"):
        credentials = profile_credentials()
        # Injected Gear profiles have always supplied their own endpoint/region.
        endpoint, region = None, None
    else:
        access_key_path = Path("/etc/gear/s3/access_key")
        secret_key_path = Path("/etc/gear/s3/secret_key")
        if access_key_path.is_file() and secret_key_path.is_file():
            credentials = {
                "aws_access_key_id": access_key_path.read_text().strip(),
                "aws_secret_access_key": secret_key_path.read_text().strip(),
            }
            if not all(credentials.values()):
                credentials = profile_credentials()
        else:
            credentials = profile_credentials()

    if not (credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key")):
        raise ValueError(f"Incomplete S3 credentials for model dependency profile {profile!r}")
    config = {
        "aws_access_key_id": credentials["aws_access_key_id"],
        "aws_secret_access_key": credentials["aws_secret_access_key"],
        "endpoint_url": endpoint or credentials.get("endpoint_url") or "https://pdx.s8k.io",
        "region_name": region or credentials.get("region") or "us-east-1",
    }
    if credentials.get("aws_session_token"):
        config["aws_session_token"] = credentials["aws_session_token"]
    return config


class ModelDependencyDownload:
    """Small boto3-compatible adapter; all payload transfers use Hydrator.

    Imports and credentials are resolved only when a cold-cache download is
    needed. Metadata is passed through unchanged for resolver-side validation.
    """

    def __init__(self, *, credentials_file: str | None = None):
        import boto3
        from botocore.config import Config

        from groot.core.data.hydrator.backends.s3_boto3 import S3Boto3Backend

        self._profile = os.environ.get("GROOT_RL_PDX_CREDENTIALS_PROFILE", "default")
        config = _client_config(self._profile, credentials_file)
        self._client = client = boto3.client(
            "s3",
            **config,
            config=Config(
                s3={"addressing_style": "path"},
                retries={"total_max_attempts": 20, "mode": "standard"},
            ),
        )

        class PinnedClientBackend(S3Boto3Backend):
            def _get_s3_client(self, bucket: str | None = None):  # noqa: ARG002
                # Metadata and payloads must use identical authentication and
                # endpoint, including an explicitly configured PDX endpoint.
                return client

        self._backend_type = PinnedClientBackend

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.head_object(**kwargs)

    def get_paginator(self, operation: str) -> Any:
        return self._client.get_paginator(operation)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        # Hydrator normalizes S3 URIs. Reject ambiguous keys rather than fetch a
        # different object than the one in the resolver's inventory/HEAD result.
        if (
            not bucket
            or any(char in bucket for char in "/\\\r\n\t ")
            or not key
            or key != key.strip()
            or "\\" in key
            or any(char in key for char in "\r\n\t")
            or any(part in {"", ".", ".."} for part in key.split("/"))
        ):
            raise ValueError("Model dependency bucket/key must be a canonical S3 object path")
        destination = Path(filename)
        with tempfile.TemporaryDirectory(
            prefix=".model-dependency-", dir=destination.parent
        ) as tmp:
            backend = self._backend_type(working_dir=tmp, credentials_profile=self._profile)
            downloaded = backend.download_file(f"s3://{bucket}/{key}")
            # The temporary directory shares the destination filesystem: no
            # extra full-size copy, and failed transfers never replace it.
            os.replace(downloaded, destination)
