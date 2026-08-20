# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Public source resolution helpers for PackDS readers.

The internal module at this package path also integrates with DeltaCAT. The OSS
implementation supports local paths and direct remote PackDS URIs only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse


def _is_remote(path: str) -> bool:
    return "://" in path


def _gear_creds_to_storage_options(creds: dict[str, str]) -> dict[str, str]:
    endpoint = creds.get("endpoint_url", "")
    return {
        "aws_access_key_id": creds["aws_access_key_id"],
        "aws_secret_access_key": creds["aws_secret_access_key"],
        "aws_region": creds.get("region", "us-east-1"),
        "aws_endpoint": endpoint,
        "allow_http": "true" if endpoint.startswith("http://") else "false",
    }


@dataclass(frozen=True)
class ResolvedSource:
    """A PackDS source after URI/path resolution."""

    source_key: str
    original_uri: str
    packds_uri: str
    storage_options: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PackDSSourceResolver:
    """Base class for PackDS source resolvers."""

    source_key: ClassVar[str]
    priority: ClassVar[int] = 0

    @classmethod
    def matches(cls, uri: str) -> bool:
        raise NotImplementedError

    @classmethod
    def resolve(
        cls,
        uri: str,
        *,
        storage_options: dict[str, str] | None = None,
        deltacat_client: Any | None = None,
        deltacat_plan_kwargs: Mapping[str, Any] | None = None,
    ) -> ResolvedSource:
        del uri, storage_options, deltacat_client, deltacat_plan_kwargs
        raise NotImplementedError


class LocalSourceResolver(PackDSSourceResolver):
    """Resolver for ordinary local filesystem paths."""

    source_key = "local"

    @classmethod
    def matches(cls, uri: str) -> bool:
        return not _is_remote(uri)

    @classmethod
    def resolve(
        cls,
        uri: str,
        *,
        storage_options: dict[str, str] | None = None,
        deltacat_client: Any | None = None,
        deltacat_plan_kwargs: Mapping[str, Any] | None = None,
    ) -> ResolvedSource:
        del deltacat_client, deltacat_plan_kwargs
        return ResolvedSource(
            source_key=cls.source_key,
            original_uri=uri,
            packds_uri=str(Path(uri)),
            storage_options=storage_options,
        )


class DirectRemoteSourceResolver(PackDSSourceResolver):
    """Resolver for direct remote PackDS URIs such as ``s3://...``."""

    source_key = "remote"

    @classmethod
    def matches(cls, uri: str) -> bool:
        parsed = urlparse(uri)
        return bool(parsed.scheme) and parsed.scheme != "dc"

    @classmethod
    def resolve(
        cls,
        uri: str,
        *,
        storage_options: dict[str, str] | None = None,
        deltacat_client: Any | None = None,
        deltacat_plan_kwargs: Mapping[str, Any] | None = None,
    ) -> ResolvedSource:
        del deltacat_client, deltacat_plan_kwargs
        resolved_storage_options = storage_options
        if resolved_storage_options is None:
            from gear import common

            resolved_storage_options = _gear_creds_to_storage_options(
                common.get_s3_credentials_from_file()
            )
        return ResolvedSource(
            source_key=cls.source_key,
            original_uri=uri,
            packds_uri=uri,
            storage_options=resolved_storage_options,
        )


class DeltaCatSourceResolver(PackDSSourceResolver):
    """Compatibility marker for the DeltaCAT backend omitted from OSS."""

    source_key = "deltacat"

    @classmethod
    def matches(cls, uri: str) -> bool:
        return urlparse(uri).scheme == "dc"

    @classmethod
    def resolve(
        cls,
        uri: str,
        *,
        storage_options: dict[str, str] | None = None,
        deltacat_client: Any | None = None,
        deltacat_plan_kwargs: Mapping[str, Any] | None = None,
    ) -> ResolvedSource:
        del uri, storage_options, deltacat_client, deltacat_plan_kwargs
        raise NotImplementedError("DeltaCAT PackDS sources are not included in the OSS package.")


SOURCE_RESOLVERS: dict[str, type[PackDSSourceResolver]] = {
    LocalSourceResolver.source_key: LocalSourceResolver,
    DirectRemoteSourceResolver.source_key: DirectRemoteSourceResolver,
}


def resolve_source(
    uri: str | Path,
    *,
    storage_options: dict[str, str] | None = None,
    deltacat_client: Any | None = None,
    deltacat_plan_kwargs: Mapping[str, Any] | None = None,
) -> ResolvedSource:
    """Resolve a local or direct-remote PackDS source."""
    uri_str = str(uri)
    if uri_str.startswith("dc://"):
        raise NotImplementedError(
            "PackDS dc:// DeltaCAT locators are not supported in the OSS package. "
            "Use a local path or direct remote PackDS URI instead."
        )
    matches = [resolver for resolver in SOURCE_RESOLVERS.values() if resolver.matches(uri_str)]
    if not matches:
        raise ValueError(f"No PackDS source resolver matched {uri_str!r}")
    resolver = min(matches, key=lambda candidate: candidate.priority)
    return resolver.resolve(
        uri_str,
        storage_options=storage_options,
        deltacat_client=deltacat_client,
        deltacat_plan_kwargs=deltacat_plan_kwargs,
    )
