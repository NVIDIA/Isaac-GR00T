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
"""Source resolution helpers for PackDS readers.

`PackDSReader` resolves a user-provided locator in two stages:

1. a *source resolver* maps the input path/URI to a concrete PackDS root
2. a *table backend* discovers a readable table representation under that root

This module owns stage 1. It keeps URI-scheme-specific logic (local paths,
plain remote URIs, DeltaCAT table locators) out of the table-format backends.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
import time as _time
from typing import Any, ClassVar
from urllib.parse import urlparse

from groot.core.data.packds import constants, utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSource:
    """A PackDS source after URI/path resolution.

    Attributes:
        source_key: Internal resolver key, e.g. ``"local"`` or ``"deltacat"``.
        original_uri: The user-provided path or URI.
        packds_uri: Concrete PackDS root path/URI to pass to table discovery.
        storage_options: Resolved storage options for reading ``packds_uri``.
        metadata: Resolver-specific side metadata, such as DeltaCAT snapshot data.
    """

    source_key: str
    original_uri: str
    packds_uri: str
    storage_options: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PackDSSourceResolver:
    """Base class for PackDS source resolvers.

    Source resolvers answer the question "what concrete PackDS root does this
    locator refer to?" They do not inspect table layouts or open datasets.
    """

    source_key: ClassVar[str]
    priority: ClassVar[int] = 0

    @classmethod
    def matches(cls, uri: str) -> bool:
        """Return whether this resolver should handle *uri*."""
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
        """Resolve *uri* into a concrete PackDS root and read options."""
        del uri, storage_options, deltacat_client, deltacat_plan_kwargs
        raise NotImplementedError


class LocalSourceResolver(PackDSSourceResolver):
    """Resolver for ordinary local filesystem paths."""

    source_key = "local"

    @classmethod
    def matches(cls, uri: str) -> bool:
        return not utils.is_remote(uri)

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
    """Resolver for direct remote PackDS URIs such as ``s3://...``.

    This covers URI schemes that already point at the PackDS root and do not
    require an external catalog lookup first.
    """

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

            resolved_storage_options = utils.gear_creds_to_lance_storage_options(
                common.get_s3_credentials_from_file()
            )
        return ResolvedSource(
            source_key=cls.source_key,
            original_uri=uri,
            packds_uri=uri,
            storage_options=resolved_storage_options,
        )


# ---------------------------------------------------------------------------
# Catalog → server-URL registry
# ---------------------------------------------------------------------------
#
# When a ``dc://<catalog>/<namespace>/<table>`` URI is resolved, the catalog
# component is mapped to a server URL via this dict — but only as a fallback.
# An explicit ``DELTACAT_SERVER_URL`` env var always wins.
#
# Tests that spin up a local DeltaCAT server should call
# :func:`register_deltacat_catalog` so a unique ``dc://<test-catalog>/...``
# URI resolves without touching any production endpoint.

#
# Two ALB families per environment, mirrored from
# ``deltacat-dev/scripts/runner/aws/gear_data_infra/deploy_deltacat.py``
# (``_EXTERNAL_ALB_URLS`` / ``_INTERNAL_ALB_URLS``):
#
# - External ALBs (internet-facing, corp users on VPN/DirectConnect).
# - Internal ALBs (RFC-6598 private IPs, worker daemons inside the VPC).
#
# Worker daemons running on gear-managed nodes typically resolve to the
# internal ALB via ``DELTACAT_SERVER_URL``. Anything resolving the
# ``iris-{env}`` keys without an explicit env var falls back to the
# external ALB so corp-network reads succeed.

DELTACAT_CATALOG_URLS: dict[str, str] = {
    "iris-alpha": "http://gear-data-infra-iris-alpha-alb-1184231570.us-east-2.elb.amazonaws.com",
    "iris-beta": "http://gear-data-infra-iris-beta-alb-1753350942.us-east-2.elb.amazonaws.com",
    "iris-prod": "http://gear-data-infra-iris-prod-alb-1122608245.us-east-2.elb.amazonaws.com",
    "iris-alpha-internal": "http://internal-gear-iris-alpha-alb-384624948.us-east-2.elb.amazonaws.com",
    "iris-beta-internal": "http://internal-gear-iris-beta-alb-1681704892.us-east-2.elb.amazonaws.com",
    "iris-prod-internal": "http://internal-gear-iris-prod-alb-293123343.us-east-2.elb.amazonaws.com",
}


def register_deltacat_catalog(name: str, server_url: str) -> None:
    """Map a catalog name in ``dc://`` URIs to a DeltaCAT server URL.

    Used by tests (``register_deltacat_catalog("local-catalog-abc", url)``)
    so a self-contained ``dc://local-catalog-abc/...`` resolves without an
    env var. Drops any cached client for this URL so the next resolve picks
    the new mapping up.
    """
    DELTACAT_CATALOG_URLS[name] = server_url
    DeltaCatSourceResolver._client_cache.pop(server_url, None)


def unregister_deltacat_catalog(name: str) -> None:
    """Remove a catalog from the registry, if present."""
    url = DELTACAT_CATALOG_URLS.pop(name, None)
    if url is not None:
        DeltaCatSourceResolver._client_cache.pop(url, None)


class DeltaCatSourceResolver(PackDSSourceResolver):
    """Resolver for ``dc://`` DeltaCAT table locators.

    It uses a DeltaCAT client to resolve the table into a concrete ``packds_uri``
    plus any plan-vended credentials needed to read it directly from storage.
    """

    source_key = "deltacat"

    # Cache of constructed clients, keyed by server URL so multiple catalogs
    # (e.g. iris-prod plus a test-local server) can coexist.
    _client_cache: ClassVar[dict[str, Any]] = {}

    # TTL-based plan cache: {uri: (ResolvedSource, expiry_time, options_fingerprint)}.
    # The fingerprint detects callers that re-resolve the same URI with
    # different storage_options / deltacat_client / deltacat_plan_kwargs.
    # That's almost always a bug — credentials and plan kwargs should be
    # stable per process — so we invalidate the entry, re-resolve fresh,
    # and warn loudly so the caller can find and fix the divergence.
    _plan_cache: ClassVar[dict[str, tuple[Any, float, tuple[Any, ...]]]] = {}
    _plan_cache_lock: ClassVar[threading.Lock] = threading.Lock()
    PLAN_CACHE_TTL_S: ClassVar[float] = constants.PLAN_CACHE_TTL_S

    @classmethod
    def matches(cls, uri: str) -> bool:
        return urlparse(uri).scheme == "dc"

    @classmethod
    def _resolve_server_url(cls, catalog: str | None) -> str | None:
        """Resolve which DeltaCAT server URL to use for *catalog*.

        Order: ``DELTACAT_SERVER_URL`` env var > registry lookup > ``None``.
        """
        import os
        env_url = os.environ.get("DELTACAT_SERVER_URL")
        if env_url:
            return env_url
        if catalog and catalog in DELTACAT_CATALOG_URLS:
            return DELTACAT_CATALOG_URLS[catalog]
        return None

    @classmethod
    def _client_from_env(cls, catalog: str | None = None) -> Any | None:
        """Auto-construct a deltacat_client from env / catalog registry.

        Returns *None* when no server URL is available for *catalog*.
        Constructed clients are cached per server URL so repeated resolves
        reuse the same connection.
        """
        server_url = cls._resolve_server_url(catalog)
        if not server_url:
            return None
        cached = cls._client_cache.get(server_url)
        if cached is not None:
            return cached
        try:
            from deltacat_client import Client

            import os
            client = Client(
                server_url,
                bearer_token=os.environ.get("DELTACAT_WORKER_TOKEN"),
            )
            cls._client_cache[server_url] = client
            return client
        except ImportError:
            return None

    @staticmethod
    def _parse_table_locator(uri: str) -> tuple[str, str, str, str | None]:
        parsed = urlparse(uri)
        catalog = parsed.netloc
        parts = [part for part in parsed.path.split("/") if part]
        if not catalog or len(parts) < 2:
            raise ValueError(
                "DeltaCAT PackDS URIs must be of the form dc://<catalog>/<namespace>/<table>"
            )
        namespace, table = parts[:2]
        table_version = parts[2] if len(parts) >= 3 else None
        if len(parts) > 3:
            raise NotImplementedError(
                "DeltaCAT PackDS URI resolution currently supports up to table-version locators "
                "(dc://<catalog>/<namespace>/<table>[/<table_version>])."
            )
        return catalog, namespace, table, table_version

    @staticmethod
    def _plan_storage_options(plan: Any) -> dict[str, str] | None:
        """Extract Lance storage_options from a DeltaCAT plan.

        Handles two credential types:
        - Direct keys: aws_access_key_id / aws_secret_access_key in the entry
        - Host profile: type=aws_profile with a profile name — resolved via
          boto3.Session(profile_name=...) to get the actual credentials
        """
        credentials = getattr(plan, "credentials", None)
        if not credentials or not isinstance(credentials, Mapping):
            return None

        # Find the credential entry matching the data root.
        # The plan has bindings like "binding:swiftstack_pdx",
        # "binding:aws_s3", etc.  Match by checking which root's
        # prefix matches the packds_uri.
        packds_uri = getattr(plan, "packds_uri", "") or ""
        available_roots = getattr(plan, "data_roots", None) or getattr(plan, "available_roots", None) or {}

        entry = credentials.get("default")
        if not isinstance(entry, Mapping):
            # Find the binding whose root is a prefix of packds_uri
            matched_binding = None
            for root_name, root_info in available_roots.items():
                if isinstance(root_info, Mapping):
                    root_path = root_info.get("root", "")
                    if packds_uri.startswith(root_path):
                        matched_binding = f"binding:{root_name}"
                        break

            if matched_binding and matched_binding in credentials:
                entry = credentials[matched_binding]
            else:
                # Fallback: first non-versioned entry
                for key, val in credentials.items():
                    if isinstance(val, Mapping) and "@" not in key:
                        entry = val
                        break

        if not isinstance(entry, Mapping):
            return None

        # Host-profile resolution: use boto3 to get actual credentials
        if entry.get("type") == "aws_profile" and entry.get("resolver") == "host_profile":
            profile = entry.get("profile")
            if profile:
                return DeltaCatSourceResolver._resolve_aws_profile(
                    profile,
                    region=entry.get("region"),
                    endpoint_url=entry.get("endpoint_url"),
                )

        # Direct credential extraction
        access_key = entry.get("aws_access_key_id") or entry.get("access_key_id")
        secret_key = entry.get("aws_secret_access_key") or entry.get("secret_access_key")
        session_token = entry.get("aws_session_token") or entry.get("session_token")
        region = entry.get("aws_region") or entry.get("region")
        endpoint = (
            entry.get("aws_endpoint")
            or entry.get("endpoint_url")
            or entry.get("aws_endpoint_url")
            or entry.get("s3_endpoint_url")
        )
        allow_http = "true" if isinstance(endpoint, str) and endpoint.startswith("http://") else "false"

        options: dict[str, str] = {}
        if access_key is not None:
            options["aws_access_key_id"] = str(access_key)
        if secret_key is not None:
            options["aws_secret_access_key"] = str(secret_key)
        if session_token is not None:
            options["aws_session_token"] = str(session_token)
        if region is not None:
            options["aws_region"] = str(region)
        if endpoint is not None:
            options["aws_endpoint"] = str(endpoint)
            options["allow_http"] = allow_http
        return options or None

    @staticmethod
    def _resolve_aws_profile(
        profile: str,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> dict[str, str] | None:
        """Resolve an AWS profile name to Lance storage_options."""
        try:
            import boto3

            session = boto3.Session(profile_name=profile)
            creds = session.get_credentials()
            if creds is None:
                return None
            frozen = creds.get_frozen_credentials()
            options: dict[str, str] = {
                "aws_access_key_id": frozen.access_key,
                "aws_secret_access_key": frozen.secret_key,
            }
            if frozen.token:
                options["aws_session_token"] = frozen.token
            # Region: use provided, fall back to profile, default us-east-1
            # for S3-compatible endpoints (SwiftStack etc.)
            resolved_region = region or session.region_name or "us-east-1"
            options["aws_region"] = resolved_region
            if endpoint_url:
                options["aws_endpoint"] = endpoint_url
                if endpoint_url.startswith("http://"):
                    options["allow_http"] = "true"
            return options
        except Exception:
            logger.debug("Failed to resolve AWS profile %s", profile, exc_info=True)
            return None

    @staticmethod
    def _fingerprint(
        storage_options: dict[str, str] | None,
        deltacat_client: Any | None,
        deltacat_plan_kwargs: Mapping[str, Any] | None,
    ) -> tuple[Any, ...]:
        """Hashable fingerprint of resolution-affecting kwargs.

        ``storage_options`` and ``deltacat_plan_kwargs`` are flattened
        to sorted item tuples; ``deltacat_client`` collapses to an
        ``id()`` so distinct client instances diverge but the same
        instance reused doesn't churn the cache.
        """
        opts = (
            tuple(sorted(storage_options.items()))
            if storage_options
            else ()
        )
        plan = (
            tuple(sorted(deltacat_plan_kwargs.items()))
            if deltacat_plan_kwargs
            else ()
        )
        client_id = id(deltacat_client) if deltacat_client is not None else None
        return (opts, client_id, plan)

    @classmethod
    def _cache_get(
        cls,
        uri: str,
        fingerprint: tuple[Any, ...],
    ) -> ResolvedSource | None:
        """Return a cached resolve result if still fresh and options match.

        On options-fingerprint mismatch, drops the stale entry and warns —
        differing options for the same URI is almost always a caller bug
        worth surfacing.
        """
        with cls._plan_cache_lock:
            entry = cls._plan_cache.get(uri)
            if entry is None:
                return None
            resolved, expiry, cached_fingerprint = entry
            if _time.monotonic() > expiry:
                del cls._plan_cache[uri]
                return None
            if cached_fingerprint != fingerprint:
                del cls._plan_cache[uri]
                logger.warning(
                    "DeltaCAT plan cache: options mismatch for %s — "
                    "re-resolving.  Same URI seen with different "
                    "storage_options / deltacat_client / plan_kwargs; "
                    "almost always a caller bug.  Cached=%r new=%r",
                    uri,
                    cached_fingerprint,
                    fingerprint,
                )
                return None
            return resolved

    @classmethod
    def _cache_put(
        cls,
        uri: str,
        resolved: ResolvedSource,
        fingerprint: tuple[Any, ...],
    ) -> None:
        with cls._plan_cache_lock:
            cls._plan_cache[uri] = (
                resolved,
                _time.monotonic() + cls.PLAN_CACHE_TTL_S,
                fingerprint,
            )

    @classmethod
    def invalidate_cache(cls, uri: str | None = None) -> None:
        """Drop cached plan results.  If *uri* is ``None``, clear everything."""
        with cls._plan_cache_lock:
            if uri is None:
                cls._plan_cache.clear()
            else:
                cls._plan_cache.pop(uri, None)

    @classmethod
    def read_companion_table(
        cls,
        resolved: ResolvedSource,
        name: str,
        *,
        columns: list[str] | None = None,
        deltacat_client: Any | None = None,
    ) -> Any | None:
        """Read the application ``_episodes`` sibling table for *resolved*.

        Translates ``name="episodes"`` to a read of the application-
        managed ``<base>_episodes`` table — the steps-table name has
        its trailing ``_steps`` replaced with ``_episodes``. Returns
        ``None`` when the table is not present, the client is
        unavailable, or the read fails.

        The DeltaCAT-auto-maintained ``<steps_table>__episodes``
        companion (frozen schema, double-underscore suffix) is
        intentionally not used here — readers should consult the app
        ``_episodes`` table for length / tasks / vendor / etc.
        """
        if resolved.source_key != cls.source_key:
            return None
        if name != "episodes":
            raise ValueError(f"Unsupported companion table name: {name!r}")

        namespace = resolved.metadata.get("namespace")
        steps_table = resolved.metadata.get("table")
        catalog = resolved.metadata.get("catalog")
        if not namespace or not steps_table:
            return None

        # Lazy import to avoid a circular import at module load time.
        from groot.core.data.packds.deltacat_dataset import (
            app_episodes_table_name,
        )
        try:
            companion = app_episodes_table_name(steps_table)
        except ValueError:
            # Steps-table name doesn't end with _steps — older / non-PackDS
            # tables. The app companion convention only applies to the
            # _steps-suffixed shape; nothing to read.
            return None

        if deltacat_client is None:
            deltacat_client = cls._client_from_env(catalog)
        if deltacat_client is None:
            return None

        cat = deltacat_client.catalog
        if not cat.table_exists(namespace=namespace, table=companion):
            return None
        return cat.read(
            namespace=namespace,
            table=companion,
            columns=columns,
            read_as="pyarrow",
        )

    @classmethod
    def resolve(
        cls,
        uri: str,
        *,
        storage_options: dict[str, str] | None = None,
        deltacat_client: Any | None = None,
        deltacat_plan_kwargs: Mapping[str, Any] | None = None,
    ) -> ResolvedSource:
        # Fast path: return cached result if still fresh AND the resolution
        # kwargs match what was cached.
        fingerprint = cls._fingerprint(
            storage_options, deltacat_client, deltacat_plan_kwargs,
        )
        cached = cls._cache_get(uri, fingerprint)
        if cached is not None:
            logger.debug("plan cache hit for %s", uri)
            return cached

        catalog, namespace, table, table_version = cls._parse_table_locator(uri)
        if deltacat_client is None:
            deltacat_client = cls._client_from_env(catalog)
            if deltacat_client is None:
                raise ValueError(
                    f"Resolving {uri!r} requires DELTACAT_SERVER_URL in the "
                    f"environment, or for {catalog!r} to be registered via "
                    f"register_deltacat_catalog(...), or an explicit "
                    f"deltacat_client=... argument."
                )
        plan_kwargs = dict(deltacat_plan_kwargs or {})
        plan_kwargs.setdefault("include_files", False)
        # Resolution is metadata-only (no episode_id filter is meaningful
        # at this layer — we're discovering the table itself, not reading
        # rows). The v5 read-scope guard rejects unscoped PackDS plans by
        # default; opt out for this metadata call. Per-episode reads still
        # push their filter through cat.read(filter_predicate=...).
        plan_kwargs.setdefault("allow_full_packds_scan", True)
        if table_version is not None:
            plan_kwargs.setdefault("table_version", table_version)
        # The catalog component from the dc:// URI identifies the server
        # (not a kwarg to plan()). Pass namespace and table only.
        plan = deltacat_client.catalog.plan(
            namespace=namespace,
            table=table,
            **plan_kwargs,
        )

        # In v5, ``packds_uri`` is no longer an authoritative plan field —
        # the reader accesses data through ``cat.read(filter_predicate=...)``
        # rather than opening a single Lance root. Fall back to the dc://
        # URI itself when the plan doesn't surface a packds_uri so downstream
        # callers that print or log the path get something useful.
        packds_uri = getattr(plan, "packds_uri", None) or uri

        resolved_storage_options = storage_options or cls._plan_storage_options(plan)
        metadata = {
            "catalog": catalog,
            "namespace": namespace,
            "table": table,
            "table_version": table_version,
            "snapshot_id": getattr(plan, "snapshot_id", None),
            "episode_index": getattr(plan, "episode_index", None),
            "raw_plan": getattr(plan, "raw", None),
        }
        resolved = ResolvedSource(
            source_key=cls.source_key,
            original_uri=uri,
            packds_uri=str(packds_uri),
            storage_options=resolved_storage_options,
            metadata=metadata,
        )
        cls._cache_put(uri, resolved, fingerprint)
        logger.debug("plan cache miss for %s, cached for %.0fs", uri, cls.PLAN_CACHE_TTL_S)
        return resolved


SOURCE_RESOLVERS: dict[str, type[PackDSSourceResolver]] = {
    LocalSourceResolver.source_key: LocalSourceResolver,
    DirectRemoteSourceResolver.source_key: DirectRemoteSourceResolver,
    DeltaCatSourceResolver.source_key: DeltaCatSourceResolver,
}


def resolve_source(
    uri: str | Path,
    *,
    storage_options: dict[str, str] | None = None,
    deltacat_client: Any | None = None,
    deltacat_plan_kwargs: Mapping[str, Any] | None = None,
) -> ResolvedSource:
    """Resolve a user-provided PackDS locator into a concrete PackDS root."""
    uri_str = str(uri)
    if uri_str.startswith("dc://"):
        raise NotImplementedError(
            "PackDS dc:// DeltaCAT locators are not supported on this branch. "
            "Use a local path or s3://... .packds URI instead."
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
