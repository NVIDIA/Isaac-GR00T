# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""DataSourceResolver: resolve local dataset paths to the best available URI.

Resolution order per path:
  0. **Whitelist gate** (local paths only): if ``path_mapping.yaml`` defines a
     non-empty ``whitelist:`` list, the path must start with one of those prefixes
     to proceed.  Paths that do not match are returned unchanged (LustreFS
     fallback) regardless of whether credentials or mappings exist.  An absent or
     empty ``whitelist:`` skips this gate and allows all paths through.
  1. Paths already starting with ``s3://`` — URI is returned unchanged; each
     tier is probed via ``HeadBucket`` in order to discover which credentials
     profile can access the bucket.  Raises ``ValueError`` if no tier succeeds.
  2. **Static mappings** (loaded from ``path_mapping.yaml`` and merged into every
     tier): each tier evaluates the same mapping table with its own credentials and
     endpoint.  Longest prefix wins.  After applying a static mapping the bucket is
     verified via ``HeadBucket``; if inaccessible, resolution continues.
  3. **Auto-derive**: if the tier has a region configured, derive an S3 bucket name
     from the local path using the deterministic scheme described below and check
     whether the bucket exists.
  4. **LustreFS fallback**: return the original local path unchanged.

Tiers are tried in order.  The first tier that returns a reachable URI wins.

This module also contains the canonical bucket-naming primitives so all
path-resolution logic lives in one place:

  - ``POOL_TO_REGION``        — pool → AWS region mapping
  - ``path_to_bucket_name``   — derive bucket name from local path
  - ``bucket_exists``         — HeadBucket check (re-raises unexpected errors)
  - ``ancestor_paths``        — parent paths longest-first

Usage::

    from groot.core.data.hydrator.data_source_resolver import (
        DataSourceConfig,
        TierConfig,
        DataSourceResolver,
    )

    config = DataSourceConfig(
        tiers=[
            TierConfig(name="aws", profile="aws", region="us-west-2"),
            TierConfig(name="pdx", profile="default"),
        ]
    )
    resolver = DataSourceResolver(config)
    uri, profile = resolver.resolve("/shared/GrootDatasets/my_dataset")
    # → ("s3://example-bucket/my_dataset", "aws")  (auto-derived)
    # → ("/shared/GrootDatasets/my_dataset", None)                              (fallback)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Literal

import boto3
from botocore import exceptions

from groot.core.data.hydrator.utils import aws as aws_utils
from groot.core.utils.distributed import worker_logging

log = worker_logging.get_unified_logger()

_DEFAULT_MAPPING_YAML = Path(__file__).parent / "migration" / "path_mapping.yaml"


def _whitelist_match(path: str, entry: str) -> bool:
    """Return True if *path* is covered by the whitelist *entry*.

    Entries without wildcards match the exact path or any path under it
    (prefix + ``/``).  Entries containing ``*``, ``?``, or ``[`` are treated
    as fnmatch globs — ``*`` matches across ``/``, so ``agibot-custom*``
    covers ``agibot-custom-converted-0829-fullres/episode_001`` as well.
    """
    if "*" in entry or "?" in entry or "[" in entry:
        import fnmatch

        return fnmatch.fnmatch(path, entry) or fnmatch.fnmatch(path, entry.rstrip("/") + "/*")
    return path == entry or path.startswith(entry.rstrip("/") + "/")


# ---------------------------------------------------------------------------
# Bucket-naming primitives
# ---------------------------------------------------------------------------


def _bucket_name_suffix(region: str) -> str:
    return f"-{region}-nvidia-gear"


def path_to_bucket_name(local_path: str, region: str) -> str:
    """Derive a bucket name from a local path and region.

    Segments are joined with hyphens; underscores within segments are also
    replaced with hyphens (AWS S3 does not allow underscores in bucket names).

    Args:
        local_path: Local directory path (e.g. ``"/shared/GrootDatasets"``).
        region: AWS region string (e.g. ``"us-west-2"``).

    Returns:
        Derived bucket name.

    Example::

        path_to_bucket_name("/shared/GrootDatasets", "us-west-2")
        → "shared-grootdatasets-us-west-2-nvidia-gear"

        path_to_bucket_name("/A_B/C/D", "us-east-1")
        → "a-b-c-d-us-east-1-nvidia-gear"
    """
    segments = [s.replace("_", "-") for s in local_path.split("/") if s]
    return "-".join(segments).lower() + _bucket_name_suffix(region)


def bucket_exists(s3_client: boto3.client, bucket_name: str) -> bool:
    """Return True if *bucket_name* exists and is accessible via *s3_client*.

    Re-raises unexpected ``ClientError`` codes (anything other than 404, 403, or
    301) so callers that care about errors (e.g. provisioning tools) receive them.

    Args:
        s3_client: Boto3 S3 client.
        bucket_name: Bucket name to check.

    Returns:
        True if the bucket is accessible, False for 404/403/301 responses.

    Raises:
        exceptions.ClientError: For unexpected error codes.
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        log.debug("bucket_exists %s → EXISTS", bucket_name)
        return True
    except exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        log.debug("bucket_exists %s → code=%s", bucket_name, code)
        if code in ("404", "NoSuchBucket", "403"):
            return False
        if code == "301":
            log.debug(
                "bucket_exists %s: exists in a different region (301) — skipping", bucket_name
            )
            return False
        raise


def ancestor_paths(local_path: str) -> list[str]:
    """Return all ancestor paths from longest to shortest (excluding the path itself).

    Args:
        local_path: Local path string (e.g. ``"/a/b/c/d"``).

    Returns:
        List of ancestor paths, longest first.
        Example: ``"/a/b/c/d"`` → ``["/a/b/c", "/a/b", "/a"]``
    """
    segments = [s for s in local_path.split("/") if s]
    return ["/" + "/".join(segments[:i]) for i in range(len(segments) - 1, 0, -1)]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TierConfig:
    """Configuration for a single resolution tier (e.g. AWS S3 or PDX/Ceph).

    Attributes:
        name: Human-readable tier name used in log messages (e.g. ``"aws"``, ``"pdx"``).
        profile: Gear credentials profile (e.g. ``"aws"`` or ``"default"``).
        region: AWS region for auto-derive (e.g. ``"us-west-2"``).  Pass ``None``
            to skip auto-derive for this tier; static mappings still apply.
        mappings: Static prefix-to-URI mappings.  Keys are local path prefixes;
            values are ``s3://`` URIs.  Longest prefix wins.
        endpoint_style: How the S3 endpoint URL is selected when building a boto3
            client for this tier.

            - ``"regional"`` *(default)*: endpoint is derived from *region* as
              ``https://s3.{region}.amazonaws.com``.  Correct for AWS S3 where
              each bucket lives in a specific regional endpoint.
            - ``"profile"``: endpoint (if any) is taken from the credential
              profile as-is — e.g. a custom PDX/Ceph URL stored in
              ``~/.gear/data_credentials``.  *region* may still be set for
              bucket-name derivation but does **not** influence the endpoint.
    """

    name: str
    profile: str
    region: str | None = None
    mappings: dict[str, str] = dataclasses.field(default_factory=dict)
    endpoint_style: Literal["regional", "profile"] = "regional"


@dataclass
class DataSourceConfig:
    """Configuration for :class:`DataSourceResolver`.

    Attributes:
        tiers: Ordered list of resolution tiers.  Tried in order; first match wins.
        mapping_yaml: Path to a ``path_mapping.yaml`` file.  Static mappings from
            the file are loaded into the **first** tier.  Set to ``None`` to skip.
            Defaults to the repo-root ``path_mapping.yaml`` if it exists.
        disable_resolution: When ``True`` the resolver is a no-op (always returns
            the original path).  Mirrors ``disable_path_mapping`` in legacy YAML.
    """

    tiers: list[TierConfig] = dataclasses.field(default_factory=list)
    mapping_yaml: str | None = None
    disable_resolution: bool = False


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class DataSourceResolver:
    """Resolve local dataset paths to the best available URI.

    See module docstring for full resolution order.

    Args:
        config: Resolver configuration.  If ``config.mapping_yaml`` is set (or
            defaults to repo-root ``path_mapping.yaml``), its ``mappings:`` section
            is merged into **every** tier's static mappings (so each tier can
            evaluate the same prefixes with its own credentials and endpoint), and
            its ``whitelist:`` section gates which local paths are eligible for
            remote resolution at all.
    """

    def __init__(self, config: DataSourceConfig | None = None) -> None:
        self._config = config or DataSourceConfig()
        self._tiers: list[_Tier] = []
        self._disabled = self._config.disable_resolution
        self._loaded = False
        self._active_profile: str | None = None
        # Maps S3 bucket name → the credentials profile that can access it.
        # Populated lazily as resolve() succeeds for each bucket.
        self._bucket_profile_map: dict[str, str] = {}
        # Local path prefixes eligible for remote resolution (local→S3 only).
        # Empty list = no restriction. Never consulted for explicit s3:// inputs.
        self._whitelist: list[str] = []
        # Pool → AWS region mapping loaded from path_mapping.yaml.
        self._pool_to_region: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, path: str) -> tuple[str, str | None]:
        """Return ``(uri, profile)`` for *path*.

        - **Local paths**: tried against each tier in order; the winning tier's
          profile is returned alongside the URI.  Falls back to the original
          path (LustreFS) if no tier matches.
        - **S3 paths** (``s3://``): the URI is returned unchanged; the whitelist
          is **not** applied (it only gates local paths).  Each tier is probed via
          ``HeadBucket`` in order to discover which credentials profile can access
          the bucket.  The result is cached in ``_bucket_profile_map`` so subsequent
          :meth:`get_profile_for_uri` calls are O(1) without re-probing.

        Args:
            path: Dataset root path (local or ``s3://``).

        Returns:
            ``(uri, profile)`` where *uri* is the best available URI and
            *profile* is the credentials profile name needed to access it, or
            ``None`` when no tier is reachable (local fallback or unknown
            bucket).
        """
        self._ensure_loaded()

        if path.startswith("s3://"):
            profile = None if self._disabled else self._probe_s3_profile(path)
            return (path, profile)

        if self._disabled:
            log.debug("DataSourceResolver: resolution disabled, returning %s unchanged", path)
            return (path, None)

        if self._whitelist and not any(_whitelist_match(path, entry) for entry in self._whitelist):
            log.info("DataSourceResolver: %s not in whitelist, using local path", path)
            return (path, None)

        for tier in self._tiers:
            result = tier.resolve(path)
            if result is not None:
                uri, profile = result
                log.info("DataSourceResolver [%s]: %s → %s", tier.name, path, uri)
                # Record which profile can access this bucket for later URI lookups.
                bucket = uri[5:].split("/")[0]
                self._bucket_profile_map[bucket] = profile
                return (uri, profile)

        log.info("DataSourceResolver: no remote copy found for %s, using local path", path)
        return (path, None)

    def _probe_s3_profile(self, s3_uri: str) -> str:
        """Return the credentials profile that can access *s3_uri*, probing tiers if needed.

        Checks ``_bucket_profile_map`` first (O(1) cache hit).  On a miss,
        tries each tier's ``_bucket_accessible`` in order and caches the
        winner.  Raises ``ValueError`` if no tier can reach the bucket.

        Args:
            s3_uri: Any ``s3://`` URI — only the bucket name is examined.

        Returns:
            Credentials profile string of the first tier that can access the
            bucket.

        Raises:
            ValueError: If no tier can access the bucket (missing credentials,
                bucket does not exist, or no tiers configured).

        Note:
            The YAML ``whitelist`` does not apply here — it only restricts when a
            **local** path may be rewritten to S3; explicit ``s3://`` URIs are always
            probed.
        """
        bucket = s3_uri[5:].split("/")[0]
        if bucket in self._bucket_profile_map:
            return self._bucket_profile_map[bucket]
        for tier in self._tiers:
            if tier._bucket_accessible(s3_uri):  # noqa: SLF001
                log.info(
                    "DataSourceResolver: s3 probe bucket=%s → profile=%s (tier=%s)",
                    bucket,
                    tier._profile,  # noqa: SLF001
                    tier.name,
                )
                self._bucket_profile_map[bucket] = tier._profile  # noqa: SLF001
                return tier._profile  # noqa: SLF001
        raise ValueError(
            f"No configured tier can access s3://{bucket} — "
            "check that credentials are available for at least one tier "
            "(GEAR_DATA_CREDENTIALS env var or ~/.gear/data_credentials) "
            "and that the bucket exists."
        )

    def get_profile_for_uri(self, uri: str) -> str | None:
        """Return the credentials profile needed to access *uri*.

        Checks ``_bucket_profile_map`` first (O(1) cache hit).  On a miss,
        probes each tier via :meth:`_probe_s3_profile` so callers do not need
        to have called :meth:`resolve` first for the bucket to be known.
        Returns ``None`` for non-S3 URIs.

        Args:
            uri: S3 URI (e.g. ``s3://my-bucket/path/to/file``).

        Returns:
            Credentials profile string, or ``None`` for non-S3 paths.
            Propagates ``ValueError`` from :meth:`_probe_s3_profile` when no
            tier can access the bucket.

        Note:
            Like :meth:`resolve` for ``s3://`` inputs, this does **not** consult the
            local-path ``whitelist`` — that list only gates local→remote mapping.
        """
        if not uri.startswith("s3://"):
            return None
        bucket = uri[5:].split("/")[0]
        if bucket in self._bucket_profile_map:
            return self._bucket_profile_map[bucket]
        self._ensure_loaded()
        if self._disabled:
            return None
        return self._probe_s3_profile(uri)

    def get_tier_endpoint_style_for_profile(
        self, profile: str
    ) -> Literal["regional", "profile"] | None:
        """Return the tier ``endpoint_style`` for *profile*, or ``None``.

        Used by ``io.resolve_s3_region_and_endpoint_style`` so PDX-tier
        buckets (whose tier carries ``endpoint_style="profile"``) keep their
        custom endpoint URL instead of being redirected to AWS regional URLs.
        """
        self._ensure_loaded()
        for tier in self._tiers:
            if tier._profile == profile:  # noqa: SLF001
                return tier._endpoint_style  # noqa: SLF001
        return None

    def reload(self) -> None:
        """Force re-load of config and YAML mappings (useful in tests)."""
        self._tiers = []
        self._loaded = False
        self._active_profile = None
        self._bucket_profile_map = {}
        self._whitelist = []
        self._pool_to_region = {}

    def _get_active_profile(self) -> str | None:
        """Return the credentials profile of the first tier with available credentials (package-internal)."""
        self._ensure_loaded()
        return self._active_profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self) -> None:
        """Build _Tier objects from config, merging YAML mappings if present."""
        yaml_mappings: dict[str, str] = {}
        yaml_disable = False

        yaml_path_str = self._config.mapping_yaml
        if yaml_path_str is None and _DEFAULT_MAPPING_YAML.exists():
            yaml_path_str = str(_DEFAULT_MAPPING_YAML)

        if yaml_path_str:
            yaml_mappings, yaml_disable, yaml_whitelist, yaml_pool_to_region = (
                self._load_yaml_mappings(yaml_path_str)
            )
            self._whitelist = yaml_whitelist
            self._pool_to_region = yaml_pool_to_region

        if yaml_disable:
            self._disabled = True

        region = self._detect_region_from_pool()

        for i, tier_cfg in enumerate(self._config.tiers):
            # Merge YAML mappings into every tier so each tier can try the mapping
            # with its own credentials/endpoint (e.g. a PDX bucket unreachable via
            # the AWS tier will still be found by the PDX tier).
            # Per-tier TierConfig.mappings take precedence over the YAML entries.
            merged_mappings = (
                {**yaml_mappings, **tier_cfg.mappings} if yaml_mappings else dict(tier_cfg.mappings)
            )

            # If region is auto-detected and tier has no explicit region, fill in
            effective_region = tier_cfg.region
            if effective_region is None and i == 0 and region:
                effective_region = region

            log.info(
                "_load: tier=%s profile=%s region=%s mappings=%s",
                tier_cfg.name,
                tier_cfg.profile,
                effective_region,
                list(merged_mappings.keys()),
            )
            self._tiers.append(
                _Tier(
                    name=tier_cfg.name,
                    profile=tier_cfg.profile,
                    region=effective_region,
                    mappings=merged_mappings,
                    endpoint_style=tier_cfg.endpoint_style,
                )
            )

        # Determine which tier has working credentials; used by callers to configure backends.
        self._active_profile = None
        for tier in self._tiers:
            try:
                aws_utils.get_boto3_client_config(profile=tier._profile)  # noqa: SLF001
                self._active_profile = tier._profile  # noqa: SLF001
                log.info(
                    "_load: active credentials profile=%r (first tier with working credentials)",
                    self._active_profile,
                )
                break
            except (ValueError, FileNotFoundError):
                log.info("_load: profile=%r unavailable", tier._profile)  # noqa: SLF001

    @staticmethod
    def _load_yaml_mappings(
        yaml_path: str,
    ) -> tuple[dict[str, str], bool, list[str], dict[str, str]]:
        """Load static mappings, whitelist, and pool_to_region from a ``path_mapping.yaml`` file.

        Args:
            yaml_path: Absolute path to the YAML file.

        Returns:
            Tuple of (mappings dict, disable_flag, whitelist, pool_to_region).
            *whitelist* is a list of local path prefixes eligible for **local→remote**
            resolution only; an empty list means all local paths are eligible.
            Explicit ``s3://`` paths never use this list.
        """
        import yaml

        path = Path(yaml_path)
        if not path.exists():
            return {}, False, [], {}
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if data.get("disable_path_mapping"):
            return {}, True, [], {}
        raw: dict = data.get("mappings") or {}
        whitelist: list[str] = [entry.rstrip("/") for entry in (data.get("whitelist") or [])]
        pool_to_region: dict[str, str] = dict(data.get("pool_to_region") or {})
        return dict(raw), False, whitelist, pool_to_region

    def _detect_region_from_pool(self) -> str | None:
        """Detect the AWS region by resolving a pool name via ``pool_to_region`` in YAML.

        Pool name comes from:

        - ``WORKFLOW_ID`` env — query ``osmo task list`` for the workflow's pool.
        - ``CLUSTER_TYPE`` env — when ``WORKFLOW_ID`` is unset (Sky / gear jobs).

        Returns:
            AWS region string (e.g. ``"us-west-2"``), or ``None`` if unavailable.
        """
        pool: str | None = None
        workflow_id = os.environ.get("WORKFLOW_ID")
        if workflow_id:
            try:
                result = subprocess.run(
                    [
                        "osmo",
                        "task",
                        "list",
                        "--workflow-id",
                        workflow_id,
                        "-a",
                        "-t",
                        "json",
                        "--count",
                        "1",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                tasks = json.loads(result.stdout).get("tasks", [])
                if tasks:
                    pool = tasks[0].get("pool")
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        else:
            pool = os.environ.get("CLUSTER_TYPE")
        if not pool:
            return None
        region = self._pool_to_region.get(pool)
        if region:
            log.info("DataSourceResolver: auto-detected region=%s from pool=%s", region, pool)
        return region


# ---------------------------------------------------------------------------
# Internal tier implementation
# ---------------------------------------------------------------------------


class _Tier:
    """One resolution tier (e.g. AWS or PDX).

    Args:
        name: Tier name for logging.
        profile: Gear credentials profile.
        region: AWS/S3 region for auto-derive (``None`` = skip auto-derive).
        mappings: Static prefix→URI mapping (longest prefix wins).
        endpoint_style: ``"regional"`` to derive endpoint from region;
            ``"profile"`` to use the credential profile's endpoint as-is.
    """

    def __init__(
        self,
        name: str,
        profile: str,
        region: str | None,
        mappings: dict[str, str],
        endpoint_style: Literal["regional", "profile"] = "regional",
    ) -> None:
        self.name = name
        self._profile = profile
        self._region = region
        self._endpoint_style = endpoint_style
        # Sort longest prefix first for greedy matching.
        self._mappings: list[tuple[str, str]] = sorted(
            ((k.rstrip("/"), v.rstrip("/")) for k, v in mappings.items()),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        # Cache of auto-derive results: path → uri (empty string = not found).
        self._derive_cache: dict[str, str] = {}
        self._s3_client = None  # lazily created

    def resolve(self, path: str) -> tuple[str, str] | None:
        """Return ``(uri, profile)`` for *path*, or ``None`` if not found in this tier.

        Args:
            path: Local path to resolve.

        Returns:
            ``(s3:// URI, credentials profile)`` if found and accessible, else ``None``.
        """
        # 1. Static mapping
        for prefix, replacement in self._mappings:
            if path == prefix or path.startswith(prefix + "/"):
                suffix = path[len(prefix) :]
                candidate = replacement + suffix
                if self._bucket_accessible(candidate):
                    log.debug("[%s] static mapping: %s → %s", self.name, path, candidate)
                    return (candidate, self._profile)
                log.warning(
                    "[%s] static mapping matched %s → %s but bucket not accessible — "
                    "skipping further static rows; will try auto-derive if tier has region",
                    self.name,
                    path,
                    candidate,
                )
                break  # prefix matched but bucket not reachable; stop trying other static rows

        # 2. Auto-derive
        if not self._region:
            return None

        if path not in self._derive_cache:
            try:
                client = self._get_s3_client()
                result = _Tier._derive_s3_path(path, self._region, client)
            except Exception as e:  # noqa: BLE001
                # Broad catch: credentials missing (FileNotFoundError), client errors,
                # network errors, etc. — skip this tier and let the next one try.
                log.warning(
                    "[%s] error resolving %s: %s — skipping tier",
                    self.name,
                    path,
                    e,
                )
                result = None
            self._derive_cache[path] = result or ""

        derived = self._derive_cache[path]
        if derived:
            return (derived, self._profile)
        return None

    # ------------------------------------------------------------------

    def _get_s3_client(self):
        if self._s3_client is None:
            self._s3_client = _Tier._make_s3_client(
                self._region, self._profile, self._endpoint_style
            )
        return self._s3_client

    def _bucket_accessible(self, s3_uri: str) -> bool:
        """Return True if the bucket in *s3_uri* is reachable with this tier's credentials."""
        if not s3_uri.startswith("s3://"):
            return False
        bucket = s3_uri[5:].split("/")[0]
        try:
            client = self._get_s3_client()
            endpoint = getattr(client.meta, "endpoint_url", None)
            log.info(
                "[%s] HeadBucket %s (endpoint=%s region=%s profile=%s)",
                self.name,
                bucket,
                endpoint,
                self._region,
                self._profile,
            )
            client.head_bucket(Bucket=bucket)
            log.info("[%s] HeadBucket %s → OK", self.name, bucket)
            return True
        except exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            log.warning(
                "[%s] HeadBucket %s → ClientError code=%s message=%s",
                self.name,
                bucket,
                code,
                e.response["Error"].get("Message", ""),
            )
            return False
        except exceptions.EndpointResolutionError as e:
            log.warning("[%s] HeadBucket %s → EndpointResolutionError: %s", self.name, bucket, e)
            return False
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[%s] HeadBucket %s → unexpected %s: %s",
                self.name,
                bucket,
                type(e).__name__,
                e,
            )
            return False

    @staticmethod
    def _make_s3_client(
        region: str | None,
        profile: str,
        endpoint_style: Literal["regional", "profile"] = "regional",
    ) -> boto3.client:
        """Build a boto3 S3 client using *profile* credentials.

        Delegates to ``aws_utils.make_s3_client``.

        Args:
            region: AWS region (e.g. ``"us-west-2"``), or ``None``.
            profile: Gear credentials profile name.
            endpoint_style: ``"regional"`` to derive endpoint from region;
                ``"profile"`` to use the credential profile's endpoint as-is.

        Returns:
            Configured boto3 S3 client.
        """
        return aws_utils.make_s3_client(
            profile=profile, region=region, endpoint_style=endpoint_style
        )

    @staticmethod
    def _apply_region_suffix(s3_uri: str, region: str) -> str:
        """Append ``-{region}-nvidia-gear`` to the bucket part of *s3_uri*.

        Args:
            s3_uri: Full S3 URI (``s3://bucket/key``).
            region: AWS region string (e.g. ``"us-west-2"``).

        Returns:
            S3 URI with region suffix appended to the bucket name.
        """
        without_scheme = s3_uri[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        full_bucket = bucket + _bucket_name_suffix(region)
        return f"s3://{full_bucket}/{key}" if key else f"s3://{full_bucket}"

    @staticmethod
    def _bucket_exists(s3_client: boto3.client, bucket_name: str) -> bool:
        """Return True if *bucket_name* is reachable (catch-all, never raises).

        Wraps :func:`bucket_exists` and silences all unexpected errors.

        Args:
            s3_client: Boto3 S3 client.
            bucket_name: Bucket name to check.

        Returns:
            True if the bucket is accessible, False otherwise.
        """
        try:
            return bucket_exists(s3_client, bucket_name)
        except Exception as e:  # noqa: BLE001
            log.debug("_bucket_exists %s unexpected error: %s", bucket_name, e)
            return False

    @staticmethod
    def _prefix_exists(s3_client: boto3.client, bucket_name: str, key_prefix: str) -> bool:
        """Return True if *key_prefix* has at least one object in *bucket_name*.

        Args:
            s3_client: Boto3 S3 client.
            bucket_name: Bucket name to query.
            key_prefix: Key prefix to check for objects.

        Returns:
            True if at least one object exists under the prefix, False otherwise.
        """
        prefix = key_prefix.rstrip("/") + "/"
        try:
            resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
            return bool(resp.get("Contents") or resp.get("CommonPrefixes"))
        except exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            log.debug("_prefix_exists %s/%s → %s", bucket_name, prefix, code)
            return False

    @staticmethod
    def _derive_s3_path(local_path: str, region: str, s3_client: boto3.client) -> str | None:
        """Derive S3 URI for *local_path* if a matching bucket exists.

        Checks exact bucket match first, then walks up ancestor paths.

        Args:
            local_path: Local path (e.g. ``"/shared/GrootDatasets/foo"``).
            region: AWS region to include in the derived bucket name.
            s3_client: Pre-built boto3 S3 client with appropriate credentials.

        Returns:
            Full S3 URI on success, or ``None`` if no matching bucket found.
        """
        # 1. Exact bucket match
        exact = path_to_bucket_name(local_path, region)
        if _Tier._bucket_exists(s3_client, exact):
            return f"s3://{exact}"

        # 2. Parent bucket match — verify the prefix has been synced.
        for anc in ancestor_paths(local_path):
            candidate = path_to_bucket_name(anc, region)
            if _Tier._bucket_exists(s3_client, candidate):
                key_prefix = os.path.relpath(local_path, anc)
                if _Tier._prefix_exists(s3_client, candidate, key_prefix):
                    return f"s3://{candidate}/{key_prefix}"
                log.debug(
                    "_derive_s3_path: bucket %s exists but prefix %s not synced",
                    candidate,
                    key_prefix,
                )
                return None

        return None


# ---------------------------------------------------------------------------
# Default resolver (module-level singleton)
# ---------------------------------------------------------------------------

_default_resolver: DataSourceResolver | None = None


def default_resolver() -> DataSourceResolver:
    """Return the shared default :class:`DataSourceResolver`.

    Configured with two tiers: AWS S3 (``profile="aws"``) and PDX/Ceph
    (``profile="default"``).  Region is auto-detected from ``WORKFLOW_ID`` for
    the AWS tier.  Static mappings are loaded from the repo-root
    ``path_mapping.yaml`` if it exists.

    The resolver is created once and reused across calls.  Call
    ``reset_default_resolver()`` to force recreation (useful in tests).

    Returns:
        Shared :class:`DataSourceResolver` instance.
    """
    global _default_resolver  # noqa: PLW0603
    if _default_resolver is None:
        _default_resolver = DataSourceResolver(
            DataSourceConfig(
                tiers=[
                    TierConfig(name="aws", profile="aws", endpoint_style="regional"),
                    TierConfig(name="pdx", profile="default", endpoint_style="profile"),
                ]
            )
        )
    return _default_resolver


def reset_default_resolver() -> None:
    """Reset the shared default resolver (useful in tests)."""
    global _default_resolver  # noqa: PLW0603
    _default_resolver = None
