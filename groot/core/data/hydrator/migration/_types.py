# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Data types and directory walk helpers shared across the migration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


_PATH_MAPPING_YAML = Path(__file__).parent / "path_mapping.yaml"


def load_pool_to_region(yaml_path: Path | None = None) -> dict[str, str]:
    """Return the pool → AWS region mapping from ``path_mapping.yaml``."""
    import yaml

    path = yaml_path or _PATH_MAPPING_YAML
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("pool_to_region") or {})

CONFLICT_SUFFIX = ".local_copy"
"""Suffix appended to S3 keys for files that already exist in S3 with a different size.

E.g. ``episode_000.parquet`` is uploaded as ``episode_000.parquet.local_copy``
so the original S3 object is left untouched.
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class WalkRequest:
    """Input to stage 1: one local path to walk."""

    local_path: str
    bucket: str
    depth: int
    key_prefix: str = ""
    """Sub-path prefix within the bucket (empty for exact-match buckets)."""


@dataclass
class SyncUnit:
    """One directory to sync as a single ``awscli s3 sync`` invocation."""

    local_dir: str
    """Absolute local directory path."""
    s3_prefix: str
    """Destination S3 prefix (without trailing slash), e.g. ``s3://bucket/a/b``."""


@dataclass
class DryRunRequest:
    """One pair of directories to compare via ``aws s3 sync --dryrun``.

    Top-level requests (emitted by the caller) have ``orig_left`` and
    ``orig_right`` equal to ``left`` and ``right``.  Sub-requests fanned out
    by :class:`WalkPairsWorker` carry the original pair for grouping.
    """

    left: str
    """Source directory (may be empty string when sub-dir exists only in right)."""
    right: str
    """Destination directory."""
    orig_left: str = ""
    """Top-level left directory this sub-pair belongs to."""
    orig_right: str = ""
    """Top-level right directory this sub-pair belongs to."""
    note: str = ""
    """``only_in_left`` | ``only_in_right`` | empty string for common sub-dirs."""


# ---------------------------------------------------------------------------
# Directory walk helpers
# ---------------------------------------------------------------------------


def walk_to_depth(
    local_path: str, bucket: str, depth: int = 1, key_prefix: str = ""
) -> list[SyncUnit]:
    """Walk *local_path* to *depth* levels and return a list of sync units.

    Each sub-directory found at the target depth becomes a sync unit.
    If the tree is shallower than *depth*, the deepest reachable directories
    are returned instead.

    The ``local_path`` itself maps to ``s3://<bucket>/``.  A subdirectory
    ``local_path/foo/bar`` maps to ``s3://<bucket>/foo/bar``.

    Args:
        local_path: Absolute local directory to copy.
        bucket: Destination S3 bucket name (no ``s3://`` prefix).
        depth: How many directory levels to descend before treating a dir as
            one sync unit.  ``depth=1`` means immediate children of
            *local_path*; ``depth=0`` means the root itself.
        key_prefix: Sub-path prefix within the bucket (empty for exact-match buckets).

    Returns:
        List of :class:`SyncUnit` instances (never empty if *local_path* exists).
    """
    local_path = os.path.normpath(local_path)

    def _s3_prefix(abs_path: str) -> str:
        parts = ([key_prefix] if key_prefix else []) + (
            [os.path.relpath(abs_path, local_path)]
            if os.path.relpath(abs_path, local_path) != "."
            else []
        )
        if parts:
            return f"s3://{bucket}/" + "/".join(parts)
        return f"s3://{bucket}"

    def _recurse(current: str, remaining: int) -> list[SyncUnit]:
        if remaining == 0:
            return [SyncUnit(local_dir=current, s3_prefix=_s3_prefix(current))]
        try:
            entries = [
                e
                for e in os.scandir(current)
                if e.is_dir(follow_symlinks=False) and not e.is_symlink()
            ]
        except PermissionError:
            return [SyncUnit(local_dir=current, s3_prefix=_s3_prefix(current))]

        if not entries:
            # Leaf directory reached before target depth
            return [SyncUnit(local_dir=current, s3_prefix=_s3_prefix(current))]

        units: list[SyncUnit] = []
        for entry in entries:
            units.extend(_recurse(entry.path, remaining - 1))
        return units

    return _recurse(local_path, depth)


def _walk_subdirs(base: str, depth: int) -> dict[str, str]:
    """Walk *base* to *depth* directory levels and return ``{rel_path: abs_path}``.

    Relative paths use ``/`` as separator and are relative to *base*.
    The base itself is represented as ``"."`` when it is a leaf (depth=0 or
    no sub-directories exist).  If *base* does not exist or is not a
    directory, an empty dict is returned.

    Args:
        base: Root directory to walk.
        depth: Maximum number of directory levels to descend.

    Returns:
        Mapping of relative path → absolute path for each leaf directory found.
    """
    base = os.path.normpath(base)
    if not os.path.isdir(base):
        return {}

    result: dict[str, str] = {}

    def _recurse(current: str, remaining: int) -> None:
        rel = os.path.relpath(current, base)
        if remaining == 0:
            result[rel] = current
            return
        try:
            entries = [
                e
                for e in os.scandir(current)
                if e.is_dir(follow_symlinks=False) and not e.is_symlink()
            ]
        except (PermissionError, FileNotFoundError):
            result[rel] = current
            return
        if not entries:
            result[rel] = current
            return
        for entry in sorted(entries, key=lambda e: e.name):
            _recurse(entry.path, remaining - 1)

    _recurse(base, depth)
    return result


# ---------------------------------------------------------------------------
# Disjoint prefix guard (utility — not called in pipeline mode)
# ---------------------------------------------------------------------------


def _assert_disjoint(units: list[SyncUnit]) -> None:
    """Raise ValueError if any S3 prefix is a prefix of another.

    Overlapping prefixes would cause double-uploading of the same objects.
    This is a standalone utility for pre-flight validation; the pipeline does
    not call it automatically since uploads start before all units are known.
    """
    prefixes = sorted(u.s3_prefix for u in units)
    for i in range(len(prefixes) - 1):
        a = prefixes[i].rstrip("/") + "/"
        b = prefixes[i + 1]
        if b.startswith(a):
            raise ValueError(
                f"Overlapping S3 prefixes detected: '{prefixes[i]}' is a prefix of '{prefixes[i + 1]}'. "
                "This would cause double-uploads. Reduce --depth or fix the source tree."
            )
