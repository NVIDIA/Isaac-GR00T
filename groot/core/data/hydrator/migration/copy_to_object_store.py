# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""Distributed data copier from local filesystem to S3.

Subcommands
-----------

sync
    Copy one or more local directories to S3 via a two-stage xenna Pipeline.

    Stage 1 — WalkWorker
      Receives a WalkRequest (local path + target bucket + depth).
      Walks the source tree to *depth* levels and returns a list of SyncUnit
      objects.  The Pipeline fans these out automatically into stage 2.

    Stage 2 — SyncWorker
      Receives a SyncUnit and runs ``aws s3 sync`` for that one directory.
      Returns a result dict.

    The two stages run concurrently with backpressure via bounded queues:
    directory listing for later paths overlaps with uploading for earlier ones.

    S3 is a flat key-value store, so syncing disjoint prefixes in parallel has
    no race conditions (there are no real directories to create).

safe_sync
    Upload local directories to S3 treating the right (S3) side as ground truth.
    Uses the same two-stage pipeline as *sync* (WalkWorker → SafeSyncWorker).

    Per sync unit the worker:

    1. Runs ``aws s3 sync --dryrun --size-only`` to discover files that differ.
    2. Lists existing S3 objects (via boto3) to classify each differing file as
       *new* (not yet in S3) or *conflicting* (different size).
    3. Uploads **new** files normally via ``aws s3 sync``.
    4. Uploads **conflicting** files as ``<s3_key>.local_copy`` via ``aws s3 cp``;
       the original S3 object is left untouched.

    Files with the same size on both sides are skipped (treated as identical).

dryrun
    Compare pairs of local directories without transferring any data using a
    two-stage xenna Pipeline:

    Stage 1 — WalkPairsWorker
      Receives a top-level DryRunRequest (left, right).  Walks both directories
      to *depth* levels and fans out one DryRunRequest per matched sub-directory
      pair, including sub-dirs that exist only on one side.

    Stage 2 — DryRunWorker
      Receives a DryRunRequest for a sub-directory pair and runs
      ``aws s3 sync <left> <right> --dryrun`` to capture what would differ.

    After the pipeline drains, results are grouped by original pair and an
    aggregate JSON plus a human-readable ``summary.txt`` are written.

    The pairs file contains one pair per line: ``<left_dir>  <right_dir>``.
    Blank lines and lines starting with ``#`` are ignored.

Usage::

    # sync — single path
    python copy_to_object_store.py sync --paths /mnt/aws-lfs-02/GrootDatasets \\
                                        --pool groot-gb200-02

    # sync — multiple paths (walked concurrently with uploads)
    python copy_to_object_store.py sync \\
        --paths /mnt/aws-lfs-02/DatasetA /mnt/aws-lfs-02/DatasetB \\
        --region us-east-1 --depth 2 --workers 16 \\
        --s3-concurrency 32 --rate-limit-mbps 500

    # safe_sync — upload preserving existing S3 files as ground truth
    python copy_to_object_store.py safe_sync \\
        --paths /mnt/aws-lfs-02/GrootDatasets --pool groot-gb200-02

    # dryrun — diff pairs of local directory trees in parallel (3 levels deep)
    python copy_to_object_store.py dryrun \\
        --pairs-file pairs.txt --output-dir /tmp/diff_reports --depth 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import re
import sys

from groot.core.utils.distributed import worker_logging
from groot.core.data.hydrator import data_source_resolver
from groot.core.data.hydrator.migration import _types, _workers, provision_object_store
from groot.core.utils.distributed import xenna

log = worker_logging.get_unified_logger()

# Re-export all public symbols so existing callers/tests that do
#   ``from ... import copy_to_object_store; copy_to_object_store.WalkRequest``
# continue to work without modification.
CONFLICT_SUFFIX = _types.CONFLICT_SUFFIX
WalkRequest = _types.WalkRequest
SyncUnit = _types.SyncUnit
DryRunRequest = _types.DryRunRequest
walk_to_depth = _types.walk_to_depth
_walk_subdirs = _types._walk_subdirs  # noqa: SLF001
_assert_disjoint = _types._assert_disjoint  # noqa: SLF001
WalkWorker = _workers.WalkWorker
SyncWorker = _workers.SyncWorker
SafeSyncWorker = _workers.SafeSyncWorker
WalkPairsWorker = _workers.WalkPairsWorker
DryRunWorker = _workers.DryRunWorker
_make_s3_client = _workers._make_s3_client  # noqa: SLF001
_parse_s3_uri = _workers._parse_s3_uri  # noqa: SLF001
_parse_dryrun_files = _workers._parse_dryrun_files  # noqa: SLF001


# ---------------------------------------------------------------------------
# Orchestration: shared helpers
# ---------------------------------------------------------------------------


def _safe_name(path: str) -> str:
    """Convert an absolute path to a filename-safe string."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", path.strip("/"))


def _build_walk_requests(
    region: str,
    depth: int,
    local_paths: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    profile: str = "aws",
) -> list[WalkRequest]:
    """Build WalkRequest objects from either local paths or explicit pairs.

    Exactly one of *local_paths* or *pairs* must be supplied.

    * **local_paths** — target bucket is resolved via
      :func:`provision_object_store.resolve_bucket`.
    * **pairs** — each ``(local_dir, s3_uri)`` is used directly; the S3 URI
      is parsed into ``(bucket, key_prefix)``.

    Args:
        region: AWS region used for bucket resolution (local_paths mode only).
        depth: Walk depth passed into each :class:`WalkRequest`.
        local_paths: Local directories whose S3 target is auto-derived.
        pairs: Explicit ``[(local_dir, s3_uri), ...]`` pairs.
        profile: Gear credentials profile for bucket resolution (local_paths mode only).

    Returns:
        List of :class:`WalkRequest` objects ready to feed into the pipeline.

    Raises:
        ValueError: If neither or both of *local_paths* and *pairs* are given,
            or if the resulting list would be empty.
    """
    if bool(local_paths) == bool(pairs):
        raise ValueError("Exactly one of local_paths or pairs must be supplied.")

    if pairs:
        return [
            WalkRequest(local_path=left, bucket=bucket, key_prefix=key_prefix, depth=depth)
            for left, right in pairs
            for bucket, key_prefix in [_parse_s3_uri(right)]
        ]

    # Load static mappings from path_mapping.yaml (longest prefix first).
    _yaml_path = data_source_resolver._DEFAULT_MAPPING_YAML  # noqa: SLF001
    static_mappings: list[tuple[str, str]] = []
    if _yaml_path.exists():
        yaml_mappings, _, _, _ = data_source_resolver.DataSourceResolver._load_yaml_mappings(  # noqa: SLF001
            str(_yaml_path)
        )
        static_mappings = sorted(
            ((k.rstrip("/"), v.rstrip("/")) for k, v in yaml_mappings.items()),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )

    s3_client = _make_s3_client(region, profile=profile)
    results = []
    for p in local_paths:  # type: ignore[union-attr]
        # 1. Try static mapping first.
        mapped: tuple[str, str] | None = None
        for prefix, replacement in static_mappings:
            if p == prefix or p.startswith(prefix + "/"):
                suffix = p[len(prefix):]
                mapped = _parse_s3_uri(replacement + suffix)
                log.info("_build_walk_requests: static mapping %s → s3://%s/%s", p, *mapped)
                break
        # 2. Fall back to dynamic bucket resolution.
        if mapped is None:
            mapped = provision_object_store.resolve_bucket(p, region, s3_client=s3_client)
        bucket, key_prefix = mapped
        results.append(WalkRequest(local_path=p, bucket=bucket, key_prefix=key_prefix, depth=depth))
    return results


# ---------------------------------------------------------------------------
# Orchestration: sync
# ---------------------------------------------------------------------------


def copy_to_object_store(
    region: str,
    *,
    local_paths: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    profile: str = "aws",
    depth: int = 1,
    num_walk_workers: int = 4,
    num_sync_workers: int = 8,
    s3_concurrency: int = 10,
    rate_limit_mbps: int | None = None,
    delete: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Copy one or more local directory trees to S3 via a two-stage pipeline.

    Stage 1 (WalkWorker) walks each path concurrently and fans SyncUnits into
    stage 2.  Stage 2 (SyncWorker) uploads each unit via ``python -m awscli s3 sync``.
    The two stages run concurrently — directory listing for later paths
    overlaps with uploading for earlier ones.

    Exactly one of *local_paths* or *pairs* must be supplied:

    * **local_paths** — target S3 bucket is auto-derived via
      :func:`provision_object_store.resolve_bucket`.
    * **pairs** — each ``(local_dir, s3_uri)`` is used directly.

    Args:
        region: AWS region (used to resolve each target bucket).
        local_paths: One or more local directories to copy.  Mutually
            exclusive with *pairs*.
        pairs: Explicit ``[(local_dir, s3_uri), ...]`` pairs.  Mutually
            exclusive with *local_paths*.
        profile: Gear credentials profile to use for S3 access.
        depth: Directory depth at which to split work into sync units.
        num_walk_workers: Workers for the walk stage (stage 1).
        num_sync_workers: Workers for the upload stage (stage 2).
        s3_concurrency: ``AWS_MAX_CONCURRENT_REQUESTS`` per sync worker.
        rate_limit_mbps: Optional per-worker bandwidth cap in MB/s.
        delete: Pass ``--delete`` to ``awscli s3 sync``.
        dry_run: Pass ``--dryrun`` to ``awscli s3 sync``.

    Returns:
        List of result dicts (one per sync unit) with keys ``local_dir``,
        ``s3_prefix``, ``returncode``, and ``stderr``.
    """
    from groot.core.utils.distributed import ray_init

    walk_requests = _build_walk_requests(
        region, depth, local_paths=local_paths, pairs=pairs, profile=profile
    )

    log.info(
        "Starting pipeline: %d path(s), walk_workers=%d, sync_workers=%d, depth=%d",
        len(walk_requests),
        num_walk_workers,
        num_sync_workers,
        depth,
    )
    if dry_run:
        log.info("[dry-run] No files will be transferred.")

    ray_init.ray_init_maybe_new_cluster()

    pipeline = xenna.Pipeline(
        stages=[
            xenna.StageSpec(
                WalkWorker,
                resources=xenna.Resources(cpus=1),
                num_workers=num_walk_workers,
            ),
            xenna.StageSpec(
                SyncWorker,
                resources=xenna.Resources(cpus=1),
                num_workers=num_sync_workers,
                worker_kwargs={
                    "region": region,
                    "profile": profile,
                    "s3_concurrency": s3_concurrency,
                    "rate_limit_mbps": rate_limit_mbps,
                    "delete": delete,
                    "dry_run": dry_run,
                },
            ),
        ]
    )

    results: list[dict] = []
    failed: list[dict] = []

    for result in pipeline.run(walk_requests):
        results.append(result)
        if result["returncode"] != 0:
            failed.append(result)
            log.error(
                "[ERROR] %s → %s\n%s", result["local_dir"], result["s3_prefix"], result["stderr"]
            )
        else:
            log.info("[OK] %s → %s", result["local_dir"], result["s3_prefix"])

    if failed:
        log.error("%d sync unit(s) failed.", len(failed))
        sys.exit(1)

    return results


# ---------------------------------------------------------------------------
# Orchestration: safe_sync
# ---------------------------------------------------------------------------


def run_safe_sync(
    region: str,
    *,
    local_paths: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    output_dir: str | None = None,
    profile: str = "aws",
    depth: int = 1,
    num_walk_workers: int = 4,
    num_sync_workers: int = 8,
    s3_concurrency: int = 10,
    rate_limit_mbps: int | None = None,
) -> list[dict]:
    """Upload local directories to S3, treating the right (S3) side as ground truth.

    Unlike :func:`copy_to_object_store`, this function never overwrites an
    existing S3 object.  Files that differ from their S3 counterpart are
    uploaded as ``<s3_key>.local_copy`` so both versions coexist.

    Exactly one of *local_paths* or *pairs* must be supplied:

    * **local_paths** — target S3 bucket is auto-derived via
      :func:`provision_object_store.resolve_bucket`.
    * **pairs** — each ``(local_dir, s3_uri)`` pair is used directly; the S3
      URI is parsed to extract the bucket and key prefix.

    If *output_dir* is given, a ``report.json`` (full per-unit detail) and a
    human-readable ``summary.txt`` are written there after the pipeline drains.

    Args:
        region: AWS region for the S3 endpoints.
        local_paths: One or more local directories whose target bucket is
            auto-derived.  Mutually exclusive with *pairs*.
        pairs: Explicit ``[(local_dir, s3_uri), ...]`` pairs.  Mutually
            exclusive with *local_paths*.
        output_dir: Directory to write ``report.json`` and ``summary.txt`` to.
            If ``None``, no report files are written.
        profile: Gear credentials profile.
        depth: Directory depth at which to split into parallel sync units.
        num_walk_workers: Workers for the walk stage.
        num_sync_workers: Workers for the safe-sync stage.
        s3_concurrency: ``AWS_MAX_CONCURRENT_REQUESTS`` per sync worker.
        rate_limit_mbps: Optional per-worker bandwidth cap in MB/s.

    Returns:
        List of result dicts (one per sync unit) with keys ``local_dir``,
        ``s3_prefix``, ``new``, ``new_files``, ``conflict``, ``conflict_files``,
        ``errors``, and ``returncode``.
    """
    from groot.core.utils.distributed import ray_init

    walk_requests = _build_walk_requests(
        region, depth, local_paths=local_paths, pairs=pairs, profile=profile
    )

    log.info(
        "Starting safe_sync pipeline: %d path(s), walk_workers=%d, sync_workers=%d, depth=%d",
        len(walk_requests),
        num_walk_workers,
        num_sync_workers,
        depth,
    )
    log.info("Conflicting files will be uploaded with suffix '%s'", CONFLICT_SUFFIX)

    ray_init.ray_init_maybe_new_cluster()

    pipeline = xenna.Pipeline(
        stages=[
            xenna.StageSpec(
                WalkWorker,
                resources=xenna.Resources(cpus=1),
                num_workers=num_walk_workers,
            ),
            xenna.StageSpec(
                SafeSyncWorker,
                resources=xenna.Resources(cpus=1),
                num_workers=num_sync_workers,
                worker_kwargs={
                    "region": region,
                    "profile": profile,
                    "s3_concurrency": s3_concurrency,
                    "rate_limit_mbps": rate_limit_mbps,
                },
            ),
        ]
    )

    results: list[dict] = []
    failed: list[dict] = []

    for result in pipeline.run(walk_requests):
        results.append(result)
        if result["returncode"] != 0:
            failed.append(result)
            log.error(
                "[ERROR] %s → %s  errors=%s",
                result["local_dir"],
                result["s3_prefix"],
                result["errors"],
            )
        else:
            log.info(
                "[OK] %s → %s  new=%d conflict=%d",
                result["local_dir"],
                result["s3_prefix"],
                result["new"],
                result["conflict"],
            )

    # Write upload report if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump({"conflict_suffix": CONFLICT_SUFFIX, "units": results}, f, indent=2)

        summary_path = os.path.join(output_dir, "summary.txt")
        total_new = sum(r["new"] for r in results)
        total_conflict = sum(r["conflict"] for r in results)
        total_errors = sum(len(r["errors"]) for r in results)
        with open(summary_path, "w") as f:
            f.write("Safe Sync Upload Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Conflict suffix: {CONFLICT_SUFFIX}\n\n")

            for r in sorted(results, key=lambda x: x["local_dir"]):
                status = "ERROR" if r["returncode"] != 0 else "OK"
                f.write(f"[{status}] {r['local_dir']} → {r['s3_prefix']}\n")
                if r["new_files"]:
                    f.write(f"  Uploaded new ({r['new']}):\n")
                    f.writelines(f"    {fn}\n" for fn in r["new_files"])
                if r["conflict_files"]:
                    f.write(f"  Uploaded as {CONFLICT_SUFFIX} ({r['conflict']}):\n")
                    f.writelines(f"    {fn}\n" for fn in r["conflict_files"])
                if not r["new_files"] and not r["conflict_files"] and r["returncode"] == 0:
                    f.write("  All files identical — nothing uploaded.\n")
                f.writelines(f"  ERROR: {err}\n" for err in r["errors"])
                f.write("\n")

            f.write("-" * 70 + "\n")
            f.write(f"Total: {total_new} new uploaded, {total_conflict} conflict copies")
            if total_errors:
                f.write(f", {total_errors} error(s)")
            f.write("\n")

        log.info("Report written to '%s'", output_dir)
        log.info("Summary: %s", summary_path)

    if failed:
        log.error("%d sync unit(s) failed.", len(failed))
        sys.exit(1)

    return results


# ---------------------------------------------------------------------------
# Orchestration: dryrun comparison
# ---------------------------------------------------------------------------


def run_dryrun_comparison(
    pairs: list[tuple[str, str]],
    output_dir: str,
    *,
    profile: str = "aws",
    region: str | None = None,
    depth: int = 3,
    num_walk_workers: int = 4,
    num_dryrun_workers: int = 8,
) -> list[dict]:
    """Compare pairs of local directories in parallel via a two-stage pipeline.

    Stage 1 (:class:`WalkPairsWorker`) walks each pair to *depth* levels and
    fans out sub-directory pairs.  Stage 2 (:class:`DryRunWorker`) runs
    ``aws s3 sync --dryrun`` for each sub-pair in parallel.

    The pipeline drains naturally when all sub-pairs have been processed —
    same as the sync pipeline.  Grouping and report writing happen after the
    loop finishes, from the results collected by the orchestrator.

    Raises:
        ValueError: If ``pairs`` is empty.

    Args:
        pairs: List of (left_dir, right_dir) tuples to compare.
        output_dir: Directory where all reports are written.
        profile: Gear credentials profile passed to :class:`DryRunWorker`.
        region: AWS region used to derive the S3 endpoint URL for the dryrun
            workers.  ``None`` falls back to the profile's own endpoint.
        depth: Directory depth at which to split comparisons (default: 3).
        num_walk_workers: Workers for the walk stage (stage 1).
        num_dryrun_workers: Workers for the dryrun stage (stage 2).

    Returns:
        List of all sub-pair result dicts.
    """
    from groot.core.utils.distributed import ray_init

    if not pairs:
        raise ValueError("pairs must not be empty")

    os.makedirs(output_dir, exist_ok=True)

    requests = [
        DryRunRequest(left=left, right=right, orig_left=left, orig_right=right)
        for left, right in pairs
    ]

    log.info(
        "Starting dryrun comparison: %d top-level pair(s), depth=%d, walk_workers=%d, dryrun_workers=%d",
        len(requests),
        depth,
        num_walk_workers,
        num_dryrun_workers,
    )

    ray_init.ray_init_maybe_new_cluster()

    pipeline = xenna.Pipeline(
        stages=[
            xenna.StageSpec(
                WalkPairsWorker,
                resources=xenna.Resources(cpus=1),
                num_workers=num_walk_workers,
                worker_kwargs={"depth": depth},
            ),
            xenna.StageSpec(
                DryRunWorker,
                resources=xenna.Resources(cpus=1),
                num_workers=num_dryrun_workers,
                worker_kwargs={"profile": profile, "region": region},
            ),
        ]
    )

    # Collect all sub-pair results as the pipeline drains — same pattern as sync.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    failed: list[dict] = []

    for result in pipeline.run(requests):
        key = (result["orig_left"], result["orig_right"])
        grouped[key].append(result)
        if result["returncode"] != 0:
            failed.append(result)
            log.error("[ERROR] '%s' vs '%s'\n%s", result["left"], result["right"], result["stderr"])

    # Write per-pair aggregate JSON and global summary once the pipeline is done.
    for (orig_left, orig_right), sub_results in sorted(grouped.items()):
        agg_name = f"aggregate__{_safe_name(orig_left)}__vs__{_safe_name(orig_right)}.json"
        with open(os.path.join(output_dir, agg_name), "w") as f:
            json.dump(
                {"orig_left": orig_left, "orig_right": orig_right, "sub_pairs": sub_results},
                f,
                indent=2,
            )

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Dryrun comparison report\n")
        f.write("=" * 70 + "\n\n")

        for (orig_left, orig_right), sub_results in sorted(grouped.items()):
            only_left = [r for r in sub_results if r["note"] == "only_in_left"]
            only_right_subs = [r for r in sub_results if r["note"] == "only_in_right"]
            differs = [r for r in sub_results if not r["note"] and r["lines"]]
            identical = [r for r in sub_results if not r["note"] and not r["lines"]]
            errors = [r for r in sub_results if r["returncode"] != 0]
            total_diff_lines = sum(len(r["lines"]) for r in sub_results)

            f.write(f"LEFT:  {orig_left}\n")
            f.write(f"RIGHT: {orig_right}\n")
            f.write(f"  Sub-pairs total:  {len(sub_results)}\n")
            f.write(f"  Identical:        {len(identical)}\n")
            f.write(f"  Differs:          {len(differs)}  ({total_diff_lines} total diff lines)\n")
            f.write(f"  Only in left:     {len(only_left)}\n")
            f.write(f"  Only in right:    {len(only_right_subs)}\n")
            if errors:
                f.write(f"  Errors:           {len(errors)}\n")
            f.write("\n")

            for r in sorted(sub_results, key=lambda x: (x.get("left") or "", x.get("right") or "")):
                left_label = r["left"] or "(missing)"
                if r["note"]:
                    status = r["note"]
                elif not r["lines"]:
                    status = "IDENTICAL"
                else:
                    status = f"{len(r['lines'])} diff(s)"
                f.write(f"  [{status}] {left_label}  vs  {r['right']}\n")
                f.writelines(f"      {line}\n" for line in r["lines"])

            f.write("\n")

    log.info("Reports written to '%s'", output_dir)
    log.info("Summary: %s", summary_path)

    if failed:
        log.error("%d sub-pair(s) failed.", len(failed))
        sys.exit(1)

    return [r for rs in grouped.values() for r in rs]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_pairs_file(path: str) -> list[tuple[str, str]]:
    """Read a pairs file and return a list of (left, right) directory tuples.

    Each non-blank, non-comment line must contain exactly two whitespace-
    separated paths.
    """
    pairs: list[tuple[str, str]] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                log.error(
                    "pairs file line %d must have exactly 2 paths, got %d: %r",
                    lineno,
                    len(parts),
                    line,
                )
                sys.exit(1)
            pairs.append((parts[0], parts[1]))
    return pairs


def _resolve_region(args) -> str:  # type: ignore[no-untyped-def]
    """Resolve AWS region from ``--pool`` or ``--region`` CLI args."""
    if args.pool:
        pool_to_region = _types.load_pool_to_region()
        if args.pool not in pool_to_region:
            known = ", ".join(sorted(pool_to_region))
            log.error("Unknown pool '%s'. Known pools:\n  %s", args.pool, known)
            sys.exit(1)
        region = pool_to_region[args.pool]
        log.info("Pool '%s' -> region '%s'", args.pool, region)
        return region
    return args.region


def _resolve_paths_or_pairs(
    args,  # type: ignore[no-untyped-def]
) -> tuple[list[str] | None, list[tuple[str, str]] | None]:
    """Resolve ``--paths`` / ``--paths-file`` / ``--path-pairs-file`` CLI args.

    Returns a ``(local_paths, pairs)`` tuple where exactly one value is set.
    """
    if args.path_pairs_file:
        pairs = _parse_pairs_file(args.path_pairs_file)
        if not pairs:
            log.error("no pairs found in '%s'", args.path_pairs_file)
            sys.exit(1)
        return None, pairs

    if args.paths_file:
        paths: list[str] = []
        with open(args.paths_file) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped and not stripped.startswith("#"):
                    paths.append(stripped)
        if not paths:
            log.error("no paths found in '%s'", args.paths_file)
            sys.exit(1)
        return paths, None

    return args.paths, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy local directories to S3 or compare directory pairs locally."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ------------------------------------------------------------------
    # sync subcommand
    # ------------------------------------------------------------------
    sync_parser = subparsers.add_parser(
        "sync",
        help="Copy local directories to S3 using a two-stage xenna pipeline.",
    )

    path_group = sync_parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument(
        "--paths",
        nargs="+",
        metavar="PATH",
        help="One or more local folders to copy (target bucket auto-derived).",
    )
    path_group.add_argument(
        "--paths-file",
        metavar="FILE",
        help="File with one local path per line (target bucket auto-derived; blank lines and # ignored).",
    )
    path_group.add_argument(
        "--path-pairs-file",
        metavar="FILE",
        help=(
            "File with explicit left/right pairs, one per line: "
            "<local_dir>  <s3_uri>  (blank lines and # comments ignored)"
        ),
    )

    region_group = sync_parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--pool",
        help="Cluster pool name (e.g. groot-gb200-02). Region is looked up automatically.",
    )
    region_group.add_argument(
        "--region",
        help="AWS region (e.g. us-east-1).",
    )

    sync_parser.add_argument(
        "--profile",
        default="aws",
        help="Gear credentials profile to use for S3 access (default: aws).",
    )
    sync_parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Directory depth at which to split into parallel sync units (default: 1).",
    )
    sync_parser.add_argument(
        "--walk-workers",
        type=int,
        default=4,
        help="Number of walk stage workers (default: 4).",
    )
    sync_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of sync stage workers (default: 8).",
    )
    sync_parser.add_argument(
        "--s3-concurrency",
        type=int,
        default=10,
        help="AWS_MAX_CONCURRENT_REQUESTS per sync worker (default: 10).",
    )
    sync_parser.add_argument(
        "--rate-limit-mbps",
        type=int,
        default=None,
        help="Per-worker bandwidth cap in MB/s (AWS_S3_TARGET_BANDWIDTH).",
    )
    sync_parser.add_argument(
        "--delete",
        action="store_true",
        help="Pass --delete to awscli s3 sync (remove objects not in source).",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dryrun to awscli s3 sync; no files are transferred.",
    )

    # ------------------------------------------------------------------
    # safe_sync subcommand
    # ------------------------------------------------------------------
    safe_sync_parser = subparsers.add_parser(
        "safe_sync",
        help=(
            "Upload local directories to S3, preserving existing S3 files as ground truth. "
            "Conflicting files are uploaded as <key>.local_copy."
        ),
    )

    safe_sync_path_group = safe_sync_parser.add_mutually_exclusive_group(required=True)
    safe_sync_path_group.add_argument(
        "--paths",
        nargs="+",
        metavar="PATH",
        help="One or more local folders to upload (target bucket auto-derived).",
    )
    safe_sync_path_group.add_argument(
        "--paths-file",
        metavar="FILE",
        help="File with one local path per line (target bucket auto-derived; blank lines and # ignored).",
    )
    safe_sync_path_group.add_argument(
        "--path-pairs-file",
        metavar="FILE",
        help=(
            "File with explicit left/right pairs, one per line: "
            "<local_dir>  <s3_uri>  (blank lines and # comments ignored)"
        ),
    )

    safe_sync_region_group = safe_sync_parser.add_mutually_exclusive_group(required=True)
    safe_sync_region_group.add_argument(
        "--pool",
        help="Cluster pool name (e.g. groot-gb200-02). Region is looked up automatically.",
    )
    safe_sync_region_group.add_argument(
        "--region",
        help="AWS region (e.g. us-east-1).",
    )

    safe_sync_parser.add_argument(
        "--profile",
        default="aws",
        help="Gear credentials profile to use for S3 access (default: aws).",
    )
    safe_sync_parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Directory depth at which to split into parallel sync units (default: 1).",
    )
    safe_sync_parser.add_argument(
        "--walk-workers",
        type=int,
        default=4,
        help="Number of walk stage workers (default: 4).",
    )
    safe_sync_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of safe-sync stage workers (default: 8).",
    )
    safe_sync_parser.add_argument(
        "--s3-concurrency",
        type=int,
        default=10,
        help="AWS_MAX_CONCURRENT_REQUESTS per sync worker (default: 10).",
    )
    safe_sync_parser.add_argument(
        "--rate-limit-mbps",
        type=int,
        default=None,
        help="Per-worker bandwidth cap in MB/s (AWS_S3_TARGET_BANDWIDTH).",
    )
    safe_sync_parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="If given, write report.json and summary.txt here after the pipeline completes.",
    )

    # ------------------------------------------------------------------
    # dryrun subcommand
    # ------------------------------------------------------------------
    dryrun_parser = subparsers.add_parser(
        "dryrun",
        help="Compare pairs of local directories in parallel using aws s3 sync --dryrun.",
    )
    dryrun_parser.add_argument(
        "--path-pairs-file",
        required=True,
        metavar="FILE",
        help=(
            "File with directory pairs to compare, one per line: "
            "<left_dir>  <right_dir>  (blank lines and # comments ignored)"
        ),
    )
    dryrun_parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where per-sub-pair JSON files, aggregate JSONs, and summary.txt are written.",
    )
    dryrun_parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Directory levels to walk before splitting into parallel sub-pairs (default: 3).",
    )
    dryrun_parser.add_argument(
        "--walk-workers",
        type=int,
        default=4,
        help="Number of walk stage workers (default: 4).",
    )
    dryrun_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dryrun stage workers (default: 8).",
    )
    dryrun_parser.add_argument(
        "--profile",
        default="aws",
        help="Gear credentials profile for S3 access (default: aws).",
    )
    dryrun_region_group = dryrun_parser.add_mutually_exclusive_group(required=False)
    dryrun_region_group.add_argument(
        "--pool",
        help="Cluster pool name (e.g. groot-gb200-02). Region is looked up automatically.",
    )
    dryrun_region_group.add_argument(
        "--region",
        help="AWS region (e.g. us-west-2). Used to construct the S3 endpoint URL.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Dispatch: sync
    # ------------------------------------------------------------------
    if args.subcommand == "sync":
        region = _resolve_region(args)
        local_paths, pairs = _resolve_paths_or_pairs(args)
        copy_to_object_store(
            region=region,
            local_paths=local_paths,
            pairs=pairs,
            profile=args.profile,
            depth=args.depth,
            num_walk_workers=args.walk_workers,
            num_sync_workers=args.workers,
            s3_concurrency=args.s3_concurrency,
            rate_limit_mbps=args.rate_limit_mbps,
            delete=args.delete,
            dry_run=args.dry_run,
        )

    # ------------------------------------------------------------------
    # Dispatch: safe_sync
    # ------------------------------------------------------------------
    elif args.subcommand == "safe_sync":
        region = _resolve_region(args)
        local_paths, pairs = _resolve_paths_or_pairs(args)
        run_safe_sync(
            region=region,
            local_paths=local_paths,
            pairs=pairs,
            output_dir=args.output_dir,
            profile=args.profile,
            depth=args.depth,
            num_walk_workers=args.walk_workers,
            num_sync_workers=args.workers,
            s3_concurrency=args.s3_concurrency,
            rate_limit_mbps=args.rate_limit_mbps,
        )

    # ------------------------------------------------------------------
    # Dispatch: dryrun
    # ------------------------------------------------------------------
    elif args.subcommand == "dryrun":
        pairs = _parse_pairs_file(args.path_pairs_file)
        if not pairs:
            log.error("no pairs found in '%s'", args.path_pairs_file)
            sys.exit(1)
        dryrun_region = _resolve_region(args) if (args.pool or args.region) else None
        run_dryrun_comparison(
            pairs=pairs,
            output_dir=args.output_dir,
            profile=args.profile,
            region=dryrun_region,
            depth=args.depth,
            num_walk_workers=args.walk_workers,
            num_dryrun_workers=args.workers,
        )


if __name__ == "__main__":
    main()
