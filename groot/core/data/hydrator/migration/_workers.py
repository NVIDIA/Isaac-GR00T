# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Worker classes and their direct helpers for the migration pipeline.

All workers are designed to be used as xenna Pipeline stage workers.
"""

from __future__ import annotations

import os
import re
import subprocess

import boto3

from groot.core.utils.distributed import worker_logging
from groot.core.data.hydrator.migration import _types
from groot.core.data.hydrator.utils import aws

log = worker_logging.get_unified_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_s3_client(region: str, profile: str = "aws"):
    """Build a boto3 S3 client using gear credentials.

    Uses GEAR_DATA_CREDENTIALS env var or ~/.gear/data_credentials.  Region is
    always forced to *region*.  The endpoint URL is taken from the credential
    profile if present (e.g. PDX/Ceph clusters); otherwise it falls back to
    the standard ``https://s3.{region}.amazonaws.com``.
    """
    client_config = aws.get_boto3_client_config(profile=profile)
    client_config["region_name"] = region
    if not client_config.get("endpoint_url"):
        client_config["endpoint_url"] = f"https://s3.{region}.amazonaws.com"
    return boto3.client("s3", **client_config)


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/prefix`` into ``(bucket, prefix)``."""
    assert s3_uri.startswith("s3://"), f"Not an S3 URI: {s3_uri}"
    path = s3_uri[5:]
    bucket, _, prefix = path.partition("/")
    return bucket, prefix


def _parse_dryrun_files(stdout: str, local_dir: str) -> list[str]:
    """Extract relative file paths from ``aws s3 sync --dryrun`` stdout.

    Each output line has the form::

        (dryrun) upload: /abs/local/path/file.txt to s3://bucket/prefix/file.txt

    Args:
        stdout: Captured stdout from ``aws s3 sync --dryrun``.
        local_dir: Absolute local directory used as the sync source.

    Returns:
        Relative paths (OS separator) relative to *local_dir*, one per file.
    """
    local_dir_norm = os.path.normpath(local_dir)
    rels: list[str] = []
    for line in stdout.splitlines():
        m = re.match(r"^\(dryrun\) (?:upload|copy): (.+?) to s3://", line.strip())
        if m:
            abs_path = os.path.normpath(m.group(1))
            rel = os.path.relpath(abs_path, local_dir_norm)
            if not rel.startswith(".."):
                rels.append(rel)
    return rels


# ---------------------------------------------------------------------------
# sync pipeline — stage 1: directory walker
# ---------------------------------------------------------------------------


class WalkWorker:
    """Stage-1 worker: walks a local path and emits SyncUnit objects.

    ``process()`` returns a list so the Pipeline fans each SyncUnit
    individually into the stage-2 queue.
    """

    def process(self, request: _types.WalkRequest) -> list[_types.SyncUnit]:
        units = _types.walk_to_depth(
            request.local_path, request.bucket, request.depth, request.key_prefix
        )
        log.info("  Walked '%s': %d sync unit(s)", request.local_path, len(units))
        return units


# ---------------------------------------------------------------------------
# sync pipeline — stage 2: S3 uploader
# ---------------------------------------------------------------------------


class SyncWorker:
    """Stage-2 worker: runs ``aws s3 sync`` for a single :class:`SyncUnit`.

    Requires the aws CLI v2 binary on PATH (installed on cluster nodes via
    the Ray cluster template). Each sync uses explicit --region and
    --endpoint-url to avoid redirects.
    """

    def __init__(
        self,
        region: str,
        profile: str = "aws",
        s3_concurrency: int = 10,
        rate_limit_mbps: int | None = None,
        delete: bool = False,
        dry_run: bool = False,
    ) -> None:
        self._region = region
        self._profile = profile
        self._s3_concurrency = s3_concurrency
        self._rate_limit_mbps = rate_limit_mbps
        self._delete = delete
        self._dry_run = dry_run
        self._env: dict[str, str] = {}

    def setup(self) -> None:
        """Configure AWS CLI environment variables and endpoint for this worker."""
        env = os.environ.copy()
        env["AWS_MAX_CONCURRENT_REQUESTS"] = str(self._s3_concurrency)
        if self._rate_limit_mbps is not None:
            env["AWS_S3_TARGET_BANDWIDTH"] = f"{self._rate_limit_mbps}MB/s"
        # Bypass proxy for S3 to avoid routing uploads through the cluster proxy.
        # Both cases are needed: NO_PROXY is read by aws CLI/botocore, no_proxy by Python urllib.
        _no_proxy = ".amazonaws.com,localhost,127.0.0.1,::1"
        env["NO_PROXY"] = env["no_proxy"] = _no_proxy

        # Override AWS credentials with the requested gear profile so the subprocess
        # does not pick up credentials from the ambient environment.
        client_config = aws.get_boto3_client_config(profile=self._profile)
        if client_config.get("aws_access_key_id"):
            env["AWS_ACCESS_KEY_ID"] = client_config["aws_access_key_id"]
        if client_config.get("aws_secret_access_key"):
            env["AWS_SECRET_ACCESS_KEY"] = client_config["aws_secret_access_key"]
        # Clear any session token that might belong to a different profile.
        env.pop("AWS_SESSION_TOKEN", None)

        self._env = env
        # Use the profile's endpoint_url if present (e.g. PDX/Ceph); otherwise
        # fall back to the standard AWS regional endpoint.
        self._endpoint_url = (
            client_config.get("endpoint_url") or f"https://s3.{self._region}.amazonaws.com"
        )

    def process(self, unit: _types.SyncUnit) -> dict:
        """Run ``python -m awscli s3 sync`` for the given sync unit.

        Returns a dict with ``local_dir``, ``s3_prefix``, ``returncode``,
        and ``stderr`` (non-empty only on failure).
        """
        cmd = [
            "aws",
            "s3",
            "sync",
            unit.local_dir,
            unit.s3_prefix,
            "--region",
            self._region,
            "--endpoint-url",
            self._endpoint_url,
        ]
        if self._delete:
            cmd.append("--delete")
        if self._dry_run:
            cmd.append("--dryrun")

        result = subprocess.run(
            cmd,
            check=False,
            env=self._env,
            capture_output=True,
            text=True,
        )
        return {
            "local_dir": unit.local_dir,
            "s3_prefix": unit.s3_prefix,
            "returncode": result.returncode,
            "stderr": result.stderr if result.returncode != 0 else "",
        }


# ---------------------------------------------------------------------------
# safe_sync pipeline — stage 2: safe S3 uploader
# ---------------------------------------------------------------------------


class SafeSyncWorker(SyncWorker):
    """Stage-2 worker for *safe_sync*: upload local files to S3 preserving right as ground truth.

    Extends :class:`SyncWorker` — reuses its ``__init__`` and ``setup`` (AWS CLI
    env + endpoint).  Adds a boto3 client for listing existing S3 objects.

    For each :class:`SyncUnit` the worker handles three cases:

    * **Identical** (same size on both sides): skipped.
    * **Only in left** (not yet in S3): uploaded normally.
    * **Conflicting** (exists in S3 with a different size): uploaded to
      ``<s3_key>.local_copy`` via ``aws s3 cp``; the original S3 object is
      left untouched.
    """

    def setup(self) -> None:
        """Configure AWS CLI env, endpoint URL, and boto3 client."""
        super().setup()
        self._s3_client = _make_s3_client(self._region, profile=self._profile)

    def _list_s3_objects(self, bucket: str, prefix: str) -> dict[str, int]:
        """Return ``{rel_key: size}`` for all objects under *bucket*/*prefix*."""
        paginator = self._s3_client.get_paginator("list_objects_v2")
        prefix_slash = prefix.rstrip("/") + "/" if prefix else ""
        result: dict[str, int] = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix_slash):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key.removeprefix(prefix_slash)
                if rel:
                    result[rel] = obj["Size"]
        return result

    def process(self, unit: _types.SyncUnit) -> dict:
        """Run safe upload for one :class:`SyncUnit`.

        Returns a dict with ``local_dir``, ``s3_prefix``, ``new``,
        ``conflict``, ``errors``, and ``returncode``.
        """
        # Step 1: dryrun to discover which local files differ from S3
        dryrun_cmd = [
            "aws",
            "s3",
            "sync",
            unit.local_dir,
            unit.s3_prefix,
            "--dryrun",
            "--size-only",
            "--region",
            self._region,
            "--endpoint-url",
            self._endpoint_url,
        ]
        dr = subprocess.run(dryrun_cmd, check=False, env=self._env, capture_output=True, text=True)
        if dr.returncode != 0:
            return {
                "local_dir": unit.local_dir,
                "s3_prefix": unit.s3_prefix,
                "new": 0,
                "new_files": [],
                "conflict": 0,
                "conflict_files": [],
                "errors": [{"type": "dryrun", "stderr": dr.stderr}],
                "returncode": dr.returncode,
            }

        to_upload_rels = _parse_dryrun_files(dr.stdout, unit.local_dir)

        if not to_upload_rels:
            log.info("  [safe_sync] %s: all identical, nothing to upload", unit.local_dir)
            return {
                "local_dir": unit.local_dir,
                "s3_prefix": unit.s3_prefix,
                "new": 0,
                "new_files": [],
                "conflict": 0,
                "conflict_files": [],
                "errors": [],
                "returncode": 0,
            }

        # Step 2: classify as new (not in S3) vs conflicting (exists with different size)
        bucket, prefix = _parse_s3_uri(unit.s3_prefix)
        s3_objs = self._list_s3_objects(bucket, prefix)

        new_rels: list[str] = []
        conflict_rels: list[str] = []
        for rel in to_upload_rels:
            if rel.replace(os.sep, "/") in s3_objs:
                conflict_rels.append(rel)
            else:
                new_rels.append(rel)

        log.info(
            "  [safe_sync] %s: %d new, %d conflict",
            unit.local_dir,
            len(new_rels),
            len(conflict_rels),
        )

        errors: list[dict] = []

        # Step 3: upload new files via sync (exclude conflict files so S3 copies are untouched)
        if new_rels:
            sync_cmd = [
                "aws",
                "s3",
                "sync",
                unit.local_dir,
                unit.s3_prefix,
                "--size-only",
                "--region",
                self._region,
                "--endpoint-url",
                self._endpoint_url,
            ]
            for rel in conflict_rels:
                sync_cmd.extend(["--exclude", rel.replace(os.sep, "/")])
            r = subprocess.run(sync_cmd, check=False, env=self._env, capture_output=True, text=True)
            if r.returncode != 0:
                errors.append({"type": "sync_new", "stderr": r.stderr})

        # Step 4: upload conflicting files under <key>.local_copy, leaving S3 originals intact
        for rel in conflict_rels:
            abs_local = os.path.join(unit.local_dir, rel)
            rel_norm = rel.replace(os.sep, "/")
            s3_dest = unit.s3_prefix.rstrip("/") + "/" + rel_norm + _types.CONFLICT_SUFFIX
            cp_cmd = [
                "aws",
                "s3",
                "cp",
                abs_local,
                s3_dest,
                "--region",
                self._region,
                "--endpoint-url",
                self._endpoint_url,
            ]
            r = subprocess.run(cp_cmd, check=False, env=self._env, capture_output=True, text=True)
            if r.returncode != 0:
                errors.append({"type": "cp_conflict", "file": rel, "stderr": r.stderr})

        return {
            "local_dir": unit.local_dir,
            "s3_prefix": unit.s3_prefix,
            "new": len(new_rels),
            "new_files": sorted(r.replace(os.sep, "/") for r in new_rels),
            "conflict": len(conflict_rels),
            "conflict_files": sorted(r.replace(os.sep, "/") for r in conflict_rels),
            "errors": errors,
            "returncode": 1 if errors else 0,
        }


# ---------------------------------------------------------------------------
# dryrun pipeline — stage 1: sub-directory pair walker
# ---------------------------------------------------------------------------


class WalkPairsWorker:
    """Stage-1 worker: walks both sides of a pair and fans out sub-pair requests.

    For a top-level ``DryRunRequest(left, right)``, both trees are walked to
    *depth* levels.  Sub-directory pairs are classified into three groups:

    * **common** — the relative path exists in both trees.
    * **only_in_left** — sub-dir is present in *left* but missing in *right*.
    * **only_in_right** — sub-dir is present in *right* but missing in *left*.

    One :class:`DryRunRequest` is emitted per sub-pair and fanned out by the
    Pipeline into the stage-2 queue.
    """

    def __init__(self, depth: int = 3) -> None:
        self._depth = depth

    @staticmethod
    def _right_sub(right_base: str, rel: str) -> str:
        """Construct the right sub-path for a given relative path.

        Works for both local paths (uses os.path.join) and S3 URIs (uses /
        since os.path.join corrupts ``s3://`` prefixes).
        """
        if right_base.startswith("s3://"):
            return right_base.rstrip("/") + "/" + rel if rel != "." else right_base
        return os.path.join(right_base, rel)

    def process(self, request: _types.DryRunRequest) -> list[_types.DryRunRequest]:
        left_subs = _types._walk_subdirs(request.left, self._depth)  # noqa: SLF001
        orig_left = request.left
        orig_right = request.right

        sub_requests: list[_types.DryRunRequest] = []

        if request.right.startswith("s3://"):
            # Right side is S3 — os.path.isdir / scandir cannot walk it.
            # Derive each right sub-path from the left sub-path by relative
            # substitution.  aws s3 sync handles the actual comparison.
            for rel, left_abs in sorted(left_subs.items()):
                sub_requests.append(
                    _types.DryRunRequest(
                        left=left_abs,
                        right=self._right_sub(orig_right, rel),
                        orig_left=orig_left,
                        orig_right=orig_right,
                    )
                )
            log.info(
                "  Walked pair '%s' / '%s' (S3 right): %d sub-pair(s)",
                orig_left,
                orig_right,
                len(sub_requests),
            )
            return sub_requests

        right_subs = _types._walk_subdirs(request.right, self._depth)  # noqa: SLF001
        left_set = set(left_subs)
        right_set = set(right_subs)

        for rel in sorted(left_set & right_set):
            sub_requests.append(
                _types.DryRunRequest(
                    left=left_subs[rel],
                    right=right_subs[rel],
                    orig_left=orig_left,
                    orig_right=orig_right,
                )
            )
        for rel in sorted(left_set - right_set):
            sub_requests.append(
                _types.DryRunRequest(
                    left=left_subs[rel],
                    right=self._right_sub(orig_right, rel),
                    orig_left=orig_left,
                    orig_right=orig_right,
                    note="only_in_left",
                )
            )
        for rel in sorted(right_set - left_set):
            sub_requests.append(
                _types.DryRunRequest(
                    left="",
                    right=right_subs[rel],
                    orig_left=orig_left,
                    orig_right=orig_right,
                    note="only_in_right",
                )
            )

        log.info(
            "  Walked pair '%s' / '%s': %d sub-pair(s) (common=%d, only_left=%d, only_right=%d)",
            orig_left,
            orig_right,
            len(sub_requests),
            len(left_set & right_set),
            len(left_set - right_set),
            len(right_set - left_set),
        )
        return sub_requests


# ---------------------------------------------------------------------------
# dryrun pipeline — stage 2: per-sub-pair dryrun executor
# ---------------------------------------------------------------------------


class DryRunWorker:
    """Stage-2 worker: runs ``aws s3 sync --dryrun`` for one sub-pair.

    For ``only_in_right`` sub-pairs (no left counterpart) the aws sync call is
    skipped and the result is returned with an empty ``lines`` list — the
    sub-dir existence is already captured in the ``note`` field.

    Returns a dict with:
        left, right        — absolute directory paths
        orig_left, orig_right — top-level pair these results belong to
        note               — ``only_in_left`` | ``only_in_right`` | ``""``
        returncode         — exit code (0 for skipped only_in_right entries)
        lines              — list of output lines from aws (one per action)
        stderr             — error text (non-empty only on failure)
    """

    def __init__(self, profile: str = "aws", region: str | None = None) -> None:
        self._profile = profile
        self._region = region
        self._env: dict[str, str] = {}
        self._endpoint_url: str | None = None

    def setup(self) -> None:
        """Configure AWS CLI credentials and endpoint for this worker."""
        env = os.environ.copy()
        _no_proxy = ".amazonaws.com,localhost,127.0.0.1,::1"
        env["NO_PROXY"] = env["no_proxy"] = _no_proxy

        client_config = aws.get_boto3_client_config(profile=self._profile)
        if client_config.get("aws_access_key_id"):
            env["AWS_ACCESS_KEY_ID"] = client_config["aws_access_key_id"]
        if client_config.get("aws_secret_access_key"):
            env["AWS_SECRET_ACCESS_KEY"] = client_config["aws_secret_access_key"]
        env.pop("AWS_SESSION_TOKEN", None)

        self._env = env
        self._endpoint_url = client_config.get("endpoint_url") or (
            f"https://s3.{self._region}.amazonaws.com" if self._region else None
        )

    def process(self, request: _types.DryRunRequest) -> dict:
        base = {
            "left": request.left,
            "right": request.right,
            "orig_left": request.orig_left,
            "orig_right": request.orig_right,
            "note": request.note,
        }

        if request.note == "only_in_right":
            # Nothing to sync from an empty source; presence is noted in report.
            log.info("  Skipped (only_in_right): '%s'", request.right)
            return {**base, "returncode": 0, "lines": [], "stderr": ""}

        # --size-only: skip ETag comparison (multipart-uploaded S3 objects have
        # a composite ETag that never matches a local MD5, causing false diffs
        # even when the file content is identical).
        cmd = ["aws", "s3", "sync", request.left, request.right, "--dryrun", "--size-only"]
        if self._endpoint_url:
            cmd += ["--endpoint-url", self._endpoint_url]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=self._env)
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]

        label = (
            request.note
            if request.note
            else ("IDENTICAL" if not lines else f"{len(lines)} diff(s)")
        )
        log.info("  Compared '%s' vs '%s': %s", request.left, request.right, label)

        return {
            **base,
            "returncode": result.returncode,
            "lines": lines,
            "stderr": result.stderr if result.returncode != 0 else "",
        }
