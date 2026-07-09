#!/usr/bin/env python3
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

"""OSMO workflow result caching for CI test retries.

Provides an S3-backed per-workflow result cache so that CI retries only rerun
workflows that actually failed.  The CLI subcommands are called from
``ci/scripts/osmo_runner_execute.sh`` and work for any osmo-based test type
(vla, control, dexmg, gear_cli, …).

Cache key: ``{commit_sha}/{workflow_name}/{sha256_of_rendered_config}.json``
Storage:   ``s3://{bucket}/{prefix}/v1/…`` (same S3 credentials CI already uses)
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CACHE_BUCKET = "GearHome"
DEFAULT_CACHE_PREFIX = "rws/ci/osmo_cache"
DEFAULT_CACHE_VERSION = "v1"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def parse_yaml_blocks(text: str) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Parse the rendered workflow YAML into environment + workflow mappings."""
    if not text.strip():
        return None, {}

    parsed = yaml.safe_load(text)
    if parsed is None:
        return None, {}
    if not isinstance(parsed, Mapping):
        raise TypeError("Workflow list YAML must parse to a top-level mapping")

    parsed_dict = dict(parsed)
    env_block = parsed_dict.pop("environment", None)
    workflow_blocks: dict[str, dict[str, Any]] = {
        key: dict(value)
        for key, value in parsed_dict.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }
    return env_block, workflow_blocks


def blocks_to_yaml(
    env_block: dict[str, Any] | None,
    blocks: dict[str, dict[str, Any]],
    keys: list[str],
) -> str:
    """Serialize environment + selected workflow mappings back to YAML."""
    output: dict[str, Any] = {}
    if env_block is not None:
        output["environment"] = env_block
    for key in keys:
        output[key] = blocks[key]
    return yaml.safe_dump(output, sort_keys=False)


def workflow_signature(workflow_config: Mapping[str, Any]) -> str:
    """Build a stable signature for the rendered workflow definition."""
    canonical = json.dumps(workflow_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def recompute_integration_summary(log_json: dict) -> dict:
    """Return a copy of *log_json* with integration summary fields recomputed."""
    workflows = list(log_json.get("integration", {}).get("workflows", []))
    passed = sum(1 for w in workflows if w.get("status") == "COMPLETED")
    total = len(workflows)

    result = dict(log_json)
    result["integration"] = dict(log_json.get("integration", {}))
    result["integration"]["workflows"] = workflows
    result["integration"]["total_workflows"] = total
    result["integration"]["passed_workflows"] = passed
    result["integration"]["failed_workflows"] = total - passed
    return result


# ---------------------------------------------------------------------------
# S3 cache adapters
# ---------------------------------------------------------------------------


@dataclass
class S3Cache:
    """Thin wrapper around s5cmd for S3 cache operations."""

    bucket: str = DEFAULT_CACHE_BUCKET
    prefix: str = DEFAULT_CACHE_PREFIX
    version: str = DEFAULT_CACHE_VERSION

    def _s3_key(
        self,
        commit_sha: str,
        workflow_name: str,
        workflow_config: Mapping[str, Any],
    ) -> str:
        signature = workflow_signature(workflow_config)
        return f"s3://{self.bucket}/{self.prefix}/{self.version}/{commit_sha}/{workflow_name}/{signature}.json"

    def read(
        self,
        commit_sha: str,
        workflow_name: str,
        workflow_config: Mapping[str, Any],
    ) -> dict | None:
        """Read a cached result from S3. Returns parsed JSON or ``None`` on miss."""
        s3_key = self._s3_key(commit_sha, workflow_name, workflow_config)
        try:
            result = subprocess.run(
                ["s5cmd", "cat", s3_key],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get("status") == "COMPLETED":
                    return data
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            logger.debug("Failed to read cached workflow data", exc_info=True)
        return None

    def write(
        self,
        commit_sha: str,
        workflow_name: str,
        workflow_config: Mapping[str, Any],
        entry: dict,
    ) -> bool:
        """Write a cache entry to S3. Returns ``True`` on success."""
        s3_key = self._s3_key(commit_sha, workflow_name, workflow_config)
        try:
            proc = subprocess.run(
                ["s5cmd", "pipe", s3_key],
                check=False,
                input=json.dumps(entry),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False


# ---------------------------------------------------------------------------
# Core cache operations
# ---------------------------------------------------------------------------


@dataclass
class CacheCheckResult:
    """Result of checking the cache for previously passed workflows."""

    cached_workflows: list[dict] = dataclasses.field(default_factory=list)
    uncached_keys: list[str] = dataclasses.field(default_factory=list)
    all_cached: bool = False


def check_cache(
    shard_yaml: str,
    commit_sha: str,
    cache: S3Cache,
) -> CacheCheckResult:
    """Check S3 cache for each workflow in the shard. Return cached vs uncached."""
    _env_block, blocks = parse_yaml_blocks(shard_yaml)
    workflow_keys = list(blocks.keys())

    cached_workflows: list[dict] = []
    for wf_name in workflow_keys:
        data = cache.read(commit_sha, wf_name, blocks[wf_name])
        if data is not None:
            cached_workflows.append(data)

    cached_names = {w["name"] for w in cached_workflows}
    uncached_keys = [k for k in workflow_keys if k not in cached_names]

    return CacheCheckResult(
        cached_workflows=cached_workflows,
        uncached_keys=uncached_keys,
        all_cached=len(uncached_keys) == 0 and len(workflow_keys) > 0,
    )


def filter_shard_yaml(shard_yaml: str, uncached_keys: list[str]) -> str:
    """Remove cached workflows from the shard YAML, keeping only uncached ones."""
    env_block, blocks = parse_yaml_blocks(shard_yaml)
    return blocks_to_yaml(env_block, blocks, uncached_keys)


def write_cache(
    log_json: dict,
    run_yaml: str,
    commit_sha: str,
    cache: S3Cache,
    pipeline_id: str = "",
    job_id: str = "",
) -> int:
    """Write COMPLETED workflow results from log.json to S3 cache. Returns count written."""
    _, workflow_blocks = parse_yaml_blocks(run_yaml)
    workflows = log_json.get("integration", {}).get("workflows", [])
    count = 0
    for wf in workflows:
        if wf.get("status") != "COMPLETED":
            continue
        workflow_name = wf["name"]
        workflow_config = workflow_blocks.get(workflow_name)
        if workflow_config is None:
            continue
        entry = {
            "name": workflow_name,
            "status": "COMPLETED",
            "workflow_id": wf.get("workflow_id", ""),
            "start_time": wf.get("start_time", ""),
            "end_time": wf.get("end_time", ""),
            "pipeline_id": pipeline_id,
            "job_id": job_id,
            "workflow_signature": workflow_signature(workflow_config),
        }
        if cache.write(commit_sha, workflow_name, workflow_config, entry):
            count += 1
    return count


def merge_results(log_json: dict, cached_workflows: list[dict]) -> dict:
    """Merge cached workflow results into log.json from osmo_runner.

    Returns a new log_json dict with updated totals and merged workflow list.
    """
    fresh_workflows = log_json.get("integration", {}).get("workflows", [])
    fresh_names = {w["name"] for w in fresh_workflows}

    merged = fresh_workflows + [w for w in cached_workflows if w["name"] not in fresh_names]

    result = dict(log_json)
    result["integration"] = dict(log_json.get("integration", {}))
    result["integration"]["workflows"] = merged
    return recompute_integration_summary(result)


def generate_log_from_cache(cached_workflows: list[dict]) -> dict:
    """Generate a complete log.json from only cached results (all-cached fast path)."""
    return {
        "integration": {
            "total_workflows": len(cached_workflows),
            "passed_workflows": len(cached_workflows),
            "failed_workflows": 0,
            "workflows": cached_workflows,
        }
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _require_s5cmd() -> bool:
    if not shutil.which("s5cmd"):
        logger.warning("s5cmd not found, skipping cache operation")
        return False
    return True


def _add_s3_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bucket", default=DEFAULT_CACHE_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_CACHE_PREFIX)


def _cmd_check_cache(args: argparse.Namespace) -> int:
    """Check cache for previously passed workflows.

    Exit codes: 0 = some to run, 2 = all cached, 1 = error / s5cmd missing.
    """
    if not _require_s5cmd():
        return 1

    cache = S3Cache(bucket=args.bucket, prefix=args.prefix)

    with open(args.shard_file) as f:
        shard_yaml = f.read()

    result = check_cache(shard_yaml, args.commit_sha, cache)

    for wf in result.cached_workflows:
        logger.info("  \u2705 Cached PASS: %s", wf["name"])
    logger.info(
        "Cache: %d passed, %d to run",
        len(result.cached_workflows),
        len(result.uncached_keys),
    )

    with open(args.cached_results_file, "w") as f:
        json.dump(result.cached_workflows, f)

    with open(args.run_file, "w") as f:
        f.write(filter_shard_yaml(shard_yaml, result.uncached_keys))

    return 2 if result.all_cached else 0


def _cmd_generate_log(args: argparse.Namespace) -> int:
    """Generate log.json entirely from cached results (all-cached fast path)."""
    with open(args.cached_results_file) as f:
        cached = json.load(f)

    log = generate_log_from_cache(cached)

    with open(args.log_file, "w") as f:
        json.dump(log, f, indent=2)

    logger.info("Generated log.json with %d cached results", len(cached))
    return 0


def _cmd_write_cache(args: argparse.Namespace) -> int:
    """Write COMPLETED workflow results to S3 cache."""
    if not _require_s5cmd():
        return 0

    cache = S3Cache(bucket=args.bucket, prefix=args.prefix)

    with open(args.log_file) as f:
        log_data = json.load(f)
    with open(args.run_file) as f:
        run_yaml = f.read()

    count = write_cache(
        log_data,
        run_yaml,
        args.commit_sha,
        cache,
        pipeline_id=args.pipeline_id,
        job_id=args.job_id,
    )
    total = len(log_data.get("integration", {}).get("workflows", []))
    logger.info("Cached %d/%d passed workflow results to S3", count, total)
    return 0


def _cmd_merge_results(args: argparse.Namespace) -> int:
    """Merge cached + fresh results and validate.

    Exit codes: 0 = all passed, 1 = failures present.
    """
    with open(args.log_file) as f:
        log_data = json.load(f)

    cached_workflows: list[dict] = []
    if args.cached_results_file and args.cached_results_file.strip():
        try:
            with open(args.cached_results_file) as f:
                cached_workflows = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    merged_log = merge_results(log_data, cached_workflows)

    with open(args.log_file, "w") as f:
        json.dump(merged_log, f, indent=2)

    fresh_names = {w["name"] for w in log_data.get("integration", {}).get("workflows", [])}
    if cached_workflows:
        cached_names = [w["name"] for w in cached_workflows if w["name"] not in fresh_names]
        if cached_names:
            logger.info("Merged %d cached results: %s", len(cached_names), ", ".join(cached_names))

    total = merged_log["integration"]["total_workflows"]
    passed = merged_log["integration"]["passed_workflows"]
    failed = merged_log["integration"]["failed_workflows"]
    logger.info("Total workflows: %d", total)
    logger.info("Passed workflows: %d", passed)
    logger.info("Failed workflows: %d", failed)

    if total == 0:
        logger.error("\u274c No workflows found")
        return 1

    if total == passed:
        logger.info("\u2705 All tests passed: %d/%d", passed, total)
        return 0

    logger.error("\u274c Not all workflows passed: %d/%d", passed, total)
    logger.error("Finding failed workflows...")
    for w in merged_log["integration"]["workflows"]:
        if w.get("status") != "COMPLETED":
            logger.error(
                "  \u274c Failed: '%s' - status: %s", w["name"], w.get("status", "UNKNOWN")
            )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osmo_cache",
        description="OSMO workflow result caching for CI test retries",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- check-cache --
    p = sub.add_parser("check-cache", help="Check S3 for previously passed workflows")
    p.add_argument("--shard-file", required=True)
    p.add_argument("--run-file", required=True)
    p.add_argument("--cached-results-file", required=True)
    p.add_argument("--commit-sha", required=True)
    _add_s3_args(p)

    # -- generate-log --
    p = sub.add_parser("generate-log", help="Build log.json from cached results only")
    p.add_argument("--cached-results-file", required=True)
    p.add_argument("--log-file", required=True)

    # -- write-cache --
    p = sub.add_parser("write-cache", help="Persist COMPLETED results to S3 cache")
    p.add_argument("--log-file", required=True)
    p.add_argument("--run-file", required=True)
    p.add_argument("--commit-sha", required=True)
    p.add_argument("--pipeline-id", default="")
    p.add_argument("--job-id", default="")
    _add_s3_args(p)

    # -- merge-results --
    p = sub.add_parser("merge-results", help="Merge cached + fresh results, validate")
    p.add_argument("--log-file", required=True)
    p.add_argument("--cached-results-file", default="")

    return parser


_DISPATCH = {
    "check-cache": _cmd_check_cache,
    "generate-log": _cmd_generate_log,
    "write-cache": _cmd_write_cache,
    "merge-results": _cmd_merge_results,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
