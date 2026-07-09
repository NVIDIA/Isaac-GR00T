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
"""Trigger OSS evaluation CI and comment on the open MR when possible.

Usage:
    uv run python evaluation.py [<run-type>-h100] [priority] --checkpoint <hf-uri-or-path>
    uv run python evaluation.py [<run-type>-h100] [priority] --post-train
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

import requests

_REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
sys.path.insert(0, str(_REPO_ROOT))

from groot.infra.ci.golden_run_config import get_variant, load_variants  # noqa: E402

_COMPUTE_TYPES = ("h100",)
_PRIORITIES = {"high", "normal", "low"}
_OPTIONAL_PIPELINE_ENV_VARS = ("OSMO_POOL",)


@dataclass(frozen=True)
class MergeRequest:
    iid: int
    source_branch: str
    web_url: str


@dataclass(frozen=True)
class Pipeline:
    web_url: str


class GitLabClient:
    def __init__(self) -> None:
        api_url = os.getenv("CI_API_V4_URL")
        if api_url:
            self.api_url = api_url.rstrip("/")
        else:
            raw_host = os.getenv("GITLAB_HOST")
            if not raw_host:
                raise ValueError("Set GITLAB_HOST or CI_API_V4_URL before triggering.")
            host = raw_host.removeprefix("https://").removeprefix("http://")
            self.api_url = f"https://{host}/api/v4"

        project = (
            os.getenv("GITLAB_PROJECT_ID")
            or os.getenv("CI_PROJECT_ID")
            or os.getenv("GITLAB_PROJECT_PATH")
            or os.getenv("CI_PROJECT_PATH")
        )
        if not project:
            raise ValueError("Set GITLAB_PROJECT_ID or GITLAB_PROJECT_PATH before triggering.")
        self.project = quote(project, safe="")

        token = os.getenv("GITLAB_TOKEN")
        job_token = os.getenv("CI_JOB_TOKEN")
        if token:
            self.headers = {"PRIVATE-TOKEN": token}
        elif job_token:
            self.headers = {"JOB-TOKEN": job_token}
        else:
            raise ValueError("Set GITLAB_TOKEN or CI_JOB_TOKEN before triggering.")

    def get_open_mrs(self, source_branch: str) -> list[MergeRequest]:
        url = f"{self.api_url}/projects/{self.project}/merge_requests"
        response = requests.get(
            url,
            headers=self.headers,
            params={"state": "opened", "source_branch": source_branch, "per_page": 20},
            timeout=30,
        )
        response.raise_for_status()
        return [
            MergeRequest(
                iid=int(item["iid"]),
                source_branch=str(item.get("source_branch", "")),
                web_url=str(item.get("web_url", "")),
            )
            for item in response.json()
        ]

    def run_pipeline(self, ref: str, variables: dict[str, str]) -> Pipeline:
        url = f"{self.api_url}/projects/{self.project}/pipeline"
        payload = {"ref": ref, "variables": [{"key": k, "value": v} for k, v in variables.items()]}
        response = requests.post(url, headers=self.headers, json=payload, timeout=30)
        response.raise_for_status()
        return Pipeline(web_url=str(response.json()["web_url"]))

    def add_mr_comment(self, mr_iid: int, body: str) -> None:
        url = f"{self.api_url}/projects/{self.project}/merge_requests/{mr_iid}/notes"
        response = requests.post(url, headers=self.headers, json={"body": body}, timeout=30)
        response.raise_for_status()


def main() -> None:
    args = _parse_args()
    run_type, compute_type, label, variables = _run_spec(args.run_type or "")
    variables.update(_pipeline_env_overrides())
    variables["OSMO_PRIORITY"] = args.priority
    if args.checkpoint:
        variables["OPEN_LOOP_EVAL_CHECKPOINT"] = args.checkpoint
    if args.post_train:
        variables.update(_post_train_variables(run_type, compute_type))

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()

    try:
        client = GitLabClient()
    except ValueError as exc:
        print(str(exc))  # noqa: T201
        sys.exit(1)
    open_mrs = client.get_open_mrs(source_branch=branch)
    mr_matches = [m for m in open_mrs if m.source_branch == branch]
    pipeline = client.run_pipeline(branch, variables=variables)
    mode = "post-train + evaluation" if args.post_train else "evaluation"
    print(f"Pipeline triggered for {label} ({compute_type.upper()}) {mode}: {pipeline.web_url}")  # noqa: T201

    if mr_matches:
        checkpoint = args.checkpoint or "post-train FINAL_CHECKPOINT_PATH"
        comment_body = (
            f"OSS {mode} triggered (priority: {args.priority}):\n\n"
            f"- {label} ({compute_type.upper()}), checkpoint: `{checkpoint}`: {pipeline.web_url}"
        )
        client.add_mr_comment(mr_matches[0].iid, comment_body)
        print(f"Comment posted to MR !{mr_matches[0].iid}: {mr_matches[0].web_url}")  # noqa: T201
    else:
        print("No open MR found for current branch; skipping MR comment.")  # noqa: T201

    print(  # noqa: T201
        f"\nOSS {mode} triggered:\n"
        f"  Branch:     {branch}\n"
        f"  Variant:    {run_type}\n"
        f"  Compute:    {compute_type}\n"
        f"  Priority:   {args.priority}\n"
        f"  Pipeline:   {pipeline.web_url}\n"
        f"  MR comment: {'posted to MR !' + str(mr_matches[0].iid) if mr_matches else 'no open MR found'}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_type",
        nargs="?",
        default="",
        help="Run type and compute suffix, for example n2-minwam-sim-h100.",
    )
    parser.add_argument(
        "priority",
        nargs="?",
        default="high",
        choices=sorted(_PRIORITIES),
        help="OSMO priority for the staging pipeline.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        help="Existing checkpoint to evaluate. Use an hf://model URI or a shared filesystem path.",
    )
    source.add_argument(
        "--post-train",
        action="store_true",
        help="Run post-train first and evaluate the produced FINAL_CHECKPOINT_PATH in the same pipeline.",
    )
    return parser.parse_args()


def _run_spec(raw: str) -> tuple[str, str, str, dict[str, str]]:
    if not raw:
        variants = load_variants()
        if not variants:
            raise ValueError("No evaluation variants are configured.")
        run_type, variant = next(iter(variants.items()))
        compute_type = next(iter(variant.platforms)).value
    else:
        run_type, compute_type = _parse_run_type_arg(raw)
        variant = get_variant(run_type)
        variant.platform(compute_type)
    variables = {
        "OPEN_LOOP_EVAL_E2E_LEAF": "true",
        "OPEN_LOOP_EVAL_VARIANT": run_type,
        "OPEN_LOOP_EVAL_RUN_TYPE": variant.run_type,
        "OPEN_LOOP_EVAL_COMPUTE_TYPE": compute_type,
    }
    return run_type, compute_type, variant.label, variables


def _post_train_variables(run_type: str, compute_type: str) -> dict[str, str]:
    variant = get_variant(run_type)
    return {
        "POST_TRAIN_E2E_LEAF": "true",
        "POST_TRAIN_VARIANT": run_type,
        "POST_TRAIN_RUN_TYPE": variant.run_type,
        "POST_TRAIN_COMPUTE_TYPE": compute_type,
        "OPEN_LOOP_EVAL_AFTER_POST_TRAIN": "true",
    }


def _pipeline_env_overrides() -> dict[str, str]:
    return {
        name: value
        for name in _OPTIONAL_PIPELINE_ENV_VARS
        if (value := os.getenv(name))
    }


def _parse_run_type_arg(raw: str) -> tuple[str, str]:
    for compute_type in _COMPUTE_TYPES:
        suffix = f"-{compute_type}"
        if raw.endswith(suffix):
            run_type = raw[: -len(suffix)]
            if run_type:
                return run_type, compute_type
    raise ValueError(
        f"Unknown evaluation argument {raw!r}. Use '<run-type>-h100', "
        "for example 'n2-minwam-sim-h100'."
    )


if __name__ == "__main__":
    main()
