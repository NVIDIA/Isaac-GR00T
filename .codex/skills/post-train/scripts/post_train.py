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
"""Trigger an OSS post-train pipeline and post a comment on the open MR if one exists.

Usage:
    uv run python post_train.py [<run-type>] [<priority>] [--eval]

Run types:
    (default)               - first active variant in golden_runs.yaml
    all                     - every active variant/platform in golden_runs.yaml
    <run-type>-h100         - Any H100 run type from golden_runs.yaml

Priority override (optional second arg): normal | low | high (default: high)
Add --eval to run OSS evaluation on the produced checkpoint in the same pipeline.
"""

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
_ALL_ARGS = {"all", "all-post-train", "all-post-trains", "all-posttrain", "all-posttrains"}
_PRIORITIES = {"high", "normal", "low"}
_OPTIONAL_PIPELINE_ENV_VARS = ("OSMO_POOL", "POST_TRAIN_CLUSTER_WAIT_TIMEOUT")


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
    args = sys.argv[1:]
    run_eval = "--eval" in args or "--evaluation" in args
    args = [arg for arg in args if arg not in {"--eval", "--evaluation"}]
    run_type_key = args[0].lower() if args else ""
    try:
        run_specs = _resolve_run_specs(run_type_key)
    except ValueError as exc:
        print(str(exc))  # noqa: T201
        sys.exit(1)

    priority = args[1] if len(args) > 1 else "high"
    if priority not in _PRIORITIES:
        print(f"Unknown priority {priority!r}. Use high, normal, or low.")  # noqa: T201
        sys.exit(1)
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
    triggered: list[tuple[str, str]] = []
    for run_type, compute_type, label, run_cfg in run_specs:
        run_cfg.update(_pipeline_env_overrides())
        run_cfg["OSMO_PRIORITY"] = priority
        if run_eval:
            run_cfg.update(_open_loop_eval_vars(run_type, compute_type))
        pipeline = client.run_pipeline(branch, variables=run_cfg)
        label_with_compute = f"{label} ({compute_type.upper()})"
        triggered.append((label_with_compute, pipeline.web_url))
        print(f"Pipeline triggered for {label_with_compute}: {pipeline.web_url}")  # noqa: T201

    if mr_matches:
        lines = "\n".join(f"- {label}: {url}" for label, url in triggered)
        mode = "post-train + evaluation" if run_eval else "post-train"
        comment_body = f"OSS {mode} triggered (priority: {priority}):\n\n{lines}"
        client.add_mr_comment(mr_matches[0].iid, comment_body)
        print(f"Comment posted to MR !{mr_matches[0].iid}: {mr_matches[0].web_url}")  # noqa: T201
    else:
        print("No open MR found for current branch; skipping MR comment.")  # noqa: T201

    summary = "\n".join(f"  - {label}: {url}" for label, url in triggered)
    print(  # noqa: T201
        f"\nOSS post-train triggered:\n"
        f"  Branch:   {branch}\n"
        f"  Priority: {priority}\n"
        f"  Eval:     {'enabled' if run_eval else 'disabled'}\n"
        f"{summary}\n"
        f"  MR comment: {'posted to MR !' + str(mr_matches[0].iid) if mr_matches else 'no open MR found'}"
    )


def _resolve_run_specs(raw: str) -> list[tuple[str, str, str, dict[str, str]]]:
    if not raw:
        variants = load_variants()
        if not variants:
            raise ValueError("No post-train variants are configured.")
        run_type, variant = next(iter(variants.items()))
        compute_type = next(iter(variant.platforms)).value
        return [_run_spec(run_type, compute_type)]

    if raw in _ALL_ARGS:
        run_specs = []
        for run_type, variant in load_variants().items():
            for compute_type in variant.platforms:
                if compute_type.value in _COMPUTE_TYPES:
                    run_specs.append(_run_spec(run_type, compute_type.value))
        if not run_specs:
            raise ValueError("No H100 post-train variants are configured.")
        return run_specs

    run_type, compute_type = _parse_run_type_arg(raw)
    return [_run_spec(run_type, compute_type)]


def _run_spec(run_type: str, compute_type: str) -> tuple[str, str, str, dict[str, str]]:
    variant = get_variant(run_type)
    variant.platform(compute_type)
    run_cfg = {
        "POST_TRAIN_E2E_LEAF": "true",
        "POST_TRAIN_VARIANT": run_type,
        "POST_TRAIN_RUN_TYPE": variant.run_type,
        "POST_TRAIN_COMPUTE_TYPE": compute_type,
    }
    return run_type, compute_type, variant.label, run_cfg


def _pipeline_env_overrides() -> dict[str, str]:
    return {
        name: value
        for name in _OPTIONAL_PIPELINE_ENV_VARS
        if (value := os.getenv(name))
    }


def _open_loop_eval_vars(run_type: str, compute_type: str) -> dict[str, str]:
    variant = get_variant(run_type)
    return {
        "OPEN_LOOP_EVAL_E2E_LEAF": "true",
        "OPEN_LOOP_EVAL_AFTER_POST_TRAIN": "true",
        "OPEN_LOOP_EVAL_VARIANT": run_type,
        "OPEN_LOOP_EVAL_RUN_TYPE": variant.run_type,
        "OPEN_LOOP_EVAL_COMPUTE_TYPE": compute_type,
    }


def _parse_run_type_arg(raw: str) -> tuple[str, str]:
    for compute_type in _COMPUTE_TYPES:
        suffix = f"-{compute_type}"
        if raw.endswith(suffix):
            run_type = raw[: -len(suffix)]
            if run_type:
                return run_type, compute_type
    raise ValueError(
        f"Unknown post-train argument {raw!r}. Use '<run-type>-h100', "
        "for example 'n2-minwam-sim-h100'."
    )


if __name__ == "__main__":
    main()
