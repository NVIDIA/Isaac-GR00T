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
"""Command-line interface for OSMO Ray cluster helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from groot.infra.job.cluster_spec import ClusterSpec
from groot.infra.job.constants import (
    DEFAULT_IMAGE,
    DEFAULT_RAY_CLUSTER_READY_INTERVAL_SECONDS,
    DEFAULT_RAY_CLUSTER_READY_TIMEOUT_SECONDS,
    DEFAULT_RAY_DASHBOARD_PORT,
    DEFAULT_RAY_PORT,
    DEFAULT_WORKDIR,
)
from groot.infra.job.ray_cluster import RayCluster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m groot.infra.job")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start-cluster",
        help="Launch an OSMO workflow that runs a Ray cluster.",
    )
    start.add_argument("--cluster-name", "--workflow-name", dest="cluster_name", required=True)
    start.add_argument("--pool", required=True, help="OSMO pool to use")
    start.add_argument(
        "--gpu-type",
        choices=("gb200", "gb300", "h100", "l40"),
        help="GPU type used only for public GPU-count defaults; pool name is supplied separately.",
    )
    start.add_argument("--num-nodes", type=int, default=1)
    start.add_argument("--image", default=DEFAULT_IMAGE)
    start.add_argument("--template", type=Path, help="OSMO workflow template path or template name")
    start.add_argument("--priority", default="high")
    start.add_argument("--workdir", default=DEFAULT_WORKDIR)
    start.add_argument("--workdir-path", default="")
    start.add_argument("--user-command", default="")
    start.add_argument("--runtime-env-json", default="{}")
    start.add_argument("--exec-timeout", default="7d")
    start.add_argument("--queue-timeout", default="1d")
    start.add_argument("--ray-port", type=int, default=DEFAULT_RAY_PORT)
    start.add_argument("--ray-dashboard-port", type=int, default=DEFAULT_RAY_DASHBOARD_PORT)
    start.add_argument("--cpus", type=int)
    start.add_argument("--gpus", type=int)
    start.add_argument("--memory-gib", type=int)
    start.add_argument("--storage-gib", type=int)
    start.add_argument(
        "--set",
        dest="extra_overrides",
        action="append",
        default=[],
        help="Additional OSMO template override in NAME=VALUE form; may be passed multiple times.",
    )
    start.add_argument(
        "--detach",
        "--no-wait",
        dest="detach",
        action="store_true",
        help="Submit the workflow without waiting for Ray dashboard readiness.",
    )
    start.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_RAY_CLUSTER_READY_TIMEOUT_SECONDS,
    )
    start.add_argument(
        "--wait-interval",
        type=float,
        default=DEFAULT_RAY_CLUSTER_READY_INTERVAL_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "start-cluster":
        result = RayCluster(
            ClusterSpec(
                cluster_name=args.cluster_name,
                pool=args.pool,
                gpu_type=args.gpu_type,
                num_nodes=args.num_nodes,
                image=args.image,
                template=args.template,
                priority=args.priority,
                workdir=args.workdir,
                workdir_path=args.workdir_path,
                user_command=args.user_command,
                runtime_env_json=args.runtime_env_json,
                exec_timeout=args.exec_timeout,
                queue_timeout=args.queue_timeout,
                ray_port=args.ray_port,
                ray_dashboard_port=args.ray_dashboard_port,
                cpus=args.cpus,
                gpus=args.gpus,
                memory_gib=args.memory_gib,
                storage_gib=args.storage_gib,
                extra_overrides=parse_overrides(args.extra_overrides),
            )
        ).launch(
            wait=not args.detach,
            readiness_timeout_seconds=args.wait_timeout,
            readiness_interval_seconds=args.wait_interval,
        )
        sys.stdout.write(f"Cluster ID: {result.cluster_id}\n")
        sys.stdout.write(f"Workflow ID: {result.workflow_id}\n")
        sys.stdout.write("Follow-up commands:\n")
        for command in result.followup_commands:
            sys.stdout.write(f"  {command}\n")
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


def parse_overrides(raw_overrides: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"Override must be in NAME=VALUE form: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key must be non-empty: {raw}")
        overrides[key] = value
    return overrides
