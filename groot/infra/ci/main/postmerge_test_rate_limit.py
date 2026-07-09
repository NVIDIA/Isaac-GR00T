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

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from groot.infra.ci import log
from groot.infra.ci.postmerge_test_rate_limit import PostmergeTestRateLimit

NO_DUE_WORKFLOWS_EXIT_CODE = 2
logger = log.init(__name__)


def _cmd_filter(args: argparse.Namespace) -> int:
    limiter = PostmergeTestRateLimit(window_hours=args.window_hours)
    due = limiter.filter_yaml(args.input, args.output)
    logger.info("Postmerge workflows due for nightly high-priority run: %d", len(due))
    for workflow_name in due:
        logger.info("  %s", workflow_name)
    if not due:
        return NO_DUE_WORKFLOWS_EXIT_CODE
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    limiter = PostmergeTestRateLimit(window_hours=args.window_hours)
    count = limiter.record_log(
        log_file=args.log_file,
        workflow_type=args.workflow_type,
        commit_sha=args.commit_sha,
        pipeline_url=args.pipeline_url,
    )
    logger.info("Recorded %d postmerge workflow result(s)", count)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily rate limit helpers for postmerge OSMO tests."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_filter = sub.add_parser(
        "filter",
        help="Filter a rendered workflow YAML to workflows without a recent result.",
    )
    p_filter.add_argument("--input", required=True, type=Path)
    p_filter.add_argument("--output", required=True, type=Path)
    p_filter.add_argument("--window-hours", type=int, default=24)
    p_filter.set_defaults(func=_cmd_filter)

    p_record = sub.add_parser(
        "record",
        help="Record workflow results from an osmo_runner log.json.",
    )
    p_record.add_argument("--log-file", required=True, type=Path)
    p_record.add_argument("--workflow-type", required=True)
    p_record.add_argument("--commit-sha", default="")
    p_record.add_argument("--pipeline-url", default="")
    p_record.add_argument("--window-hours", type=int, default=24)
    p_record.set_defaults(func=_cmd_record)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
