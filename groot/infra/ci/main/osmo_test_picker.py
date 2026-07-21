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
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
r"""CLI for tach-based OSMO workflow filtering.

Usage (in CI):
    uv run --active -m groot.infra.ci.main.osmo_test_picker \\
        --workflow-dir tests/vla_workflows \\
        --input tests/vla_workflows/list_rendered.yaml \\
        --output tests/vla_workflows/list_tach_filtered.yaml \\
        --base $CI_MERGE_REQUEST_DIFF_BASE_SHA

Exit codes:
    0  - Success (workflows written, or validate mode)
    2  - No workflows selected (sentinel: nothing to run for this MR)
    1  - Unexpected error
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from groot.infra.ci.osmo_filter_result import OsmoFilterResult
from groot.infra.ci.osmo_test_picker import OsmoTestPicker
from groot.infra.ci.osmo_workflow import OsmoWorkflow
from groot.infra.ci.pipeline import Pipeline
from groot.infra.ci.schedule_type import POSTMERGE, PREMERGE

logger = logging.getLogger(__name__)


def _names_with_schedule_type(
    workflows: list[OsmoWorkflow],
    schedule_type: str | None,
) -> set[str] | None:
    if schedule_type is None:
        return None
    return {wf.name for wf in workflows if schedule_type in wf.schedule_types}


def _apply_schedule_type_filter(
    result: OsmoFilterResult,
    workflows: list[OsmoWorkflow],
    schedule_type: str | None,
) -> OsmoFilterResult:
    """Restrict selected workflows to those that include schedule_type."""
    if schedule_type is None:
        return result

    wf_by_name = {wf.name: wf for wf in workflows}
    schedule_excluded = [
        name
        for name in result.selected
        if schedule_type not in wf_by_name.get(name, OsmoWorkflow("", Path(""), [])).schedule_types
    ]
    if not schedule_excluded:
        return result

    logger.info(
        "[tach-filter] schedule_type excludes (no %s): %s",
        schedule_type,
        ", ".join(schedule_excluded),
    )
    excluded_set = set(schedule_excluded)
    return OsmoFilterResult(
        total=result.total,
        selected=[n for n in result.selected if n not in excluded_set],
        skipped=result.skipped + schedule_excluded,
    )


def _write_filter_result(path: Path | None, mode: str, result: OsmoFilterResult) -> None:
    if path is None:
        return
    data = {
        "mode": mode,
        "total": result.total,
        "selected_count": len(result.selected),
        "skipped_count": len(result.skipped),
        "selected": result.selected,
        "skipped": result.skipped,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    logger.info("[tach-filter] filter result written to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter OSMO workflows using tach dependency analysis."
    )
    parser.add_argument(
        "--workflow-dir",
        required=True,
        type=Path,
        help="Directory containing list.template.yaml and workflow YAML files.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Rendered (envsubst) workflow list YAML to filter.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the filtered workflow list YAML.",
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Base git ref to diff against (default: main).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Validate mode: log what would be skipped but write all workflows "
            "(observe impact without filtering)."
        ),
    )
    parser.add_argument(
        "--schedule-type",
        choices=[PREMERGE, POSTMERGE],
        default=None,
        help=(
            "Restrict output to workflows containing this schedule_type. MR pipelines "
            "default to premerge when this is omitted."
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Path to write tach_filter_result.json for observability.",
    )
    args = parser.parse_args()

    picker = OsmoTestPicker(args.workflow_dir)
    workflows = picker.get_workflows()

    pipeline = Pipeline()
    schedule_type = args.schedule_type
    if schedule_type is None and pipeline.get_pipeline_source() == "merge_request_event":
        schedule_type = PREMERGE
    schedule_type_names = _names_with_schedule_type(workflows, schedule_type)

    # Get changed files and build dep graph
    changed_all = OsmoTestPicker.get_changed_files(args.base)
    if changed_all is None:
        logger.warning("[tach-filter] WARNING: git diff failed, falling back to all workflows")
        count = picker.write_filtered_yaml(
            args.input,
            args.output,
            selected_names=schedule_type_names,
        )
        _write_filter_result(
            args.metrics_output, "fallback", OsmoFilterResult(total=count, selected=[], skipped=[])
        )
        return

    changed_py = [f for f in changed_all if f.endswith(".py")]

    logger.info(
        "[tach-filter] changed files: %d total, %d Python",
        len(changed_all),
        len(changed_py),
    )

    dep_graph = picker.get_dep_graph()
    if not dep_graph:
        logger.warning(
            "[tach-filter] WARNING: tach dep graph is empty, falling back to all workflows"
        )
        count = picker.write_filtered_yaml(
            args.input,
            args.output,
            selected_names=schedule_type_names,
        )
        _write_filter_result(
            args.metrics_output, "fallback", OsmoFilterResult(total=count, selected=[], skipped=[])
        )
        logger.info("[tach-filter] fallback: writing all %d workflows", count)
        return

    affected = picker.compute_affected(changed_py, dep_graph)
    result = picker.select_workflows(workflows, affected, changed_all)
    result = _apply_schedule_type_filter(result, workflows, schedule_type)

    logger.info(
        "[tach-filter] %d workflows total: %d selected, %d skipped",
        result.total,
        len(result.selected),
        len(result.skipped),
    )
    if result.skipped:
        logger.info("[tach-filter] skipping: %s", ", ".join(result.skipped))
    if result.selected:
        logger.info("[tach-filter] selected: %s", ", ".join(result.selected))

    # Validate mode: log impact but write all workflows
    if args.validate:
        _write_filter_result(args.metrics_output, "validate", result)
        count = picker.write_filtered_yaml(
            args.input,
            args.output,
            selected_names=schedule_type_names,
        )
        logger.info("[tach-filter] validate mode: writing all %d workflows (not filtering)", count)
        return

    # Filter mode: write only selected workflows
    _write_filter_result(args.metrics_output, "filter", result)
    if not result.selected:
        logger.info("[tach-filter] no workflows affected by this MR — skipping")
        sys.exit(2)

    count = picker.write_filtered_yaml(args.input, args.output, selected_names=set(result.selected))
    logger.info("[tach-filter] wrote %d/%d workflows to %s", count, result.total, args.output)
    if count == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
