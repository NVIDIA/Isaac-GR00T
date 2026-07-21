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
"""Tach-based OSMO workflow filter.

Uses tach map --direction dependents to build a reverse dependency graph.
Auto-detects each workflow's Python entry point from its YAML definition,
then selects only workflows whose entry point is transitively reached by
the set of changed files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import subprocess

from groot.infra.ci.osmo_filter_result import OsmoFilterResult
from groot.infra.ci.osmo_workflow import OsmoWorkflow
from groot.infra.ci.osmo_workflow_owner_template import OsmoWorkflowOwnerTemplate

logger = logging.getLogger(__name__)

# Match the workflow: field inside a YAML block (indented key)
_WORKFLOW_FILE_RE = re.compile(r'^\s+workflow:\s*["\']?([^"\'#\n\s]+)["\']?', re.MULTILINE)
# Match schedule_type: [values] inside a YAML block (inline list)
_SCHEDULE_TYPE_RE = re.compile(r"^\s+schedule_type:\s*\[([^\]]*)\]", re.MULTILINE)
# Match an entire schedule_type: line (for stripping from output)
_SCHEDULE_TYPE_LINE_RE = re.compile(r"^[ \t]+schedule_type:.*\n?", re.MULTILINE)


class OsmoTestPicker:
    def __init__(self, workflow_dir: Path, tach_toml: Path = Path("tach.toml")):
        self.workflow_dir = workflow_dir
        # Script root is the project root (where groot/ lives).
        # Derived from tach_toml location; defaults to CWD via Path(".").
        self.script_root: Path = tach_toml.parent

    @staticmethod
    def get_changed_files(base: str) -> list[str] | None:
        """Get files changed between base ref and HEAD.

        Returns None if git diff fails (caller should fall back to all workflows).
        """
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["git", "diff", "--name-only", base, "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning("Failed to get changed files: %s", result.stderr)
                return None
        return [f for f in result.stdout.strip().split("\n") if f]

    @staticmethod
    def _parse_top_level_blocks(text: str) -> tuple[str | None, dict[str, str]]:
        """Parse flat YAML into (environment_block, {workflow_name: block_text}).

        Uses the same line-by-line heuristic as the zombie/shard filters in
        osmo_runner_execute.sh: a top-level key is a non-indented, non-comment
        line containing ':'.
        """
        lines = text.splitlines(keepends=True)
        blocks: dict[str, list[str]] = {}
        current_key: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if line and not line[0].isspace() and not line.startswith("#") and ":" in line:
                if current_key is not None:
                    blocks[current_key] = current_lines
                current_key = line.split(":")[0].strip()
                current_lines = [line]
            elif current_key is not None:
                current_lines.append(line)

        if current_key is not None:
            blocks[current_key] = current_lines

        env_lines = blocks.pop("environment", None)
        env_block = "".join(env_lines) if env_lines else None
        return env_block, {k: "".join(v) for k, v in blocks.items()}

    def get_workflows(self) -> list[OsmoWorkflow]:
        """Parse list.template.yaml; resolve each workflow's YAML and entry points."""
        template = self.workflow_dir / "list.template.yaml"
        text = template.read_text()
        _, blocks = self._parse_top_level_blocks(text)
        owners_by_workflow = {
            owner.workflow_name: owner.owners
            for owner in OsmoWorkflowOwnerTemplate.init_from_path(template).get_all()
        }

        workflows: list[OsmoWorkflow] = []
        for name, block_text in blocks.items():
            m = _WORKFLOW_FILE_RE.search(block_text)
            if not m:
                continue
            yaml_filename = m.group(1)
            yaml_path = self.workflow_dir / yaml_filename
            workflow = OsmoWorkflow.from_workflow_yaml(name, yaml_path, self.script_root)
            workflow.owner = owners_by_workflow.get(name, [])

            # Parse optional schedule_type list from the template block
            st_match = _SCHEDULE_TYPE_RE.search(block_text)
            if st_match:
                items = [s.strip().strip("'\"") for s in st_match.group(1).split(",") if s.strip()]
                if items:
                    workflow.schedule_types = frozenset(items)

            workflows.append(workflow)
        return workflows

    def get_dep_graph(self) -> dict[str, list[str]]:
        """Run `tach map --direction dependents`. No tach.toml modification needed."""
        try:
            result = subprocess.run(
                ["uv", "run", "tach", "map", "--direction", "dependents"],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except OSError as e:
            logger.warning("tach map failed: %s", e)
            return {}
        if result.returncode != 0:
            logger.warning("tach map failed: %s", result.stderr)
            return {}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.warning("tach map output invalid JSON: %s", e)
            return {}

    def compute_affected(self, changed_py: list[str], dep_graph: dict[str, list[str]]) -> set[str]:
        """DFS transitive closure — mirrors tach_file_level_picker.py exactly."""
        affected: set[str] = set()
        stack = list(changed_py)
        while stack:
            f = stack.pop()
            if f in affected:
                continue
            affected.add(f)
            for dependent in dep_graph.get(f, []):
                if dependent not in affected:
                    stack.append(dependent)
        return affected

    def select_workflows(
        self,
        workflows: list[OsmoWorkflow],
        affected_files: set[str],
        changed_all: list[str],
    ) -> OsmoFilterResult:
        """Decide which workflows to run.

        A workflow is included if:
          1. Its entry point is in affected_files (tach dep graph trace), OR
          2. No entry points could be detected (conservative: always run).

        Safety bypass: if any changed file is either non-Python or a Python file
        outside the first-party source root (groot/), run all workflows since
        the tach graph can't reason about those changes (e.g. gear_cli Python files
        under external_dependencies/).
        """
        # Any non-Python change, or a Python file outside tach source roots
        # (groot/) -> can't rely on tach graph, run everything.
        if any(
            not f.endswith(".py") or not f.startswith("groot/")
            for f in changed_all
        ):
            return OsmoFilterResult(
                total=len(workflows),
                selected=[wf.name for wf in workflows],
                skipped=[],
            )

        selected: list[str] = []
        skipped: list[str] = []

        for wf in workflows:
            if not wf.entry_points:
                # Conservative: can't determine entry point, always run
                selected.append(wf.name)
                continue

            ep_strs = {str(ep) for ep in wf.entry_points}
            if ep_strs & affected_files:
                selected.append(wf.name)
                continue

            skipped.append(wf.name)

        return OsmoFilterResult(
            total=len(workflows),
            selected=selected,
            skipped=skipped,
        )

    def write_filtered_yaml(
        self,
        rendered_path: Path,
        output_path: Path,
        selected_names: set[str] | None,
    ) -> int:
        """Write filtered YAML with only the selected workflows.

        Args:
            rendered_path: Path to the envsubst-rendered list YAML.
            output_path: Where to write the filtered YAML.
            selected_names: Set of workflow names to include, or None to include all.

        Returns:
            Number of workflows written.
        """
        text = rendered_path.read_text()
        env_block, blocks = self._parse_top_level_blocks(text)

        if selected_names is None:
            keys_to_write = list(blocks.keys())
        else:
            keys_to_write = [k for k in blocks if k in selected_names]

        parts: list[str] = []
        if env_block:
            parts.append(env_block)
            parts.append("\n")
        for key in keys_to_write:
            # Strip schedule_type: lines so downstream consumers (osmo_runner,
            # osmo_cache, ci_state_filter, sharding) never see the field.
            cleaned = _SCHEDULE_TYPE_LINE_RE.sub("", blocks[key])
            parts.append(cleaned)
            parts.append("\n")

        output_path.write_text("".join(parts))
        return len(keys_to_write)
