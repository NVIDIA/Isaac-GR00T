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

"""Owner metadata operations for OSMO workflow templates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import ClassVar

import yaml

from groot.infra.ci.codeowners import Codeowners
from groot.infra.ci.osmo_workflow import OsmoWorkflow
from groot.infra.ci.osmo_workflow_owner import OsmoWorkflowOwner


@dataclass
class OsmoWorkflowOwnerTemplate:
    """An OSMO workflow list file with owner metadata operations."""

    path: Path
    repo_root: Path = Path(".")
    codeowners: Codeowners | None = None

    _ROOT_OWNERS: ClassVar[set[str]] = {"@zhezh", "@cuonguyen"}
    _INACTIVE_OWNERS: ClassVar[set[str]] = {"@sreed"}

    @classmethod
    def init_from_path(
        cls,
        path: Path,
        repo_root: Path = Path("."),
        codeowners_path: Path | None = None,
    ) -> OsmoWorkflowOwnerTemplate:
        """Create a template object, optionally loading CODEOWNERS for backfill."""
        codeowners = Codeowners.init_from_file(codeowners_path) if codeowners_path else None
        return cls(path=path, repo_root=repo_root, codeowners=codeowners)

    @classmethod
    def init_all_from_paths(
        cls,
        paths: list[Path],
        repo_root: Path = Path("."),
        codeowners_path: Path | None = None,
    ) -> list[OsmoWorkflowOwnerTemplate]:
        """Create template objects for several workflow list files."""
        return [
            cls.init_from_path(path=path, repo_root=repo_root, codeowners_path=codeowners_path)
            for path in paths
        ]

    def get_all(self) -> list[OsmoWorkflowOwner]:
        """Read owner metadata from active workflow blocks."""
        return [
            OsmoWorkflowOwner.init_from_workflow_config(workflow_name, workflow_config)
            for workflow_name, workflow_config in self._workflow_configs()
            if "workflow" in workflow_config
        ]

    def get_test_owner(self, workflow_name: str) -> str:
        """Return the persisted Test owner string for a workflow."""
        for owner in self.get_all():
            if owner.workflow_name == workflow_name:
                return owner.as_test_owner()
        return "ci"

    def validate(self) -> list[str]:
        """Return validation errors for active workflow blocks without owners."""
        errors: list[str] = []
        for workflow_name, workflow_config in self._workflow_configs():
            if "workflow" not in workflow_config:
                continue
            owner = OsmoWorkflowOwner.init_from_workflow_config(workflow_name, workflow_config)
            if not owner.owners:
                errors.append(f"{self.path}: workflow '{workflow_name}' missing owner field")
        return errors

    def backfill(self, dry_run: bool = False) -> int:
        """Insert missing owner fields into active workflow blocks."""
        if self.codeowners is None:
            raise ValueError("codeowners is required for backfill")

        text = self.path.read_text()
        replacements: list[tuple[str, str]] = []

        for workflow_name, block_text, workflow_config in self._workflow_blocks_with_config():
            owner = OsmoWorkflowOwner.init_from_workflow_config(workflow_name, workflow_config)
            if "workflow" not in workflow_config or owner.owners:
                continue
            owners = self._resolve_workflow_owners(workflow_name, workflow_config)
            if not owners:
                raise ValueError(
                    f"{self.path}: unable to resolve owner for workflow '{workflow_name}'"
                )
            lines = block_text.splitlines(keepends=True)
            lines.insert(1, f"  owner: [{', '.join(owners)}]\n")
            replacements.append((block_text, "".join(lines)))

        if replacements and not dry_run:
            for old, new in replacements:
                text = text.replace(old, new, 1)
            self.path.write_text(text)

        return len(replacements)

    def to_json(self) -> str:
        """Return active workflow owners as sorted JSON for shell integrations."""
        return json.dumps(
            {owner.workflow_name: owner.owners for owner in self.get_all() if owner.owners},
            sort_keys=True,
        )

    def write_osmo_runner_yaml(self, output_path: Path) -> int:
        """Write a runner-compatible workflow list without owner metadata."""
        parsed = yaml.safe_load(self.path.read_text()) or {}
        if not isinstance(parsed, dict):
            raise TypeError(f"{self.path}: workflow list YAML must be a mapping")

        stripped_count = 0
        for workflow_name, workflow_config in parsed.items():
            if workflow_name == "environment" or not isinstance(workflow_config, dict):
                continue
            if "owner" in workflow_config:
                workflow_config.pop("owner")
                stripped_count += 1

        output_path.write_text(yaml.safe_dump(parsed, sort_keys=False))
        return stripped_count

    def _workflow_blocks(self) -> list[tuple[str, str]]:
        text = self.path.read_text()
        return [
            (key, block_text)
            for key, block_text in self._parse_top_level_blocks(text)
            if key != "environment"
        ]

    def _workflow_configs(self) -> list[tuple[str, dict[str, object]]]:
        return [
            (workflow_name, workflow_config)
            for workflow_name, _, workflow_config in self._workflow_blocks_with_config()
        ]

    def _workflow_blocks_with_config(self) -> list[tuple[str, str, dict[str, object]]]:
        return [
            (workflow_name, block_text, self._parse_workflow_config(block_text))
            for workflow_name, block_text in self._workflow_blocks()
        ]

    def _parse_top_level_blocks(self, text: str) -> list[tuple[str, str]]:
        lines = text.splitlines(keepends=True)
        blocks: list[tuple[str, str]] = []
        current_key: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if line and not line[0].isspace() and not line.startswith("#") and ":" in line:
                if current_key is not None:
                    blocks.append((current_key, "".join(current_lines)))
                current_key = line.split(":")[0].strip()
                current_lines = [line]
            elif current_key is not None:
                current_lines.append(line)

        if current_key is not None:
            blocks.append((current_key, "".join(current_lines)))

        return blocks

    def _parse_workflow_config(self, block_text: str) -> dict[str, object]:
        config: dict[str, object] = {}
        lines = block_text.splitlines(keepends=True)
        index = 1
        while index < len(lines):
            line = lines[index]
            field_match = re.match(r"^([ \t]+)([A-Za-z_][A-Za-z0-9_-]*):(.*)$", line)
            if not field_match:
                index += 1
                continue

            indent = len(field_match.group(1).replace("\t", "  "))
            key = field_match.group(2)
            value_lines = [line]
            index += 1

            while index < len(lines):
                next_line = lines[index]
                next_match = re.match(r"^([ \t]+)([A-Za-z_][A-Za-z0-9_-]*):(.*)$", next_line)
                if next_match and len(next_match.group(1).replace("\t", "  ")) <= indent:
                    break
                if next_line.strip():
                    value_lines.append(next_line)
                index += 1

            if key in {"owner", "workflow"}:
                parsed = yaml.safe_load("".join(value_lines))
                if isinstance(parsed, dict) and key in parsed:
                    config[key] = parsed[key]

        return config

    def _resolve_workflow_owners(
        self,
        workflow_name: str,
        workflow_config: dict[str, object],
    ) -> list[str]:
        if self.codeowners is None:
            raise ValueError("codeowners is required to resolve owners")

        workflow_filename = workflow_config.get("workflow")
        if not isinstance(workflow_filename, str):
            return []

        workflow_yaml = self.path.parent / workflow_filename
        workflow = OsmoWorkflow.from_workflow_yaml(workflow_name, workflow_yaml, self.repo_root)
        owner_path = workflow.entry_points[0] if workflow.entry_points else self.path.parent
        return self._resolve_owners(self._repo_relative(owner_path))

    def _repo_relative(self, path: Path) -> str:
        if path.is_absolute():
            try:
                return path.relative_to(self.repo_root).as_posix()
            except ValueError:
                return path.as_posix().lstrip("/")
        return path.as_posix()

    def _resolve_owners(self, file_path: str) -> list[str]:
        if self.codeowners is None:
            raise ValueError("codeowners is required to resolve owners")

        all_owners = self.codeowners.owners_for(file_path)
        specific = [
            owner.lstrip("@") for owner in all_owners if self._is_active_person_owner(owner)
        ]
        if specific:
            return specific[:2]
        fallback = [owner.lstrip("@") for owner in all_owners if owner not in self._INACTIVE_OWNERS]
        return fallback[:1]

    def _is_active_person_owner(self, owner: str) -> bool:
        return (
            owner not in self._ROOT_OWNERS
            and owner not in self._INACTIVE_OWNERS
            and owner != "@GR00TCodeOwners"
            and not owner.startswith("@project_")
        )
