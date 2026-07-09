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
"""OsmoWorkflow dataclass and entry-point extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

from groot.infra.ci.schedule_type import DEFAULT_SCHEDULE_TYPES

# Match groot/...py (not followed by a word char, so .pyc doesn't match)
_PY_RE = re.compile(r"(groot/[^\s]+\.py)(?!\w)")
# Match bash groot/...sh
_SH_RE = re.compile(r"\bbash\s+(groot/[^\s]+\.sh)(?!\w)")


@dataclass
class OsmoWorkflow:
    """An OSMO workflow with its auto-detected Python entry points."""

    name: str
    yaml_path: Path
    entry_points: list[Path]
    schedule_types: frozenset[str] = field(default_factory=lambda: DEFAULT_SCHEDULE_TYPES)
    owner: list[str] = field(default_factory=list)

    @staticmethod
    def extract_python_entry_points(yaml_text: str, script_root: Path) -> list[Path]:
        """Extract Python entry points from workflow YAML text.

        Finds groot/...py paths directly in the YAML. For bash groot/...sh
        references, reads the shell script and extracts Python paths from it.
        Returns a deduplicated sorted list.
        """
        found: set[str] = set()

        # Direct Python paths in the YAML
        for m in _PY_RE.finditer(yaml_text):
            found.add(m.group(1))

        # Bash script references — recursively parse the script
        for m in _SH_RE.finditer(yaml_text):
            sh_path = script_root / m.group(1)
            if sh_path.exists():
                try:
                    sh_text = sh_path.read_text()
                    for py_m in _PY_RE.finditer(sh_text):
                        found.add(py_m.group(1))
                except OSError as e:
                    logging.warning("Failed to read shell script %s: %s", sh_path, e)

        return [Path(p) for p in sorted(found)]

    @classmethod
    def from_workflow_yaml(cls, name: str, yaml_path: Path, script_root: Path) -> OsmoWorkflow:
        """Create an OsmoWorkflow by auto-detecting entry points from the YAML."""
        try:
            yaml_text = yaml_path.read_text()
        except OSError:
            return cls(name=name, yaml_path=yaml_path, entry_points=[])
        entry_points = cls.extract_python_entry_points(yaml_text, script_root)
        return cls(name=name, yaml_path=yaml_path, entry_points=entry_points)

    def get_source_dirs(self) -> list[Path]:
        """Return the parent directories of all entry points (for non-Python fallback)."""
        seen: set[Path] = set()
        result: list[Path] = []
        for ep in self.entry_points:
            p = ep.parent
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result
