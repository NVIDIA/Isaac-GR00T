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

"""OSMO workflow template selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from groot.infra.job.constants import DEFAULT_TEMPLATE_NAME


@dataclass(frozen=True)
class WorkflowTemplate:
    """A concrete OSMO workflow template path."""

    path: Path

    @classmethod
    def init_from_reference(cls, template: str | Path | None) -> WorkflowTemplate:
        template_dir = Path(__file__).with_name("templates")
        if template is None:
            return cls(path=template_dir / f"{DEFAULT_TEMPLATE_NAME}.yaml")

        candidate = Path(template)
        if candidate.is_file():
            return cls(path=candidate)
        if candidate.suffix:
            raise FileNotFoundError(f"OSMO workflow template does not exist: {candidate}")

        named_template = template_dir / f"{candidate.name}.yaml"
        if named_template.is_file():
            return cls(path=named_template)
        raise FileNotFoundError(f"Unknown OSMO workflow template: {template}")
