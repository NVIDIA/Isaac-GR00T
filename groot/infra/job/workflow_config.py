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

"""Typed rendered OSMO workflow config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class WorkflowConfig:
    """Rendered OSMO workflow YAML."""

    data: Mapping[str, Any]

    @classmethod
    def init_from_yaml_text(cls, text: str) -> WorkflowConfig:
        raw = yaml.safe_load(text) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("OSMO dry-run output must be a YAML mapping")
        return cls(data=raw)

    def to_yaml_text(self) -> str:
        return yaml.safe_dump(dict(self.data))
