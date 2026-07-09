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

"""Owner metadata for one OSMO integration test workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class OsmoWorkflowOwner:
    """Owner metadata for one OSMO workflow block."""

    workflow_name: str
    owners: list[str]

    @classmethod
    def init_from_workflow_config(
        cls,
        workflow_name: str,
        workflow_config: Mapping[str, object],
    ) -> OsmoWorkflowOwner:
        """Create owner metadata from parsed OSMO workflow fields."""
        owner_value = workflow_config.get("owner")
        if owner_value is None:
            return cls(workflow_name=workflow_name, owners=[])
        if isinstance(owner_value, str):
            owners = [owner_value.strip().lstrip("@")] if owner_value.strip() else []
        elif isinstance(owner_value, list):
            owners = [item.strip().lstrip("@") for item in owner_value if isinstance(item, str)]
        else:
            owners = []
        return cls(workflow_name=workflow_name, owners=owners)

    def as_test_owner(self) -> str:
        """Return the persisted Test owner string."""
        return ",".join(self.owners) if self.owners else "ci"
