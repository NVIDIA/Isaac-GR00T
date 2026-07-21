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

"""Ray cluster launch result."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClusterLaunchResult:
    """Result returned after submitting an OSMO Ray cluster workflow."""

    cluster_name: str
    workflow_id: str
    outputs: tuple[str, ...]
    followup_commands: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cluster_id(self) -> str:
        """Stable cluster identifier used by downstream launchers."""
        return self.workflow_id
