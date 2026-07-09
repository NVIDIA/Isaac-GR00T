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

"""Evaluation report serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from groot.infra.eval.metric_summary import MetricSummary
from groot.infra.eval.resolved_artifacts import ResolvedArtifacts


@dataclass(frozen=True)
class EvalReport:
    artifacts: ResolvedArtifacts
    traj_ids: list[int]
    steps: int
    action_horizon: int
    results: dict[str, MetricSummary]

    def to_json_text(self) -> str:
        payload = {
            "model_path": str(self.artifacts.model_path),
            "dataset_path": str(self.artifacts.dataset_path),
            "traj_ids": self.traj_ids,
            "steps": self.steps,
            "action_horizon": self.action_horizon,
            "results": {
                label: {
                    "per_trajectory": summary.per_trajectory,
                    "average": summary.average,
                }
                for label, summary in self.results.items()
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_text())
