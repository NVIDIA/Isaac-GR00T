# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
