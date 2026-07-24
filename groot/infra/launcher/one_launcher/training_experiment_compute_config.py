# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute config object for Python-authored one-launcher experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig


@dataclass(frozen=True)
class TrainingExperimentComputeConfig:
    families: Mapping[str, ExperimentComputeConfig]
    env_vars: Mapping[str, str] = field(default_factory=dict)

    def family(self, *, experiment: str, compute_type: str) -> ExperimentComputeConfig:
        try:
            return self.families[compute_type]
        except KeyError as exc:
            valid = ", ".join(sorted(self.families))
            raise ValueError(
                f"Experiment {experiment} has no compute entry for {compute_type}; "
                f"valid compute types: {valid}"
            ) from exc

    def env_section(self) -> Mapping[str, str]:
        return {str(key): str(value) for key, value in self.env_vars.items()}
