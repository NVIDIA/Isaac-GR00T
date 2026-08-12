# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
        lookup_compute_type = "gb300" if compute_type == "colossus" else compute_type
        try:
            return self.families[lookup_compute_type]
        except KeyError as exc:
            valid = ", ".join(sorted(self.families))
            raise ValueError(
                f"Experiment {experiment} has no compute entry for {compute_type}; "
                f"valid compute types: {valid}"
            ) from exc

    def env_section(self) -> Mapping[str, str]:
        return {str(key): str(value) for key, value in self.env_vars.items()}
