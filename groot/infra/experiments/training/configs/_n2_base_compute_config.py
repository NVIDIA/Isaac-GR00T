# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared N2 compute config fragment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)


@dataclass(frozen=True)
class N2BaseComputeConfig:
    h100_batch_size: int | None = None
    gb300_batch_size: int | None = None
    env_vars: Mapping[str, str] = field(default_factory=dict)

    def to_compute_config(self) -> TrainingExperimentComputeConfig:
        families = {}
        if self.h100_batch_size is not None:
            families["h100"] = ExperimentComputeConfig(per_device_batch_size=self.h100_batch_size)
        if self.gb300_batch_size is not None:
            families["gb300"] = ExperimentComputeConfig(per_device_batch_size=self.gb300_batch_size)
        if not families:
            raise ValueError("N2 compute config requires at least one compute-family batch size")
        return TrainingExperimentComputeConfig(families=families, env_vars=self.env_vars)
