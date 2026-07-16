# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared N1.7ES compute config fragment."""

from __future__ import annotations

from dataclasses import dataclass

from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)


@dataclass(frozen=True)
class N17ESBaseComputeConfig:
    h100_batch_size: int
    gb300_batch_size: int | None = None

    def to_compute_config(self) -> TrainingExperimentComputeConfig:
        families = {"h100": ExperimentComputeConfig(per_device_batch_size=self.h100_batch_size)}
        if self.gb300_batch_size is not None:
            families["gb300"] = ExperimentComputeConfig(per_device_batch_size=self.gb300_batch_size)
        return TrainingExperimentComputeConfig(families=families)
