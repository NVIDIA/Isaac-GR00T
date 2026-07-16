# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared N2 data config fragment."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from groot.infra.launcher.one_launcher.training_experiment_data_config import TrainingExperimentDataConfig


@dataclass(frozen=True)
class N2BaseDataConfig:
    data: Mapping[str, Any]

    def to_data_config(self) -> TrainingExperimentDataConfig:
        return TrainingExperimentDataConfig(
            data=deepcopy(self.data),
            launcher_data={"no_download_cache": True},
        )
