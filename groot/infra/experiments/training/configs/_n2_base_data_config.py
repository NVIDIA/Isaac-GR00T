# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
