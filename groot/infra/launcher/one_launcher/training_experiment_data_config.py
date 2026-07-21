# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data config object for Python-authored one-launcher experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingExperimentDataConfig:
    data: Mapping[str, Any]
    launcher_data: Mapping[str, Any] = field(default_factory=dict)
