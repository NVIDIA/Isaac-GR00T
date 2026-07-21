# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training config object for Python-authored one-launcher experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingExperimentTrainingConfig:
    training: Any
    global_batch_size: int
    model_name: str
    model: Mapping[str, Any]
