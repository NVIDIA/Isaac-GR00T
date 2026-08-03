# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation config for Python-authored one-launcher training experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingExperimentEvaluationConfig:
    """Describe an evaluation to run against a training experiment checkpoint."""

    task: str
    num_trials: int
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("Evaluation task must be a non-empty string")
        if isinstance(self.num_trials, bool) or not isinstance(self.num_trials, int):
            raise TypeError("Evaluation num_trials must be an integer")
        if self.num_trials <= 0:
            raise ValueError("Evaluation num_trials must be greater than 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("Evaluation seed must be an integer")
        if self.seed < 0:
            raise ValueError("Evaluation seed must be greater than or equal to 0")
