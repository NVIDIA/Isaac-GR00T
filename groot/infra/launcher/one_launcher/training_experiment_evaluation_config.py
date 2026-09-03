# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Evaluation config for Python-authored one-launcher training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class TrainingExperimentEvaluationConfig:
    """Describe an evaluation to run against a training experiment checkpoint."""

    task: str
    num_trials: int
    num_parallel_trials: int = 1
    inference_steps: int | None = None
    seed: int = 0
    persist: bool = False
    local_checkpoint: str | None = None
    remote_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("Evaluation task must be a non-empty string")
        if isinstance(self.num_trials, bool) or not isinstance(self.num_trials, int):
            raise TypeError("Evaluation num_trials must be an integer")
        if self.num_trials <= 0:
            raise ValueError("Evaluation num_trials must be greater than 0")
        if isinstance(self.num_parallel_trials, bool) or not isinstance(
            self.num_parallel_trials, int
        ):
            raise TypeError("Evaluation num_parallel_trials must be an integer")
        if self.num_parallel_trials <= 0:
            raise ValueError("Evaluation num_parallel_trials must be greater than 0")
        if self.inference_steps is not None:
            if isinstance(self.inference_steps, bool) or not isinstance(self.inference_steps, int):
                raise TypeError("Evaluation inference_steps must be an integer")
            if self.inference_steps <= 0:
                raise ValueError("Evaluation inference_steps must be greater than 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("Evaluation seed must be an integer")
        if self.seed < 0:
            raise ValueError("Evaluation seed must be greater than or equal to 0")
        if not isinstance(self.persist, bool):
            raise TypeError("Evaluation persist must be a boolean")
        self.validate_local_checkpoint()
        self.validate_remote_checkpoint()
        if self.local_checkpoint is not None and self.remote_checkpoint is not None:
            local_name = Path(self.local_checkpoint).name
            remote_name = self.remote_checkpoint.rstrip("/").rsplit("/", maxsplit=1)[-1]
            if local_name != remote_name:
                raise ValueError(
                    "Evaluation local_checkpoint and remote_checkpoint must identify the same checkpoint directory"
                )

    def validate_local_checkpoint(self) -> None:
        if self.local_checkpoint is None:
            return
        if not isinstance(self.local_checkpoint, str) or not self.local_checkpoint.strip():
            raise ValueError("Evaluation local_checkpoint must be a non-empty absolute path")
        path = Path(self.local_checkpoint)
        if (
            not path.is_absolute()
            or "://" in self.local_checkpoint
            or "{" in self.local_checkpoint
            or "}" in self.local_checkpoint
        ):
            raise ValueError("Evaluation local_checkpoint must be a non-empty absolute path")

    def validate_remote_checkpoint(self) -> None:
        if self.remote_checkpoint is None:
            return
        if not isinstance(self.remote_checkpoint, str):
            raise TypeError("Evaluation remote_checkpoint must be an exact msc:// checkpoint URI")
        parsed = urlsplit(self.remote_checkpoint)
        checkpoint_name = parsed.path.rsplit("/", maxsplit=1)[-1]
        if (
            parsed.scheme != "msc"
            or not parsed.netloc
            or not parsed.path.removeprefix("/")
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", checkpoint_name) is None
            or self.remote_checkpoint != self.remote_checkpoint.strip()
            or self.remote_checkpoint.endswith("/")
            or parsed.query
            or parsed.fragment
            or any(character in self.remote_checkpoint for character in "*?[]{}")
        ):
            raise ValueError("Evaluation remote_checkpoint must be an exact msc:// checkpoint URI")
