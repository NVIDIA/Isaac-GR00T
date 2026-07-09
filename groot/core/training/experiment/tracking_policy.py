# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experiment Hub registration policy helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

_TRACKING_POLICIES = {"auto", "enabled", "disabled"}
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def experiment_tracking_policy(config: Any) -> str:
    """Resolve dashboard registration policy for rank-0 fallback only."""
    if _env_flag_enabled("SKIP_EXPERIMENT_REGISTRATION") or _env_flag_enabled("DRY_RUN"):
        return "disabled"
    raw = (
        os.environ.get("GEAR_EXPERIMENT_TRACKING")
        or getattr(config.training, "experiment_tracking", "auto")
        or "auto"
    )
    policy = str(raw).strip().lower()
    if policy not in _TRACKING_POLICIES:
        logging.warning(
            "Invalid training.experiment_tracking=%r; falling back to 'auto'",
            raw,
        )
        return "auto"
    return policy


def _has_remote_checkpoint_intent(config: Any) -> bool:
    training = config.training
    if getattr(training, "upload_checkpoints", False):
        return True
    return bool(getattr(training, "msc_checkpoint_url", None))


def should_fallback_register_experiment(config: Any) -> tuple[bool, str]:
    """Return whether rank 0 should create a fallback dashboard experiment.

    Launcher-side registration is preferred because it creates identity-backed
    experiment and training-run records. This fallback is only for legacy real
    training jobs that still produce durable checkpoints.
    """
    policy = experiment_tracking_policy(config)
    if policy == "disabled":
        return False, "experiment tracking disabled"
    if policy == "enabled":
        return True, "experiment tracking explicitly enabled"
    if _has_remote_checkpoint_intent(config):
        return True, "auto policy detected remote checkpoint intent"
    return False, "auto policy skipped non-checkpoint-producing fallback"
