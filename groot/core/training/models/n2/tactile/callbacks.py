# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Trainer callbacks for tactile GrootN2 training."""

from __future__ import annotations

from typing import Any

import torch
from transformers.trainer_callback import TrainerCallback


class TactileLossWarmupCallback(TrainerCallback):
    """Ramp the future-force loss weight linearly from 0 to 1 over the first steps.

    Keeps the new future-force objective from dominating the video/action losses early in
    training. Writes ``model.tactile_loss_warmup_factor`` in place at every step begin.

    Args:
        model: A GrootN2 with ``tactile_cfg.predict_future`` (it owns the factor buffer).
        warmup_steps: Steps over which the factor ramps to 1; must be positive.
    """

    def __init__(self, model: Any, warmup_steps: int):
        if not isinstance(getattr(model, "tactile_loss_warmup_factor", None), torch.Tensor):
            raise TypeError(
                "TactileLossWarmupCallback needs a model with tactile_cfg.predict_future"
            )
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        self.model = model
        self.warmup_steps = warmup_steps

    def on_step_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        # Read the buffer each step: moving the model to another dtype/device replaces it.
        # fill_ keeps its identity for compiled graphs.
        factor = min(1.0, state.global_step / self.warmup_steps)
        self.model.tactile_loss_warmup_factor.fill_(factor)


def trainer_callbacks(model: Any) -> list[TrainerCallback]:
    """Return the tactile callbacks ``model`` trains with (none without a loss warmup)."""
    spec = getattr(model, "tactile_spec", None)
    if spec is None or not spec.predict_future or spec.tactile_loss_warmup_steps <= 0:
        return []
    return [TactileLossWarmupCallback(model, spec.tactile_loss_warmup_steps)]
