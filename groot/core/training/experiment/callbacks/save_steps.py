# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Keep configured checkpoint cadence when restoring Trainer state."""

import logging

from transformers import TrainerCallback


class OverrideSaveStepsCallback(TrainerCallback):
    """Reapply the configured cadence after HF loads trainer_state.json."""

    def __init__(self, save_steps: int):
        self.save_steps = save_steps

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if state.save_steps != self.save_steps:
            logging.info(
                "Overriding save_steps from checkpoint state (%s) with config value (%s)",
                state.save_steps,
                self.save_steps,
            )
        args.save_steps = self.save_steps
        state.save_steps = self.save_steps
