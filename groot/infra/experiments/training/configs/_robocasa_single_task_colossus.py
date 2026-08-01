# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared construction for single-task N2 Turbo RoboCasa Colossus runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from groot.infra.experiments.training.configs.n2_turbo_robocasa365_colossus import (
    COMPUTE_CONFIG,
)
from groot.infra.experiments.training.configs.n2_turbo_robocasa365_colossus import (
    DATA_CONFIG as BASE_DATA_CONFIG,
)
from groot.infra.experiments.training.configs.n2_turbo_robocasa365_colossus import (
    TRAINING_CONFIG as BASE_TRAINING_CONFIG,
)
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment

SINGLE_TASK_MAX_STEPS = 500


def make_robocasa_single_task_experiment(
    *,
    dataset_path: str,
    wandb_project: str,
) -> TrainingExperiment:
    """Build a one-GB300 experiment for one staged RoboCasa pretrain task."""
    data = deepcopy(BASE_DATA_CONFIG.data)
    data["datasets"][0]["dataset_paths"] = [dataset_path]
    data_config = replace(BASE_DATA_CONFIG, data=data)

    training_config = replace(
        BASE_TRAINING_CONFIG,
        training=replace(
            BASE_TRAINING_CONFIG.training,
            # At global batch 128, 500 steps expose 64k fixed-window samples.
            # DivideBuffetTrays reached a sustained all-component loss plateau
            # at this budget, so use it consistently for single-task variants.
            max_steps=SINGLE_TASK_MAX_STEPS,
            logging_steps=10,
            use_wandb=True,
            wandb_project=wandb_project,
        ),
        # One GB300 x 128 samples/GPU x one gradient-accumulation step.
        global_batch_size=128,
    )

    return TrainingExperiment(
        data_config=data_config,
        training_config=training_config,
        compute_config=COMPUTE_CONFIG,
    )
