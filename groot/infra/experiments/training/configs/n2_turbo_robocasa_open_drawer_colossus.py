# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drawer opening for the protected Auto Research suite."""

from groot.infra.experiments.training.configs._robocasa_single_task_colossus import (
    make_robocasa_single_task_experiment,
)

EXPERIMENT = make_robocasa_single_task_experiment(
    dataset_path=(
        "datasets/robocasa_benchmark/new-robocasa/robocasa-datasets/v1.0/"
        "pretrain/atomic/OpenDrawer/20250819/lerobot"
    ),
    evaluation_task="OpenDrawer",
    evaluation_num_trials=20,
    evaluation_seed=0,
    max_steps=1250,
    wandb_project="n2-turbo-robocasa-open-drawer-colossus",
)
