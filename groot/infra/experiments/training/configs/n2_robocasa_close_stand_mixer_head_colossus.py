# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-mixer head closing on one Colossus GB300."""

from groot.infra.experiments.training.configs._robocasa_single_task_colossus import (
    make_robocasa_single_task_experiment,
)

EXPERIMENT = make_robocasa_single_task_experiment(
    dataset_path=(
        "datasets/robocasa_benchmark/new-robocasa/robocasa-datasets/v1.0/"
        "pretrain/atomic/CloseStandMixerHead/20250820/lerobot"
    ),
    evaluation_task="CloseStandMixerHead",
    evaluation_num_trials=20,
    evaluation_seed=0,
    max_steps=800,
    wandb_project="n2-robocasa-close-stand-mixer-head-colossus",
)
