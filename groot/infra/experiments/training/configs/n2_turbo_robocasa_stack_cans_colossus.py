# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precision can stacking on one Colossus GB300."""

from groot.infra.experiments.training.configs._robocasa_single_task_colossus import (
    make_robocasa_single_task_experiment,
)

EXPERIMENT = make_robocasa_single_task_experiment(
    dataset_path=(
        "datasets/robocasa_benchmark/new-robocasa/robocasa-datasets/v1.0/"
        "pretrain/composite/StackCans/20250731/lerobot"
    ),
    evaluation_task="StackCans",
    evaluation_num_trials=20,
    evaluation_local_checkpoint=(
        "/data/groot_ol/shared/checkpoints/agent007/"
        "train_eval_n2_turbo_robocasa_stack_cans_colossus_colossus_97d632d6b7/"
        "checkpoint-1250"
    ),
    evaluation_remote_checkpoint=(
        "msc://GearCheckpoints/trinity-ckpt-repository/agent007/"
        "colossus-auto-research-cache/stack-cans/97d632d6b7/checkpoint-1250"
    ),
    max_steps=1250,
    wandb_project="n2-turbo-robocasa-stack-cans-colossus",
)
