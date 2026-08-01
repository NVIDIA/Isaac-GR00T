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
    wandb_project="n2-turbo-robocasa-stack-cans-colossus",
)
