# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contact-rich cutting-board scrubbing for the protected Auto Research suite."""

from groot.infra.experiments.training.configs._robocasa_single_task_colossus import (
    make_robocasa_single_task_experiment,
)

EXPERIMENT = make_robocasa_single_task_experiment(
    dataset_path=(
        "datasets/robocasa_benchmark/new-robocasa/robocasa-datasets/v1.0/"
        "pretrain/composite/ScrubCuttingBoard/20250805/lerobot"
    ),
    evaluation_task="ScrubCuttingBoard",
    evaluation_num_trials=20,
    evaluation_local_checkpoint=(
        "/data/groot_ol/shared/checkpoints/agent007/"
        "train_eval_n2_turbo_robocasa_scrub_cutting_board_colossus_colossus_640cdb7948/"
        "checkpoint-750"
    ),
    evaluation_remote_checkpoint=(
        "msc://GearCheckpoints/trinity-ckpt-repository/agent007/"
        "colossus-auto-research-cache/scrub-cutting-board/640cdb7948/checkpoint-750"
    ),
    max_steps=750,
    wandb_project="n2-turbo-robocasa-scrub-cutting-board-colossus",
)
