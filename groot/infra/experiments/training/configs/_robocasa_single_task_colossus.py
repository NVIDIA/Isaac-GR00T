# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared construction for single-task N2 Turbo RoboCasa Colossus runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from groot.infra.experiments.training.configs.n2_robocasa365 import (
    EXPERIMENT as BASE_ROBOCASA365_EXPERIMENT,
)
from groot.infra.experiments.training.configs.n2_turbo_airplane_colossus import (
    TRAINING_CONFIG as BASE_TURBO_TRAINING_CONFIG,
)
from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_evaluation_config import (
    TrainingExperimentEvaluationConfig,
)

SINGLE_TASK_MAX_STEPS = 500

# Reuse the canonical RoboCasa-365 dataset metadata while keeping this module a
# construction helper rather than exposing an unsupported aggregate experiment.
BASE_DATA_CONFIG = replace(
    BASE_ROBOCASA365_EXPERIMENT.data_config,
    data={
        **deepcopy(BASE_ROBOCASA365_EXPERIMENT.data_config.data),
        "multiprocessing_context": "fork",
        # Fail fast when a staged dataset is incomplete instead of generating
        # statistics independently on each Ray worker.
        "auto_generate_stats": False,
    },
)

_ROBOCASA365_TRAINING_CONFIG = BASE_ROBOCASA365_EXPERIMENT.training_config
_TURBO_ROBOCASA_MODEL = deepcopy(BASE_TURBO_TRAINING_CONFIG.model)
# RoboCasa provides 16 action steps. Keep the checkpoint-compatible 48-token
# action head, but do not pad every sample with 32 masked tokens.
_TURBO_ROBOCASA_MODEL["action-cfg"]["max-action-horizon"] = 16
# RoboCasa has exactly three real camera modalities. Avoid appending the fully
# masked zero view inherited from the four-camera Airplane profile.
_TURBO_ROBOCASA_MODEL["max_n_cameras"] = 3

BASE_TRAINING_CONFIG = replace(
    _ROBOCASA365_TRAINING_CONFIG,
    training=replace(
        _ROBOCASA365_TRAINING_CONFIG.training,
        output_dir="/data/groot_ol/shared/checkpoints/{USER_SLUG}/{JOB_NAME}",
        logging_steps=1000,
        torch_compile=False,
        dataloader_num_workers=4,
        use_wandb=False,
        # The RoboCasa checkpoint-80000 is Wan2.1 and is not Turbo-compatible.
        start_from_checkpoint=BASE_TURBO_TRAINING_CONFIG.training.start_from_checkpoint,
        upload_checkpoints=False,
        msc_checkpoint_url="",
        msc_profile_url="",
        enable_profiling=False,
        save_final_model=True,
        # The async saver forks inside a Ray actor and has triggered
        # task_receiver objects_valid crashes; prefer resumable sync saves.
        use_async_checkpoint_save=False,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        log_distributed_grad_norm=False,
        straggler=replace(
            _ROBOCASA365_TRAINING_CONFIG.training.straggler,
            enabled=False,
        ),
        use_ddp=True,
        ddp_bucket_cap_mb=256,
    ),
    model_name=BASE_TURBO_TRAINING_CONFIG.model_name,
    model=_TURBO_ROBOCASA_MODEL,
)

# One Launcher maps Colossus to the gb300 compute family while enforcing one
# training GPU per station.
COMPUTE_CONFIG = TrainingExperimentComputeConfig(
    families={"gb300": ExperimentComputeConfig(per_device_batch_size=128)},
    env_vars={
        "GEAR_PRESERVE_CHECKPOINT_ROOT": "1",
        "GROOT_DISTRIBUTED_TIMEOUT_MINUTES": "30",
        "NCCL_SOCKET_NTHREADS": "4",
        "NCCL_NSOCKS_PERTHREAD": "4",
        "TORCH_NCCL_HIGH_PRIORITY": "1",
    },
)


def make_robocasa_single_task_experiment(
    *,
    dataset_path: str,
    evaluation_task: str,
    evaluation_num_trials: int,
    evaluation_seed: int = 0,
    max_steps: int = SINGLE_TASK_MAX_STEPS,
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
            max_steps=max_steps,
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
        evaluation_config=TrainingExperimentEvaluationConfig(
            task=evaluation_task,
            num_trials=evaluation_num_trials,
            seed=evaluation_seed,
        ),
    )
