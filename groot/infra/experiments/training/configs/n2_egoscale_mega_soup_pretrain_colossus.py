# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Colossus-optimized N2 egoscale mega-soup pretrain experiment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from groot.infra.experiments.training.configs.n2_egoscale_mega_soup_pretrain import (
    DATA_CONFIG as BASE_DATA_CONFIG,
)
from groot.infra.experiments.training.configs.n2_egoscale_mega_soup_pretrain import (
    TRAINING_CONFIG as BASE_TRAINING_CONFIG,
)
from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)

DATA_CONFIG = replace(
    BASE_DATA_CONFIG,
    data={
        **deepcopy(BASE_DATA_CONFIG.data),
        # The current Colossus sources resolve to local LeRobot datasets.
        "multiprocessing_context": "fork",
    },
)

TRAINING_CONFIG = replace(
    BASE_TRAINING_CONFIG,
    training=replace(
        BASE_TRAINING_CONFIG.training,
        output_dir="/data/groot_ol/shared/checkpoints/{USER_SLUG}/{JOB_NAME}",
        gradient_accumulation_steps=1,
        logging_steps=1000,
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        optim="adamw_torch",
        dataloader_num_workers=2,
        use_wandb=False,
        upload_checkpoints=False,
        msc_checkpoint_url="",
        msc_profile_url="",
        enable_profiling=False,
        save_steps=1000,
        save_total_limit=5,
        save_final_model=True,
        # The async saver forks inside a Ray actor and has triggered
        # task_receiver objects_valid crashes; prefer resumable sync saves.
        use_async_checkpoint_save=False,
        use_ddp=False,
        deepspeed_stage=2,
        # DeepSpeed sizes these buckets in elements, not bytes. The smaller
        # bucket overlaps better with the 25 Gb/s inter-node fabric.
        deepspeed_reduce_bucket_size=100_000_000,
    ),
    global_batch_size=32,
)

# One Launcher maps Colossus to the existing "gb300" compute-family key while
# enforcing one GPU per node in its Colossus launch profile.
COMPUTE_CONFIG = TrainingExperimentComputeConfig(
    families={"gb300": ExperimentComputeConfig(per_device_batch_size=8)},
    env_vars={
        "GEAR_PRESERVE_CHECKPOINT_ROOT": "1",
        "NCCL_SOCKET_NTHREADS": "4",
        "NCCL_NSOCKS_PERTHREAD": "4",
    },
)

EXPERIMENT = TrainingExperiment(
    data_config=DATA_CONFIG,
    training_config=TRAINING_CONFIG,
    compute_config=COMPUTE_CONFIG,
)
