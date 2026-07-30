# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 Turbo RoboCasa-365 posttraining on four Colossus stations."""

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

# Reuse the canonical RoboCasa-365 dataset list, embodiment, modalities, and
# sampling settings. Restore the following archive under /data/groot_ol/datasets
# on every Colossus station:
# s3://GrootDatasets/robocasa_benchmark/new-robocasa/robocasa-datasets/v1.0/
# archives/pretrain-112958637747-227001.tar
# SHA-256: 0de7d998712edee3f0503de858defefb25cf9877d19ecacbcf94122abf8a290e
DATA_CONFIG = replace(
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
# action head, but do not pad every training sample with 32 masked tokens.
_TURBO_ROBOCASA_MODEL["action-cfg"]["max-action-horizon"] = 16

TRAINING_CONFIG = replace(
    _ROBOCASA365_TRAINING_CONFIG,
    training=replace(
        _ROBOCASA365_TRAINING_CONFIG.training,
        output_dir="/data/groot_ol/shared/checkpoints/{USER_SLUG}/{JOB_NAME}",
        logging_steps=1000,
        # Preserve the canonical RoboCasa choice: torch.compile is a net loss
        # for its category-specific action projectors.
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
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        log_distributed_grad_norm=False,
        straggler=replace(
            _ROBOCASA365_TRAINING_CONFIG.training.straggler,
            enabled=False,
        ),
        # DDP both keeps this checkpoint numerically stable and avoids ZeRO-2
        # gradient partitioning on every accumulation microbatch.
        use_ddp=True,
        ddp_bucket_cap_mb=256,
    ),
    # Preserve RoboCasa-365's canonical global batch on four one-GPU stations:
    # 4 GPUs x 64 samples/GPU x 2 gradient-accumulation steps.
    global_batch_size=512,
    model_name=BASE_TURBO_TRAINING_CONFIG.model_name,
    model=_TURBO_ROBOCASA_MODEL,
)

# One Launcher maps Colossus to the "gb300" compute-family key while enforcing
# one GPU per station.
COMPUTE_CONFIG = TrainingExperimentComputeConfig(
    families={"gb300": ExperimentComputeConfig(per_device_batch_size=64)},
    env_vars={
        "GEAR_PRESERVE_CHECKPOINT_ROOT": "1",
        "GROOT_DISTRIBUTED_TIMEOUT_MINUTES": "30",
        "NCCL_SOCKET_NTHREADS": "4",
        "NCCL_NSOCKS_PERTHREAD": "4",
    },
)

EXPERIMENT = TrainingExperiment(
    data_config=DATA_CONFIG,
    training_config=TRAINING_CONFIG,
    compute_config=COMPUTE_CONFIG,
)
