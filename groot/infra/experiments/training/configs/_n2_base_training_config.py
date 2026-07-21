# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared N2 training config fragment."""

from __future__ import annotations

from dataclasses import dataclass

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.experiments.training.configs._n2_model_config import N2ModelConfig
from groot.infra.launcher.one_launcher.training_experiment_training_config import (
    TrainingExperimentTrainingConfig,
)

N2_PRETRAIN_CHECKPOINT = "ruijiez/n2_yam_pretrain-bidrectional-full-200k-source/checkpoint-146000"


@dataclass(frozen=True)
class N2BaseTrainingConfig:
    wandb_project: str
    max_n_cameras: int = 4
    global_batch_size: int = 384
    max_steps: int = 10000
    save_steps: int = 2000
    upload_every: int = 2000
    start_from_checkpoint: str = N2_PRETRAIN_CHECKPOINT
    torch_compile: bool = False
    torch_compile_mode: str = "default"

    def to_training_config(self) -> TrainingExperimentTrainingConfig:
        return TrainingExperimentTrainingConfig(
            training=TrainingConfig(
                learning_rate=5e-5,
                weight_decay=1e-5,
                logging_steps=10,
                save_steps=self.save_steps,
                save_total_limit=5,
                max_steps=self.max_steps,
                warmup_ratio=0.05,
                bf16=True,
                tf32=True,
                torch_compile=self.torch_compile,
                torch_compile_mode=self.torch_compile_mode,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                use_wandb=True,
                wandb_project=self.wandb_project,
                dataloader_num_workers=4,
                start_from_checkpoint=self.start_from_checkpoint,
                upload_checkpoints=True,
                upload_every=self.upload_every,
                eval_strategy="no",
            ),
            global_batch_size=self.global_batch_size,
            model_name="GrootN2",
            model=N2ModelConfig(max_n_cameras=self.max_n_cameras).to_mapping(),
        )
