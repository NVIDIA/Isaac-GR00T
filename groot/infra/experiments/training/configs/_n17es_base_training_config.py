# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared N1.7ES training config fragment."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.launcher.one_launcher.training_experiment_training_config import (
    TrainingExperimentTrainingConfig,
)

N17ES_PRETRAIN_CHECKPOINT = (
    "{PATHS.ckpts}/ruijiez/galaxea_sharpa/stable_checkpoints/"
    "n15_thirdstage_checkpoint-200000_patched/"
)


@dataclass(frozen=True)
class N17ESBaseTrainingConfig:
    global_batch_size: int

    def to_training_config(self) -> TrainingExperimentTrainingConfig:
        return TrainingExperimentTrainingConfig(
            training=TrainingConfig(
                learning_rate=5e-5,
                weight_decay=1e-5,
                logging_steps=10,
                save_steps=1000,
                save_total_limit=5,
                max_steps=10000,
                warmup_ratio=0.05,
                tf32=True,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                use_wandb=True,
                wandb_project="n17-eef",
                dataloader_num_workers=4,
                start_from_checkpoint=N17ES_PRETRAIN_CHECKPOINT,
                upload_checkpoints=True,
                upload_every=1000,
            ),
            global_batch_size=self.global_batch_size,
            model_name="GrootN1d7",
            model=self.model_config(),
        )

    def model_config(self) -> Mapping[str, Any]:
        return deepcopy(
            {
                "eagle_path": (
                    "NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops"
                ),
                "load_bf16": False,
                "reproject_vision": False,
                "max_action_dim": 132,
                "max_state_dim": 132,
                "action_horizon": 40,
                "use_percentiles": True,
                "use_mean_std": False,
                "exclude_state": False,
                "state_dropout_prob": 0.5,
                "letter_box_transform": True,
                "random_history_crop": True,
                "color_jitter_params": {
                    "brightness": 0.3,
                    "contrast": 0.4,
                    "saturation": 0.5,
                    "hue": 0.08,
                },
                "tune_llm": True,
                "tune_visual": True,
            }
        )
