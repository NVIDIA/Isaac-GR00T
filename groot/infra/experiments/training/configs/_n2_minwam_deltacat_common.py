# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared N2 Min-WAM DeltaCAT experiment fragments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment_compute_config import TrainingExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment_data_config import TrainingExperimentDataConfig
from groot.infra.launcher.one_launcher.training_experiment_training_config import (
    TrainingExperimentTrainingConfig,
)

EMBODIMENT = "xdof_relative_eef_relative_joint"
STEPS_TABLE = "dc://default/xdof/gearyamrawdata_v1_full_20260608_steps"
VIDEO_DELTA_INDICES = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]


def data_config(
    *,
    session_ids: tuple[str, ...],
    video_config: Mapping[str, Any] | None = None,
) -> TrainingExperimentDataConfig:
    sessions = ",\n            ".join(f"'{session_id}'" for session_id in session_ids)
    modality_video = {"delta_indices": list(VIDEO_DELTA_INDICES)}
    if video_config:
        modality_video.update(video_config)
    return TrainingExperimentDataConfig(
        data={
            "datasets": [
                {
                    "dataset_paths": [STEPS_TABLE],
                    "embodiment_tag": EMBODIMENT,
                    "mix_ratio": 1.0,
                    "deltacat": {
                        "episode_query": f"session_id IN (\n            {sessions}\n          )",
                        "steps_query": "has_valid_action = true",
                    },
                }
            ],
            "seed": 24,
            "mode": "single_turn",
            "dataset_factory_type": "default",
            "allow_padding": True,
            "auto_generate_stats": False,
            "video_backend": "torchcodec",
            "override_pretraining_statistics": True,
            "deltacat_stats": {
                "sample_anchor_steps_per_embodiment": 50000,
                "min_anchor_steps_per_stratum": 1024,
                "seed": None,
            },
            "modality_configs": {
                EMBODIMENT: {
                    "video": modality_video,
                }
            },
        },
        launcher_data={"no_download_cache": True},
    )


def training_config(
    *,
    max_n_cameras: int,
    global_batch_size: int = 128,
    gradient_accumulation_steps: int = 1,
) -> TrainingExperimentTrainingConfig:
    return TrainingExperimentTrainingConfig(
        training=TrainingConfig(
            learning_rate=5e-5,
            weight_decay=1e-5,
            logging_steps=10,
            save_steps=1000,
            save_total_limit=5,
            max_steps=10000,
            warmup_ratio=0.05,
            bf16=True,
            tf32=True,
            torch_compile=True,
            torch_compile_mode="reduce-overhead",
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            use_wandb=True,
            wandb_project="groot_n2_posttrain",
            dataloader_num_workers=0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            start_from_checkpoint="ruijiez/n2_yam_pretrain-bidrectional-full-200k-source/checkpoint-200000",
            upload_checkpoints=True,
            upload_every=1000,
            eval_strategy="no",
        ),
        global_batch_size=global_batch_size,
        model_name="GrootN2",
        model=model_config(max_n_cameras=max_n_cameras),
    )


def model_config(*, max_n_cameras: int) -> Mapping[str, Any]:
    config: dict[str, Any] = {
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "train_architecture": "full",
        "image_target_size": [224, 320],
        "tokenizer_path": "cosmos25/cosmos_reason1/",
        "cond_timestep": 0.1,
        "tiled": False,
        "view_embed_dim": 7,
        "max_n_cameras": max_n_cameras,
        "video_scheduler_type": "beta",
        "video_noise_beta_alpha": 7.0,
        "video_noise_beta_beta": 1.0,
        "video_noise_s": 0.999,
        "diffusion-model-cfg": "diffusion-model-config",
        "diffusion_model_cfg": {
            "diffusion_model_pretrained_path": "cosmos25/Cosmos-Predict2.5-2B-diffusers/transformer",
            "model_size": "2B",
        },
        "text-encoder-cfg": "text-encoder-config",
        "text_encoder_cfg": {
            "text_encoder_pretrained_path": "cosmos25/cosmos_reason1/",
        },
        "vae-cfg": "vae-config",
        "vae_cfg": {
            "vae_pretrained_path": "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        },
        "action-cfg": {
            "_config": "action-config",
            "max-action-dim": 128,
            "max-state-dim": 128,
            "action-horizon": 40,
            "max-action-horizon": 40,
            "num-embodiments": 1,
            "use_percentiles": True,
            "clip_outliers": True,
            "apply-sincos-state-encoding": False,
            "state-dropout-prob": 0.5,
            "attention-mask-pattern": ["bidirectional_video_action_full", 1.0],
        },
    }
    return deepcopy(config)


def h100_compute_config(*, per_device_batch_size: int) -> TrainingExperimentComputeConfig:
    return TrainingExperimentComputeConfig(
        families={
            "h100": ExperimentComputeConfig(per_device_batch_size=per_device_batch_size),
        }
    )
