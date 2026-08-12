# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Public-safe N2 OSS pretraining experiment."""

from __future__ import annotations

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_data_config import (
    TrainingExperimentDataConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_training_config import (
    TrainingExperimentTrainingConfig,
)

PUBLIC_CHECKPOINT = (
    "hf://model/nvidia/GR00T2Internal@4fe662abaf302ee46916881c447377da8c4bd15a/base_200k"
)
PUBLIC_TOKENIZER = "hf://model/nvidia/Cosmos-Reason1-7B@375e24000b24baed78f4618d3dd779e47cd96323"
PUBLIC_TRANSFORMER = (
    "hf://model/nvidia/Cosmos-Predict2.5-2B@0d37c7498f54cee3c599d438d895a0a4a8608064/transformer"
)
PUBLIC_VAE = (
    "hf://model/Wan-AI/Wan2.1-I2V-14B-480P@6b73f84e66371cdfe870c72acd6826e1d61cf279/Wan2.1_VAE.pth"
)

DATA_CONFIG = TrainingExperimentDataConfig(
    data={
        "datasets": [
            {
                "dataset_paths": ["datasets/replace_with_your_lerobot_dataset"],
                "mix_ratio": 1.0,
                "embodiment_tag": "xdof_relative_eef_relative_joint",
            }
        ],
        "modality_configs": {
            "xdof_relative_eef_relative_joint": {
                "video": {"delta_indices": [-24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 36, 42, 48]},
                "action": {"delta_indices": list(range(48))},
            }
        },
        "seed": 24,
        "mode": "single_turn",
        "allow_padding": True,
        "video_backend": "torchcodec",
    },
    launcher_data={"no_download_cache": True},
)

TRAINING_CONFIG = TrainingExperimentTrainingConfig(
    training=TrainingConfig(
        learning_rate=3e-5,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=5,
        max_steps=300000,
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        optim="adamw_torch",
        lr_scheduler_type="wsd",
        wsd_decay_type="cosine",
        use_wandb=False,
        wandb_project="isaac_gr00t_n2",
        dataloader_num_workers=1,
        start_from_checkpoint=PUBLIC_CHECKPOINT,
        upload_checkpoints=False,
        upload_every=None,
        msc_profile_url="",
        experiment_tracking="disabled",
        eval_strategy="no",
    ),
    global_batch_size=2048,
    model_name="GrootN2",
    model={
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "train_architecture": "full",
        "image_target_size": [224, 320],
        "tokenizer_path": PUBLIC_TOKENIZER,
        "cond_timestep": 0.1,
        "tiled": False,
        "view_embed_dim": 7,
        "max_n_cameras": 4,
        "language_dropout_prob": 0.1,
        "video_scheduler_type": "beta",
        "video_noise_beta_alpha": 7.0,
        "video_noise_beta_beta": 1.0,
        "video_noise_s": 0.999,
        "diffusion-model-cfg": "diffusion-model-config",
        "diffusion_model_cfg": {
            "diffusion_model_pretrained_path": PUBLIC_TRANSFORMER,
            "model_size": "2B",
        },
        "text-encoder-cfg": "text-encoder-config",
        "text_encoder_cfg": {"text_encoder_pretrained_path": PUBLIC_TOKENIZER},
        "vae-cfg": "vae-config",
        "vae_cfg": {"vae_pretrained_path": PUBLIC_VAE},
        "action-cfg": {
            "_config": "action-config",
            "max-action-dim": 128,
            "max-state-dim": 128,
            "action-horizon": 48,
            "max-action-horizon": 48,
            "num-embodiments": 32,
            "state-dropout-prob": 1.0,
            "action-loss-scale": 1.0,
            "use_percentiles": True,
            "clip_outliers": True,
            "apply-sincos-state-encoding": False,
            "use-temporal-pos-emb": True,
            "attention-mask-pattern": ["bidirectional_video_action_full", 1.0],
        },
    },
)

COMPUTE_CONFIG = TrainingExperimentComputeConfig(
    families={
        "h100": ExperimentComputeConfig(per_device_batch_size=8),
        "gb300": ExperimentComputeConfig(per_device_batch_size=8),
    }
)

EXPERIMENT = TrainingExperiment(
    data_config=DATA_CONFIG,
    training_config=TRAINING_CONFIG,
    compute_config=COMPUTE_CONFIG,
)
