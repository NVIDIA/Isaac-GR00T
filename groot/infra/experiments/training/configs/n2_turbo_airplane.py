# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 Turbo Airplane posttraining experiment.

Launch on H100 with:

    uv run ol train --experiment n2_turbo_airplane --compute-type h100
"""

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

DATA_CONFIG = TrainingExperimentDataConfig(
    data={
        "datasets": [
            {
                "dataset_paths": [
                    "datasets/xdof_yam_annotated_full/"
                    "xdof.xdof_airplane_04_21_v6_200traj_2026-04-22_05-29-30_v4_USA"
                ],
                "mix_ratio": 1.0,
                "embodiment_tag": "xdof_relative_eef_relative_joint_umi_keys_4views",
            }
        ],
        "modality_configs": {
            "xdof_relative_eef_relative_joint_umi_keys_4views": {
                "video": {
                    "delta_indices": [
                        -24,
                        -18,
                        -12,
                        -6,
                        0,
                        6,
                        12,
                        18,
                        24,
                        30,
                        36,
                        42,
                        48,
                    ]
                },
                "action": {"delta_indices": list(range(48))},
            }
        },
        "seed": 24,
        "mode": "single_turn",
        "allow_padding": True,
        "override_pretraining_statistics": True,
        "video_backend": "torchcodec",
        "dataloader_prefetch_factor": 8,
    },
    launcher_data={"no_download_cache": True},
)

TRAINING_CONFIG = TrainingExperimentTrainingConfig(
    training=TrainingConfig(
        output_dir="{PATHS.checkpoints}/{USER_SLUG}/{JOB_NAME}",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_steps=5000,
        save_total_limit=5,
        max_steps=20000,
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        use_wandb=True,
        wandb_project="groot_n2_video_finetune",
        dataloader_num_workers=2,
        start_from_checkpoint=(
            "msc://GearCheckpoints/trinity-ckpt-repository/"
            "jingwang_n2_cosmosnano_megasoup-action_loss_megasoup_0710_g1_patched/"
            "checkpoint-200000"
        ),
        upload_checkpoints=True,
        upload_every=5000,
        msc_checkpoint_url=("msc://GearCheckpoints/trinity-ckpt-repository/{USER_SLUG}_{JOB_NAME}"),
        msc_execution_mode="local",
        msc_delete_checkpoint_after_upload=False,
        eval_strategy="no",
    ),
    global_batch_size=256,
    model_name="GrootN2",
    model={
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "vae_type": "wan22",
        "train_architecture": "full",
        "image_target_size": [224, 320],
        "tokenizer_path": "cosmos25/cosmos_reason1/",
        "language_dropout_prob": 0.1,
        "cond_timestep": 0.1,
        "video_scheduler_type": "beta",
        "video_noise_beta_alpha": 7.0,
        "video_noise_beta_beta": 1.0,
        "video_noise_s": 0.999,
        "tiled": False,
        "view_embed_dim": 7,
        "max_n_cameras": 4,
        "diffusion-model-cfg": "diffusion-model-config",
        "diffusion_model_cfg": {
            "diffusion_model_pretrained_path": (
                "cosmos25/Cosmos-Predict2.5-2B-diffusers/transformer"
            ),
            "model_size": "2B",
        },
        "text-encoder-cfg": "text-encoder-config",
        "text_encoder_cfg": {
            "text_encoder_pretrained_path": "cosmos25/cosmos_reason1/",
        },
        "vae-cfg": "vae-config",
        "vae_cfg": {
            "vae_pretrained_path": "cosmos3/Wan2.2_VAE.pth",
        },
        "action-cfg": {
            "_config": "action-config",
            "max-action-dim": 128,
            "max-state-dim": 128,
            "action-horizon": 48,
            "max-action-horizon": 48,
            "num-embodiments": 32,
            "state-dropout-prob": 0.5,
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
