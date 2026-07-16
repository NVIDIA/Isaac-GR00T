# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 egoscale mega-soup pretrain experiment."""

from __future__ import annotations

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_data_config import TrainingExperimentDataConfig
from groot.infra.launcher.one_launcher.training_experiment_training_config import (
    TrainingExperimentTrainingConfig,
)

XDOF_VIDEO_DELTA_INDICES = (-24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 36, 42, 48)
XDOF_ACTION_DELTA_INDICES = tuple(range(48))
DROID_VIDEO_DELTA_INDICES = (-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24)
DROID_ACTION_DELTA_INDICES = tuple(range(24))
SHARPA_VIDEO_DELTA_INDICES = (-16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32)
SHARPA_ACTION_DELTA_INDICES = tuple(range(32))


DATA_CONFIG = TrainingExperimentDataConfig(
    data={
        "datasets": [
            {
                "dataset_paths": [
                    "xdof_yam_annotated_full/xdof.xdof_yam_annotated_full_14khrs"
                ],
                "mix_ratio": 0.25,
                "embodiment_tag": "xdof_relative_eef_relative_joint_umi_keys",
            },
            {
                "dataset_paths": [
                    "yam_v7/xdof.assembly_2026-01-25_13-45-27_all_v5_USA_annotated_umi_keys"
                ],
                "mix_ratio": 0.05,
                "embodiment_tag": "xdof_relative_eef_relative_joint_umi_keys",
            },
            {
                "dataset_paths": ["yam_v7/xdof.posttrain_inlab_merged_4view"],
                "mix_ratio": 0.05,
                "embodiment_tag": "xdof_relative_eef_relative_joint_umi_keys_4views",
            },
            {
                "dataset_paths": ["droid_101_success_idlefiltered_n17"],
                "mix_ratio": 0.05,
                "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
            },
            {
                "dataset_paths": [
                    "galaxea_sharpa/real_r1_pro_sharpa.inlab_play_real_robot_batch_1",
                    "galaxea_sharpa/real_r1_pro_sharpa.inlab_play_real_robot_batch_2",
                    "galaxea_sharpa/real_r1_pro_sharpa.miscellaneous_1k_trajectories",
                ],
                "embodiment_tag": "real_r1_pro_sharpa_relative_eef",
                "mix_ratio": 0.05,
            },
            {
                "dataset_paths": ["dexmate_sharpa/merged_5532_dexmate_play_data_0515"],
                "mix_ratio": 0.05,
                "embodiment_tag": "real_dexmate_sharpa_relative_eef",
            },
            {
                "dataset_paths": [
                    "human_pretraining/mecka_46khrs_merged_egocentric_only"
                ],
                "embodiment_tag": "real_r1_pro_sharpa_relative_eef_mecka",
                "mix_ratio": 0.4,
            },
            {
                "dataset_paths": [
                    "galaxea_sharpa/maxinsights_lerobot_updated/1530hrs/"
                    "real_r1_pro_sharpa.maxinsights_1530hrs_updated_train_set_merged"
                ],
                "embodiment_tag": "real_r1_pro_sharpa_relative_eef_maxinsights",
                "mix_ratio": 0.1,
            },
            {
                "dataset_paths": [
                    "dexmate_sharpa/merged_human_diverse_lang_0505",
                    "dexmate_sharpa/merged_egoscale_human_inlab_0505",
                ],
                "embodiment_tag": "real_r1_pro_sharpa_relative_eef_human",
                "mix_ratio": 0.1,
            },
        ],
        "modality_configs": {
            "xdof_relative_eef_relative_joint_umi_keys": {
                "video": {"delta_indices": list(XDOF_VIDEO_DELTA_INDICES)},
                "action": {"delta_indices": list(XDOF_ACTION_DELTA_INDICES)},
                "language": {
                    "modality_keys": ["annotation.task", "annotation.sub_task"]
                },
            },
            "xdof_relative_eef_relative_joint_umi_keys_4views": {
                "video": {"delta_indices": list(XDOF_VIDEO_DELTA_INDICES)},
                "action": {"delta_indices": list(XDOF_ACTION_DELTA_INDICES)},
            },
            "oxe_droid_relative_eef_relative_joint": {
                "video": {
                    "delta_indices": list(DROID_VIDEO_DELTA_INDICES),
                    "modality_keys": [
                        "exterior_image_1_left",
                        "exterior_image_2_left",
                        "wrist_image_left",
                    ],
                },
                "action": {"delta_indices": list(DROID_ACTION_DELTA_INDICES)},
            },
            "real_r1_pro_sharpa_relative_eef": {
                "video": {"delta_indices": list(SHARPA_VIDEO_DELTA_INDICES)},
                "action": {"delta_indices": list(SHARPA_ACTION_DELTA_INDICES)},
            },
            "real_dexmate_sharpa_relative_eef": {
                "video": {"delta_indices": list(SHARPA_VIDEO_DELTA_INDICES)},
                "action": {"delta_indices": list(SHARPA_ACTION_DELTA_INDICES)},
            },
            "real_r1_pro_sharpa_relative_eef_mecka": {
                "video": {"delta_indices": list(SHARPA_VIDEO_DELTA_INDICES)},
                "action": {"delta_indices": list(SHARPA_ACTION_DELTA_INDICES)},
            },
            "real_r1_pro_sharpa_relative_eef_maxinsights": {
                "video": {"delta_indices": list(SHARPA_VIDEO_DELTA_INDICES)},
                "action": {"delta_indices": list(SHARPA_ACTION_DELTA_INDICES)},
            },
            "real_r1_pro_sharpa_relative_eef_human": {
                "video": {"delta_indices": list(SHARPA_VIDEO_DELTA_INDICES)},
                "action": {"delta_indices": list(SHARPA_ACTION_DELTA_INDICES)},
            },
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
        use_wandb=True,
        wandb_project="groot_n2_video_finetune",
        dataloader_num_workers=1,
        start_from_checkpoint=(
            "ruijiez/n2_yam_short_plus_recap_long_50human_50robot-batch1536-100k-"
            "video-only-double-horizon/checkpoint-100000-corecompat"
        ),
        upload_checkpoints=True,
        upload_every=1000,
        eval_strategy="no",
    ),
    global_batch_size=2048,
    model_name="GrootN2",
    model={
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "train_architecture": "full",
        "image_target_size": [224, 320],
        "tokenizer_path": "cosmos25/cosmos_reason1/",
        "cond_timestep": 0.1,
        "tiled": False,
        "codec_aug_params": [
            "mixture_identity",
            0.85,
            "mixture_jpeg",
            0.10,
            "mixture_mp4v",
            0.05,
            "jpeg_quality_lower",
            60,
            "jpeg_quality_upper",
            95,
            "mp4v_rate",
            30,
            "p",
            1.0,
        ],
        "view_embed_dim": 7,
        "max_n_cameras": 4,
        "language_dropout_prob": 0.1,
        "video_scheduler_type": "beta",
        "video_noise_beta_alpha": 7.0,
        "video_noise_beta_beta": 1.0,
        "video_noise_s": 0.999,
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
            "vae_pretrained_path": "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        },
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
    families={"gb300": ExperimentComputeConfig(per_device_batch_size=8)}
)

EXPERIMENT = TrainingExperiment(
    data_config=DATA_CONFIG,
    training_config=TRAINING_CONFIG,
    compute_config=COMPUTE_CONFIG,
)
