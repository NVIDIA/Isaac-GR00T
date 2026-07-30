# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Colossus N2 Turbo airplane posttraining experiment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.experiments.training.configs.n2_turbo_airplane import (
    DATA_CONFIG as BASE_DATA_CONFIG,
)
from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_training_config import (
    TrainingExperimentTrainingConfig,
)

# Colossus prerequisite: stage the patched checkpoint, Wan2.2 VAE, airplane
# dataset, and this validated horizon-48 metadata on every station:
# s3://GearCheckpoints/model_deps/n2turbo_airplane_200traj_0421/relative_stats_h48.json
# SHA-256: 0fa540aacbadb843bf6bc62f2d779e15c403a725421d1839522be5acc1f3a36d
# Inherit the canonical dataset loading and path normalization, then overlay
# only the behavior required by node-local Colossus data.
DATA_CONFIG = replace(
    BASE_DATA_CONFIG,
    data={
        **deepcopy(BASE_DATA_CONFIG.data),
        "multiprocessing_context": "fork",
        # The cluster must provide complete horizon-48 statistics beside the
        # local dataset. Fail fast instead of spawning a Ray actor pool.
        "auto_generate_stats": False,
    },
)

TRAINING_CONFIG = TrainingExperimentTrainingConfig(
    training=TrainingConfig(
        output_dir="/data/groot_ol/shared/checkpoints/{USER_SLUG}/{JOB_NAME}",
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1000,
        save_steps=5000,
        save_total_limit=5,
        max_steps=20_000,
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        enable_profiling=False,
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        optim="adamw_torch",
        dataloader_num_workers=2,
        lr_scheduler_type="cosine",
        use_wandb=False,
        start_from_checkpoint=(
            "jingwang_n2_cosmosnano_megasoup-action_loss_megasoup_0710_g1_patched/checkpoint-200000"
        ),
        upload_checkpoints=False,
        msc_checkpoint_url="",
        msc_profile_url="",
        eval_strategy="no",
        save_final_model=True,
        # The async saver forks inside a Ray actor and has triggered
        # task_receiver objects_valid crashes; prefer resumable sync saves.
        use_async_checkpoint_save=False,
        gradient_accumulation_steps=4,
        use_ddp=False,
        deepspeed_stage=2,
        # DeepSpeed sizes these buckets in elements, not bytes.
        deepspeed_reduce_bucket_size=100_000_000,
    ),
    # Match the canonical global batch 256 on four one-GPU Colossus stations:
    # 4 GPUs x 16 samples/GPU x 4 gradient-accumulation steps. Keep the
    # microbatch at 16: batch 32 produced nonfinite losses and batch 64
    # exhausted HBM on this topology.
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
        "use_gradient_checkpointing": True,
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
    families={"gb300": ExperimentComputeConfig(per_device_batch_size=16)},
    env_vars={
        "GEAR_PRESERVE_CHECKPOINT_ROOT": "1",
        # Leave headroom for Wan2.2's long first compiled step.
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
