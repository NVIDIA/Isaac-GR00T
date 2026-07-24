# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 TTT Egoscale mega-soup midtrain experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.core.training.configs.ttt_compile import resolve_ttt_compile_env
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

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DATA_CONFIG_PATH = (
    _REPO_ROOT
    / "groot/core/training/configs/experiments/n2/pretrain/egoscale_mega_soup_anneal_final.yaml"
)
_DATA_CONFIG_PAYLOAD = yaml.safe_load(_DATA_CONFIG_PATH.read_text())
if not isinstance(_DATA_CONFIG_PAYLOAD, dict) or not isinstance(
    _DATA_CONFIG_PAYLOAD.get("data"),
    dict,
):
    raise TypeError(f"{_DATA_CONFIG_PATH} must contain a data mapping")

N2_TTT_MEGASOUP_MIDTRAIN_DATA = deepcopy(_DATA_CONFIG_PAYLOAD["data"])
# One Launcher exports LFS_CACHE_PATH as the cluster data-volume root.  The
# legacy TTT launcher exported its ``datasets`` child instead, so preserve the
# same resolved dataset locations when reusing the legacy data YAML.
for _dataset in N2_TTT_MEGASOUP_MIDTRAIN_DATA["datasets"]:
    _dataset["dataset_paths"] = [
        path if path.startswith("datasets/") else f"datasets/{path}"
        for path in _dataset["dataset_paths"]
    ]
N2_TTT_MEGASOUP_MIDTRAIN_DATA.update(
    {
        "seed": 24,
        "mode": "trajectory",
        "max_len": 1024,
        "trajectory_step_stride": 30,
        "allow_padding": True,
        "video_backend": "torchcodec",
        "shard_size": 32,
    }
)

DATA_CONFIG = TrainingExperimentDataConfig(
    data=N2_TTT_MEGASOUP_MIDTRAIN_DATA,
    launcher_data={"no_download_cache": True},
)

TRAINING_CONFIG = TrainingExperimentTrainingConfig(
    training=TrainingConfig(
        output_dir="{PATHS.checkpoints}/{USER_SLUG}/{JOB_NAME}",
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=5,
        max_steps=50000,
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        torch_compile=False,
        optim="adamw_torch",
        lr_scheduler_type="wsd",
        wsd_decay_type="cosine",
        use_wandb=True,
        wandb_project="groot_n2_ttt",
        dataloader_num_workers=1,
        start_from_checkpoint=(
            "msc://GearCheckpoints/trinity-ckpt-repository/"
            "ruijiez_n2_egoscale_megasoup-anneal200k-robot-upweight-batch2048-"
            "wsd-lr1e-5k-0703/checkpoint-200000"
        ),
        upload_checkpoints=True,
        upload_every=1000,
        msc_checkpoint_url=("msc://GearCheckpoints/trinity-ckpt-repository/{USER_SLUG}_{JOB_NAME}"),
        msc_delete_checkpoint_after_upload=False,
        eval_strategy="no",
        enable_profiling=False,
        enable_worker_profiling=False,
    ),
    global_batch_size=128,
    model_name="GrootN2TTT",
    model={
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "train_architecture": "full",
        "image_target_size": [224, 320],
        "tokenizer_path": "cosmos25/cosmos_reason1/",
        "cond_timestep": 0.1,
        "tiled": False,
        "view_embed_dim": 7,
        "max_n_cameras": 4,
        "language_dropout_prob": 0.1,
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
        "video_scheduler_type": "beta",
        "video_noise_beta_alpha": 7.0,
        "video_noise_beta_beta": 1.0,
        "video_noise_s": 0.999,
        "ttt_layer_type": "ttt_mlp",
        "ttt_num_heads": 16,
        "ttt_mini_batch_size": 1,
        "ttt_base_lr": 0.1,
        "ttt_gating_alpha_init": 0.01,
        "ttt_state_decay": 1.0,
        "ttt_num_register_tokens": 64,
        "ttt_register_only": True,
        "ttt_register_attention": "full",
        "collator_seq_len_handle_strategy": "fixed",
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
            "use_percentiles": True,
            "clip_outliers": True,
            "use-temporal-pos-emb": True,
            "apply-sincos-state-encoding": False,
            "state-dropout-prob": 0.2,
            "action-loss-scale": 1.0,
            "attention-mask-pattern": ["bidirectional_video_action_full", 1.0],
        },
    },
)

COMPUTE_CONFIG = TrainingExperimentComputeConfig(
    families={"gb300": ExperimentComputeConfig(per_device_batch_size=1)},
    env_vars=resolve_ttt_compile_env({}),
)

EXPERIMENT = TrainingExperiment(
    data_config=DATA_CONFIG,
    training_config=TRAINING_CONFIG,
    compute_config=COMPUTE_CONFIG,
)
