# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared N2 model configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class N2ModelConfig:
    max_n_cameras: int

    def to_mapping(self) -> Mapping[str, Any]:
        return deepcopy(
            {
                "model_dtype": "bfloat16",
                "model_size": "2B",
                "train_architecture": "full",
                "image_target_size": [224, 320],
                "tokenizer_path": "cosmos25/cosmos_reason1/",
                "cond_timestep": 0.1,
                "tiled": False,
                "view_embed_dim": 7,
                "max_n_cameras": self.max_n_cameras,
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
        )
