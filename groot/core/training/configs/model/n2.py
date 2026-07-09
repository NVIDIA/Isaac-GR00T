# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import TypedDict

from transformers import PretrainedConfig

from groot.core.training.configs.model import register_model_config


class DiffusionModelConfig(TypedDict, total=False):
    diffusion_model_pretrained_path: str
    model_size: str


class TextEncoderConfig(TypedDict, total=False):
    text_encoder_pretrained_path: str


class VAEConfig(TypedDict, total=False):
    vae_pretrained_path: str


class ActionConfig(TypedDict, total=False):
    """Action prediction config. Set to None on GrootN2Config for video-only."""

    max_action_dim: int  # padded action dimension for multi-embodiment (default 32)
    max_state_dim: int  # padded state dimension, after sincos if enabled (default 64)
    action_horizon: int  # action timesteps total (default 48)
    max_action_horizon: int  # fixed action horizon for padding (default None = no padding)
    num_embodiments: int  # for CategorySpecificLinear (default 1)
    # StateActionProcessor settings
    apply_sincos_state_encoding: bool  # sin/cos encoding doubles state dim (default True)
    # When True, q01/q99 are used instead of min/max for normalization — for both
    # absolute and relative actions (default False). Pre-fix checkpoints trained
    # before relative-action q01/q99 routing became unconditional must be migrated
    # with groot/core/training/scripts/patch_eval/patch_rel_stats_q99.py.
    use_percentiles: bool
    clip_outliers: bool  # clip normalized values to [-1, 1] (default True)
    state_dropout_prob: float  # probability of zeroing out state during training (default 0.0)
    # Beta flow matching noise schedule (N1.7-style)
    noise_beta_alpha: float  # Beta distribution alpha (default 1.5)
    noise_beta_beta: float  # Beta distribution beta (default 1.0)
    noise_s: float  # time scaling factor (default 0.999)
    # Weighting of video and action losses in the total loss:
    #   loss = video_loss_scale * video_loss + action_loss_scale * action_loss
    # Both default to 1.0 (no change). Set <1 to down-weight one term during
    # staged pretrain (e.g. video_loss_scale=0 for action-only finetune, or
    # action_loss_scale=0 for video-only pretrain while keeping the action head
    # trainable for later stages).
    video_loss_scale: float
    action_loss_scale: float
    # Attention mask pattern: list of (pattern_name, probability) pairs.
    # Per batch element, one pattern is sampled. See AttentionMaskPattern enum.
    # Default (when absent): [("disjoint_video_action", 1.0)]
    attention_mask_pattern: list[tuple[str, float]]
    # Additive learned positional embedding along the action sequence (time)
    # axis, applied inside ActionEncoder *in addition to* the 1D RoPE inside
    # attention. Zero-init so model output is identical at step 0. Default False.
    use_temporal_pos_emb: bool


def _normalize_action_cfg(action_cfg: ActionConfig | None) -> ActionConfig | None:
    if action_cfg is None:
        return None

    normalized = dict(action_cfg)
    pattern = normalized.get("attention_mask_pattern")
    if pattern is not None:
        normalized["attention_mask_pattern"] = [tuple(pair) for pair in pattern]
    return normalized


@dataclass
class GrootN2Config(PretrainedConfig):
    model_type: str = "GrootN2"
    model_dtype: str = "bfloat16"

    # Transformer
    model_size: str = "2B"
    use_gradient_checkpointing: bool = True

    # Video
    tiled: bool = True
    tile_size_height: int = 34
    tile_size_width: int = 34
    tile_stride_height: int = 18
    tile_stride_width: int = 16

    # Noise / Conditioning
    train_video_sigma_shift: float = 5.0
    cond_timestep: float = 0.1  # matches Cosmos Predict 2.5 convention
    num_timestep_buckets: int = 1000

    # Video flow-matching scheduler. "unipc" (default — shifted sigma + UniPC
    # solver, many-step inference) or "beta" (Beta-distribution timesteps
    # biased toward the high-noise regime + Euler solver; matches DreamZero's
    # video schedule and supports few/1-step inference).
    # Beta scheduler samples η ~ Beta(alpha, beta), then t_video = (1 - η)*noise_s
    # with convention t=0 → high noise, t=1 → clean data. alpha=7, beta=1 peaks
    # density sharply at t near 0 (exact DreamZero default).
    video_scheduler_type: str = "unipc"
    video_noise_beta_alpha: float = 7.0
    video_noise_beta_beta: float = 1.0
    video_noise_s: float = 0.999

    # Training
    train_architecture: str = "full"  # "full" or "lora"
    tune_dit: bool = True
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"

    # Multiview (n_cameras derived from data at runtime — supports cross-embodiment)
    max_n_cameras: int = 3
    view_embed_dim: int = 0  # per-view embedding dim concatenated to input channels (7 for official AV multiview)
    # FlexAttention kernel settings. Cluster training uses the compiled 128-token
    # path; local smoke can override these explicitly for smaller GPUs.
    flex_attention_block_size: int = 128
    compile_flex_attention: bool = True

    # Image processing
    image_target_size: list[int] | None = None
    image_crop_fraction: float = 0.95
    color_jitter_params: dict[str, float] | None = None
    codec_aug_params: dict | None = None

    # Language
    tokenizer_path: str = ""
    formalize_language: bool = False
    max_language_length: int = 512
    # Classifier-free-guidance training: at each step, with this probability,
    # replace the sample's language with `null_prompt` before tokenization.
    # Default 0.0 (no dropout; back-compat with checkpoints trained before this field).
    language_dropout_prob: float = 0.0
    # Null prompt used (a) during training when language dropout fires, (b) at
    # inference as the default `text_negative` for CFG. Tying these ensures the
    # "unconditional" branch at inference is exactly the one trained.
    null_prompt: str = "low quality, blurry, static"

    # Sub-configs
    diffusion_model_cfg: DiffusionModelConfig | None = None
    text_encoder_cfg: TextEncoderConfig | None = None
    vae_cfg: VAEConfig | None = None
    action_cfg: ActionConfig | None = None  # None = video-only (no action modules)

    def __init__(self, **kwargs):
        if "action_cfg" in kwargs:
            kwargs["action_cfg"] = _normalize_action_cfg(kwargs["action_cfg"])
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


register_model_config("GrootN2", GrootN2Config)
