# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass, field

from transformers import PretrainedConfig

from groot.core.training.configs.model import register_model_config


@dataclass
class GrootN1d7Config(PretrainedConfig):
    """Unified configuration for GrootN1d7 model with Eagle/Qwen backbone and action head."""

    # Model identification
    model_type: str = "GrootN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # VLM backbone selection: "eagle", "qwen3", or "qwen2_5"
    vlm_backend: str = "eagle"
    # Path to VLM model (Qwen). When None, falls back to eagle_path for Eagle backend.
    vlm_model_path: str | None = None

    # Backbone configuration (Eagle)
    eagle_path: str = "NVEagle/eagle_er-qwen3_1.7B-Siglip2_400M_stage1_5_128gpu_er_v6_1mlp_nops"
    backbone_embedding_dim: int = 1536  # project_to_dim
    tune_llm: bool = False
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    tune_visual: bool = False
    tune_linear: bool = True
    select_layer: int = 12
    reproject_vision: bool = True
    use_flash_attention: bool = True
    load_bf16: bool = True  # Enable BF16 loading for Eagle
    exclude_state: bool = False

    ### Processing parameters
    image_crop_size: tuple[int, int] = (230, 230)
    image_target_size: tuple[int, int] = (256, 256)
    random_rotation_angle: int = 0
    color_jitter_params: dict[str, float] = field(
        default_factory=lambda: {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.01,
        }
    )
    center_crop_augmentation: bool = False
    # Top-camera extra albumentations (rotation / perspective); 0..3 => v0..v3 (see GrootN1d7Processor).
    top_view_augment_version: int = 0
    # When True, use albumentations resize/crop path (center_crop_augmentation and top_view_augment_version apply).
    use_albumentations: bool = False
    shortest_image_edge: int | None = None
    crop_fraction: float | None = None
    formalize_language: bool = True
    action_space_prompt: bool = False
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    random_history_crop: bool = False
    letter_box_transform: bool = False
    use_percentiles: bool = True  # Use q01/q99 percentiles instead of min/max for normalization
    use_mean_std: bool = False  # Use mean/std normalization for all modalities instead of min/max
    use_relative_action: bool = True  # N1.7 always uses relative actions in StateActionProcessor

    state_gaussian_noise_std: float = 0.0

    # Action head configuration parameters
    max_state_dim: int = 29  # Default from state_shape
    max_action_dim: int = 29  # Default from action_shape
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    # When set, overrides input_embedding_dim for state/action encoding in the action head.
    # Needed when backbone_embedding_dim != input_embedding_dim (e.g. Qwen3 2048D backbone
    # with 1536D action head latent space).
    dit_latent_dim: int | None = None
    state_dropout_prob: float = 0.0

    # Global parameters from YAML
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    use_vl_self_attention: bool = False
    max_seq_len: int = 1024

    # Future tokens
    # Only used when loading OSS N1.5 converted checkpoint for zero-shot.
    use_future_tokens: bool = False

    # DiT variant: when True, uses AlternateVLDiT that separates image and
    # non-image tokens in cross-attention
    use_alternate_vl_dit: bool = False

    # Self attention configuration
    vl_self_attention_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 4,
            "num_attention_heads": 24,
            "attention_head_dim": 64,
            "dropout": 0.2,
            "final_dropout": True,
        }
    )

    # Diffusion model configuration
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
            "cross_attention_dim": 1536,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    # Unified embodiment ID: when True, all embodiments share the same ID (0)
    # so that CategorySpecificMLP uses shared weights across all embodiments
    unified_embodiment_id: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


register_model_config("GrootN1d7", GrootN1d7Config)
