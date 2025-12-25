from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d6Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d6 model with backbone and action head."""

    # Model identification
    model_type: str = "Gr00tN1d6"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # backbone configuration
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_model_type: str = "eagle"
    model_revision: str | None = None
    tune_top_llm_layers: int = 4  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True  # Enable BF16 loading
    collator_overwrite_image_inputs: bool = False  # Deprecated; use eagle_collator.
    eagle_collator: bool = (
        False  # this allows model to change image size in collator, needed for eagle any-res
    )
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = None
    image_target_size: tuple[int, int] | None = None

    shortest_image_edge: int | None = 256
    crop_fraction: float | None = 0.95

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 29  # Default from state_shape
    max_action_dim: int = 29  # Default from action_shape
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # Global parameters from YAML
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    # Diffusion model type selection
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    # Diffusion model configuration with 32 layers (main difference from N15)
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # 32 layers instead of 16
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
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

    # State Augmentation parameters
    state_dropout_prob: float = 0.0  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            # PATCH: Backward compatibility for legacy argument "collator_overwrite_image_inputs"
            if key == "collator_overwrite_image_inputs":
                setattr(self, "eagle_collator", value)
            # /PATCH
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("GrootN1d6", Gr00tN1d6Config)
