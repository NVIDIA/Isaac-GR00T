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
from typing import Any

from transformers import AutoConfig, PretrainedConfig

from groot.core.training.configs.model import register_model_config
from groot.core.training.configs.model.n1d6 import GrootN1d6Config
from groot.core.training.configs.model.n1d7 import GrootN1d7Config
from groot.core.training.configs.model.n2 import GrootN2Config

_N2_IQL_FIELD_DEFAULTS = {
    "q_loss_coeff": 1.0,
    "v_loss_coeff": 1.0,
    "advantage_beta_inverse": 1.0,
    "awr_weight_clip": 100.0,
    "q_model_ensemble_size": 2,
    "q_model_polyak": 0.995,
    "q_model_polyak_fp32": True,
    "q_model_gamma": 0.99,
    "v_model_iql_quantile": 0.7,
    "reward_key": "next.delta_progress",
    "reward_scale": 1.0,
    "include_video_loss_in_iql": True,
}

_N2_ALWAYS_STALE_RL_FIELD_NAMES = (
    "pretrained_q_model_path",
    "pretrained_v_model_path",
    "tune_q_model",
    "tune_v_model",
    "q_model_pooling_method",
    "q_model_pooling_qformer_cfg",
    "v_model_pooling_method",
    "v_model_pooling_qformer_cfg",
    "residual_action_scale",
    "residual_use_base_action",
    "residual_pooling_method",
    "residual_pooling_qformer_cfg",
    "hidden_size",
)

_N2_STALE_RL_FIELD_NAMES = (
    *_N2_IQL_FIELD_DEFAULTS,
    *_N2_ALWAYS_STALE_RL_FIELD_NAMES,
)

_N2_ACTOR_ONLY_LOSS_MODES = {
    "BC",
    "BC_NORMAL_ONLY",
    "BC_HUMAN_NORMAL_ONLY",
    "BC_QUALITY_WEIGHTED",
    "BC_CONTINUOUS_WEIGHTED",
}


def _is_neutral_stale_value(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, int | float) and value == 0:
        return True
    return value in ("", [], {}, ())


@dataclass
class GrootN2RLConfig(PretrainedConfig):
    """Actor-only N2 weighted-BC wrapper config."""

    model_type: str = "GrootN2RL"
    model_dtype: str = "bfloat16"

    base_model_cfg: GrootN2Config = field(default_factory=GrootN2Config)
    pretrained_base_model_path: str | None = None

    loss_mode: str = "BC_QUALITY_WEIGHTED"
    actor_loss_coeff: float = 1.0
    quality_loss_weight_map: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 0.5, 2: 0.0}
    )
    quality_w0: float | None = None
    quality_w1: float | None = None
    quality_w2: float | None = None

    def __init__(self, **kwargs):
        actor_only_checkpoint = kwargs.get("loss_mode") in _N2_ACTOR_ONLY_LOSS_MODES
        stale_fields = {
            key: kwargs[key]
            for key in _N2_STALE_RL_FIELD_NAMES
            if key in kwargs
            and not _is_neutral_stale_value(kwargs[key])
            and not (
                actor_only_checkpoint
                and key in _N2_IQL_FIELD_DEFAULTS
                and kwargs[key] == _N2_IQL_FIELD_DEFAULTS[key]
            )
        }
        if stale_fields:
            raise ValueError(
                "GrootN2RL is actor-only weighted BC. Remove stale critic/residual RL "
                f"fields from the config: {sorted(stale_fields)}"
            )
        if kwargs.get("residual", False):
            raise ValueError("GrootN2RL does not support residual=True in weighted-BC MVP")

        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in _N2_STALE_RL_FIELD_NAMES and key != "residual"
        }
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "model_dtype"):
            self.model_dtype = "bfloat16"
        if not hasattr(self, "pretrained_base_model_path"):
            self.pretrained_base_model_path = None
        if not hasattr(self, "loss_mode"):
            self.loss_mode = "BC_QUALITY_WEIGHTED"
        if not hasattr(self, "actor_loss_coeff"):
            self.actor_loss_coeff = 1.0
        for key in ("quality_w0", "quality_w1", "quality_w2"):
            if not hasattr(self, key):
                setattr(self, key, None)
        if not hasattr(self, "base_model_cfg") or self.base_model_cfg is None:
            self.base_model_cfg = GrootN2Config()
        self._finalize_quality_weight_map()

    def _finalize_quality_weight_map(self):
        if not hasattr(self, "quality_loss_weight_map") or self.quality_loss_weight_map is None:
            self.quality_loss_weight_map = {0: 1.0, 1: 0.5, 2: 0.0}
        self.quality_loss_weight_map = {
            int(key): float(value) for key, value in dict(self.quality_loss_weight_map).items()
        }
        if any(getattr(self, f"quality_w{i}", None) is not None for i in range(3)):
            self.quality_loss_weight_map = {
                i: float(getattr(self, f"quality_w{i}") or 0.0) for i in range(3)
            }

    def to_dict(self):
        output = super().to_dict()
        if isinstance(self.base_model_cfg, PretrainedConfig):
            output["base_model_cfg"] = self.base_model_cfg.to_dict()
        else:
            output["base_model_cfg"] = self.base_model_cfg
        output["quality_loss_weight_map"] = {
            int(key): float(value) for key, value in self.quality_loss_weight_map.items()
        }
        return output


@dataclass
class RLWrapperConfig(PretrainedConfig):
    model_type: str = field(default="RLWrapper", metadata={"help": "Model type identifier."})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})

    # MLP for residual, q, and v models
    hidden_size: int = field(default=1024, metadata={"help": "Hidden size of the MLP."})

    # Base model
    base_model_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Base model configuration."}
    )
    pretrained_base_model_path: str | None = field(
        default=None, metadata={"help": "Path to pretrained base model."}
    )

    # RL info key
    reward_key: str = field(
        default="next.delta_progress", metadata={"help": "Key for delta progress in rl_info."}
    )

    # Residual RL
    residual: bool = field(default=False, metadata={"help": "Whether to use residual RL."})
    residual_action_scale: float = field(
        default=0.1, metadata={"help": "Scale of the residual actions."}
    )
    residual_use_base_action: bool = field(
        default=True, metadata={"help": "Whether to use base action to predict residual."}
    )
    residual_pooling_method: str = field(
        default="mean", metadata={"help": "Pooling method for residual model."}
    )
    residual_pooling_qformer_cfg: dict | None = field(
        default=None, metadata={"help": "Qformer configuration for residual model."}
    )

    loss_mode: str = field(default="IQL", metadata={"help": "Loss mode."})
    actor_loss_coeff: float = field(default=1.0, metadata={"help": "Coefficient for actor loss."})
    q_loss_coeff: float = field(default=1.0, metadata={"help": "Coefficient for Q loss."})
    v_loss_coeff: float = field(default=1.0, metadata={"help": "Coefficient for V loss."})
    pretrained_q_model_path: str | None = field(
        default=None, metadata={"help": "Path to pretrained q model."}
    )
    pretrained_v_model_path: str | None = field(
        default=None, metadata={"help": "Path to pretrained value model."}
    )
    tune_q_model: bool = field(default=True, metadata={"help": "Whether to train the q model."})
    tune_v_model: bool = field(default=True, metadata={"help": "Whether to train the v model."})
    advantage_beta_inverse: float = field(
        default=1.0,
        metadata={"help": "Inverse beta (temperature) for advantage weighting, i.e. 1/beta."},
    )

    # For training q model
    q_model_pooling_method: str = field(default="mean", metadata={"help": "Pooling method."})
    q_model_pooling_qformer_cfg: dict | None = field(
        default=None, metadata={"help": "Qformer configuration."}
    )
    q_model_ensemble_size: int = field(
        default=2, metadata={"help": "Number of critics in the ensemble."}
    )
    q_model_polyak: float = field(default=0.995, metadata={"help": "Polyak coefficient."})
    q_model_gamma: float = field(default=0.99, metadata={"help": "Gamma for q model."})

    # For training v model (IQL)
    v_model_pooling_method: str = field(
        default="mean", metadata={"help": "Pooling method for value model"}
    )
    v_model_pooling_qformer_cfg: dict | None = field(
        default=None, metadata={"help": "Qformer configuration for value model"}
    )
    v_model_iql_quantile: float = field(
        default=0.7, metadata={"help": "Expectile parameter for value function (tau in IQL)"}
    )

    # Per-step loss weighting for BC_QUALITY_WEIGHTED mode.
    # Maps quality mask value (int) -> loss multiplier (float).
    quality_loss_weight_map: dict = field(
        default_factory=lambda: {0: 1.0, 1: 0.5, 2: 0.0},
        metadata={"help": "Quality value -> loss weight mapping for BC_QUALITY_WEIGHTED"},
    )

    # CLI-friendly scalar overrides for quality_loss_weight_map.
    quality_w0: float | None = field(
        default=None, metadata={"help": "Loss weight for quality=0 (clear_positive/normal)"}
    )
    quality_w1: float | None = field(
        default=None, metadata={"help": "Loss weight for quality=1 (slowly_positive/suboptimal)"}
    )
    quality_w2: float | None = field(
        default=None, metadata={"help": "Loss weight for quality=2 (clear_bad/failure)"}
    )

    # BC_CONTINUOUS_WEIGHTED mode: use delta_progress as continuous per-step weights.
    continuous_weight_reduction: str = field(
        default="sum",
        metadata={
            "help": "Reduction for continuous weights: 'sum' (sample-wise) or 'none' (step-wise)."
        },
    )
    delta_progress_weight_clamp_max: float | None = field(
        default=None,
        metadata={"help": "Optional upper clamp for continuous delta_progress weights."},
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._finalize_quality_weight_map()

    def _finalize_quality_weight_map(self):
        if not hasattr(self, "quality_loss_weight_map") or self.quality_loss_weight_map is None:
            self.quality_loss_weight_map = {0: 1.0, 1: 0.0, 2: 0.0}
        if isinstance(self.quality_loss_weight_map, dict):
            self.quality_loss_weight_map = {
                int(k): float(v) for k, v in self.quality_loss_weight_map.items()
            }
        if any(getattr(self, f"quality_w{i}", None) is not None for i in range(3)):
            self.quality_loss_weight_map = {
                i: float(getattr(self, f"quality_w{i}") or 0.0) for i in range(3)
            }


@dataclass
class GrootN1d6RLConfig(RLWrapperConfig):
    model_type: str = field(default="GrootN1d6RL", metadata={"help": "Model type identifier."})
    base_model_cfg: GrootN1d6Config = field(
        default_factory=GrootN1d6Config, metadata={"help": "Base model configuration."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "base_model_cfg") or self.base_model_cfg is None:
            self.base_model_cfg = GrootN1d6Config()
        self._finalize_quality_weight_map()


@dataclass
class GrootN1d7RLConfig(RLWrapperConfig):
    model_type: str = field(default="GrootN1d7RL", metadata={"help": "Model type identifier."})
    base_model_cfg: GrootN1d7Config = field(
        default_factory=GrootN1d7Config, metadata={"help": "Base model configuration."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "base_model_cfg") or self.base_model_cfg is None:
            self.base_model_cfg = GrootN1d7Config()
        self._finalize_quality_weight_map()


register_model_config("GrootN1d6RL", GrootN1d6RLConfig)
register_model_config("GrootN1d7RL", GrootN1d7RLConfig)
register_model_config("GrootN2RL", GrootN2RLConfig)
AutoConfig.register("GrootN2RL", GrootN2RLConfig, exist_ok=True)
