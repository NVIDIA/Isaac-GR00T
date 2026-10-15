# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Config schema for GrootN2 tactile F6-force and deform-map conditioning.

Torch-free and independent of n2.py, so the config registry, the TRT contract and the
tactile model package can all import it without pulling in the model stack.
"""

import json
import math
from typing import Annotated, Any, TypedDict

from tyro.constructors import PrimitiveConstructorSpec

# Pinned architecture. These used to be `.get()` defaults on tactile_cfg/deform_cfg, but
# no trained tactile checkpoint stores an override, so they are constants, not keys.
TACTILE_LATENT_DIM = 256  # Tactile-VAE per-finger latent
TACTILE_INPUT_WINDOW = 20  # force frames per VAE window; the future window has the same length
TACTILE_N_HANDS = 2
TACTILE_N_FINGERS = 5  # tactile tokens per hand
TACTILE_N_TOKENS = TACTILE_N_HANDS * TACTILE_N_FINGERS
TACTILE_F6_DIM = 6  # [Fx, Fy, Fz, Tx, Ty, Tz] per fingertip
DEFORM_N_MAPS = 10  # one deform map, and one token, per fingertip
DEFORM_FEATURE_DIM = 512
DEFORM_INPUT_HW = 240

# What an absent key means. Changing one silently changes every stored config that omits
# the key.
DEFAULT_PREDICT_FUTURE = False
DEFAULT_TACTILE_LOSS_SCALE = 0.2
DEFAULT_TACTILE_LOSS_WARMUP_STEPS = 1000
DEFAULT_FUTURE_PAD_TOLERANCE = 0

# The tactile attention rules extend exactly these two patterns. PATTERN_PREDICT is the
# name predict_future checkpoints saved; for the base tokens it is PATTERN_COND's table.
PATTERN_COND = "bidirectional_video_action_full"
PATTERN_PREDICT = "bidirectional_video_tactile_action_full"


class TactileConfig(TypedDict, total=False):
    """F6 force conditioning through a frozen Tactile-VAE; None on GrootN2Config disables it.

    The keys are exactly those tactile checkpoints store. The three loss knobs are only
    valid with predict_future=True.
    """

    tactile_vae_ckpt_path: str  # frozen Tactile-VAE .pt checkpoint
    # Also denoise future force as extra tokens with an auxiliary flow-matching loss.
    predict_future: bool  # default False
    tactile_loss_scale: float  # weight of the tactile loss (default 0.2)
    tactile_loss_warmup_steps: int  # linear warmup of the tactile loss (default 1000)
    # Future frames that may be clamped past the episode end and still be supervised,
    # holding the last frame. 0 drops any sample with a clamped future frame.
    future_pad_tolerance: int  # default 0


class DeformConfig(TypedDict, total=False):
    """Deform-map conditioning through a frozen HB1 encoder; None on GrootN2Config disables it."""

    deform_encoder_ckpt_path: str  # frozen HB1 deform-encoder .pth checkpoint


# On the command line each is ONE flag taking a JSON object (or null), e.g.
#   --model.tactile_cfg '{"tactile_vae_ckpt_path": "...", "predict_future": true}'
# not a tyro subcommand: an optional nested config adds a subcommand level to GrootN2Config,
# and every command line that selects a later subcommand (the RL configs nest GrootN2Config
# and select their own after it) would then have to select this one first.
_JSON_OBJECT = PrimitiveConstructorSpec(
    nargs=1,
    metavar="JSON",
    instance_from_str=lambda args: json.loads(args[0]),
    is_instance=lambda value: value is None or isinstance(value, dict),
    str_from_instance=lambda value: [json.dumps(value)],
)
TactileConfigArg = Annotated[TactileConfig | None, _JSON_OBJECT]
DeformConfigArg = Annotated[DeformConfig | None, _JSON_OBJECT]


_PREDICT_ONLY_KEYS = ("tactile_loss_scale", "tactile_loss_warmup_steps", "future_pad_tolerance")
# Keys earlier tactile branches accepted; named in the error so a stale recipe or
# checkpoint fails with a pointer to the constants above.
_PINNED_KEYS = frozenset(
    {
        "latent_dim",
        "input_window",
        "n_hands",
        "n_finger_tokens",
        "online_calibration",
        "scheduler_type",
        "hand_model",
        "feature_dim",
        "input_hw",
    }
)


def _check_keys(name: str, cfg: object, schema: type) -> None:
    if not isinstance(cfg, dict):
        raise TypeError(f"{name} must be a dict or None, got {type(cfg).__name__}")
    allowed = schema.__annotations__
    unknown = set(cfg) - set(allowed)
    if unknown:
        pinned = sorted(unknown & _PINNED_KEYS)
        hint = f"; {pinned} are pinned constants in n2_tactile.py" if pinned else ""
        raise ValueError(
            f"{name} has unsupported keys {sorted(unknown)}{hint}; allowed: {sorted(allowed)}"
        )


def _check_path(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string, got {value!r}")


def _check_number(name: str, value: object, *, integer: bool) -> None:
    kinds = int if integer else (int, float)
    # bool subclasses int, so a stray true/false would otherwise pass as 1/0.
    if isinstance(value, bool) or not isinstance(value, kinds):
        raise TypeError(f"{name} must be {'an int' if integer else 'a number'}, got {value!r}")


def _check_tactile(cfg: dict) -> None:
    _check_keys("tactile_cfg", cfg, TactileConfig)
    _check_path("tactile_cfg.tactile_vae_ckpt_path", cfg.get("tactile_vae_ckpt_path"))
    predict = cfg.get("predict_future", DEFAULT_PREDICT_FUTURE)
    if not isinstance(predict, bool):
        raise TypeError(f"tactile_cfg.predict_future must be a bool, got {predict!r}")
    stray = sorted(k for k in _PREDICT_ONLY_KEYS if k in cfg)
    if not predict and stray:
        # Without predict_future nothing reads them, so a set value is a silent no-op.
        raise ValueError(f"tactile_cfg {stray} only apply with predict_future=True")
    scale = cfg.get("tactile_loss_scale", DEFAULT_TACTILE_LOSS_SCALE)
    _check_number("tactile_cfg.tactile_loss_scale", scale, integer=False)
    if not (math.isfinite(scale) and scale >= 0):
        raise ValueError(f"tactile_cfg.tactile_loss_scale must be finite and >= 0, got {scale}")
    warmup = cfg.get("tactile_loss_warmup_steps", DEFAULT_TACTILE_LOSS_WARMUP_STEPS)
    _check_number("tactile_cfg.tactile_loss_warmup_steps", warmup, integer=True)
    if warmup < 0:
        raise ValueError(f"tactile_cfg.tactile_loss_warmup_steps must be >= 0, got {warmup}")
    tolerance = cfg.get("future_pad_tolerance", DEFAULT_FUTURE_PAD_TOLERANCE)
    _check_number("tactile_cfg.future_pad_tolerance", tolerance, integer=True)
    if not 0 <= tolerance <= TACTILE_INPUT_WINDOW:
        raise ValueError(
            f"tactile_cfg.future_pad_tolerance must be in [0, {TACTILE_INPUT_WINDOW}], "
            f"got {tolerance}"
        )


def validate_tactile_model_config(kwargs: dict[str, Any], *, exact_groot_n2: bool) -> None:
    """Reject tactile/deform settings the model would refuse later or silently misread.

    Args:
        kwargs: Raw GrootN2Config constructor kwargs (config.json, tyro or code), read for
            tactile_cfg, deform_cfg, action_cfg and token_merge_kernel.
        exact_groot_n2: Whether the config class is exactly GrootN2Config. Subclass
            models give additional-token ids 4+ their own meaning, so they cannot carry
            tactile or deform groups.

    Raises:
        TypeError: On a cfg that is not a dict or a knob of the wrong type.
        ValueError: On unknown or pinned keys, a missing encoder path, out-of-range knobs,
            loss knobs without predict_future, a subclass config, token merge, or an
            attention_mask_pattern other than the single one the tactile rules extend.
    """
    tactile = kwargs.get("tactile_cfg")
    deform = kwargs.get("deform_cfg")
    action_cfg = kwargs.get("action_cfg") or {}
    raw_pattern = action_cfg.get("attention_mask_pattern")
    pattern = [tuple(entry) for entry in raw_pattern or ()]
    if tactile is None and deform is None:
        if any(entry[0] == PATTERN_PREDICT for entry in pattern):
            raise ValueError(f"{PATTERN_PREDICT!r} requires tactile_cfg.predict_future=True")
        return
    if not exact_groot_n2:
        raise ValueError("tactile_cfg/deform_cfg are only supported on GrootN2, not subclasses")
    if kwargs.get("token_merge_kernel") is not None:
        # Tactile checkpoint parity with token merging has not been validated.
        raise ValueError("tactile_cfg/deform_cfg do not support token_merge_kernel")
    if tactile is not None:
        _check_tactile(tactile)
    if deform is not None:
        _check_keys("deform_cfg", deform, DeformConfig)
        _check_path("deform_cfg.deform_encoder_ckpt_path", deform.get("deform_encoder_ckpt_path"))
    predict = tactile is not None and tactile.get("predict_future", DEFAULT_PREDICT_FUTURE)
    want = PATTERN_PREDICT if predict else PATTERN_COND
    if pattern != [(want, 1.0)]:
        raise ValueError(
            f"with predict_future={predict}, tactile/deform need "
            f"action_cfg.attention_mask_pattern == [({want!r}, 1.0)], got {raw_pattern!r}"
        )
