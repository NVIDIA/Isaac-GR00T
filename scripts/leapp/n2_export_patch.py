"""Trace-time substitutions used only by the N2 LEAPP exporter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import os
from typing import Any
from unittest.mock import patch

import cv2
from leapp import FunctionPatch, TensorSemantics
import numpy as np
import torch
import torch.nn.functional as F

from groot.core.data.state_action import state_action_processor
from groot.core.data.types import ActionFormat, ActionRepresentation, ActionType
from groot.core.data.utils import _align_stats_bounds
from groot.core.training.models.dream_cosmos.attention_utils.flex_attention import (
    torch_attention_op,
)
from groot.core.training.models.n2 import processor as n2_processor
from groot.core.training.models.n2.components import sequence_packing, transformer_cosmos
from groot.core.training.models.n2.components.wan_video_vae import CausalConv3d, RMS_norm


def cv2_resize_for_leapp(
    src: np.ndarray,
    dsize: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Trace N2's RGB INTER_LINEAR resize without a TensorRT-unsupported uint8 tensor."""
    if interpolation != cv2.INTER_LINEAR:
        raise ValueError("N2 export only supports cv2.INTER_LINEAR")

    image = torch.from_numpy(src).permute(2, 0, 1).unsqueeze(0).float()
    image = F.interpolate(
        image,
        size=(dsize[1], dsize[0]),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    return image[0].round().clamp(0, 255).to(torch.float32).permute(1, 2, 0).numpy()


def apply_eval_image_transform_for_leapp(transform, images, replay=None):
    """Apply deterministic eval transforms while keeping pixels as float32."""
    if replay is not None or hasattr(transform, "replay"):
        raise ValueError("N2 export only supports deterministic eval image transforms")

    transformed = []
    for image in images:
        image_array = transform(image=np.array(image))["image"]
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32)
        elif image_array.dtype != np.float32:
            raise ValueError(f"Unexpected transformed image data type: {image_array.dtype}")
        transformed.append(torch.from_numpy(image_array).permute(2, 0, 1))
    return transformed, None


def dense_create_block_mask_for_leapp(
    mask_mod,
    *,
    B: int,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    BLOCK_SIZE: int,
    device: str | torch.device,
):
    """Evaluate an N2 FlexAttention mask function into a dense boolean mask."""
    del H, BLOCK_SIZE
    batch_idx = torch.arange(B, device=device).view(B, 1, 1)
    query_idx = torch.arange(Q_LEN, device=device).view(1, Q_LEN, 1)
    key_idx = torch.arange(KV_LEN, device=device).view(1, 1, KV_LEN)
    head_idx = torch.zeros((), dtype=torch.long, device=device)
    return mask_mod(batch_idx, head_idx, query_idx, key_idx).unsqueeze(1)


def dense_attention_for_leapp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor,
    pad: int,
    flatten_heads: bool,
    compile_attention: bool,
) -> torch.Tensor:
    """Use GR00T's dense attention implementation for a dense export mask."""
    del pad, compile_attention
    attn_mask = attn_mask[..., : q.shape[1], : k.shape[1]]
    return torch_attention_op(
        q,
        k,
        v,
        attn_mask=attn_mask,
        flatten_heads=flatten_heads,
    )


def normalize_values_minmax_for_leapp(values, params):
    """Traceable equivalent of GR00T's min/max normalization."""
    max_vals = _align_stats_bounds(values, params["max"])
    min_vals = _align_stats_bounds(values, params["min"])
    mask = ~np.isclose(max_vals, min_vals)
    safe_range = np.where(mask, max_vals - min_vals, 1)
    normalized = 2 * (values - min_vals) / safe_range - 1
    return np.where(mask, normalized, 0).astype(values.dtype)


def normalize_values_meanstd_for_leapp(values, params):
    """Traceable equivalent of GR00T's mean/std normalization."""
    mean_vals = _align_stats_bounds(values, params["mean"])
    std_vals = _align_stats_bounds(values, params["std"])
    mask = std_vals != 0
    safe_std = np.where(mask, std_vals, 1)
    normalized = (values - mean_vals) / safe_std
    return np.where(mask, normalized, values).astype(values.dtype)


def n2_function_patches() -> tuple[FunctionPatch, ...]:
    """Let LEAPP substitute module functions only when their inputs are traced."""
    return (
        FunctionPatch(cv2, "resize", cv2_resize_for_leapp),
        FunctionPatch(transformer_cosmos, "flex_attention_op", dense_attention_for_leapp),
        FunctionPatch(
            state_action_processor, "normalize_values_minmax", normalize_values_minmax_for_leapp
        ),
        FunctionPatch(
            state_action_processor, "normalize_values_meanstd", normalize_values_meanstd_for_leapp
        ),
    )


def relative_joint_action_to_absolute_for_leapp(action, reference_state):
    """Add the current joint position to every predicted relative-action step."""
    if action.ndim != 2 or reference_state.ndim != 1:
        raise ValueError(
            "Expected relative joint action [horizon, joints] and reference state [joints]"
        )
    if action.shape[1] != reference_state.shape[0]:
        raise ValueError(
            f"Action width {action.shape[1]} does not match state width "
            f"{reference_state.shape[0]}"
        )
    return action + reference_state[None, :]


def causal_conv3d_for_leapp(module, x, cache_x=None):
    """Keep causal padding out of the exported Conv attributes."""
    padding = list(module._padding)  # noqa: SLF001
    if cache_x is not None and module._padding[4] > 0:  # noqa: SLF001
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]

    if padding[4] > 0:
        prefix = x.new_zeros((x.shape[0], x.shape[1], padding[4], x.shape[3], x.shape[4]))
        x = torch.cat([prefix, x], dim=2)

    if padding[0] != padding[1] or padding[2] != padding[3] or padding[5] != 0:
        raise ValueError(f"Unsupported CausalConv3d padding for ONNX export: {padding}")

    return F.conv3d(
        x,
        module.weight,
        module.bias,
        module.stride,
        (0, padding[2], padding[0]),
        module.dilation,
        module.groups,
    )


def rms_norm_for_leapp(module, x):
    """Run the reduction in FP32 while preserving the VAE tensor dtype."""
    dim = 1 if module.channel_first else -1
    normalized = F.normalize(x.float(), dim=dim).type_as(x)
    return normalized * module.scale * module.gamma + module.bias


def zero_pad2d_for_leapp(module, x):
    """Express the VAE downsample padding as concatenation for ONNX Runtime."""
    if tuple(module.padding) != (0, 1, 0, 1):
        raise ValueError(f"Unsupported ZeroPad2d padding for ONNX export: {module.padding}")

    x = torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)
    return torch.cat([x, torch.zeros_like(x[..., :1, :])], dim=-2)


@contextmanager
def annotate_action_noise_for_leapp(model, annotate, node_name, original_randn):
    """Promote the seeded action-noise draw to an FP32 LEAPP input."""
    action_config = model.config.action_cfg
    action_shape = (
        action_config.get("action_horizon", 48),
        action_config.get("max_action_dim", 32),
    )
    captured = False

    def randn(*args, **kwargs):
        nonlocal captured
        value = original_randn(*args, **kwargs)
        if not captured and value.ndim == 3 and tuple(value.shape[1:]) == action_shape:
            value = annotate.input_tensors(
                node_name,
                TensorSemantics(
                    name="initial_noise",
                    ref=value.float(),
                ),
            ).to(device=value.device, dtype=value.dtype)
            captured = True
        return value

    with patch.object(torch, "randn", randn):
        yield

    if not captured:
        raise RuntimeError(f"Did not capture action noise with trailing shape {action_shape}")


@contextmanager
def partial_eef_actions_for_leapp(processor, embodiment_tag: str):
    """Omit only relative EEF pose composition during this processor's decode."""
    modalities = processor.modality_configs[embodiment_tag]
    action_modality = modalities["action"]
    configs = action_modality.action_configs
    outputs = {}
    export_configs = []
    for key, config in zip(action_modality.modality_keys, configs or [], strict=False):
        if (
            processor.use_relative_action
            and config.rep == ActionRepresentation.RELATIVE
            and config.type == ActionType.EEF
        ):
            outputs[f"action.{key}"] = {
                "representation": "relative",
                # The native converter accepts xyz + rot6d even when a different
                # absolute output format was requested. No format conversion runs here.
                "format": ActionFormat.XYZ_ROT6D.value,
                "element_names": ["x", "y", "z", *[f"rot6d_{i}" for i in range(6)]],
                "rotation_convention": "first two rows of the rotation matrix",
                "reference_state": f"state.{config.state_key or key}",
                "reference_timestep": "last observed timestep, shared by the entire action chunk",
                "composition": "T_absolute = T_reference @ T_relative",
                "requested_absolute_format": config.format.value,
                "units": "checkpoint training-data units; no unit conversion is applied",
            }
            # Change only the decode-time conversion predicate. In particular,
            # keep use_relative_action and its already-loaded statistics intact.
            export_configs.append(replace(config, rep=ActionRepresentation.ABSOLUTE))
        else:
            export_configs.append(config)

    if not outputs:
        yield outputs
        return

    export_modalities = {
        **processor.modality_configs,
        embodiment_tag: {
            **modalities,
            "action": replace(action_modality, action_configs=export_configs),
        },
    }
    original_unapply = processor.unapply_action

    def unapply_action(action, embodiment_tag, state=None):
        # Only decoding sees the modified config; preprocessing stays unchanged.
        with patch.object(processor, "modality_configs", export_modalities):
            return original_unapply(action, embodiment_tag, state=state)

    with patch.object(processor, "unapply_action", unapply_action):
        yield outputs


@contextmanager
def patch_n2_for_leapp(model: Any) -> Iterator[None]:
    """Scope method and no-traced-argument patches not handled by FunctionPatch."""
    decoder_class = type(model.action_decoder)
    original_decoder_forward = decoder_class.forward
    original_to_absolute = (
        state_action_processor.StateActionProcessor._convert_to_absolute_action  # noqa: SLF001
    )

    def action_decoder_forward(decoder, tokens, cat_ids):
        tokens = tokens.to(dtype=next(decoder.parameters()).dtype)
        return original_decoder_forward(decoder, tokens, cat_ids)

    def convert_to_absolute(processor, action, reference_state, action_type, action_format):
        if action_type == ActionType.NON_EEF and action_format == ActionFormat.DEFAULT:
            return relative_joint_action_to_absolute_for_leapp(action, reference_state)
        if action_type == ActionType.EEF:
            raise NotImplementedError(
                "Unexpected relative end-effector pose conversion during LEAPP export. "
                "The partial EEF export must bypass this step; pose conversion belongs "
                "in the deployment application."
            )
        return original_to_absolute(
            processor,
            action,
            reference_state,
            action_type,
            action_format,
        )

    with (
        patch.dict(os.environ, {"ENABLE_TENSORRT": "true"}),
        # Traced tensors live in mask_mod's closure, not in the call arguments.
        patch.object(
            sequence_packing,
            "create_block_mask",
            dense_create_block_mask_for_leapp,
        ),
        patch.object(
            n2_processor,
            "apply_with_replay",
            apply_eval_image_transform_for_leapp,
        ),
        patch.object(
            state_action_processor.StateActionProcessor,
            "_convert_to_absolute_action",
            convert_to_absolute,
        ),
        patch.object(CausalConv3d, "forward", causal_conv3d_for_leapp),
        patch.object(RMS_norm, "forward", rms_norm_for_leapp),
        patch.object(torch.nn.ZeroPad2d, "forward", zero_pad2d_for_leapp),
        patch.object(decoder_class, "forward", action_decoder_forward),
        # AOT ONNX cannot save inference tensors; no_grad keeps export inference-only
        # without setting PyTorch's permanent inference-tensor bit.
        patch.object(torch, "inference_mode", torch.no_grad),
    ):
        yield
