# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tactile/deform conditioning at predict, for GrootN2's inference packing hooks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch.nn import functional as F

from groot.core.training.configs.model.n2_tactile import TACTILE_LATENT_DIM, TACTILE_N_TOKENS
from groot.core.training.models.n2.components.sequence_packing import unpack_contiguous
from groot.core.training.models.n2.tactile import guards, tokens
from groot.core.training.models.n2.tactile.attention import (
    MODALITY_TACTILE,
    PATTERN_VIDEO_TACTILE_ACTION_FULL,
    tactile_pattern_name,
)
from groot.core.training.models.n2.tactile.modules import encode_deform, encode_tactile


def check_future_inference(model: Any, data: dict, *, joint_denoise: bool) -> None:
    """Require the joint solver for checkpoints trained with future force."""
    spec = getattr(model, "tactile_spec", None)
    if spec is None or not spec.predict_future:
        return
    if not joint_denoise:
        raise guards.TactileUnsupportedError(
            "Future-F6 prediction requires joint_denoise=True (GROOT_N2_JOINT_DENOISE=1)."
        )
    if getattr(model.config, "token_merge_kernel", None) is not None:
        raise guards.TactileUnsupportedError("Future-F6 prediction does not support token merging.")
    if tactile_pattern_name(model.attention_mask_pattern) != PATTERN_VIDEO_TACTILE_ACTION_FULL:
        raise guards.TactileUnsupportedError(
            f"Future-F6 prediction requires {PATTERN_VIDEO_TACTILE_ACTION_FULL}."
        )
    guards.check_inference_batch(model.config, data)
    if not model.has_action_head or "state" not in data:
        raise guards.MissingTactileInputError(
            "Future-F6 prediction requires an action head and state."
        )


@dataclass
class FutureTactileInference:
    """Per-prediction future-force latents and solver; never stored on the model."""

    latents: torch.Tensor
    scheduler: Any

    def pack(self, model, kwargs, *, step, dtype) -> dict:
        """Insert future force between the observed force and deformation groups."""
        groups = kwargs["additional_token_groups"]
        future = tokens.tactile_denoise_group(
            model, self.latents, self.scheduler.sigmas[step], out_dtype=dtype
        )
        # Both force groups are a fixed snapshot during action-only substeps.
        # The joint loop rebuilds their K/V at every video/force update; neither
        # group reads action, so action substeps cannot change this snapshot.
        groups = (groups[0], replace(future, kv_cacheable=True), *groups[1:])
        return tokens.pack_kwargs(groups, model.attention_mask_pattern)

    def decode_velocity(self, model, out_tokens: torch.Tensor, packed) -> torch.Tensor:
        """Decode the trained future-force head with the training dtype handling."""
        hidden = unpack_contiguous(out_tokens, packed, MODALITY_TACTILE)
        head = model.tactile_decoder_head
        return F.linear(hidden, head.weight.to(hidden.dtype), head.bias.to(hidden.dtype)).float()

    def advance(self, velocity: torch.Tensor, step: int) -> None:
        """Advance force once with each joint video/action step."""
        self.latents = self.scheduler.step(
            velocity, self.scheduler.timesteps[step], self.latents, step_index=step
        )[0]


def prepare_future_inference(
    model: Any, *, batch_size: int, device, num_steps: int, shift: float, generator
) -> FutureTactileInference | None:
    """Draw future noise after video/action noise and create an independent solver."""
    spec = getattr(model, "tactile_spec", None)
    if spec is None or not spec.predict_future:
        return None
    scheduler = type(model.tactile_scheduler)(model.tactile_scheduler.config)
    scheduler.set_timesteps(num_steps, device=device, shift=shift)
    latents = torch.randn(
        batch_size,
        TACTILE_N_TOKENS,
        TACTILE_LATENT_DIM,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return FutureTactileInference(latents, scheduler)


def inference_pack_kwargs(model: Any, data: dict, *, video_tokens: torch.Tensor) -> dict:
    """Packer kwargs for the conditioning groups at predict (``{}`` without tactile configs).

    Emits the present-force and deform groups. The future-force joint solver inserts
    its evolving group. Both force groups are cached together between joint
    updates; the encoders are deterministic and the observed inputs stay fixed.

    Args:
        model: The GrootN2 (``getattr``: duck-typed runtimes may lack ``tactile_spec``).
        data: The inference batch.
        video_tokens: This pack's video tokens (device and token dtype).

    Returns:
        ``{}``, or the groups and attention rule to splat into ``packer.pack``.

    Raises:
        MissingTactileInputError: If the batch lacks an input the model consumes.
    """
    spec = getattr(model, "tactile_spec", None)
    if spec is None:
        return {}
    guards.check_inference_batch(model.config, data)
    device, dtype = video_tokens.device, video_tokens.dtype
    groups = []
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        if spec.has_tactile:
            mu = encode_tactile(model, data["tactile_raw"].to(device))
            groups.append(tokens.tactile_cond_group(model, mu, out_dtype=dtype, sigma_dtype=dtype))
        if spec.has_deform:
            mu = encode_deform(model, data["deform_raw"].to(device))
            groups.append(tokens.deform_group(model, mu, out_dtype=dtype, sigma_dtype=dtype))
    return tokens.pack_kwargs(groups, getattr(model, "attention_mask_pattern", None))
