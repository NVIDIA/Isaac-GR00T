# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tactile/deform in the GrootN2 training forward: three calls, in forward order.

``prepare_training_inputs`` checks the batch and encodes it before GrootN2 draws its video
and action noise, ``training_pack_kwargs`` adds the token groups to the single pack, and
``add_training_loss`` folds the future-force loss into the additional training loss.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F

from groot.core.training.models.n2.components.sequence_packing import unpack
from groot.core.training.models.n2.tactile import guards, tokens
from groot.core.training.models.n2.tactile.attention import MODALITY_TACTILE
from groot.core.training.models.n2.tactile.modules import (
    encode_deform,
    encode_tactile,
    pin_tactile_vae,
)


class TactileTrainInputs(NamedTuple):
    """One training batch's encoded tactile/deform inputs; None where the model has none."""

    tactile_mu: Tensor | None
    deform_mu: Tensor | None
    # predict_future only: noised future latents, the per-sample t and sigma (B, 1), the
    # flow-matching target, and the per-sample loss mask (B,).
    future_x_t: Tensor | None = None
    future_t: Tensor | None = None
    future_sigma: Tensor | None = None
    future_target: Tensor | None = None
    future_valid: Tensor | None = None


def prepare_training_inputs(
    model: Any,
    data: dict,
    *,
    batch_size: int,
    device: torch.device,
    has_action: bool,
    action_loss_details: bool,
) -> TactileTrainInputs | None:
    """Check the batch, encode the tactile/deform inputs and noise the future force window.

    Call before GrootN2 samples its video/action noise: the tactile checkpoints were trained
    with the future-force timesteps and noise drawn first.

    Args:
        model: The GrootN2 being trained.
        data: The training batch.
        batch_size: B.
        device: The device of the video latents.
        has_action: Whether the batch packs action tokens.
        action_loss_details: Whether forward returns action-loss details.

    Returns:
        None for a model without tactile configs, else the encoded inputs.
    """
    spec = model.tactile_spec
    if spec is None:
        return None
    guards.check_training_inputs(
        spec.config_fields,
        spec.training_input_keys,
        data,
        has_action=has_action,
        action_loss_details=action_loss_details,
    )
    if spec.has_tactile:
        # Before the first encode: see pin_tactile_vae (compiled forward + CUDA-graph trees).
        pin_tactile_vae(model, device)
    tactile_mu = encode_tactile(model, data["tactile_raw"].to(device)) if spec.has_tactile else None
    deform_mu = encode_deform(model, data["deform_raw"].to(device)) if spec.has_deform else None
    if not spec.predict_future:
        return TactileTrainInputs(tactile_mu, deform_mu)

    scheduler = model.tactile_scheduler
    # fp32 for the scheduler math, like the action head.
    x0 = encode_tactile(model, data["tactile_raw_future"].to(device)).float()
    n_tokens = x0.shape[1]
    t, sigma = scheduler.sample_timesteps(batch_size, device, x0.dtype)
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(
        x0.flatten(0, 1), noise.flatten(0, 1), t.expand(batch_size, n_tokens).flatten(0, 1)
    ).unflatten(0, (batch_size, n_tokens))
    n_padded = data["tactile_future_n_padded"].to(device=device)
    return TactileTrainInputs(
        tactile_mu,
        deform_mu,
        future_x_t=x_t,
        future_t=t,
        future_sigma=sigma,
        future_target=scheduler.training_target(x0, noise),
        # Supervise windows with at most future_pad_tolerance frames clamped past the
        # episode end (their target holds the last frame).
        future_valid=(n_padded <= spec.future_pad_tolerance).to(torch.float32),
    )


def training_pack_kwargs(
    model: Any, inputs: TactileTrainInputs | None, *, dtype: torch.dtype
) -> dict:
    """Packer kwargs for the training pack (``{}`` without tactile inputs).

    Call inside the forward's bf16 autocast block, where the tokens were always projected.

    Args:
        model: The GrootN2 being trained.
        inputs: ``prepare_training_inputs``'s result.
        dtype: The video token dtype.

    Returns:
        ``{}``, or the tactile groups and attention rule to splat into ``packer.pack``.
    """
    if inputs is None:
        return {}
    groups = []
    if inputs.tactile_mu is not None:
        groups.append(tokens.tactile_cond_group(model, inputs.tactile_mu, out_dtype=dtype))
    if inputs.future_x_t is not None:
        groups.append(
            tokens.tactile_denoise_group(
                model, inputs.future_x_t, inputs.future_sigma, out_dtype=dtype
            )
        )
    if inputs.deform_mu is not None:
        groups.append(tokens.deform_group(model, inputs.deform_mu, out_dtype=dtype))
    return tokens.pack_kwargs(groups, getattr(model, "attention_mask_pattern", None))


def add_training_loss(
    model: Any,
    inputs: TactileTrainInputs | None,
    *,
    packed: Any,
    out_tokens: Tensor,
    additional_loss: Tensor | None,
    additional_metrics: dict[str, Tensor],
) -> tuple[Tensor | None, dict[str, Tensor]]:
    """Add the future-force flow-matching loss to GrootN2's additional training loss.

    Call outside the autocast block, like the video and action losses. Without
    predict_future, returns ``additional_loss`` and ``additional_metrics`` unchanged.

    Args:
        model: The GrootN2 being trained.
        inputs: ``prepare_training_inputs``'s result.
        packed: The training ``PackedSequence``.
        out_tokens: The transformer output.
        additional_loss: The loss from ``_compute_additional_training_loss``.
        additional_metrics: Its metrics.

    Returns:
        The additional loss plus ``tactile_loss_scale * warmup * tactile_loss``, and the
        metrics plus ``tactile_loss``.
    """
    if inputs is None or inputs.future_x_t is None:
        return additional_loss, additional_metrics
    denoise_out = unpack(out_tokens, packed.modality_ids, MODALITY_TACTILE)  # (B, n, D)
    head = model.tactile_decoder_head
    # Compute in the fp32 of the transformer output: under DeepSpeed bf16 the head's weights
    # are bf16 (the action decoder casts its weights the same way).
    pred_v = F.linear(
        denoise_out, head.weight.to(denoise_out.dtype), head.bias.to(denoise_out.dtype)
    )
    loss_per_sample = F.mse_loss(
        pred_v.float(), inputs.future_target.float(), reduction="none"
    ).mean(dim=(1, 2))
    batch_size = loss_per_sample.shape[0]
    weight = (
        model.tactile_scheduler.training_weight(inputs.future_t.reshape(batch_size))
        .to(loss_per_sample.device)
        .reshape(batch_size)
    ) * inputs.future_valid
    tactile_loss = (loss_per_sample * weight).sum() / weight.sum().clamp(min=1.0)
    term = model.tactile_spec.tactile_loss_scale * model.tactile_loss_warmup_factor * tactile_loss
    loss = term if additional_loss is None else additional_loss + term
    return loss, {**additional_metrics, "tactile_loss": tactile_loss}
