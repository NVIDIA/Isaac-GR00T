# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tactile/deform AdditionalTokenGroups, packed after action as (4, 5, 6).

Projection, positional embedding, timesteps and RoPE reproduce the tokens the tactile
checkpoints were trained and served with, bit for bit. Callers own the autocast
context: those tokens were projected inside the bf16 autocast block on every path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from groot.core.training.models.n2.components.sequence_packing import AdditionalTokenGroup
from groot.core.training.models.n2.tactile.attention import (
    MODALITY_DEFORM_COND,
    MODALITY_TACTILE,
    MODALITY_TACTILE_COND,
    TACTILE_RULE_MODALITIES,
    attention_rule_for,
    tactile_pattern_name,
)


def _project(
    norm: nn.Module, embed: nn.Module, pos_emb: Tensor, x: Tensor, out_dtype: torch.dtype
) -> Tensor:
    # Cast after adding the fp32 pos_emb: under bf16 autocast the sum promotes back to
    # fp32 and would mismatch the bf16 video tokens at the packer's cat.
    return (embed(norm(x.float())) + pos_emb).to(out_dtype)


def tactile_cond_group(
    model: Any, mu: Tensor, *, out_dtype: torch.dtype, sigma_dtype: torch.dtype | None = None
) -> AdditionalTokenGroup:
    """Clean present-window force tokens at RoPE positions 0..n-1.

    Args:
        model: Holds tactile_norm / tactile_token_embed / tactile_pos_emb / tactile_rope
            and cond_timestep.
        mu: (B, n, latent) encoder latents.
        out_dtype: Token dtype (the video token dtype).
        sigma_dtype: Timestep dtype. None (fp32) in training; the inference dtype at predict.

    Returns:
        The kv_cacheable group, id 4.
    """
    tokens = _project(
        model.tactile_norm, model.tactile_token_embed, model.tactile_pos_emb, mu, out_dtype
    )
    return AdditionalTokenGroup(
        tokens=tokens,
        timesteps=torch.full(
            tokens.shape[:2], model.cond_timestep, device=tokens.device, dtype=sigma_dtype
        ),
        rope_freqs=model.tactile_rope(tokens),
        modality_id=MODALITY_TACTILE_COND,
        # Static conditioning: cached action-only steps keep attending to it.
        kv_cacheable=True,
    )


def tactile_denoise_group(
    model: Any, x_t: Tensor, sigma: Tensor, *, out_dtype: torch.dtype
) -> AdditionalTokenGroup:
    """Noised future-force latents (predict_future) at RoPE positions n..2n-1.

    Args:
        model: Same modules as ``tactile_cond_group``; tactile_rope spans 2n positions.
        x_t: (B, n, latent) noised future latents.
        sigma: (B, 1) flow-matching sigma, broadcast to the n tokens for AdaLN.
        out_dtype: Token dtype (the video token dtype).

    Returns:
        The non-cacheable group, id 5.
    """
    batch_size, n_tokens = x_t.shape[:2]
    tokens = _project(
        model.tactile_norm, model.tactile_token_embed, model.tactile_pos_emb, x_t, out_dtype
    )
    # The present window holds positions 0..n-1. ActionRoPE1D reads only the length and
    # device, so a length-2n placeholder yields both halves; keep the future one.
    cos, sin = model.tactile_rope(x_t.new_zeros(batch_size, 2 * n_tokens, 1))
    return AdditionalTokenGroup(
        tokens=tokens,
        timesteps=sigma.expand(batch_size, n_tokens),
        rope_freqs=(cos[n_tokens:], sin[n_tokens:]),
        modality_id=MODALITY_TACTILE,
        # Evolving future force is never a static conditioning cache entry.
        kv_cacheable=False,
    )


def deform_group(
    model: Any, mu: Tensor, *, out_dtype: torch.dtype, sigma_dtype: torch.dtype | None = None
) -> AdditionalTokenGroup:
    """Clean deform-map feature tokens, one per finger, at RoPE positions 0..n-1.

    Args:
        model: Holds deform_norm / deform_token_embed / deform_pos_emb / deform_rope and
            cond_timestep.
        mu: (B, n, feature) encoder features.
        out_dtype: Token dtype (the video token dtype).
        sigma_dtype: Timestep dtype. None (fp32) in training; the inference dtype at predict.

    Returns:
        The kv_cacheable group, id 6.
    """
    tokens = _project(
        model.deform_norm, model.deform_token_embed, model.deform_pos_emb, mu, out_dtype
    )
    return AdditionalTokenGroup(
        tokens=tokens,
        timesteps=torch.full(
            tokens.shape[:2], model.cond_timestep, device=tokens.device, dtype=sigma_dtype
        ),
        rope_freqs=model.deform_rope(tokens),
        modality_id=MODALITY_DEFORM_COND,
        kv_cacheable=True,
    )


def pack_kwargs(groups: Sequence[AdditionalTokenGroup], pattern_config: Any) -> dict:
    """Return packer kwargs for tactile groups: ``{}`` without groups.

    Args:
        groups: Tactile groups in packing order.
        pattern_config: The model's ``attention_mask_pattern``, read at pack time (it can be
            reassigned after ``__init__``).

    Returns:
        ``{}``, or ``{"additional_token_groups", "attention_rule"}`` to splat into
        ``packer.pack``.

    Raises:
        ValueError: If the ids are not ascending, or the pattern has no edges for one of
            them.
    """
    if not groups:
        return {}
    rule = attention_rule_for(pattern_config)
    ids = [group.modality_id for group in groups]
    # (4, 5, 6) is the layout verified against the trained checkpoints.
    if ids != sorted(set(ids)):
        raise ValueError(f"Tactile groups must be packed in ascending id order, got {ids}")
    name = tactile_pattern_name(pattern_config)
    unconnected = sorted(set(ids) - TACTILE_RULE_MODALITIES[name])
    if unconnected:
        raise ValueError(f"{name!r} gives tactile modality ids {unconnected} no attention edges")
    return {"additional_token_groups": tuple(groups), "attention_rule": rule}
