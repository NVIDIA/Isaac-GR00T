# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Attention rules for the tactile/deform token groups appended after action.

The packer keeps its 4x4 ``to_table()``. Each rule is the packer's own edge for ids 0..3
plus the tactile edges of the 7x7 tables the tactile checkpoints were trained with, so
it matches those tables on every raw id pair, padding id -1 included.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from groot.core.training.models.n2.components.sequence_packing import (
    MODALITY_ACTION,
    MODALITY_VIDEO_COND,
    AttentionRule,
)

# Runtime-only ids, never serialized. TTT registers and subtask completion queries also
# use id 4, which is why attach() refuses tactile on every GrootN2 subclass.
MODALITY_TACTILE_COND = 4
MODALITY_TACTILE = 5  # future-force denoise tokens (predict_future)
MODALITY_DEFORM_COND = 6

# Saved-config names (action_cfg.attention_mask_pattern). Plain strings, so resolving a
# rule never depends on which members the packer's enum has.
PATTERN_VIDEO_ACTION_FULL = "bidirectional_video_action_full"
PATTERN_VIDEO_TACTILE_ACTION_FULL = "bidirectional_video_tactile_action_full"


def _ordinary(b: Tensor, q: Tensor, kv: Tensor, base_table: Tensor) -> Tensor:
    """The packer's own 4x4 edge (its MOD_TO_TABLE [1, 0, 2, 3] lookup) for ids 0..3.

    The clamp mirrors the packer's lookup, so padding id -1 reads the video row like it
    does there (the packer masks -1 itself), and ids >= 4 get no ordinary edge.
    """
    q4 = q.clamp(min=0, max=MODALITY_ACTION)
    kv4 = kv.clamp(min=0, max=MODALITY_ACTION)
    # MOD_TO_TABLE as arithmetic (swap 0/1): no captured lookup tensor in the mask_mod.
    q_idx = torch.where(q4 < 2, 1 - q4, q4)
    kv_idx = torch.where(kv4 < 2, 1 - kv4, kv4)
    return (q <= MODALITY_ACTION) & (kv <= MODALITY_ACTION) & base_table[b, q_idx, kv_idx]


def _is_video(m: Tensor) -> Tensor:
    # Ids 0/1, and padding -1 as in the packer's clamp(min=0) lookup (it masks -1 itself).
    return m <= MODALITY_VIDEO_COND


def video_action_full_rule(b: Tensor, q: Tensor, kv: Tensor, base_table: Tensor) -> Tensor:
    """Rule for bidirectional_video_action_full (conditioning-only checkpoints).

    action -> {tactile_cond, deform}; tactile_cond -> {all video, tactile_cond};
    deform -> {deform}. Video and state never read ids >= 4; id 5 has no edges.
    """
    return (
        _ordinary(b, q, kv, base_table)
        | ((q == MODALITY_ACTION) & ((kv == MODALITY_TACTILE_COND) | (kv == MODALITY_DEFORM_COND)))
        | ((q == MODALITY_TACTILE_COND) & (_is_video(kv) | (kv == MODALITY_TACTILE_COND)))
        | ((q == MODALITY_DEFORM_COND) & (kv == MODALITY_DEFORM_COND))
    )


def video_tactile_action_full_rule(b: Tensor, q: Tensor, kv: Tensor, base_table: Tensor) -> Tensor:
    """Rule for bidirectional_video_tactile_action_full (predict_future checkpoints).

    action -> {tactile_cond, tactile_denoise, deform}; {tactile_cond, tactile_denoise} ->
    {all video, tactile_cond, tactile_denoise}; deform -> {deform}. Video and state never
    read ids >= 4. The packer's table for this pattern equals bidirectional_video_action_full's.
    """
    tactile_q = (q == MODALITY_TACTILE_COND) | (q == MODALITY_TACTILE)
    tactile_kv = (kv == MODALITY_TACTILE_COND) | (kv == MODALITY_TACTILE)
    return (
        _ordinary(b, q, kv, base_table)
        | ((q == MODALITY_ACTION) & (tactile_kv | (kv == MODALITY_DEFORM_COND)))
        | (tactile_q & (_is_video(kv) | tactile_kv))
        | ((q == MODALITY_DEFORM_COND) & (kv == MODALITY_DEFORM_COND))
    )


TACTILE_ATTENTION_RULES: dict[str, AttentionRule] = {
    PATTERN_VIDEO_ACTION_FULL: video_action_full_rule,
    PATTERN_VIDEO_TACTILE_ACTION_FULL: video_tactile_action_full_rule,
}

# Group ids each rule connects. A group outside its pattern's set would be packed with no
# edges at all (e.g. denoise tokens under bidirectional_video_action_full).
TACTILE_RULE_MODALITIES: dict[str, frozenset[int]] = {
    PATTERN_VIDEO_ACTION_FULL: frozenset({MODALITY_TACTILE_COND, MODALITY_DEFORM_COND}),
    PATTERN_VIDEO_TACTILE_ACTION_FULL: frozenset(
        {MODALITY_TACTILE_COND, MODALITY_TACTILE, MODALITY_DEFORM_COND}
    ),
}


def tactile_pattern_name(pattern_config: Any) -> str:
    """Return the one deterministic tactile pattern in an attention_mask_pattern config.

    Args:
        pattern_config: ``[(name, prob)]`` as the packer takes it; ``name`` may be a
            string or an ``AttentionMaskPattern`` member.

    Returns:
        The pattern name, a key of ``TACTILE_ATTENTION_RULES``.

    Raises:
        ValueError: Unless there is exactly one entry, with probability 1.0, naming a
            pattern that has a tactile rule. A stochastic mix would change the base
            table per sample under fixed tactile edges, and the other patterns never
            gave tactile tokens any attention.
    """
    if pattern_config is None or isinstance(pattern_config, str) or len(pattern_config) != 1:
        raise ValueError(
            "Tactile/deform need exactly one attention_mask_pattern entry with probability "
            f"1.0, got {pattern_config!r}"
        )
    name, prob = pattern_config[0]
    name = getattr(name, "value", name)  # AttentionMaskPattern member -> its saved name
    if float(prob) != 1.0:
        raise ValueError(f"Tactile/deform need a deterministic attention pattern, got {prob=}")
    if name not in TACTILE_ATTENTION_RULES:
        raise ValueError(
            f"No tactile attention rule for {name!r}; use one of {list(TACTILE_ATTENTION_RULES)}"
        )
    return name


def attention_rule_for(pattern_config: Any) -> AttentionRule:
    """Return the tactile rule for ``pattern_config`` (see ``tactile_pattern_name``).

    Resolve at pack time from the model's current ``attention_mask_pattern``: setup.py
    and the RL pipelines can reassign it after ``__init__``.
    """
    return TACTILE_ATTENTION_RULES[tactile_pattern_name(pattern_config)]
