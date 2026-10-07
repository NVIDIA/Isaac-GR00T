# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Sequence packing: combine per-modality tokens into a single transformer input.

Each packer class takes tokenized modalities and returns a PackedSequence ready
for CosmosTransformer3DModel.forward(). All packers share the same signature:
video args are required, action/state args are optional. When absent, the packer
produces a video-only sequence.

Sequence order: [video | state | action]. Action is last because it attends to
everything; video and state only self-attend.

Packers:
  ConcatPacker — [video | state | action] with BlockMask attention pattern.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

MODALITY_VIDEO = 0
MODALITY_VIDEO_COND = 1
MODALITY_STATE = 2
MODALITY_ACTION = 3

FLEX_BLOCK_SIZE = 128


class AttentionMaskPattern(str, Enum):
    """Named attention mask patterns for the packed sequence.

    Each pattern defines a 4x4 boolean table: rows = query modality,
    cols = key modality, in order [video_cond, video_denoise, state, action].
    True means the query can attend to the key.

    To add a new pattern, add an enum member and its table in to_table().
    """

    DISJOINT_VIDEO_ACTION = "disjoint_video_action"
    """Action tokens attend to video_cond + state + action (NOT video_denoise).
    Video and action predictions are decoupled."""

    JOINT_VIDEO_ACTION = "joint_video_action"
    """Action tokens attend to all video + state + action.
    Action head can exploit future info from denoising tokens."""

    BIDIRECTIONAL_VIDEO_ACTION_COND = "bidirectional_video_action_cond"
    """Original attention pattern used before the mask refactor.
    video_cond attends to ALL video (cond + denoise); action is disjoint
    from video_denoise. Use for checkpoints trained before the v1 mask change."""

    BIDIRECTIONAL_VIDEO_ACTION_FULL = "bidirectional_video_action_full"
    """Video tokens have bidirectional self-attention;
    Action tokens attend to all video + state + action.
    Action tokens can exploit future info from denoising tokens."""

    FULL = "full"
    """All tokens attend to all tokens. No restricted attention."""

    def to_table(self) -> list[list[bool]]:
        """Return 4x4 attention table [q_modality][kv_modality].

        Order: [video_cond, video_denoise, state, action].
        """
        if self == AttentionMaskPattern.DISJOINT_VIDEO_ACTION:
            return [
                # kv:  v_cond  v_denoise  state  action
                [True, False, False, False],  # q: video_cond
                [True, True, False, False],  # q: video_denoise
                [False, False, True, False],  # q: state
                [True, False, True, True],  # q: action (no video_denoise)
            ]
        elif self == AttentionMaskPattern.JOINT_VIDEO_ACTION:
            return [
                # kv:  v_cond  v_denoise  state  action
                [True, False, False, False],  # q: video_cond
                [True, True, False, False],  # q: video_denoise
                [False, False, True, False],  # q: state
                [True, True, True, True],  # q: action (sees video_denoise)
            ]
        elif self == AttentionMaskPattern.BIDIRECTIONAL_VIDEO_ACTION_COND:
            return [
                # kv:  v_cond  v_denoise  state  action
                [True, True, False, False],  # q: video_cond (sees all video)
                [True, True, False, False],  # q: video_denoise
                [False, False, True, False],  # q: state
                [True, False, True, True],  # q: action (no video_denoise)
            ]
        elif self == AttentionMaskPattern.BIDIRECTIONAL_VIDEO_ACTION_FULL:
            return [
                # kv:  v_cond  v_denoise  state  action
                [True, True, False, False],  # q: video_cond (sees all video)
                [True, True, False, False],  # q: video_denoise
                [False, False, True, False],  # q: state
                [True, True, True, True],  # q: action (sees all video)
            ]
        elif self == AttentionMaskPattern.FULL:
            return [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ]
        raise ValueError(f"Unknown pattern: {self}")


# Extension attention receives (batch, query modality, key modality, base table).
AttentionRule = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class AdditionalTokenGroup:
    """Project-owned non-video tokens appended by the canonical packer.

    ``kv_cacheable`` marks static conditioning that cached action-only steps keep
    attending to. By default a group's K/V is dropped from the conditioning cache
    (right for per-step queries such as completion tokens). Setting it asserts, via
    the caller's attention_rule, that action attends to the group and that the group
    never attends to action or to a non-cacheable group.

    Cached inference also requires batch-uniform validity at retained positions.
    The caller's rule must preserve the ordinary attention table, let action read
    exactly the cacheable extra groups, and keep all static tokens independent of
    action/non-cacheable groups. Rules must be pure; sample dropout outside them.
    Whole-predict CUDA graph replay and GrootN2TTT do not support these groups.
    """

    tokens: Tensor
    timesteps: Tensor
    rope_freqs: tuple[Tensor, Tensor]
    modality_id: int
    token_valid_mask: Tensor | None = None
    kv_cacheable: bool = False


@dataclass
class PackedSequence:
    """Result of packing modalities into a single sequence for the transformer."""

    tokens: Tensor  # (B, L_total, D)
    timesteps: Tensor  # (B, L_total)   raw sigma per token
    rope_freqs: tuple[Tensor, Tensor]  # (cos, sin) each (L_total, D_head)
    modality_ids: Tensor  # (B, L_total)   int — 0=video, 1=video_cond, 2=state, 3=action
    attention_mask: BlockMask | None  # flex attention mask (None = full attention)
    extra_pos_emb: Tensor | None  # (B, L_total, D) learnable pos embed, or None
    pad: int = 0  # padding added for BlockMask alignment
    # Packer-derived routing metadata. modality_ids remains the semantic source
    # of truth; these indices are compile-friendly derivatives of the layout.
    video_token_indices: Tensor | None = None
    state_token_indices: Tensor | None = None
    action_token_indices: Tensor | None = None
    non_video_token_indices: Tensor | None = None
    has_video_cond_tokens: bool = False
    # Validity the BlockMask was built from, covering appended groups (the caller's
    # own mask object when there are none), and the ids of kv_cacheable groups.
    token_valid_mask: Tensor | None = None
    kv_cacheable_modalities: tuple[int, ...] = ()
    # Reuse the extension's semantics when building a reduced token-merge mask.
    attention_rule: AttentionRule | None = None

    def unpack(self, packed_output: Tensor, modality: int | tuple[int, ...]) -> Tensor:
        """Extract contiguous modality tokens without building a boolean index mask."""
        return unpack_contiguous(packed_output, self, modality)


def unpack(
    packed_output: Tensor,
    modality_ids: Tensor,
    modality: int | tuple[int, ...],
) -> Tensor:
    """Extract tokens for one or more modalities from packed transformer output.

    Assumes all batch elements have the same packing layout.
    """
    ids = modality_ids[0]  # (L,)
    if isinstance(modality, int):
        mask = ids == modality
    else:
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for m in modality:
            mask |= ids == m
    return packed_output[:, mask]  # (B, L_modality, D)


def unpack_contiguous(
    packed_output: Tensor,
    packed: PackedSequence,
    modality: int | tuple[int, ...],
) -> Tensor:
    """Extract modality tokens using packer-provided routing indices.

    Falls back to mask-based unpacking for legacy/manual PackedSequence objects
    or modality combinations that do not have a dedicated routing index.
    """
    if isinstance(modality, int):
        indices = None
        if modality == MODALITY_STATE:
            indices = packed.state_token_indices
        elif modality == MODALITY_ACTION:
            indices = packed.action_token_indices
        elif modality == MODALITY_VIDEO and not packed.has_video_cond_tokens:
            indices = packed.video_token_indices
        if indices is not None:
            return packed_output[:, indices].contiguous()
    elif set(modality) == {MODALITY_VIDEO, MODALITY_VIDEO_COND}:
        if packed.video_token_indices is not None:
            return packed_output[:, packed.video_token_indices].contiguous()

    return unpack(packed_output, packed.modality_ids, modality)


def _build_attention_mask(
    modality_ids: Tensor,
    device: torch.device,
    pattern_config: list[tuple[str, float]] | None = None,
    token_valid_mask: Tensor | None = None,
    block_size: int = FLEX_BLOCK_SIZE,
    attention_rule: AttentionRule | None = None,
) -> tuple[BlockMask, int]:
    """Build a BlockMask from an AttentionMaskPattern config.

    Args:
        modality_ids: (B, L) int tensor of modality IDs.
        device: torch device.
        block_size: FlexAttention alignment size.
        attention_rule: Optional project modality rule; padding remains packer-owned.
        pattern_config: list of (pattern_name, probability) pairs. Per batch element,
            one pattern is sampled according to the probabilities. Defaults to
            [("bidirectional_video_action_cond", 1.0)] which matches the original training
            behavior (video_cond attends to all video, action disjoint from video_denoise).
        token_valid_mask: (B, L) bool tensor. True = real token, False = padded.
            When provided, padded kv tokens are masked out so no query attends to them.
            When None, all tokens are considered valid (backward compatible).

    Returns (block_mask, pad) where pad is the number of tokens added
    for FlexAttention block-size alignment.
    """
    if pattern_config is None:
        pattern_config = [("bidirectional_video_action_cond", 1.0)]
    if block_size <= 0:
        raise ValueError(f"FlexAttention block size must be positive, got {block_size}")

    ids = modality_ids[0]  # (L,)
    L = ids.shape[0]
    B = modality_ids.shape[0]

    pad = (block_size - L % block_size) % block_size
    L_padded = L + pad

    if pad > 0:
        ids = torch.cat([ids, torch.full((pad,), -1, device=device)])

    # Handle token_valid_mask: default to all-valid when not provided
    if token_valid_mask is None:
        token_valid_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Pad token_valid_mask to L_padded (False for alignment padding)
    if pad > 0:
        token_valid = torch.cat(
            [token_valid_mask, torch.zeros(B, pad, dtype=torch.bool, device=device)], dim=1
        )
    else:
        token_valid = token_valid_mask

    # Build per-sample (B, 4, 4) attention table
    patterns = [(AttentionMaskPattern(name), prob) for name, prob in pattern_config]
    if len(patterns) == 1:
        # Single pattern — no sampling needed, same table for all batch elements
        table = torch.tensor(patterns[0][0].to_table(), dtype=torch.bool, device=device)
        # (4, 4) → (B, 4, 4)
        table = table.unsqueeze(0).expand(B, -1, -1)
    else:
        # Stochastic: sample per batch element
        probs = torch.tensor([p for _, p in patterns], device=device)
        indices = torch.multinomial(probs.unsqueeze(0).expand(B, -1), 1).squeeze(1)  # (B,)
        tables = torch.stack(
            [torch.tensor(p.to_table(), dtype=torch.bool, device=device) for p, _ in patterns]
        )  # (num_patterns, 4, 4)
        table = tables[indices]  # (B, 4, 4)

    # Modality IDs use: 0=video_denoise, 1=video_cond, 2=state, 3=action
    # Table columns use: 0=video_cond, 1=video_denoise, 2=state, 3=action
    # Remap modality IDs to table indices
    MOD_TO_TABLE = torch.tensor([1, 0, 2, 3], dtype=torch.long, device=device)

    def mask_fn(b, h, q_idx, kv_idx):
        del h
        q_mod = ids[q_idx]
        kv_mod = ids[kv_idx]
        q_valid = q_mod >= 0
        kv_valid = kv_mod >= 0
        if attention_rule is not None:
            return (
                q_valid
                & kv_valid
                & attention_rule(b, q_mod, kv_mod, table)
                & token_valid[b, kv_idx]
            )
        q_table_idx = MOD_TO_TABLE[q_mod.clamp(min=0)]
        kv_table_idx = MOD_TO_TABLE[kv_mod.clamp(min=0)]
        return q_valid & kv_valid & table[b, q_table_idx, kv_table_idx] & token_valid[b, kv_idx]

    block_mask = create_block_mask(
        mask_fn,
        B=B,
        H=None,
        Q_LEN=L_padded,
        KV_LEN=L_padded,
        BLOCK_SIZE=block_size,
        device=str(device),
    )
    return block_mask, pad


class ConcatPacker:
    """Concatenation packer: [video | state | action], masked attention.

    Attention pattern is controlled by attention_mask_pattern config.
    Default (disjoint_video_action):
    - Video cond tokens self-attend to video cond only (NOT video_denoise)
    - Video denoise tokens attend to all video (cond + denoise)
    - State tokens self-attend only
    - Action tokens attend to video_cond + state + action (NOT video_denoise)

    When action/state are None, produces a video-only sequence with no mask
    (full attention, no flex_attention overhead).
    """

    def __init__(self, *, flex_attention_block_size: int = FLEX_BLOCK_SIZE):
        if flex_attention_block_size <= 0:
            raise ValueError(
                f"FlexAttention block size must be positive, got {flex_attention_block_size}"
            )
        self.flex_attention_block_size = flex_attention_block_size

    def pack(
        self,
        # Video (always present)
        video_tokens: Tensor,  # (B, L_video, D)
        video_timesteps: Tensor,  # (B, L_video)
        video_rope: tuple[Tensor, Tensor],  # (cos, sin) each (L_video, D_head)
        video_extra_pos_emb: Tensor | None = None,  # (B, L_video, D) or None
        video_cond_token_mask: Tensor | None = None,  # (B, L_video) bool — True = conditioning
        # Action (optional)
        action_tokens: Tensor | None = None,  # (B, L_action, D)
        action_timesteps: Tensor | None = None,  # (B, L_action)
        action_rope: tuple[Tensor, Tensor] | None = None,
        # State (optional)
        state_tokens: Tensor | None = None,  # (B, L_state, D)
        state_timesteps: Tensor | None = None,  # (B, L_state)
        state_rope: tuple[Tensor, Tensor] | None = None,
        # Attention mask pattern
        attention_mask_pattern: list[tuple[str, float]] | None = None,
        # Token validity mask for padded tokens (e.g. padded camera views, padded action steps)
        token_valid_mask: Tensor | None = None,  # ordinary [video | state | action] validity
        additional_token_groups: Sequence[AdditionalTokenGroup] = (),
        attention_rule: AttentionRule | None = None,
    ) -> PackedSequence:
        B = video_tokens.shape[0]
        device = video_tokens.device
        L_video = video_tokens.shape[1]
        L_state = 0 if state_tokens is None else state_tokens.shape[1]
        L_action = 0 if action_tokens is None else action_tokens.shape[1]

        # Build video modality ids: split cond vs denoise when mask is provided
        video_ids = torch.full((B, L_video), MODALITY_VIDEO, dtype=torch.long, device=device)
        if video_cond_token_mask is not None:
            video_ids[video_cond_token_mask] = MODALITY_VIDEO_COND

        all_tokens = [video_tokens]
        all_timesteps = [video_timesteps]
        all_cos = [video_rope[0]]
        all_sin = [video_rope[1]]
        all_ids = [video_ids]

        has_multimodal = state_tokens is not None or action_tokens is not None

        if state_tokens is not None:
            assert state_timesteps is not None
            assert state_rope is not None
            all_tokens.append(state_tokens)
            all_timesteps.append(state_timesteps)
            all_cos.append(state_rope[0])
            all_sin.append(state_rope[1])
            all_ids.append(
                torch.full(
                    (B, state_tokens.shape[1]), MODALITY_STATE, dtype=torch.long, device=device
                )
            )

        if action_tokens is not None:
            assert action_timesteps is not None
            assert action_rope is not None
            all_tokens.append(action_tokens)
            all_timesteps.append(action_timesteps)
            all_cos.append(action_rope[0])
            all_sin.append(action_rope[1])
            all_ids.append(
                torch.full(
                    (B, action_tokens.shape[1]), MODALITY_ACTION, dtype=torch.long, device=device
                )
            )

        kv_cacheable_modalities: tuple[int, ...] = ()
        if additional_token_groups:
            if attention_rule is None:
                raise ValueError("Additional modalities require an explicit attention_rule")
            # This host check may graph-break under compilation, but must run on
            # every batch: the attention mask uses the first row's modality ids.
            if not torch.equal(video_ids, video_ids[0:1].expand_as(video_ids)):
                raise ValueError("Additional tokens require a batch-uniform packed layout")
            # Generator expressions, not list comprehensions: Python 3.12 inlines the latter,
            # and Dynamo then fails to resume this frame after the mask builder's graph break.
            group_ids = tuple(g.modality_id for g in additional_token_groups)
            if len(set(group_ids)) != len(group_ids):
                raise ValueError(f"Additional groups need distinct modality ids, got {group_ids}")
            original_length = L_video + L_state + L_action
            if token_valid_mask is None:
                token_valid_mask = torch.ones(B, original_length, dtype=torch.bool, device=device)
            elif token_valid_mask.shape != (B, original_length):
                raise ValueError("Token-valid mask must match the ordinary packed sequence")
            all_valid = [token_valid_mask.to(device=device, dtype=torch.bool)]
            for group in additional_token_groups:
                if group.modality_id <= MODALITY_ACTION:
                    raise ValueError("Additional groups must use a distinct non-video modality")
                if group.tokens.shape[0] != B or group.tokens.shape[2] != video_tokens.shape[2]:
                    raise ValueError("Additional tokens must match the batch and hidden dimensions")
                if group.timesteps.shape != group.tokens.shape[:2]:
                    raise ValueError("Additional token timesteps have the wrong shape")
                if any(
                    rope.shape != (group.tokens.shape[1], video_rope[0].shape[1])
                    for rope in group.rope_freqs
                ):
                    raise ValueError("Additional token RoPE has the wrong shape")
                valid = group.token_valid_mask
                if valid is None:
                    valid = torch.ones(group.tokens.shape[:2], dtype=torch.bool, device=device)
                elif valid.shape != group.tokens.shape[:2]:
                    raise ValueError("Additional token validity has the wrong shape")
                all_tokens.append(group.tokens)
                all_timesteps.append(group.timesteps)
                all_cos.append(group.rope_freqs[0])
                all_sin.append(group.rope_freqs[1])
                all_ids.append(
                    torch.full(
                        group.tokens.shape[:2], group.modality_id, dtype=torch.long, device=device
                    )
                )
                all_valid.append(valid.to(device=device, dtype=torch.bool))
            token_valid_mask = torch.cat(all_valid, dim=1)
            has_multimodal = True
            kv_cacheable_modalities = tuple(
                g.modality_id for g in additional_token_groups if g.kv_cacheable
            )

        tokens = torch.cat(all_tokens, dim=1)
        timesteps = torch.cat(all_timesteps, dim=1)
        rope_freqs = (torch.cat(all_cos, dim=0), torch.cat(all_sin, dim=0))
        modality_ids = torch.cat(all_ids, dim=1)

        extra_pos_emb = None
        if video_extra_pos_emb is not None:
            L_total = tokens.shape[1]
            D = video_extra_pos_emb.shape[2]
            if L_total == L_video:
                extra_pos_emb = video_extra_pos_emb
            else:
                extra_pos_emb = torch.cat(
                    [
                        video_extra_pos_emb,
                        torch.zeros(
                            B, L_total - L_video, D, device=device, dtype=video_extra_pos_emb.dtype
                        ),
                    ],
                    dim=1,
                )

        attention_mask = None
        pad = 0
        if has_multimodal:
            attention_mask, pad = _build_attention_mask(
                modality_ids,
                device,
                attention_mask_pattern,
                token_valid_mask,
                block_size=self.flex_attention_block_size,
                attention_rule=attention_rule,
            )

        return PackedSequence(
            tokens=tokens,
            timesteps=timesteps,
            rope_freqs=rope_freqs,
            modality_ids=modality_ids,
            attention_mask=attention_mask,
            extra_pos_emb=extra_pos_emb,
            pad=pad,
            video_token_indices=torch.arange(L_video, device=device),
            state_token_indices=torch.arange(L_video, L_video + L_state, device=device),
            action_token_indices=torch.arange(
                L_video + L_state,
                L_video + L_state + L_action,
                device=device,
            ),
            non_video_token_indices=(
                torch.arange(L_video, tokens.shape[1], device=device)
                if tokens.shape[1] > L_video
                else None
            ),
            has_video_cond_tokens=video_cond_token_mask is not None,
            token_valid_mask=token_valid_mask,
            kv_cacheable_modalities=kv_cacheable_modalities,
            attention_rule=attention_rule,
        )
