# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""cuDNN prefix attention for the ``bidirectional_video_action_full`` mask.

The packed sequence is [video | state | action] with a per-token validity mask (padded
camera views, episode-end frames, padded action steps). Under that pattern:

* video queries attend to the valid video tokens only,
* state queries attend to the valid state tokens only (one token: its own value),
* action queries attend to valid video + state + action.

Like the flex mask this masks keys only: a padded video or action query row still attends the
valid keys (its loss is masked), and the state row is zero when the state token is invalid.
Attention does not care about the order of the keys, so reordering only K and V per sample
as [valid video | valid state | valid action | invalid] turns every key set above into a
prefix, which cuDNN's fused kernel takes as a per-sample sequence length
(``cudnn_prefix_sdpa``); queries and the output stay in the original order. This replaces
the FlexAttention Triton template, which reached 22 % of the bf16 GEMM ceiling at 37 frames
on GB300. Every shape below is static and every data
dependence is tensor data (permutation, lengths, validity), so the path traces under
torch.compile and replays inside CUDA graphs like the rest of the DiT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VideoActionAttentionPlan:
    """Per-batch key permutation and prefix lengths for the attention above.

    Built inside the compiled forward like the FlexAttention BlockMask it replaces; all
    tensors have static shapes, only their contents depend on the batch.
    """

    layout: tuple[int, int, int]  # video / state / action token counts of the packed sequence
    perm: torch.Tensor  # (B, L) long: position in the reordered key sequence -> original index
    inv_perm: torch.Tensor  # (B, L) long: original index -> reordered position
    seq_video: torch.Tensor  # (B,) int32 valid video tokens
    seq_action_kv: torch.Tensor  # (B,) int32 valid video + state + action tokens
    state_valid: torch.Tensor  # (B, n_state) bool

    def self_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return video_action_attention(q, k, v, self)


def build_video_action_attention_plan(
    modality_ids: torch.Tensor,
    token_valid_mask: torch.Tensor | None,
    *,
    layout: tuple[int, int, int],
    state_id: int,
    action_id: int,
) -> VideoActionAttentionPlan:
    """Plan for ``modality_ids`` (B, L) packed as [video | state | action] with ``layout`` counts."""
    B, L = modality_ids.shape
    n_video, n_state, n_action = layout
    if n_video + n_state + n_action != L:
        raise ValueError(f"layout {layout} does not add up to the sequence length {L}")
    if n_state > 1:
        # state queries attend the valid state tokens: with one token that is its own value
        raise ValueError(f"the video-action plan supports at most one state token, got {n_state}")
    if token_valid_mask is None:
        token_valid_mask = torch.ones(B, L, dtype=torch.bool, device=modality_ids.device)
    is_state = modality_ids == state_id
    is_action = modality_ids == action_id
    is_video = ~(is_state | is_action)
    # Stable sort by (0 video, 1 state, 2 action, 3 invalid) keeps the original order inside
    # each class: the reordered keys are [valid video | valid state | valid action | invalid].
    order_key = torch.where(is_state, 1, torch.where(is_action, 2, 0))
    order_key = torch.where(token_valid_mask, order_key, 3)
    perm = torch.argsort(order_key, dim=1, stable=True)
    return VideoActionAttentionPlan(
        layout=layout,
        perm=perm,
        inv_perm=torch.argsort(perm, dim=1),
        seq_video=(is_video & token_valid_mask).sum(1).to(torch.int32),
        seq_action_kv=token_valid_mask.sum(1).to(torch.int32),
        state_valid=token_valid_mask[:, n_video : n_video + n_state],
    )


def _gather_tokens(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rows of x (B, L, H, D) at index (B, n)."""
    return torch.gather(x, 1, index[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3]))


class _PermuteTokens(torch.autograd.Function):
    """x[perm] along the token axis for a permutation.

    Autograd's backward of a gather is a scatter-add (atomics, 16 ms/step at 37 frames); for a
    permutation it is exactly the gather with the inverse permutation.
    """

    @staticmethod
    def forward(ctx, x, perm, inv_perm):
        ctx.save_for_backward(inv_perm)
        return _gather_tokens(x, perm)

    @staticmethod
    def backward(ctx, grad):
        (inv_perm,) = ctx.saved_tensors
        return _gather_tokens(grad, inv_perm), None, None


def video_action_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: VideoActionAttentionPlan
) -> torch.Tensor:
    """Masked self-attention as two prefix attentions. q, k, v and the result are (B, L, H, D)."""
    # lazy, so CPU-only imports of the packer stay free of the cuDNN custom ops
    from groot.core.training.models.n2.components.cudnn_sdpa import cudnn_prefix_sdpa

    n_video, n_state, n_action = plan.layout
    kp = _PermuteTokens.apply(k, plan.perm, plan.inv_perm)
    vp = _PermuteTokens.apply(v, plan.perm, plan.inv_perm)
    q_video, _, q_action = torch.split(q, [n_video, n_state, n_action], dim=1)
    # Both calls take the full reordered keys: the prefix length masks the rest, and the kernel
    # writes zero dK/dV there, so no sliced views and no zero-padding backward. Every query row
    # of the block is computed (seq_len_q = the full block), so padded views, episode-end frames
    # and padded action steps may sit anywhere in the original order.
    full = torch.full_like(plan.seq_video, n_video)
    video_out = cudnn_prefix_sdpa(q_video, kp, vp, full, plan.seq_video)
    state_out = v[:, n_video : n_video + n_state] * plan.state_valid[:, :, None, None]
    action_out = cudnn_prefix_sdpa(
        q_action, kp, vp, torch.full_like(full, n_action), plan.seq_action_kv
    )
    return torch.cat([video_out, state_out, action_out], dim=1)
