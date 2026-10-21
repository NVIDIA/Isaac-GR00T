# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The text-padding mask of the DiT cross-attention (``GrootN2Config.dit_text_keys="real"``).

Every prompt is right-padded to ``max_language_length`` with ``<|endoftext|>``, and the causal text
encoder runs those pads after the prompt, so their embeddings carry the prompt. Under ``"all"``
(the default) every query cross-attends all slots, pads included. Under ``"real"`` it attends its
sample's real prompt tokens only: a prefix attention over the keys with the full query block,
:class:`TextPadMask`, in training and inference alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def text_seq_len(text_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """(B,) int32 real tokens of right-padded prompts ``text_ids`` (B, N)."""
    real = text_ids != pad_token_id
    seq_len = real.sum(1, dtype=torch.int32)
    capturing = text_ids.is_cuda and torch.cuda.is_current_stream_capturing()
    if not (torch.compiler.is_compiling() or capturing):
        # load-bearing: the mask keeps a prefix, so a pad before a real token would drop that
        # token and keep a pad; checked where a host sync is allowed (eager, not under capture)
        prefix = torch.arange(text_ids.shape[1], device=text_ids.device)[None, :] < seq_len[:, None]
        if not torch.equal(real, prefix):
            raise ValueError(
                "dit_text_keys='real' needs right-padded prompts; a pad precedes a token"
            )
    return seq_len


def dense_prefix_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
) -> torch.Tensor:
    """Per sample, the first ``seq_len_q[b]`` queries over the first ``seq_len_kv[b]`` keys.

    q (B, S_q, H, D), k/v (B, S_kv, H, D) -> (B, S_q, H, D), zero past the query prefix.
    ``F.scaled_dot_product_attention`` over the boolean mask, in the inputs' dtype.
    """
    q_ok = torch.arange(q.shape[1], device=q.device)[None, :] < seq_len_q[:, None].long()
    kv_ok = torch.arange(k.shape[1], device=q.device)[None, :] < seq_len_kv[:, None].long()
    allowed = q_ok[:, :, None] & kv_ok[:, None, :]  # (B, S_q, S_kv)
    empty = ~allowed.any(dim=2)
    allowed = allowed | empty[:, :, None]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed[:, None]
    ).transpose(1, 2)
    keep = q_ok & ~empty
    return torch.where(
        keep[:, :, None, None], out, torch.zeros((), dtype=out.dtype, device=out.device)
    )


@dataclass
class TextPadMask:
    """Cross-attention of every query to its sample's real prompt tokens, pads masked.

    The prompts are right-padded, so sample ``b``'s keys ``[0, seq_len[b])`` are its prompt and the
    rest pads. Pad keys get no attention weight and zero dK/dV.
    """

    seq_len: torch.Tensor  # (B,) int32 real prompt tokens per sample
    # "cudnn" (training under GrootN2Config.dit_masked_attention="cudnn") or "dense"
    kernel: str = "dense"

    def __post_init__(self) -> None:
        if self.kernel not in ("dense", "cudnn"):
            raise ValueError(f"TextPadMask kernel must be 'dense' or 'cudnn', got {self.kernel!r}")

    def cross_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Masked cross-attention: q (B, S_q, H, D), k/v (B, N, H, D) -> (B, S_q, H, D)."""
        full = torch.full_like(self.seq_len, q.shape[1])
        if self.kernel == "cudnn":
            from groot.core.training.models.n2.components.cudnn_sdpa import cudnn_prefix_sdpa

            # the fused kernel is bf16; fp32 callers round-trip
            q16, k16, v16 = (t.to(torch.bfloat16) for t in (q, k, v))
            return cudnn_prefix_sdpa(q16, k16, v16, full, self.seq_len).to(q.dtype)
        return dense_prefix_sdpa(q, k, v, full, self.seq_len)

    def repeat(self, n: int) -> TextPadMask:
        """The mask of ``n`` stacked copies of the batch, row ``i * B + b`` (the per-view fan-out)."""
        return TextPadMask(self.seq_len.repeat(n), self.kernel)
