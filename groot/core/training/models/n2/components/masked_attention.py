# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The kernel choice for the DiT's masked (video-action) self-attention in training.

The ``bidirectional_video_action_full`` mask reduces to *segment attention*: per sample, video
queries attend the valid video keys and action queries attend the valid video + state + action
keys (``video_action_attention.py``), each a prefix of the sample's reordered keys
(``GrootN2Config.dit_masked_attention``):

* ``flex`` (default): no plan; the packer builds the FlexAttention ``BlockMask``.
* ``cudnn``: a segment plan whose prefixes run on cudnn-frontend graphs (``cudnn_sdpa.py``,
  ``cudnn_prefix_sdpa``).

``cudnn_varlen`` is accepted as an alias of ``cudnn`` (the name earlier checkpoints saved). The
segment plan serves only the single ``bidirectional_video_action_full`` pattern with at most one
state token, no appended token groups and no token merging; every other batch keeps the
``BlockMask``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

KERNELS = ("flex", "cudnn")
_ALIASES = {"cudnn_varlen": "cudnn"}


def normalize_kernel(name: str) -> str:
    """The canonical kernel name for a config value or an alias; raises on anything else."""
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in KERNELS:
        raise ValueError(
            f"masked attention kernel must be one of {KERNELS} (or an alias "
            f"{sorted(_ALIASES)}), got {name!r}"
        )
    return key


@runtime_checkable
class MaskedAttentionPlan(Protocol):
    """What the attention layer asks of a plan: run the masked self-attention."""

    def self_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Masked self-attention: q, k, v (B, L, H, D) -> (B, L, H, D)."""
