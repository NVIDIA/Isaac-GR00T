# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""KV cache for GrootN2 inference.

Stores per-transformer-layer self-attention K/V for a token prefix.
Follows the transformers DynamicCache convention: a single update() method
handles both capture (first write) and splice (concat with existing),
controlled by a store flag for immutable reuse in diffusion loops.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class CondKVCache:
    """Per-layer self-attention K/V cache.

    Tensors are stored post qk-norm and post RoPE, shape (B, H, L, D_head),
    ready to concatenate with freshly computed K/V inside Attention.forward.
    """

    key_cache: list[torch.Tensor] = field(default_factory=list)
    value_cache: list[torch.Tensor] = field(default_factory=list)

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        *,
        store: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new K/V and return the (possibly concatenated) result.

        First call for a given layer_idx appends; subsequent calls concatenate
        the cached prefix with new K/V along the sequence dimension (dim=2).

        Args:
            k: (B, H, L_new, D_head) key tensor.
            v: (B, H, L_new, D_head) value tensor.
            layer_idx: Transformer layer index.
            store: If True (default), update the internal cache with the
                concatenated result. If False, return the concatenation but
                leave the cache unchanged — used for diffusion action denoising
                where action K/V changes every step but the static prefix is
                reused immutably.

        Returns:
            (k_out, v_out) — concatenated key/value tensors.
        """
        if layer_idx < len(self.key_cache):
            k_out = torch.cat([self.key_cache[layer_idx], k], dim=2)
            v_out = torch.cat([self.value_cache[layer_idx], v], dim=2)
            if store:
                self.key_cache[layer_idx] = k_out
                self.value_cache[layer_idx] = v_out
            return k_out, v_out
        else:
            if store:
                self.key_cache.append(k)
                self.value_cache.append(v)
            return k, v

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct access to cached (K, V) for a layer — used by _slice_cache."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
