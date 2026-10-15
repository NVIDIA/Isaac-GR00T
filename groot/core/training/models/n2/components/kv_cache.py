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

from __future__ import annotations

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


@dataclass
class FrozenPrefixKVCache(CondKVCache):
    """A ``CondKVCache`` whose capture holds a fixed set of sequence positions.

    ``frozen_index`` selects positions along the sequence axis (dim 2) of the
    captured K/V. On the first ``update`` for a layer (capture), those positions
    are replaced by ``frozen_key[layer_idx]`` / ``frozen_value[layer_idx]`` before
    the tensors are stored and returned, so every later query attends to the
    frozen values. Splice reuse (``layer_idx`` already cached) is unchanged.

    Joint denoising uses it to hold the conditioning-video tokens' K/V at the
    values captured on an earlier denoising step: those tokens' inputs do not
    change between steps and their predicted velocity is discarded, so the only
    effect is the drift of their K/V, which the frozen values approximate.
    """

    frozen_index: torch.Tensor | None = None
    frozen_key: list[torch.Tensor] = field(default_factory=list)
    frozen_value: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def from_captured(cls, cache: CondKVCache, index: torch.Tensor) -> FrozenPrefixKVCache:
        """Freeze the ``index`` positions of a fully captured cache (detached copies)."""
        if len(cache) == 0:
            raise ValueError("FrozenPrefixKVCache.from_captured needs a captured cache")
        if index.ndim != 1 or index.numel() == 0:
            raise ValueError("FrozenPrefixKVCache needs a non-empty 1-D position index")
        keys = [k.index_select(2, index).detach().clone() for k in cache.key_cache]
        values = [v.index_select(2, index).detach().clone() for v in cache.value_cache]
        return cls(frozen_index=index, frozen_key=keys, frozen_value=values)

    def fresh(self) -> FrozenPrefixKVCache:
        """An empty capture cache sharing this cache's frozen K/V."""
        return FrozenPrefixKVCache(
            frozen_index=self.frozen_index,
            frozen_key=self.frozen_key,
            frozen_value=self.frozen_value,
        )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        *,
        store: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == len(self.key_cache):
            # Capture: substitute before the base class stores/returns.
            if self.frozen_index is None or layer_idx >= len(self.frozen_key):
                raise ValueError(
                    f"FrozenPrefixKVCache has no frozen K/V for layer {layer_idx} "
                    f"(frozen layers: {len(self.frozen_key)})"
                )
            k = k.clone()
            v = v.clone()
            k[:, :, self.frozen_index, :] = self.frozen_key[layer_idx].to(k.dtype)
            v[:, :, self.frozen_index, :] = self.frozen_value[layer_idx].to(v.dtype)
        return super().update(k, v, layer_idx, store=store)
