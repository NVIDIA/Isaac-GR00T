# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Freezes installer: pokes GrootN2 internals by design (it memoizes the model's
# own private constructions), patch-target lambdas carry fixed signatures, and
# the picked docstring/zip style predates the lint gate.
# ruff: noqa: B905, D205, D209, SLF001
"""Steady-state freezes for GrootN2 inference (opt-in, GROOT_N2_PREDICT_FREEZES=1).

Memoizes the per-predict constructions that are identical on every call at
fixed deployment shapes: the joint BlockMask (vmapped mask evaluation + index
kernels rebuilt per joint step), shape-only RoPE/positional tensors, and
the static-prefix cache-slice index (a nonzero() host sync per joint step).

Ported from the freeze helpers of Jing Wang's whole-predict graph runner
(``n2_predict_graph.py``, commit 7b287970d9), decoupled from CUDA-graph
capture — the memoization carries its saving standalone.

Contract (same as the source): one policy instance per process; the model is
not reused for training or variable-config inference. Memo keys cover tensor
shapes and scalar arguments, so a genuine layout change rebuilds rather than
mis-serves.
"""

from __future__ import annotations

from types import MethodType

import torch


def _freeze_attention_mask_builder() -> None:
    """Memoize the packer's BlockMask builder keyed on argument signature."""
    import groot.core.training.models.n2.components.sequence_packing as packing

    if getattr(packing._build_attention_mask, "_predict_freeze", False):
        return
    real_build = packing._build_attention_mask
    cache: dict = {}

    def describe(x):
        return tuple(x.shape) if torch.is_tensor(x) else repr(x)

    def cached_build(*args, **kwargs):
        key = (
            tuple(describe(x) for x in args),
            tuple((name, describe(x)) for name, x in sorted(kwargs.items())),
        )
        if key not in cache:
            cache[key] = real_build(*args, **kwargs)
        return cache[key]

    cached_build._predict_freeze = True
    packing._build_attention_mask = cached_build


def _freeze_static_geometry(model) -> None:
    """Memoize shape-only RoPE and positional forwards on the model instance."""

    def install(module) -> None:
        if module is None or getattr(module, "_predict_freeze", False):
            return
        original = module.forward
        cache: dict = {}

        def forward(_module, value, *args, **kwargs):
            key = (
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
                tuple(args),
                tuple(sorted(kwargs.items())),
            )
            if key not in cache:
                result = original(value, *args, **kwargs)
                if isinstance(result, tuple):
                    result = tuple(tensor.detach().clone() for tensor in result)
                else:
                    result = result.detach().clone()
                cache[key] = result
            return cache[key]

        module.forward = MethodType(forward, module)
        module._predict_freeze = True

    install(getattr(model, "video_rope", None))
    install(getattr(model, "action_rope", None))
    install(getattr(model, "state_rope", None))
    install(getattr(getattr(model, "video_tokenizer", None), "learnable_pos_embed", None))


def _freeze_cache_slicing(model) -> None:
    """Replace the static-prefix cache slice's per-call nonzero()/assert host
    syncs with a memoized keep-index (identical ascending order)."""
    import groot.core.training.models.n2.model as n2_model

    if getattr(model._slice_cache_to_static_prefix, "_predict_freeze", False):
        return
    idx_cache: dict = {}

    def graph_safe_slice(cache, modality_ids, token_valid_mask=None, token_merge=None):
        if token_merge is not None:
            raise RuntimeError(
                "predict freezes do not support token merge (the memoized cache slice keeps "
                "one keep-index for every layer); run with GROOT_N2_PREDICT_FREEZES=0"
            )
        key = tuple(modality_ids[0].shape)
        if key not in idx_cache:
            pattern = model._resolve_attention_mask_pattern()
            table = pattern.to_table()
            # Same invariant as the stock slice: video rows (0, 1) must not attend
            # to state (col 2) or action (col 3), or the cached K/V would be unsafe.
            if table[0][2] or table[0][3] or table[1][2] or table[1][3]:
                raise ValueError(
                    f"KV cache requires a pattern where video is self-contained; "
                    f"{pattern.value!r} has video attending to state/action."
                )
            action_row = table[3]
            keep_modalities = []
            if action_row[0]:
                keep_modalities.append(n2_model.MODALITY_VIDEO_COND)
            if action_row[1]:
                keep_modalities.append(n2_model.MODALITY_VIDEO)
            if action_row[2]:
                keep_modalities.append(n2_model.MODALITY_STATE)
            ids = modality_ids[0]
            keep = torch.zeros_like(ids, dtype=torch.bool)
            for m in keep_modalities:
                keep |= ids == m
            if token_valid_mask is not None:
                assert torch.all(token_valid_mask == token_valid_mask[:1])
                keep &= token_valid_mask[0, : keep.shape[0]]
            idx_cache[key] = keep.nonzero(as_tuple=True)[0]
        keep_idx = idx_cache[key]
        sliced = n2_model.CondKVCache()
        for k_layer, v_layer in zip(cache.key_cache, cache.value_cache):
            sliced.key_cache.append(k_layer.index_select(2, keep_idx).clone())
            sliced.value_cache.append(v_layer.index_select(2, keep_idx).clone())
        return sliced

    graph_safe_slice._predict_freeze = True
    model._slice_cache_to_static_prefix = graph_safe_slice


def install_predict_freezes(model) -> None:
    """Install all steady-state freezes on a policy-owned model instance."""
    _freeze_attention_mask_builder()
    _freeze_static_geometry(model)
    _freeze_cache_slicing(model)
