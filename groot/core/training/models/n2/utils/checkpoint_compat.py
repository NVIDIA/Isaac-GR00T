# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Checkpoint compatibility helpers for cross-version weight loading."""

from collections.abc import Iterable
import re
from typing import Any

import torch
import torch.nn.functional as F
import transformers as _tx_module

# Detect target Qwen layout from the installed transformers version. 4.55+
# refactored Qwen2_5_VLForConditionalGeneration to nest `visual` and the LLM
# tower under `.model.*`. Computed once at import — the model class layout
# follows the runtime transformers version, so this matches what the loaded
# model expects.
_TX_VERSION = tuple(int(p) for p in _tx_module.__version__.split(".")[:2])
_TARGET_IS_457 = _TX_VERSION >= (4, 55)

# Hardcoded Qwen wrapper prefix in groot's GrootN2 (CosmosTextEncoder owns
# a single `Qwen2_5_VLForConditionalGeneration` named `text_encoder`, itself
# attached to GrootN2 as `self.text_encoder`). For any other Qwen wrapper,
# fall back to the whole-dict `remap_qwen2_5_vl_keys` below which detects
# wrappers dynamically from `target_keys`.
_GROOT_QWEN_PREFIX = "text_encoder.text_encoder."


def resize_temporal_pos_embedding(
    checkpoint_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    *,
    key: str = "action_encoder.temporal_pos_emb",
) -> torch.Tensor:
    """Resize a learned ``[1, time, hidden]`` table along time only.

    Rows are treated as samples over normalized horizon progress, so linear
    interpolation with aligned corners preserves the learned first and last
    positions exactly. All non-temporal dimensions must already match.
    """
    if checkpoint_tensor.shape == target_tensor.shape:
        return checkpoint_tensor

    source_shape = tuple(checkpoint_tensor.shape)
    target_shape = tuple(target_tensor.shape)
    valid_layout = (
        checkpoint_tensor.ndim == 3
        and target_tensor.ndim == 3
        and checkpoint_tensor.shape[0] == target_tensor.shape[0] == 1
        and checkpoint_tensor.shape[2] == target_tensor.shape[2]
    )
    if not valid_layout:
        raise ValueError(
            f"Cannot resize {key} from {source_shape} to {target_shape}: "
            "only the temporal axis may differ"
        )
    if checkpoint_tensor.shape[1] == 0 or target_tensor.shape[1] == 0:
        raise ValueError(
            f"Cannot resize {key} from {source_shape} to {target_shape}: "
            "temporal dimensions must be non-zero"
        )
    if not checkpoint_tensor.is_floating_point():
        raise TypeError(
            f"Cannot resize non-floating {key} tensor with dtype {checkpoint_tensor.dtype}"
        )

    resized = F.interpolate(
        checkpoint_tensor.float().transpose(1, 2),
        size=target_tensor.shape[1],
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    return resized.to(dtype=checkpoint_tensor.dtype).contiguous()


def remap_qwen2_5_vl_key(key: str) -> str:
    """Per-key Qwen2.5-VL 4.51.3 ↔ 4.57.6 remap (singular).

    Suitable for HF's `_fix_state_dict_key_on_load(key) -> (new_key, changed)`
    hook, which fires per key on every internal load path. Direction is
    fixed by the installed `transformers` version: a model running under
    transformers 4.55+ expects `<P>.model.visual.*` / `<P>.model.language_model.*`,
    so a 4.51.3-saved checkpoint's `<P>.visual.*` / `<P>.model.<X>` keys
    must be rewritten to that layout (and vice versa under 4.51.3).

    Hardcodes `<P> = "text_encoder.text_encoder."` for groot's
    `CosmosTextEncoder` wrapper; non-Qwen keys pass through untouched.
    For matched-version loads the function is a no-op.
    """
    if not key.startswith(_GROOT_QWEN_PREFIX):
        return key
    suffix = key[len(_GROOT_QWEN_PREFIX) :]

    if _TARGET_IS_457:
        # 4.51.3-form key being loaded into a 4.57.6 model: insert `model.`
        # before `visual` and `language_model.` after `model.` for LLM keys.
        if suffix.startswith("visual."):
            suffix = "model." + suffix
        elif (
            suffix.startswith("model.")
            and not suffix.startswith("model.visual.")
            and not suffix.startswith("model.language_model.")
        ):
            suffix = "model.language_model." + suffix[len("model.") :]
    # 4.57.6-form key being loaded into a 4.51.3 model: drop `model.`
    # before `visual.` and `language_model.` after `model.`.
    elif suffix.startswith("model.visual."):
        suffix = suffix[len("model.") :]
    elif suffix.startswith("model.language_model."):
        suffix = "model." + suffix[len("model.language_model.") :]

    return _GROOT_QWEN_PREFIX + suffix


def remap_qwen2_5_vl_keys(
    state_dict: dict[str, Any],
    target_keys: Iterable[str],
) -> dict[str, Any]:
    """Remap Qwen2.5-VL keys between transformers 4.51.3 and 4.57.6+ layouts.

    transformers 4.55+ refactored Qwen2_5_VLForConditionalGeneration in two
    places (lm_head position and inside-visual / inside-LLM substructures are
    otherwise unchanged):

      1. Visual encoder moved from `<P>.visual.*` to `<P>.model.visual.*`.
      2. LLM tower moved from `<P>.model.X` to `<P>.model.language_model.X`
         (where `X` is anything except `visual` or `language_model`).

    `<P>` is whatever wrapper prefix Qwen lives under — empty for a bare
    `Qwen2_5_VLForConditionalGeneration`, `text_encoder.text_encoder.` when
    wrapped inside groot's `CosmosTextEncoder`. The function detects every
    Qwen prefix in `target_keys` (anything immediately before `.model.visual.`
    or, in 4.51.3, `.visual.`) and rewrites only those subtrees of
    `state_dict`. Non-Qwen subtrees — VAE, DiT, action head — are passed
    through untouched, even when they have their own unrelated `.model.X`
    keys (e.g. `vae.model.encoder.conv1.weight`).

    Inspects `target_keys` (typically `model.state_dict().keys()`) to decide
    which layout the destination model uses, then rewrites `state_dict` to
    match. Bidirectional. If the target layout is ambiguous (both or neither
    set of markers present), the input is returned unchanged.

    For matched-version loads (4.51.3↔4.51.3 or 4.57.6↔4.57.6) the function
    is a no-op: the relevant key forms are already in the target form.
    """
    target_keys = list(target_keys)

    # Detect every Qwen wrapper prefix present in the target. Only those
    # subtrees are subject to the remap.
    qwen_prefixes = _find_qwen_prefixes(target_keys)
    if not qwen_prefixes:
        return state_dict

    target_set = set(target_keys)
    has_dot_model_visual = any(".model.visual." in k for k in target_set)
    has_dot_lang_model = any(".model.language_model." in k for k in target_set)
    has_bare_visual = any(
        re.search(r"(^|\.)visual\.", k) and ".model.visual." not in k for k in target_set
    )

    target_is_457 = has_dot_model_visual and has_dot_lang_model
    target_is_413 = has_bare_visual and not has_dot_lang_model

    if target_is_457 and not target_is_413:
        return _remap_with_prefixes(state_dict, qwen_prefixes, _qwen_tail_to_457)
    if target_is_413 and not target_is_457:
        return _remap_with_prefixes(state_dict, qwen_prefixes, _qwen_tail_to_413)
    return state_dict


def _find_qwen_prefixes(target_keys: list[str]) -> set[str]:
    """Return the set of wrapper prefixes that hold a Qwen2.5-VL submodule.

    Detection uses two unambiguous Qwen markers:
      - 4.57.6 layout: `<P>.model.visual.` (with `.model.language_model.` siblings)
      - 4.51.3 layout: `<P>.visual.` at the boundary AND a sibling `<P>.model.<X>`
        with X != visual/language_model (the LLM tower).

    The prefix is everything before the matched marker (including the trailing
    dot, so callers can do `k.startswith(prefix)` cheaply). Returns an empty
    set if no Qwen subtree is found, which short-circuits the remap.
    """
    prefixes_457 = set()
    prefixes_413_visual = set()
    prefixes_413_model = set()
    for k in target_keys:
        # 4.57.6 layout: "<P>.model.visual.<rest>"
        m = re.match(r"(.+?)\.model\.visual\.", k)
        if m:
            prefixes_457.add(m.group(1) + ".")
            continue
        # bare Qwen 4.57.6 (no wrapper): "model.visual.<rest>"
        if k.startswith("model.visual."):
            prefixes_457.add("")
            continue
        # 4.51.3 layout: "<P>.visual.<rest>" (visual at top)
        m = re.match(r"(.+?)\.visual\.", k)
        if m and ".model.visual." not in k:
            prefixes_413_visual.add(m.group(1) + ".")
            continue
        if k.startswith("visual.") and ".model.visual." not in k:
            prefixes_413_visual.add("")
            continue
    # For 4.51.3, also require a sibling LLM tower under the same prefix
    for k in target_keys:
        m = re.match(
            r"(.+?)\.model\.(?!visual(\.|$)|language_model(\.|$))[^.]+\.",
            k,
        )
        if m:
            prefixes_413_model.add(m.group(1) + ".")
            continue
        if re.match(
            r"^model\.(?!visual(\.|$)|language_model(\.|$))[^.]+\.",
            k,
        ):
            prefixes_413_model.add("")
    if prefixes_457:
        return prefixes_457
    return prefixes_413_visual & prefixes_413_model


def _remap_with_prefixes(
    state_dict: dict[str, Any],
    prefixes: set[str],
    tail_fn,
) -> dict[str, Any]:
    """Apply ``tail_fn`` to matching Qwen key suffixes.

    Keys outside the detected Qwen prefixes pass through unchanged.
    """
    # Sort prefixes longest-first so e.g. "text_encoder.text_encoder." beats
    # the bare-Qwen "" when both somehow apply.
    sorted_prefixes = sorted(prefixes, key=len, reverse=True)
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        new_k = k
        for p in sorted_prefixes:
            if (p == "" and "." in k) or (p and k.startswith(p)):
                tail = k[len(p) :]
                new_tail = tail_fn(tail)
                new_k = p + new_tail
                break
        out[new_k] = v
    return out


def _qwen_tail_to_457(tail: str) -> str:
    """Rewrite a Qwen-internal key from 4.51.3 layout to 4.57.6 layout."""
    # 1) visual.X → model.visual.X (only if not already nested)
    if "model.visual." not in tail and tail.startswith("visual."):
        tail = "model." + tail
    # 2) model.X → model.language_model.X  (X != visual, language_model)
    tail = re.sub(
        r"^model\.(?!visual(\.|$)|language_model(\.|$))",
        r"model.language_model.",
        tail,
        count=1,
    )
    return tail


def _qwen_tail_to_413(tail: str) -> str:
    """Rewrite a Qwen-internal key from 4.57.6 layout to 4.51.3 layout."""
    tail = re.sub(r"^model\.language_model\.", "model.", tail, count=1)
    tail = re.sub(r"^model\.visual\.", "visual.", tail, count=1)
    return tail


# Back-compat alias — older code imported the visual-only name.
remap_qwen2_5_vl_visual_keys = remap_qwen2_5_vl_keys


def remap_n2_vae_key(key: str) -> str:
    """Remap Wan VAE weights from the pre-refactor layout to the current layout.

    GrootN2 used to nest the frozen Wan VAE under ``video_tokenizer.vae``; it
    now lives at top-level ``vae``. Checkpoints saved before the move (including
    GrootN2RL actor checkpoints) still carry the old prefix.
    """
    old_prefix = "video_tokenizer.vae."
    if key.startswith(old_prefix):
        return "vae." + key[len(old_prefix) :]
    return key


def remap_n2_vae_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Bulk ``remap_n2_vae_key`` for explicit ``load_state_dict`` callers."""
    return {remap_n2_vae_key(k): v for k, v in state_dict.items()}
