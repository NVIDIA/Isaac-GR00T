# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Language-keyed conditioning of the cached-action engine's text inputs.

The fixed ``2v4a`` schedule runs two cached-action engine calls per prediction.
Under the ``raw`` action context mode each call receives the raw text embeddings
and recomputes the transformer's ``crossattn_proj`` inside the engine. The
``projected`` mode moves that projection out of the engine: this module runs the
loaded transformer's own BF16 ``crossattn_proj`` once per prompt and hands the
engine the projected context from a runtime-owned buffer with a stable address.
The ``kv`` mode goes one step further and also runs every block's cross-attention
``to_k`` / ``to_v`` projections and key RMS-normalization once per prompt, so the
engine reads the normalized keys and values directly.

Cache identity is the prompt tensor *object*. ``Gr00tN2Policy`` reuses one prompt
tensor while the language is unchanged and produces a new tensor on a language
change or reset, so object identity means exactly "the same prompt as the last
call". A strong reference to the keyed tensor is retained so its storage cannot be
recycled underneath the cache. Callers that mutate a prompt buffer in place (the
predict CUDA graph replay copies new embeddings into one static buffer) must call
``invalidate`` or ``refresh`` themselves: the runtime never infers a content
change and never falls back to the raw-context engine path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from groot.deployment.n2_trt.contract import ACTION_CONTEXT_MODES

PROJECTED_CONTEXT_INPUT = "context_proj"
CROSS_KEY_INPUT = "cross_k"
CROSS_VALUE_INPUT = "cross_v"
# Engine input names added by each non-raw mode (the raw ``context`` input is removed).
ACTION_CONTEXT_INPUT_NAMES: dict[str, frozenset[str]] = {
    "raw": frozenset({"context"}),
    "projected": frozenset({PROJECTED_CONTEXT_INPUT}),
    "kv": frozenset({CROSS_KEY_INPUT, CROSS_VALUE_INPUT}),
}


def project_context(projection: torch.nn.Module, prompt_embs: torch.Tensor) -> torch.Tensor:
    """Run the transformer's exact BF16 cross-attention projection on raw text embeddings.

    Shared by the exporter (validation samples) and the runtime conditioner so both
    sides derive the projected context with the same eager operators the production
    forward uses (BF16 linear plus GELU under BF16 autocast).
    """
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=prompt_embs.is_cuda),
    ):
        projected = projection(prompt_embs.to(torch.bfloat16))
    return projected.to(torch.bfloat16).contiguous()


def cross_attention_kv(
    projection: torch.nn.Module,
    cross_attentions: Sequence[torch.nn.Module],
    prompt_embs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every block's normalized cross-attention keys and values for one prompt.

    Mirrors ``Attention.forward`` for the cross-attention path: ``to_k`` / ``to_v`` on
    the projected context, the head split to ``(B, H, S, D)``, and the key RMS-norm.
    Cross-attention applies no rotary embedding. Output shape is ``(L, B, H, S, D)``
    for both tensors, contiguous BF16.
    """
    if not cross_attentions:
        raise ValueError("cross attention K/V conditioning requires at least one block")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=prompt_embs.is_cuda),
    ):
        context = projection(prompt_embs.to(torch.bfloat16))
        keys = []
        values = []
        for attention in cross_attentions:
            key = attention.to_k(context)
            value = attention.to_v(context)
            batch, tokens, _ = key.shape
            dim_head = key.shape[-1] // attention.heads
            key = key.view(batch, tokens, attention.heads, dim_head).transpose(1, 2)
            value = value.view(batch, tokens, attention.heads, dim_head).transpose(1, 2)
            keys.append(attention.norm_k(key))
            values.append(value)
        stacked_keys = torch.stack(keys, dim=0)
        stacked_values = torch.stack(values, dim=0)
    return (
        stacked_keys.to(torch.bfloat16).contiguous(),
        stacked_values.to(torch.bfloat16).contiguous(),
    )


def _first_linear(module: torch.nn.Module) -> torch.nn.Linear:
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            return child
    raise TypeError("cross-attention projection has no Linear layer")


def _require_bf16_on(module: torch.nn.Module, device: torch.device, label: str) -> None:
    for parameter in module.parameters():
        if parameter.device != device or parameter.dtype != torch.bfloat16:
            raise RuntimeError(
                f"{label} weights must be bfloat16 on {device}, "
                f"got {parameter.dtype} on {parameter.device}"
            )


class ActionContextConditioner:
    """Compute the action engine's language-side inputs once per prompt and reuse them."""

    def __init__(
        self,
        *,
        mode: str,
        projection: torch.nn.Module,
        device: torch.device | str,
        engine_inputs: dict[str, dict[str, Any]],
        cross_attentions: Sequence[torch.nn.Module] | None = None,
    ) -> None:
        if mode not in ACTION_CONTEXT_MODES:
            raise ValueError(
                f"unsupported action context mode {mode!r}; expected {ACTION_CONTEXT_MODES}"
            )
        if mode == "raw":
            raise ValueError(
                "the raw action context mode feeds the engine directly; no conditioner"
            )
        if mode == "kv" and not cross_attentions:
            raise ValueError("kv action context requires the blocks' cross-attention modules")
        self.mode = mode
        self.device = torch.device(device)
        self.projection = projection.eval()
        _require_bf16_on(self.projection, self.device, "projection")
        linear = _first_linear(projection)
        self.cross_attentions: tuple[torch.nn.Module, ...] = ()
        self._buffers: dict[str, torch.Tensor] = {}
        for name in sorted(self.input_names):
            spec = engine_inputs.get(name)
            if spec is None:
                raise RuntimeError(
                    f"action engine has no {name!r} input; bindings={sorted(engine_inputs)}"
                )
            if spec["dtype"] != torch.bfloat16:
                raise RuntimeError(f"{name} must be bfloat16, got {spec['dtype']}")
            self._buffers[name] = torch.empty(
                tuple(int(dim) for dim in spec["shape"]), dtype=torch.bfloat16, device=self.device
            )
        if mode == "projected":
            shape = tuple(self._buffers[PROJECTED_CONTEXT_INPUT].shape)
            if linear.out_features != shape[-1]:
                raise RuntimeError(
                    "cross-attention projection width does not match the action engine: "
                    f"{linear.out_features} != {shape[-1]}"
                )
            self.expected_prompt_shape = (*shape[:-1], linear.in_features)
        else:
            self.cross_attentions = tuple(cross_attentions)
            for index, attention in enumerate(self.cross_attentions):
                _require_bf16_on(attention.to_k, self.device, f"block {index} to_k")
                _require_bf16_on(attention.to_v, self.device, f"block {index} to_v")
                _require_bf16_on(attention.norm_k, self.device, f"block {index} norm_k")
            key_shape = tuple(self._buffers[CROSS_KEY_INPUT].shape)
            value_shape = tuple(self._buffers[CROSS_VALUE_INPUT].shape)
            if key_shape != value_shape or len(key_shape) != 5:
                raise RuntimeError(
                    f"cross K/V bindings must share one (L, B, H, S, D) shape: {key_shape} vs {value_shape}"
                )
            layers, batch, heads, tokens, dim_head = key_shape
            first = self.cross_attentions[0]
            expected = (
                len(self.cross_attentions),
                batch,
                first.heads,
                tokens,
                first.to_k.out_features // first.heads,
            )
            if key_shape != expected:
                raise RuntimeError(
                    f"cross K/V binding {key_shape} does not match the loaded blocks {expected}"
                )
            for attention in self.cross_attentions:
                if attention.to_k.in_features != linear.out_features:
                    raise RuntimeError("cross-attention key width does not match the projection")
            self.expected_prompt_shape = (batch, tokens, linear.in_features)
        self._source: torch.Tensor | None = None
        self._valid = False
        self.hit_count = 0
        self.miss_count = 0
        self.invalidate_count = 0

    @property
    def input_names(self) -> frozenset[str]:
        return ACTION_CONTEXT_INPUT_NAMES[self.mode]

    @property
    def is_valid(self) -> bool:
        return self._valid

    def buffers(self) -> dict[str, torch.Tensor]:
        """Return the runtime-owned output buffers (stable addresses for the engine and graphs)."""
        return dict(self._buffers)

    def condition(self, prompt_embs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the engine inputs for ``prompt_embs``: a hit for the keyed object, else recompute."""
        self._require_prompt(prompt_embs)
        if self._valid and prompt_embs is self._source:
            self.hit_count += 1
        else:
            self._compute(prompt_embs)
            self.miss_count += 1
        return dict(self._buffers)

    def refresh(self, prompt_embs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Recompute unconditionally from ``prompt_embs`` and key the cache to that object.

        Required after a prompt buffer was mutated in place, which identity cannot see.
        """
        self._require_prompt(prompt_embs)
        self._compute(prompt_embs)
        self.miss_count += 1
        return dict(self._buffers)

    def invalidate(self) -> None:
        """Drop the keyed prompt; the next ``condition`` call recomputes."""
        self._source = None
        self._valid = False
        self.invalidate_count += 1

    def state(self) -> dict[str, Any]:
        """JSON-safe counters and buffer identities for measurement receipts."""
        return {
            "mode": self.mode,
            "valid": self._valid,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "invalidate_count": self.invalidate_count,
            "prompt_shape": list(self.expected_prompt_shape),
            "layers": len(self.cross_attentions),
            "buffers": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "data_ptr": tensor.data_ptr(),
                }
                for name, tensor in self._buffers.items()
            },
        }

    def _require_prompt(self, prompt_embs: Any) -> None:
        if not isinstance(prompt_embs, torch.Tensor):
            raise TypeError(f"prompt embeddings must be a tensor, got {type(prompt_embs).__name__}")
        reasons = []
        if prompt_embs.device != self.device:
            reasons.append(f"device:{prompt_embs.device}!={self.device}")
        if tuple(prompt_embs.shape) != self.expected_prompt_shape:
            reasons.append(f"shape:{tuple(prompt_embs.shape)}!={self.expected_prompt_shape}")
        if prompt_embs.dtype != torch.bfloat16:
            reasons.append(f"dtype:{prompt_embs.dtype}!=torch.bfloat16")
        if reasons:
            raise RuntimeError(
                "action context conditioner rejected the prompt: " + ";".join(reasons)
            )

    def _outputs(self, prompt_embs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.mode == "projected":
            return {PROJECTED_CONTEXT_INPUT: project_context(self.projection, prompt_embs)}
        keys, values = cross_attention_kv(self.projection, self.cross_attentions, prompt_embs)
        return {CROSS_KEY_INPUT: keys, CROSS_VALUE_INPUT: values}

    def _compute(self, prompt_embs: torch.Tensor) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "action context conditioner miss during CUDA graph capture; "
                "condition the prompt before capture"
            )
        outputs = self._outputs(prompt_embs)
        for name, tensor in outputs.items():
            buffer = self._buffers[name]
            if tuple(tensor.shape) != tuple(buffer.shape) or tensor.dtype != buffer.dtype:
                raise RuntimeError(
                    f"conditioned {name!r} does not match the action engine binding: "
                    f"{tuple(tensor.shape)}/{tensor.dtype} != {tuple(buffer.shape)}/{buffer.dtype}"
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise RuntimeError(f"action context conditioner produced non-finite {name!r}")
        for name, tensor in outputs.items():
            self._buffers[name].copy_(tensor)
        self._source = prompt_embs
        self._valid = True
