#!/usr/bin/env python3
# ruff: noqa: SLF001, T201
"""Export the admitted N2 joint/action transformer graphs without quantization."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
import transformers

from groot.core.training.models.n2.components.flash_attention import (
    flash_attn_func_v23,
    scaled_dot_product_attention_bf16,
)
from groot.core.training.models.n2.components.kv_cache import CondKVCache, FrozenPrefixKVCache
from groot.core.training.models.n2.components.text_pad_mask import TextPadMask
from groot.core.training.models.n2.utils.cosmos_model_deps import ensure_cosmos_model_deps
from groot.deployment.n2_trt.conditioner import (
    CROSS_KEY_INPUT,
    CROSS_VALUE_INPUT,
    PROJECTED_CONTEXT_INPUT,
    cross_attention_kv,
    project_context,
)
from groot.deployment.n2_trt.contract import (
    ACTION_CONTEXT_MODES,
    TEXT_SEQ_LEN_INPUT,
    N2TensorRTContract,
    TransformerPrecision,
)
from groot.deployment.n2_trt.tools.conditioning import export_conditioning

DEFAULT_EMBODIMENT = "xdof_relative_eef_relative_joint"
DEFAULT_CODEC_ROUNDTRIP = "mp4v"
# Literal twin of contract.ACTION_CONTEXT_MODES for the argparse ``choices``: the Thor runbook
# CLI-contract test compiles this parser from literal module assignments only.
_ACTION_CONTEXT_CHOICES = ("raw", "projected", "kv")


def _prepare_checkpoint(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    return Path(ensure_cosmos_model_deps(checkpoint_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--embodiment",
        default=DEFAULT_EMBODIMENT,
        help="Lowercase YAM XDOF embodiment tag present in the checkpoint processor.",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC_ROUNDTRIP,
        choices=("h264", "mp4v", "none"),
        help="Observation codec roundtrip used only for the deterministic export probe.",
    )
    parser.add_argument(
        "--action-context-mode",
        default="raw",
        choices=_ACTION_CONTEXT_CHOICES,
        help=(
            "Language input of the cached-action engine: 'raw' projects the text "
            "embeddings inside the engine on every call; 'projected' takes the "
            "transformer's crossattn_proj output, computed once per prompt by the "
            "runtime conditioner. The joint engine always takes raw context."
        ),
    )
    parser.add_argument(
        "--cond-kv-freeze",
        action="store_true",
        help=(
            "Also export the conditioning-cached joint graph: the second joint step runs "
            "only the denoised-video, state, and action tokens as queries against the first "
            "step's conditioning-video K/V (serves GROOT_N2_COND_KV_FREEZE=1)."
        ),
    )
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument("--num-action-steps", type=int, default=4)
    args = parser.parse_args()
    # Other export options retain their production defaults.
    args.mode = "production"
    args.embodiment_tag = args.embodiment
    args.policy_embodiment_tag = args.embodiment.upper()
    args.video_mode = "gradient"
    args.language = "pick up the object"
    args.opset = 20
    args.shift = 5.0
    args.seed = 42
    args.save_samples = True
    args.joint_mask_export = "segmented-sdpa"
    return args


def _resolve_routing_indices(
    modality_ids: torch.Tensor,
    video_token_indices: torch.Tensor | None = None,
    non_video_token_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve static routing indices before ONNX tracing starts."""
    if modality_ids.ndim != 2:
        raise ValueError(
            f"modality_ids must have shape (batch, sequence); got {tuple(modality_ids.shape)}"
        )
    if not torch.equal(modality_ids, modality_ids[:1].expand_as(modality_ids)):
        raise ValueError("ONNX export requires one shared modality layout per batch")

    ids = modality_ids[0]
    if video_token_indices is None:
        video_token_indices = torch.where((ids == 0) | (ids == 1))[0]
    if non_video_token_indices is None:
        non_video_token_indices = torch.where((ids != 0) & (ids != 1))[0]
    return (
        video_token_indices.to(device=modality_ids.device, dtype=torch.long),
        non_video_token_indices.to(device=modality_ids.device, dtype=torch.long),
    )


class _ExportTextPadMask(TextPadMask):
    """Export form of :class:`TextPadMask` (``text_keys="real"``) with the same semantics.

    Every query attends its sample's first ``seq_len`` keys; a sample without real tokens
    yields zeros, as ``dense_prefix_sdpa`` does. The key mask enters SDPA as an additive
    bias in the query dtype, broadcast over heads and queries, so the strongly typed graph
    carries no boolean-to-float32 mask conversion.
    """

    def cross_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = self.seq_len.long()
        has_keys = seq_len > 0
        kv_ok = torch.arange(k.shape[1], device=q.device)[None, :] < seq_len[:, None]
        kv_ok = kv_ok | ~has_keys[:, None]
        bias = torch.where(
            kv_ok,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), float("-inf"), dtype=q.dtype, device=q.device),
        )
        out = scaled_dot_product_attention_bf16(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=bias[:, None, None, :],
        ).transpose(1, 2)
        return torch.where(
            has_keys[:, None, None, None], out, torch.zeros((), dtype=out.dtype, device=out.device)
        )

    def repeat(self, n: int) -> _ExportTextPadMask:
        return _ExportTextPadMask(self.seq_len.repeat(n))


def _text_inputs(payload: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    """The ``text_seq_len`` graph input of a captured call (none for ``text_keys="all"``)."""
    seq_len = payload.get(TEXT_SEQ_LEN_INPUT)
    return () if seq_len is None else (seq_len,)


class _CrossKVSource(torch.nn.Module):
    """Per-forward holder of the precomputed ``(L, B, H, S, D)`` cross-attention K/V inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None


class _PrecomputedCrossAttention(torch.nn.Module):
    """Export-only cross-attention that reads normalized keys/values from graph inputs.

    Mirrors ``Attention.forward`` for the action-only cross path: query projection,
    head split, query RMS-norm, attention, output projection. Cross-attention carries
    no rotary embedding and no self-attention cache; the keys and values arrive
    pre-projected and pre-normalized, so ``to_k`` / ``to_v`` / ``norm_k`` leave the graph.
    """

    def __init__(self, attention: torch.nn.Module, layer_idx: int, source: _CrossKVSource):
        super().__init__()
        self.attention = attention
        self.layer_idx = layer_idx
        self.source = source

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: Any = None,
        n_cameras: int = 1,
        video_idx: torch.Tensor | None = None,
        non_video_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del encoder_hidden_states, n_cameras, non_video_idx
        if image_rotary_emb is not None:
            raise RuntimeError("cross-attention K/V inputs do not carry rotary embeddings")
        if video_idx is not None and video_idx.numel() > 0:
            raise RuntimeError("precomputed cross-attention K/V is an action-only export path")
        if self.source.key is None or self.source.value is None:
            raise RuntimeError("cross-attention K/V inputs were not bound before the forward")
        attention = self.attention
        query = attention.to_q(hidden_states)
        batch, tokens, _ = query.shape
        dim_head = query.shape[-1] // attention.heads
        query = query.view(batch, tokens, attention.heads, dim_head).transpose(1, 2)
        query = attention.norm_q(query)
        key = self.source.key[self.layer_idx]
        value = self.source.value[self.layer_idx]
        out = flash_attn_func_v23(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), flatten=False
        )
        out = out.reshape(batch, tokens, -1)
        out = attention.to_out[0](out)
        return attention.to_out[1](out)


class _N2TransformerExportAdapterBase(torch.nn.Module):
    """Call the compile-friendly transformer entry point with static routing."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
        token_merge: Any = None,
        projected_context: bool = False,
        cross_kv: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        # Keep the original implementation while routing through this adapter.
        self._routing_forward = transformer._forward_with_routing_indices.__func__
        self.token_merge = token_merge
        # ``projected_context``: the graph receives the already-projected text
        # context, so the transformer is traced with its crossattn_proj replaced by
        # identity. ``cross_kv`` additionally replaces every block's cross-attention
        # with the export-only module that reads precomputed K/V inputs. Modules are
        # swapped only for the duration of the call.
        self.projected_context = projected_context
        self.cross_kv = cross_kv
        if (projected_context or cross_kv) and not getattr(
            transformer.config, "use_crossattn_projection", False
        ):
            raise RuntimeError(
                "projected action context requires a transformer with crossattn_proj"
            )
        if cross_kv and getattr(transformer, "per_view_text", False):
            raise RuntimeError("cross-attention K/V inputs require a single text copy")
        self.kv_source = _CrossKVSource() if cross_kv else None
        self.register_buffer("video_token_indices", video_token_indices)
        self.register_buffer("non_video_token_indices", non_video_token_indices)

    def _forward_transformer(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        *,
        extra_pos_emb: torch.Tensor | None,
        attention_mask: Any,
        pad: int,
        kv_cache: Any,
        store_kv: bool,
        text_seq_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_merge_kwargs = (
            {"sequence_transform": self.token_merge} if self.token_merge is not None else {}
        )
        if text_seq_len is not None:
            if self.cross_kv:
                raise RuntimeError("cross-attention K/V inputs do not support text_keys='real'")
            token_merge_kwargs["text_pad_mask"] = _ExportTextPadMask(text_seq_len)
        projection = None
        swapped_attentions: list[tuple[torch.nn.Module, torch.nn.Module]] = []
        if self.projected_context or self.cross_kv:
            projection = self.transformer.crossattn_proj
            self.transformer.crossattn_proj = torch.nn.Identity()
        if self.cross_kv:
            for index, block in enumerate(self.transformer.transformer_blocks):
                swapped_attentions.append((block, block.attn2))
                block.attn2 = _PrecomputedCrossAttention(block.attn2, index, self.kv_source)
        try:
            return self._routing_forward(
                self.transformer,
                latents,
                timesteps,
                (rope_cos, rope_sin),
                context,
                modality_ids,
                extra_pos_emb=extra_pos_emb,
                attention_mask=attention_mask,
                pad=pad,
                kv_cache=kv_cache,
                store_kv=store_kv,
                video_token_indices=self.video_token_indices,
                non_video_token_indices=self.non_video_token_indices,
                **token_merge_kwargs,
            )
        finally:
            for block, attention in swapped_attentions:
                block.attn2 = attention
            if projection is not None:
                self.transformer.crossattn_proj = projection


class N2TransformerJointCaptureExportAdapter(_N2TransformerExportAdapterBase):
    """Production joint transformer call that returns output tokens and captured KV cache."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        attention_mask: Any,
        pad: int,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
        token_merge: Any = None,
    ):
        super().__init__(
            transformer,
            video_token_indices,
            non_video_token_indices,
            token_merge=token_merge,
        )
        self.attention_mask = attention_mask
        self.pad = pad

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        text_seq_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        from groot.core.training.models.n2.components.kv_cache import CondKVCache

        latents = latents.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)
        context = context.to(torch.bfloat16)
        cache = CondKVCache()
        output = self._forward_transformer(
            latents,
            timesteps,
            rope_cos,
            rope_sin,
            context,
            modality_ids,
            extra_pos_emb=None,
            attention_mask=self.attention_mask,
            pad=self.pad,
            kv_cache=cache,
            store_kv=True,
            text_seq_len=text_seq_len,
        )
        if self.token_merge is None:
            return (
                output,
                torch.stack(cache.key_cache, dim=0),
                torch.stack(cache.value_cache, dim=0),
            )
        from groot.deployment.n2_trt.runtime import _stack_token_merge_cache

        return (
            output,
            *_stack_token_merge_cache(
                cache,
                (self.token_merge.start_layer, self.token_merge.end_layer),
            ),
        )


class _ConditioningPrefixCache(CondKVCache):
    """Export-only splice cache of the conditioning-cached joint graph.

    Every layer's cached prefix is the conditioning-video K/V; the fresh K/V the query
    tokens produce are retained so the graph can return them for the cached action steps.
    """

    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        super().__init__(key_cache=list(key.unbind(0)), value_cache=list(value.unbind(0)))
        self.fresh_key: list[torch.Tensor] = []
        self.fresh_value: list[torch.Tensor] = []

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        *,
        store: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if store or layer_idx != len(self.fresh_key) or layer_idx >= len(self.key_cache):
            raise RuntimeError("conditioning-prefix export expects one splice per layer, in order")
        self.fresh_key.append(k)
        self.fresh_value.append(v)
        return super().update(k, v, layer_idx, store=False)


class N2TransformerJointCondCachedExportAdapter(_N2TransformerExportAdapterBase):
    """Later joint step with the first step's conditioning-video K/V as a cached prefix.

    Only the denoised-video, state, and action tokens are queries: the model discards the
    conditioning tokens' velocity, so the graph never computes them. Returns the query
    outputs and their fresh K/V; the runtime rebuilds the full-layout cache from
    ``[cached prefix | fresh]`` for the cached action steps.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        attention_mask: Any,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
    ):
        super().__init__(transformer, video_token_indices, non_video_token_indices)
        # Routes the blocks through the masked attention path; the segmented exporter
        # dispatches on sequence lengths and never reads the mask's contents.
        self.attention_mask = attention_mask

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        cond_key: torch.Tensor,
        cond_value: torch.Tensor,
        text_seq_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = latents.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)
        context = context.to(torch.bfloat16)
        cache = _ConditioningPrefixCache(cond_key.to(torch.bfloat16), cond_value.to(torch.bfloat16))
        output = self._forward_transformer(
            latents,
            timesteps,
            rope_cos,
            rope_sin,
            context,
            modality_ids,
            extra_pos_emb=None,
            attention_mask=self.attention_mask,
            pad=0,
            kv_cache=cache,
            store_kv=False,
            text_seq_len=text_seq_len,
        )
        if len(cache.fresh_key) != len(cache.key_cache):
            raise RuntimeError("conditioning-prefix export did not splice every layer")
        return (
            output,
            torch.stack(cache.fresh_key, dim=0),
            torch.stack(cache.fresh_value, dim=0),
        )


class N2TransformerActionCachedExportAdapter(_N2TransformerExportAdapterBase):
    """Production action-only transformer call with sliced KV cache input."""

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        kv_cache_key: torch.Tensor,
        kv_cache_value: torch.Tensor,
        text_seq_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from groot.core.training.models.n2.components.kv_cache import CondKVCache

        latents = latents.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)
        context = context.to(torch.bfloat16)
        kv_cache_key = kv_cache_key.to(torch.bfloat16)
        kv_cache_value = kv_cache_value.to(torch.bfloat16)
        cache = CondKVCache(
            key_cache=list(kv_cache_key.unbind(0)),
            value_cache=list(kv_cache_value.unbind(0)),
        )
        return self._forward_transformer(
            latents,
            timesteps,
            rope_cos,
            rope_sin,
            context,
            modality_ids,
            extra_pos_emb=None,
            attention_mask=None,
            pad=0,
            kv_cache=cache,
            store_kv=False,
            text_seq_len=text_seq_len,
        )


def _context_placeholder(transformer: torch.nn.Module) -> torch.Tensor:
    """A never-consumed stand-in for the text context of a cross-K/V action graph."""
    device = next(transformer.parameters()).device
    return torch.zeros(
        (1, 1, transformer.config.text_embed_dim), dtype=torch.bfloat16, device=device
    )


class N2TransformerActionCachedCrossKVExportAdapter(_N2TransformerExportAdapterBase):
    """Action-only call whose cross-attention keys/values arrive as inputs (no text context)."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
    ):
        super().__init__(transformer, video_token_indices, non_video_token_indices, cross_kv=True)
        self.register_buffer("context_placeholder", _context_placeholder(transformer))

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        modality_ids: torch.Tensor,
        kv_cache_key: torch.Tensor,
        kv_cache_value: torch.Tensor,
    ) -> torch.Tensor:
        from groot.core.training.models.n2.components.kv_cache import CondKVCache

        self.kv_source.key = cross_k.to(torch.bfloat16)
        self.kv_source.value = cross_v.to(torch.bfloat16)
        cache = CondKVCache(
            key_cache=list(kv_cache_key.to(torch.bfloat16).unbind(0)),
            value_cache=list(kv_cache_value.to(torch.bfloat16).unbind(0)),
        )
        return self._forward_transformer(
            latents.to(torch.bfloat16),
            timesteps.to(torch.bfloat16),
            rope_cos,
            rope_sin,
            self.context_placeholder,
            modality_ids,
            extra_pos_emb=None,
            attention_mask=None,
            pad=0,
            kv_cache=cache,
            store_kv=False,
        )


class N2TransformerTokenMergeActionCachedCrossKVExportAdapter(_N2TransformerExportAdapterBase):
    """Token-merge action-only call with fine/coarse KV inputs and cross-attention K/V inputs."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
        token_merge_layers: tuple[int, int],
    ):
        super().__init__(transformer, video_token_indices, non_video_token_indices, cross_kv=True)
        self.token_merge_layers = token_merge_layers
        self.register_buffer("context_placeholder", _context_placeholder(transformer))

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        modality_ids: torch.Tensor,
        kv_cache_key_fine: torch.Tensor,
        kv_cache_value_fine: torch.Tensor,
        kv_cache_key_coarse: torch.Tensor,
        kv_cache_value_coarse: torch.Tensor,
    ) -> torch.Tensor:
        from groot.deployment.n2_trt.runtime import _restore_token_merge_cache

        self.kv_source.key = cross_k.to(torch.bfloat16)
        self.kv_source.value = cross_v.to(torch.bfloat16)
        cache = _restore_token_merge_cache(
            kv_cache_key_fine.to(torch.bfloat16),
            kv_cache_value_fine.to(torch.bfloat16),
            kv_cache_key_coarse.to(torch.bfloat16),
            kv_cache_value_coarse.to(torch.bfloat16),
            self.token_merge_layers,
        )
        return self._forward_transformer(
            latents.to(torch.bfloat16),
            timesteps.to(torch.bfloat16),
            rope_cos,
            rope_sin,
            self.context_placeholder,
            modality_ids,
            extra_pos_emb=None,
            attention_mask=None,
            pad=0,
            kv_cache=cache,
            store_kv=False,
        )


class N2TransformerTokenMergeActionCachedExportAdapter(_N2TransformerExportAdapterBase):
    """Action-only transformer call with heterogeneous fine/coarse KV inputs."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
        token_merge_layers: tuple[int, int],
        projected_context: bool = False,
    ):
        super().__init__(
            transformer,
            video_token_indices,
            non_video_token_indices,
            projected_context=projected_context,
        )
        self.token_merge_layers = token_merge_layers

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        kv_cache_key_fine: torch.Tensor,
        kv_cache_value_fine: torch.Tensor,
        kv_cache_key_coarse: torch.Tensor,
        kv_cache_value_coarse: torch.Tensor,
        text_seq_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from groot.deployment.n2_trt.runtime import _restore_token_merge_cache

        cache = _restore_token_merge_cache(
            kv_cache_key_fine.to(torch.bfloat16),
            kv_cache_value_fine.to(torch.bfloat16),
            kv_cache_key_coarse.to(torch.bfloat16),
            kv_cache_value_coarse.to(torch.bfloat16),
            self.token_merge_layers,
        )
        return self._forward_transformer(
            latents.to(torch.bfloat16),
            timesteps.to(torch.bfloat16),
            rope_cos,
            rope_sin,
            context.to(torch.bfloat16),
            modality_ids,
            extra_pos_emb=None,
            attention_mask=None,
            pad=0,
            kv_cache=cache,
            store_kv=False,
            text_seq_len=text_seq_len,
        )


def _dense_token_mask_from_block_mask(block_mask: Any) -> torch.Tensor:
    device = block_mask.kv_num_blocks.device
    q_len, kv_len = block_mask.seq_lengths
    batch = int(block_mask.kv_num_blocks.shape[0])
    b = torch.arange(batch, device=device)[:, None, None]
    h = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
    q_idx = torch.arange(q_len, device=device)[None, :, None]
    kv_idx = torch.arange(kv_len, device=device)[None, None, :]
    return block_mask.mask_mod(b, h, q_idx, kv_idx)[:, None, :, :].to(torch.bool)


@dataclass(frozen=True)
class _BidirectionalVideoActionFullSegments:
    """Fixed ``[video | state | action]`` layout for segmented attention export.

    ``video_tokens`` counts conditioning and denoising video tokens across all
    camera slots, including padded slots. State and action tokens follow that
    complete video segment.

    ``valid_video_indices`` contains the ordered, zero-based token positions
    within the video segment whose keys and values attention may use. These
    positions are inferred from the captured attention mask and checked against
    the contract's camera layout before export. They cover tokens from real
    camera inputs; tokens from padded camera slots are excluded. ``None`` means
    every video token has a valid key and value. The layout is shared across
    batch items and video/action queries.

    Video and action attention gather video keys and values at these positions.
    All query positions remain present, including queries for padded video
    tokens, so sequence length and state/action offsets stay unchanged. For
    example, three input cameras in four slots with eight tokens per slot have
    ``video_tokens=32`` and ``valid_video_indices=tuple(range(24))``.

    Fine and merged/coarse attention use separate instances, with indices in
    their respective token resolutions. ``alignment_pad`` counts extra mask
    positions beyond ``total_tokens`` used for block alignment.
    """

    video_tokens: int
    state_tokens: int
    action_tokens: int
    alignment_pad: int
    valid_video_indices: tuple[int, ...] | None = None

    @property
    def total_tokens(self) -> int:
        return self.video_tokens + self.state_tokens + self.action_tokens

    def to_manifest(self) -> dict[str, int | str]:
        return {
            "pattern": "bidirectional_video_action_full",
            "video_tokens": self.video_tokens,
            "state_tokens": self.state_tokens,
            "action_tokens": self.action_tokens,
            "total_tokens": self.total_tokens,
            "alignment_pad": self.alignment_pad,
            "valid_video_tokens": (
                self.video_tokens
                if self.valid_video_indices is None
                else len(self.valid_video_indices)
            ),
        }


def _model_latent_shape(
    model: Any,
    *,
    conditioning_frames: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Per-camera VAE latent (C, T, H, W) for this checkpoint, measured not assumed.

    Base N2 (/8 into 16ch) and the Wan2.2 f16t4c48 VAE (turbo) disagree here, and the token
    count downstream follows it. Derived from the VAE directly rather than from the
    captured sequence, so the contract check stays a real check.
    """
    probe = torch.zeros(
        (1, 3, model.max_n_cameras * conditioning_frames, *image_size),
        device=model.device,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        latents = model.encode_video(probe)
    channels, all_frames, height, width = latents.shape[1:]
    return (channels, all_frames // model.max_n_cameras, height, width)


def _require_segment_counts(
    segments: _BidirectionalVideoActionFullSegments,
    *,
    total_camera_slot_count: int,
    action_horizon: int,
    latent_shape: tuple[int, int, int, int],
    total_latent_frames: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> None:
    from groot.deployment.n2_trt.runtime import n2_segment_token_counts

    actual = (segments.video_tokens, segments.state_tokens, segments.action_tokens)
    expected = n2_segment_token_counts(
        total_camera_slot_count,
        action_horizon,
        latent_shape,
        total_latent_frames,
        patch_size,
    )
    if actual != expected:
        raise ValueError(
            f"segmented-sdpa N2 runtime contract requires exact token counts {expected}; got {actual}"
        )


def _require_segment_camera_validity(
    segments: _BidirectionalVideoActionFullSegments,
    contract: N2TensorRTContract,
    *,
    layout: str,
) -> None:
    """Require captured video-key validity to match the runtime camera layout."""
    total_camera_slot_count = contract.total_camera_slot_count
    input_camera_count = contract.input_camera_count
    padded_camera_slot_count = contract.padded_camera_slot_count
    # Real camera inputs occupy the first slots; trailing slots contain masked padding.
    if segments.video_tokens % total_camera_slot_count:
        raise ValueError(
            f"segmented-sdpa {layout} video tokens cannot be evenly split across "
            f"{total_camera_slot_count} total camera slots"
        )
    video_tokens_per_camera_slot = segments.video_tokens // total_camera_slot_count
    expected_valid_video_token_count = video_tokens_per_camera_slot * input_camera_count
    expected_valid_video_indices = tuple(range(expected_valid_video_token_count))
    captured_valid_video_indices = segments.valid_video_indices
    if captured_valid_video_indices is None:
        captured_valid_video_indices = tuple(range(segments.video_tokens))
    if captured_valid_video_indices != expected_valid_video_indices:
        raise ValueError(
            f"segmented-sdpa {layout} camera padding differs from the runtime contract: "
            f"expected the first {expected_valid_video_token_count} of "
            f"{segments.video_tokens} video keys for {input_camera_count} input cameras and "
            f"{padded_camera_slot_count} padded camera slots "
            f"({total_camera_slot_count} total camera slots); "
            f"captured {len(captured_valid_video_indices)} valid video keys"
        )


def _validate_bidirectional_video_action_full_segments(
    modality_ids: torch.Tensor,
    dense_mask: torch.Tensor,
) -> _BidirectionalVideoActionFullSegments:
    """Prove the mask is three contiguous segments with fixed video-key validity."""
    if modality_ids.ndim != 2:
        raise ValueError(
            f"segmented-sdpa requires modality_ids with shape (batch, sequence); got {tuple(modality_ids.shape)}"
        )
    if dense_mask.ndim != 4 or dense_mask.shape[1] != 1 or dense_mask.dtype != torch.bool:
        raise ValueError(
            "segmented-sdpa requires a boolean mask with shape (batch, 1, query, key); "
            f"got shape={tuple(dense_mask.shape)} dtype={dense_mask.dtype}"
        )

    batch, sequence = modality_ids.shape
    if (
        dense_mask.shape[0] != batch
        or dense_mask.shape[2] != dense_mask.shape[3]
        or dense_mask.shape[2] < sequence
    ):
        raise ValueError(
            "segmented-sdpa mask shape does not cover the captured modality layout: "
            f"mask={tuple(dense_mask.shape)} modality_ids={tuple(modality_ids.shape)}"
        )
    if not torch.equal(modality_ids, modality_ids[0:1].expand_as(modality_ids)):
        raise ValueError("segmented-sdpa requires the same modality layout for every batch item")

    ids = modality_ids[0]
    is_video_cond = ids == 1
    is_video_denoise = ids == 0
    is_video = is_video_cond | is_video_denoise
    is_state = ids == 2
    is_action = ids == 3
    if not torch.all(is_video | is_state | is_action).item():
        raise ValueError("segmented-sdpa encountered an unsupported modality ID")
    if not torch.any(is_video_cond).item() or not torch.any(is_video_denoise).item():
        raise ValueError("segmented-sdpa requires both conditioning and denoising video tokens")

    video_tokens = int(is_video.sum().item())
    state_tokens = int(is_state.sum().item())
    action_tokens = int(is_action.sum().item())
    if min(video_tokens, state_tokens, action_tokens) <= 0:
        raise ValueError("segmented-sdpa requires non-empty video, state, and action segments")

    state_end = video_tokens + state_tokens
    if not (
        torch.all(is_video[:video_tokens]).item()
        and torch.all(is_state[video_tokens:state_end]).item()
        and torch.all(is_action[state_end:]).item()
    ):
        raise ValueError(
            "segmented-sdpa requires contiguous [video | state | action] modality segments"
        )

    expected_mask = torch.zeros_like(dense_mask)
    expected_mask[:, :, :video_tokens, :video_tokens] = True
    expected_mask[:, :, video_tokens:state_end, video_tokens:state_end] = True
    expected_mask[:, :, state_end:sequence, :sequence] = True
    key_validity = dense_mask[:, :, state_end : state_end + 1, :sequence]
    if not key_validity[..., video_tokens:].all() or not key_validity[..., :video_tokens].any():
        raise ValueError(
            "segmented-sdpa requires valid state/action keys and at least one real video key"
        )
    if not torch.equal(key_validity, key_validity[:1].expand_as(key_validity)):
        raise ValueError("segmented-sdpa requires batch-uniform camera padding")
    expected_mask[:, :, :, :sequence] &= key_validity
    if not torch.equal(dense_mask, expected_mask):
        raise ValueError(
            "segmented-sdpa only supports bidirectional_video_action_full with fixed video-key padding"
        )

    return _BidirectionalVideoActionFullSegments(
        video_tokens=video_tokens,
        state_tokens=state_tokens,
        action_tokens=action_tokens,
        alignment_pad=int(dense_mask.shape[2] - sequence),
        valid_video_indices=(
            tuple(torch.where(key_validity[0, 0, 0, :video_tokens])[0].tolist())
            if not key_validity[..., :video_tokens].all()
            else None
        ),
    )


def _segmented_bidirectional_video_action_full_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    segments: _BidirectionalVideoActionFullSegments,
    flatten_heads: bool,
) -> torch.Tensor:
    """Exact unmasked-SDPA decomposition of ``bidirectional_video_action_full``."""
    batch, sequence, heads, head_dim = q.shape
    if sequence != segments.total_tokens or k.shape[1] != sequence or v.shape[1] != sequence:
        raise ValueError(
            "segmented-sdpa input sequence does not match the validated export layout: "
            f"q={sequence}, k={k.shape[1]}, v={v.shape[1]}, expected={segments.total_tokens}"
        )

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    video_end = segments.video_tokens
    state_end = video_end + segments.state_tokens

    video_k, video_v = k[:, :, :video_end], v[:, :, :video_end]
    action_k, action_v = k, v
    if segments.valid_video_indices is not None:
        video_indices = torch.tensor(
            segments.valid_video_indices, dtype=torch.long, device=k.device
        )
        # Materialize the entire fixed layout once. Concatenating a traced
        # arange with a constant tensor can mix CPU/CUDA during ONNX folding.
        key_indices = torch.tensor(
            (*segments.valid_video_indices, *range(video_end, segments.total_tokens)),
            dtype=torch.long,
            device=k.device,
        )
        video_k = video_k.index_select(2, video_indices)
        video_v = video_v.index_select(2, video_indices)
        action_k = k.index_select(2, key_indices)
        action_v = v.index_select(2, key_indices)
    video_out = scaled_dot_product_attention_bf16(q[:, :, :video_end], video_k, video_v)
    state_out = scaled_dot_product_attention_bf16(
        q[:, :, video_end:state_end],
        k[:, :, video_end:state_end],
        v[:, :, video_end:state_end],
    )
    action_out = scaled_dot_product_attention_bf16(q[:, :, state_end:], action_k, action_v)
    out = torch.cat((video_out, state_out, action_out), dim=2)
    out = out.permute(0, 2, 1, 3).contiguous()
    if flatten_heads:
        return out.reshape(batch, sequence, heads * head_dim)
    return out


def _install_segmented_sdpa_flex_patch(
    modality_ids: torch.Tensor,
    dense_mask: torch.Tensor,
    *,
    contract: N2TensorRTContract,
    total_camera_slot_count: int,
    action_horizon: int,
    latent_shape: tuple[int, int, int, int],
    total_latent_frames: int,
    token_merge: Any = None,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> tuple[
    Any,
    _BidirectionalVideoActionFullSegments,
    _BidirectionalVideoActionFullSegments | None,
]:
    """Install the exact YAM segmented exporter after validating its camera contract."""
    from groot.core.training.models.n2.components import transformer_cosmos

    if total_camera_slot_count != contract.total_camera_slot_count:
        raise ValueError(
            "segmented-sdpa model camera slots differ from the runtime contract: "
            f"{total_camera_slot_count} != {contract.total_camera_slot_count}"
        )
    segments = _validate_bidirectional_video_action_full_segments(modality_ids, dense_mask)
    _require_segment_counts(
        segments,
        total_camera_slot_count=total_camera_slot_count,
        action_horizon=action_horizon,
        latent_shape=latent_shape,
        total_latent_frames=total_latent_frames,
        patch_size=patch_size,
    )
    _require_segment_camera_validity(segments, contract, layout="fine")
    coarse_segments = None
    if token_merge is not None:
        if token_merge.reduced_attention_mask is None:
            raise ValueError("segmented-sdpa token merge requires a coarse attention mask")
        coarse_modality_ids = modality_ids[:, token_merge.reduced_token_indices]
        coarse_dense_mask = _dense_token_mask_from_block_mask(token_merge.reduced_attention_mask)
        coarse_segments = _validate_bidirectional_video_action_full_segments(
            coarse_modality_ids, coarse_dense_mask
        )
        expected_coarse = (
            segments.video_tokens // (token_merge.kernel_height * token_merge.kernel_width),
            segments.state_tokens,
            segments.action_tokens,
        )
        actual_coarse = (
            coarse_segments.video_tokens,
            coarse_segments.state_tokens,
            coarse_segments.action_tokens,
        )
        if actual_coarse != expected_coarse:
            raise ValueError(
                "segmented-sdpa token-merge layout requires exact coarse token counts "
                f"{expected_coarse}; got {actual_coarse}"
            )
        _require_segment_camera_validity(coarse_segments, contract, layout="coarse")
    original = transformer_cosmos.flex_attention_op

    def segmented_sdpa_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Any = None,
        pad: int = 0,
        flatten_heads: bool = True,
        compile_attention: bool = False,
    ) -> torch.Tensor:
        del compile_attention  # Export always uses the explicit segmented SDPA path.
        if attn_mask is None:
            raise ValueError("segmented-sdpa is only valid for a masked joint transformer call")
        sequence = q.shape[1]
        if sequence == segments.total_tokens:
            active_segments = segments
        elif coarse_segments is not None and sequence == coarse_segments.total_tokens:
            active_segments = coarse_segments
        else:
            supported = [segments.total_tokens]
            if coarse_segments is not None:
                supported.append(coarse_segments.total_tokens)
            raise ValueError(
                f"segmented-sdpa sequence {sequence} is not one of the exported layouts {supported}"
            )
        if pad != active_segments.alignment_pad:
            raise ValueError(
                "segmented-sdpa alignment pad differs from the validated export layout: "
                f"{pad} != {active_segments.alignment_pad}"
            )
        return _segmented_bidirectional_video_action_full_sdpa(
            q,
            k,
            v,
            segments=active_segments,
            flatten_heads=flatten_heads,
        )

    transformer_cosmos.flex_attention_op = segmented_sdpa_attention_op
    return original, segments, coarse_segments


def _restore_flex_patch(original: Any) -> None:
    if original is None:
        return
    from groot.core.training.models.n2.components import transformer_cosmos

    transformer_cosmos.flex_attention_op = original


@dataclass(frozen=True)
class _ConditioningPrefixLayout:
    """Joint layout of a later denoising step served with cached conditioning-video K/V.

    Keys keep the full ``segments`` layout with the conditioning-video tokens spliced in
    first (``CondKVCache.update`` places the cached prefix before the fresh K/V); queries
    are the remaining ``[denoised video | state | action]`` tokens in their original order.

    ``valid_video_key_indices`` selects real-camera video keys in this reordered
    key layout, rather than the original positions in ``segments``. Padded query
    rows remain present to preserve the engine shape, but padded keys are excluded.
    """

    segments: _BidirectionalVideoActionFullSegments
    prefix_tokens: int
    valid_video_key_indices: tuple[int, ...] | None = None

    @property
    def query_tokens(self) -> int:
        return self.segments.total_tokens - self.prefix_tokens

    @property
    def key_tokens(self) -> int:
        return self.segments.total_tokens

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pattern": "bidirectional_video_action_full",
            "prefix_tokens": self.prefix_tokens,
            "query_tokens": self.query_tokens,
            "key_tokens": self.key_tokens,
            "video_query_tokens": self.segments.video_tokens - self.prefix_tokens,
            "state_tokens": self.segments.state_tokens,
            "action_tokens": self.segments.action_tokens,
        }
        if self.valid_video_key_indices is not None:
            payload["valid_video_key_indices"] = list(self.valid_video_key_indices)
        return payload


def _conditioning_prefix_indices(
    modality_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(conditioning-video positions, every other position) of one shared modality layout."""
    ids = modality_ids[0]
    return torch.nonzero(ids == 1).flatten(), torch.nonzero(ids != 1).flatten()


def _validate_conditioning_prefix_layout(
    modality_ids: torch.Tensor,
    segments: _BidirectionalVideoActionFullSegments,
) -> _ConditioningPrefixLayout:
    """Prove the conditioning-video tokens are a strict subset of the validated video segment."""
    if modality_ids.ndim != 2 or not torch.equal(
        modality_ids, modality_ids[0:1].expand_as(modality_ids)
    ):
        raise ValueError("conditioning-prefix export requires one shared modality layout per batch")
    if modality_ids.shape[1] != segments.total_tokens:
        raise ValueError(
            "conditioning-prefix export layout does not match the validated joint layout: "
            f"{modality_ids.shape[1]} != {segments.total_tokens}"
        )
    cond_index, query_index = _conditioning_prefix_indices(modality_ids)
    prefix_tokens = int(cond_index.numel())
    if prefix_tokens == 0 or prefix_tokens >= segments.video_tokens:
        raise ValueError(
            "conditioning-prefix export requires conditioning and denoised video tokens; got "
            f"{prefix_tokens} conditioning of {segments.video_tokens} video tokens"
        )
    if bool((cond_index >= segments.video_tokens).any().item()):
        raise ValueError(
            "conditioning-prefix export found conditioning tokens outside the video segment"
        )
    valid_video_key_indices = None
    if segments.valid_video_indices is not None:
        video_key_order = torch.cat((cond_index, query_index[query_index < segments.video_tokens]))
        valid_video_positions = set(segments.valid_video_indices)
        valid_video_key_indices = tuple(
            index
            for index, position in enumerate(video_key_order.tolist())
            if position in valid_video_positions
        )
    return _ConditioningPrefixLayout(
        segments=segments,
        prefix_tokens=prefix_tokens,
        valid_video_key_indices=valid_video_key_indices,
    )


def _require_conditioning_prefix_count(
    layout: _ConditioningPrefixLayout,
    *,
    num_cameras: int,
    latent_shape: tuple[int, int, int, int],
    conditioning_latent_frames: int,
) -> None:
    from groot.deployment.n2_trt.runtime import video_tokens_per_camera

    expected = video_tokens_per_camera(latent_shape, conditioning_latent_frames) * num_cameras
    if layout.prefix_tokens != expected:
        raise ValueError(
            "conditioning-prefix N2 runtime contract requires exactly "
            f"{expected} conditioning-video tokens; got {layout.prefix_tokens}"
        )


def _segmented_conditioning_prefix_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layout: _ConditioningPrefixLayout,
    flatten_heads: bool,
) -> torch.Tensor:
    """``bidirectional_video_action_full`` for query tokens over keys with the cached prefix."""
    batch, queries, heads, head_dim = q.shape
    if (
        queries != layout.query_tokens
        or k.shape[1] != layout.key_tokens
        or v.shape[1] != layout.key_tokens
    ):
        raise ValueError(
            "conditioning-prefix sdpa input does not match the validated export layout: "
            f"q={queries}, k={k.shape[1]}, v={v.shape[1]}, "
            f"expected q={layout.query_tokens}, k=v={layout.key_tokens}"
        )

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    video_q_end = layout.segments.video_tokens - layout.prefix_tokens
    state_q_end = video_q_end + layout.segments.state_tokens
    video_k_end = layout.segments.video_tokens
    state_k_end = video_k_end + layout.segments.state_tokens

    video_k, video_v = k[:, :, :video_k_end], v[:, :, :video_k_end]
    action_k, action_v = k, v
    if layout.valid_video_key_indices is not None:
        video_indices = torch.tensor(
            layout.valid_video_key_indices, dtype=torch.long, device=k.device
        )
        key_indices = torch.tensor(
            (*layout.valid_video_key_indices, *range(video_k_end, layout.key_tokens)),
            dtype=torch.long,
            device=k.device,
        )
        video_k = video_k.index_select(2, video_indices)
        video_v = video_v.index_select(2, video_indices)
        action_k = k.index_select(2, key_indices)
        action_v = v.index_select(2, key_indices)
    video_out = scaled_dot_product_attention_bf16(q[:, :, :video_q_end], video_k, video_v)
    state_out = scaled_dot_product_attention_bf16(
        q[:, :, video_q_end:state_q_end],
        k[:, :, video_k_end:state_k_end],
        v[:, :, video_k_end:state_k_end],
    )
    action_out = scaled_dot_product_attention_bf16(q[:, :, state_q_end:], action_k, action_v)
    out = torch.cat((video_out, state_out, action_out), dim=2)
    out = out.permute(0, 2, 1, 3).contiguous()
    if flatten_heads:
        return out.reshape(batch, queries, heads * head_dim)
    return out


def _install_conditioning_prefix_flex_patch(layout: _ConditioningPrefixLayout) -> None:
    """Teach the installed segmented exporter the cached-prefix joint call.

    Wraps whatever ``flex_attention_op`` is installed (the full-layout segmented exporter):
    a ``(query_tokens, key_tokens)`` masked call takes the prefix decomposition, every other
    call falls through unchanged. ``_restore_flex_patch`` with the pre-segmented op removes both.
    """
    from groot.core.training.models.n2.components import transformer_cosmos

    inner = transformer_cosmos.flex_attention_op

    def conditioning_prefix_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Any = None,
        pad: int = 0,
        flatten_heads: bool = True,
        compile_attention: bool = False,
    ) -> torch.Tensor:
        if (
            attn_mask is not None
            and q.shape[1] == layout.query_tokens
            and k.shape[1] == layout.key_tokens
        ):
            if pad != 0:
                raise ValueError(f"conditioning-prefix export runs unpadded; got pad={pad}")
            return _segmented_conditioning_prefix_sdpa(
                q, k, v, layout=layout, flatten_heads=flatten_heads
            )
        return inner(
            q,
            k,
            v,
            attn_mask=attn_mask,
            pad=pad,
            flatten_heads=flatten_heads,
            compile_attention=compile_attention,
        )

    transformer_cosmos.flex_attention_op = conditioning_prefix_attention_op


def _tensor_record(tensor: Any) -> dict[str, Any] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _production_eager_oracle() -> str:
    from groot.deployment.n2_trt.manifest import TRANSFORMER_EAGER_ORACLE

    if os.environ.get("ENABLE_TENSORRT", "False").lower() == "true":
        raise RuntimeError(
            "refusing to record a transformer parity oracle through the TensorRT export surrogate"
        )
    return TRANSFORMER_EAGER_ORACLE


def _joint_sample_extras(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Retain the layout needed to reconstruct the static action cache."""
    extras = {}
    valid_mask = getattr(payload["packed"], "token_valid_mask", None)
    if valid_mask is not None:
        extras["token_valid_mask"] = valid_mask
    token_merge = payload.get("token_merge")
    if token_merge is not None:
        extras["coarse_token_idx"] = token_merge.reduced_token_indices
    return extras


def _save_capture(path: Path, payload: dict[str, Any]) -> None:
    packed = payload["packed"]
    cache = payload.get("kv_cache")
    output_name = payload.get("output_name", "output")
    if output_name not in {"output", "output1"}:
        raise ValueError(f"unsupported production capture output name: {output_name!r}")
    sample = {
        "oracle": _production_eager_oracle(),
        "latents": packed.tokens.detach().cpu(),
        "timesteps": packed.timesteps.detach().cpu(),
        "rope_cos": packed.rope_freqs[0].detach().cpu(),
        "rope_sin": packed.rope_freqs[1].detach().cpu(),
        "context": payload["prompt_embs"].detach().cpu(),
        "modality_ids": packed.modality_ids.detach().cpu(),
        "video_token_indices": getattr(packed, "video_token_indices", None),
        "non_video_token_indices": getattr(packed, "non_video_token_indices", None),
        output_name: payload["output"].detach().cpu(),
        "pad": int(packed.pad),
    }
    if isinstance(payload.get(TEXT_SEQ_LEN_INPUT), torch.Tensor):
        sample[TEXT_SEQ_LEN_INPUT] = payload[TEXT_SEQ_LEN_INPUT].detach().cpu()
    for name in ("video_token_indices", "non_video_token_indices"):
        if isinstance(sample[name], torch.Tensor):
            sample[name] = sample[name].detach().cpu()
    for name in (PROJECTED_CONTEXT_INPUT, CROSS_KEY_INPUT, CROSS_VALUE_INPUT):
        if isinstance(payload.get(name), torch.Tensor):
            # Derived from the same eager conditioning the runtime runs; the production
            # sample then binds a projected-context or cross-K/V action engine directly.
            sample[name] = payload[name].detach().cpu()
    if cache is not None and getattr(cache, "key_cache", None):
        token_merge_layers = payload.get("token_merge_layers")
        if token_merge_layers is None:
            sample["kv_cache_key"] = torch.stack(cache.key_cache).detach().cpu()
            sample["kv_cache_value"] = torch.stack(cache.value_cache).detach().cpu()
        else:
            from groot.deployment.n2_trt.runtime import _stack_token_merge_cache

            cache_tensors = _stack_token_merge_cache(cache, token_merge_layers)
            for name, tensor in zip(
                (
                    "kv_cache_key_fine",
                    "kv_cache_value_fine",
                    "kv_cache_key_coarse",
                    "kv_cache_value_coarse",
                ),
                cache_tensors,
                strict=True,
            ):
                sample[name] = tensor.detach().cpu()
    sample.update(
        {name: tensor.detach().cpu() for name, tensor in _joint_sample_extras(payload).items()}
    )
    if isinstance(payload.get("cond_index"), torch.Tensor):
        # Conditioning-cached joint step: the frozen prefix the graph receives as inputs.
        for name in ("cond_index", "cond_key", "cond_value"):
            sample[name] = payload[name].detach().cpu()
    torch.save(sample, path)


def _install_call_probe(
    model: torch.nn.Module,
    sample_dir: Path,
    *,
    action_context_mode: str = "raw",
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Capture the first joint/action call for ONNX export."""
    if action_context_mode not in ACTION_CONTEXT_MODES:
        raise ValueError(f"unsupported action context mode {action_context_mode!r}")
    captured: dict[str, dict[str, Any]] = {}
    sample_paths: dict[str, Path] = {}
    original = model._call_transformer
    model._conditioning_probe_calls = []
    token_merge_layers = (
        tuple(model.config.token_merge_layers)
        if model.config.token_merge_kernel is not None
        else None
    )

    def wrapped_call(
        packed: Any,
        prompt_embs: torch.Tensor,
        *,
        denoising_step: int = 0,
        is_cfg: bool = False,
        To: int = 1,
        total_seqlen: Any = None,
        last_mini_batch_params_dict: Any = None,
        state_collector: Any = None,
        **transformer_kwargs: Any,
    ) -> torch.Tensor:
        if packed.attention_mask is not None and transformer_kwargs.get("kv_cache") is None:
            from groot.core.training.models.n2.components.kv_cache import CondKVCache

            transformer_kwargs["kv_cache"] = CondKVCache()
        output = original(
            packed,
            prompt_embs,
            denoising_step=denoising_step,
            is_cfg=is_cfg,
            To=To,
            total_seqlen=total_seqlen,
            last_mini_batch_params_dict=last_mini_batch_params_dict,
            state_collector=state_collector,
            **transformer_kwargs,
        )
        cache = transformer_kwargs.get("kv_cache")
        store_kv = transformer_kwargs.get("store_kv", True)
        mode = None
        if packed.attention_mask is not None and cache is not None and store_kv:
            # A frozen-prefix cache marks a later joint step the conditioning-cached graph serves.
            mode = (
                "joint_cond_capture"
                if isinstance(cache, FrozenPrefixKVCache)
                else "joint_full_capture"
            )
        elif cache is not None and not store_kv:
            mode = "action_cached"
        if mode is not None:
            sigmas = packed.timesteps.detach().to(device="cpu", dtype=torch.bfloat16).clone()
            if not torch.isfinite(sigmas).all():
                raise ValueError("production probe supplied nonfinite conditioning sigmas")
            model._conditioning_probe_calls.append(
                {
                    "mode": mode,
                    "denoising_step": denoising_step,
                    "shape": list(sigmas.shape),
                    "sigmas": torch.unique(sigmas),
                }
            )
        if mode is not None and mode not in captured:
            payload = {
                "packed": packed,
                "prompt_embs": prompt_embs,
                "output": output,
                "output_name": "output1" if mode == "action_cached" else "output",
                "kv_cache": cache,
                # The merged model API threads the canonical TokenMergePlan as
                # ``sequence_transform``.  Retain the historical artifact key
                # consumed by the TensorRT exporter/runtime, but observe it at
                # the production call boundary rather than rebuilding a plan.
                "token_merge": transformer_kwargs.get("sequence_transform"),
                "token_merge_layers": token_merge_layers,
            }
            text_pad_mask = transformer_kwargs.get("text_pad_mask")
            if text_pad_mask is not None:
                payload[TEXT_SEQ_LEN_INPUT] = text_pad_mask.seq_len
            if mode == "action_cached" and action_context_mode == "projected":
                payload[PROJECTED_CONTEXT_INPUT] = project_context(
                    model.transformer.crossattn_proj, prompt_embs
                )
            elif mode == "action_cached" and action_context_mode == "kv":
                keys, values = cross_attention_kv(
                    model.transformer.crossattn_proj,
                    [block.attn2 for block in model.transformer.transformer_blocks],
                    prompt_embs,
                )
                payload[CROSS_KEY_INPUT] = keys
                payload[CROSS_VALUE_INPUT] = values
            elif mode == "joint_cond_capture":
                payload["cond_index"] = cache.frozen_index
                payload["cond_key"] = torch.stack(cache.frozen_key, dim=0)
                payload["cond_value"] = torch.stack(cache.frozen_value, dim=0)
            captured[mode] = payload
            path = sample_dir / f"{mode}.pt"
            _save_capture(path, payload)
            sample_paths[mode] = path
        return output

    model._call_transformer = wrapped_call
    return captured, sample_paths


# ── synthetic probe observation ─────────────────────────────────────────────
# Checkpoint-shaped input used only to capture the fixed export graph.

DEFAULT_EMBODIMENT = "xdof_relative_eef_relative_joint"
LANGUAGE = "pick up the object"


def _load_metadata(
    checkpoint: Path,
    embodiment: str = DEFAULT_EMBODIMENT,
) -> tuple[dict[str, Any], dict[str, Any], N2TensorRTContract]:
    """Read the checkpoint workload and check it is exportable."""
    processor = json.loads((checkpoint / "processor_config.json").read_text())
    statistics_config = json.loads((checkpoint / "statistics.json").read_text())
    kwargs = processor["processor_kwargs"]
    configs = kwargs["modality_configs"]
    if embodiment not in configs or embodiment not in statistics_config:
        raise ValueError(f"Checkpoint does not contain the {embodiment!r} workload")

    config = configs[embodiment]
    contract = N2TensorRTContract.from_checkpoint(
        checkpoint,
        embodiment=embodiment,
        video_denoising_steps=2,
        action_denoising_steps=4,
        joint_denoise=True,
        conditional_kv_cache=True,
    )
    action_deltas = tuple(config["action"]["delta_indices"])
    checks = {
        "state deltas": (tuple(config["state"]["delta_indices"]), (0,)),
        "action horizon": (action_deltas, tuple(range(len(action_deltas)))),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                "unsupported N2 TensorRT contract (not supported yet): "
                f"{name} must be {expected}, got {actual}"
            )
    return config, statistics_config[embodiment], contract


def _gradient_video(shape: tuple[int, int, int, int]) -> np.ndarray:
    frames, height, width, _channels = shape
    x = np.linspace(0, 255, width, dtype=np.uint16)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint16)[:, None]
    video = np.empty(shape, dtype=np.uint8)
    for index in range(frames):
        video[index, :, :, 0] = ((x + index * 7) % 256).astype(np.uint8)
        video[index, :, :, 1] = ((y + index * 13) % 256).astype(np.uint8)
        video[index, :, :, 2] = ((x // 2 + y // 2 + index * 17) % 256).astype(np.uint8)
    return video


def make_synthetic_observation(
    checkpoint: Path,
    embodiment_tag: str = DEFAULT_EMBODIMENT,
    *,
    language: str = LANGUAGE,
    video_mode: str = "gradient",
    seed: int = 0,
) -> dict[str, Any]:
    """Build the checkpoint-shaped fixed input (extra args retain caller compatibility)."""
    if video_mode != "gradient":
        raise ValueError("Only the deterministic gradient video is supported")
    del seed  # The historical gradient is deterministic and seed-independent.

    config, stats, contract = _load_metadata(checkpoint, embodiment_tag.lower())
    video_shape = (
        contract.conditioning_frames,
        contract.image_height,
        contract.image_width,
        3,
    )
    observation: dict[str, Any] = {}
    for key in config["video"]["modality_keys"]:
        observation[f"video.{key}"] = _gradient_video(video_shape)
    for key in config["state"]["modality_keys"]:
        mean = np.asarray(stats["state"][key]["mean"], dtype=np.float32)
        observation[f"state.{key}"] = mean[None, :]
    for key in config["language"]["modality_keys"]:
        observation[key] = language
    return observation


def _run_policy_probe(
    args: argparse.Namespace, output_dir: Path
) -> tuple[torch.nn.Module, dict[str, Path], dict[str, dict[str, Any]]]:
    from groot.core.data import schema
    from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
    from groot.core.policy.n2_policy import Gr00tN2Policy

    os.environ["GROOT_N2_JOINT_DENOISE"] = "1"
    os.environ["GROOT_N2_KV_CACHE_COND"] = "1"
    os.environ["GROOT_N2_NUM_VIDEO_STEPS"] = str(args.num_video_steps)
    os.environ["GROOT_N2_SHIFT"] = "5"
    os.environ["GROOT_N2_CODEC_ROUNDTRIP"] = args.codec
    cond_kv_freeze = bool(getattr(args, "cond_kv_freeze", False))
    if cond_kv_freeze:
        os.environ["GROOT_N2_COND_KV_FREEZE"] = "1"
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    policy = Gr00tN2Policy(
        schema.EmbodimentTag[args.policy_embodiment_tag],
        args.checkpoint,
        num_inference_timesteps=args.num_action_steps,
        device="cuda",
        strict=True,
        use_cuda_graph=False,
    )
    original_call_transformer = policy.model._call_transformer
    captured, sample_paths = _install_call_probe(
        policy.model, sample_dir, action_context_mode=args.action_context_mode
    )
    observation = make_synthetic_observation(
        Path(args.checkpoint),
        args.embodiment_tag,
        language=args.language,
        video_mode="gradient",
        seed=0,
    )
    try:
        with torch.inference_mode():
            Gr00tFlatPolicyWrapper(policy, strict=True).get_action(observation)
            if "action_cached" not in captured:
                _probe_unused_action_engine(policy.model, captured["joint_full_capture"])
    finally:
        policy.model._call_transformer = original_call_transformer
    required = {"joint_full_capture", "action_cached"}
    if cond_kv_freeze:
        required.add("joint_cond_capture")
    missing = required - set(captured)
    if missing:
        raise RuntimeError(
            f"policy probe did not capture required transformer calls: {sorted(missing)}"
        )
    return policy.model, sample_paths, captured


def _probe_unused_action_engine(model, joint_payload):
    """Capture the unused cached-action ABI for equal video/action schedules."""
    packed = joint_payload["packed"]
    action_indices = torch.where(packed.modality_ids[0] == 3)[0]
    positions = torch.arange(action_indices.numel(), device=packed.tokens.device)
    action_packed = replace(
        packed,
        tokens=packed.tokens.index_select(1, action_indices),
        timesteps=packed.timesteps.index_select(1, action_indices),
        rope_freqs=tuple(freq.index_select(0, action_indices) for freq in packed.rope_freqs),
        modality_ids=packed.modality_ids.index_select(1, action_indices),
        attention_mask=None,
        pad=0,
        extra_pos_emb=None,
        video_token_indices=positions[:0],
        state_token_indices=positions[:0],
        action_token_indices=positions,
        non_video_token_indices=positions,
        has_video_cond_tokens=False,
        token_valid_mask=None,
    )
    kwargs = {}
    if joint_payload.get("token_merge") is not None:
        kwargs["token_merge"] = joint_payload["token_merge"]
    cache = model._slice_cache_to_static_prefix(
        joint_payload["kv_cache"],
        packed.modality_ids,
        token_valid_mask=packed.token_valid_mask,
        **kwargs,
    )
    if joint_payload.get(TEXT_SEQ_LEN_INPUT) is not None:
        kwargs = {"text_pad_mask": TextPadMask(joint_payload[TEXT_SEQ_LEN_INPUT])}
    else:
        kwargs = {}
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        model._call_transformer(
            action_packed, joint_payload["prompt_embs"], kv_cache=cache, store_kv=False, **kwargs
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _runtime_manifest(args: argparse.Namespace, *, contract: N2TensorRTContract) -> dict[str, Any]:
    import onnx

    from groot.deployment.n2_trt.manifest import SourceRevision

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
    }
    cond_kv_freeze = bool(getattr(args, "cond_kv_freeze", False))
    return {
        "checkpoint": args.checkpoint,
        "source": asdict(
            SourceRevision.capture("exporter", __file__, revision=args.source_revision)
        ),
        "contract_schema_version": contract.schema_version,
        "input_camera_names": list(contract.input_camera_names),
        "input_camera_count": contract.input_camera_count,
        "total_camera_slot_count": contract.total_camera_slot_count,
        "padded_camera_slot_count": contract.padded_camera_slot_count,
        "mode": args.mode,
        "transformer_precision": TransformerPrecision.BF16.value,
        "joint_mask_export": args.joint_mask_export,
        "cond_kv_freeze": cond_kv_freeze,
        "embodiment_tag": args.embodiment_tag,
        "policy_embodiment_tag": args.policy_embodiment_tag,
        "codec_roundtrip": args.codec,
        "action_context_mode": args.action_context_mode,
        "text_keys": contract.text_keys,
        "capture_samples": 1,
        "opset": args.opset,
        "predict_kwargs": {
            "num_action_steps": args.num_action_steps,
            "num_video_steps": args.num_video_steps,
            "joint_denoise": True,
            "kv_cache_cond": True,
            "cond_kv_freeze": cond_kv_freeze,
            "predict_video": False,
            "shift": args.shift,
            "seed": args.seed,
        },
        "versions": versions,
        "environment": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "TRITON_PTXAS_PATH": os.environ.get("TRITON_PTXAS_PATH"),
            "ENABLE_TENSORRT": os.environ.get("ENABLE_TENSORRT"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def _onnx_summary(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": opset.domain, "version": opset.version} for opset in model.opset_import
        ],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(sorted(op_counts.items())),
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stack_cache(cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if not key_cache or not value_cache:
        raise ValueError("Captured action_cached payload has no populated KV cache")
    return torch.stack(key_cache, dim=0), torch.stack(value_cache, dim=0)


def _action_inputs(
    payload: dict[str, Any],
    *,
    token_merge_layers: tuple[int, int] | None = None,
    action_context_mode: str = "raw",
) -> tuple[torch.Tensor, ...]:
    packed = payload["packed"]
    if action_context_mode == "raw":
        context_inputs = (payload["prompt_embs"],)
    elif action_context_mode == "projected":
        context_inputs = (payload[PROJECTED_CONTEXT_INPUT],)
    elif action_context_mode == "kv":
        context_inputs = (payload[CROSS_KEY_INPUT], payload[CROSS_VALUE_INPUT])
    else:
        raise ValueError(f"unsupported action context mode {action_context_mode!r}")
    if token_merge_layers is None:
        cache_inputs = _stack_cache(payload["kv_cache"])
    else:
        from groot.deployment.n2_trt.runtime import _stack_token_merge_cache

        cache_inputs = _stack_token_merge_cache(payload["kv_cache"], token_merge_layers)
    return (
        packed.tokens,
        packed.timesteps,
        packed.rope_freqs[0],
        packed.rope_freqs[1],
        *context_inputs,
        packed.modality_ids,
        *cache_inputs,
    )


def _action_context_input_names(action_context_mode: str) -> list[str]:
    """Engine input names that carry the action graph's language conditioning."""
    names = {
        "raw": ["context"],
        "projected": [PROJECTED_CONTEXT_INPUT],
        "kv": [CROSS_KEY_INPUT, CROSS_VALUE_INPUT],
    }
    if action_context_mode not in names:
        raise ValueError(f"unsupported action context mode {action_context_mode!r}")
    return names[action_context_mode]


def _static_cache_outputs(
    joint_output: Any,
    joint_payload: dict[str, Any],
    *,
    token_merge_layers: tuple[int, int] | None,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(joint_output, tuple):
        raise TypeError("joint export adapter must return hidden state and KV tensors")
    packed = joint_payload["packed"]
    modality_ids = packed.modality_ids[0]
    keep = modality_ids != 3
    if getattr(packed, "token_valid_mask", None) is not None:
        keep &= packed.token_valid_mask[0, : keep.numel()]
    fine_indices = torch.nonzero(keep).flatten()

    if token_merge_layers is None:
        if len(joint_output) != 3:
            raise ValueError(f"joint export returned {len(joint_output)} tensors; expected 3")
        return tuple(tensor.index_select(3, fine_indices) for tensor in joint_output[1:])

    if len(joint_output) != 5:
        raise ValueError(
            f"token-merge joint export returned {len(joint_output)} tensors; expected 5"
        )
    token_merge = joint_payload.get("token_merge")
    if token_merge is None:
        raise ValueError("token-merge joint export is missing its token layout")
    coarse_indices = torch.where(keep[token_merge.reduced_token_indices.long()])[0]
    return (
        joint_output[1].index_select(3, fine_indices),
        joint_output[2].index_select(3, fine_indices),
        joint_output[3].index_select(3, coarse_indices),
        joint_output[4].index_select(3, coarse_indices),
    )


def _joint_inputs(payload: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    packed = payload["packed"]
    prompt_embs = payload["prompt_embs"]
    return (
        packed.tokens,
        packed.timesteps,
        packed.rope_freqs[0],
        packed.rope_freqs[1],
        prompt_embs,
        packed.modality_ids,
    )


def _joint_cond_inputs(payload: dict[str, Any], joint_output: Any) -> tuple[torch.Tensor, ...]:
    """Inputs of the conditioning-cached joint graph.

    Query tokens come from the captured later joint step; the cached prefix is the matched
    joint export's conditioning-video K/V, so the two exported graphs compose exactly.
    """
    if not isinstance(joint_output, tuple) or len(joint_output) != 3:
        raise ValueError(
            "conditioning-cached export requires the (output, key, value) joint export"
        )
    packed = payload["packed"]
    cond_index, query_index = _conditioning_prefix_indices(packed.modality_ids)
    recorded = payload.get("cond_index")
    if recorded is not None and not torch.equal(recorded.to(cond_index.device), cond_index):
        raise ValueError("captured frozen positions differ from the conditioning-video tokens")
    return (
        packed.tokens[:, query_index],
        packed.timesteps[:, query_index],
        packed.rope_freqs[0][query_index],
        packed.rope_freqs[1][query_index],
        payload["prompt_embs"],
        packed.modality_ids[:, query_index],
        joint_output[1].index_select(3, cond_index),
        joint_output[2].index_select(3, cond_index),
    )


def _captured_layout_record(payload: dict[str, Any]) -> dict[str, Any]:
    packed = payload["packed"]
    modality_ids = packed.modality_ids
    ids = modality_ids[0]
    return {
        "tokens": _tensor_record(packed.tokens),
        "timesteps": _tensor_record(packed.timesteps),
        "rope_cos": _tensor_record(packed.rope_freqs[0]),
        "rope_sin": _tensor_record(packed.rope_freqs[1]),
        "context": _tensor_record(payload["prompt_embs"]),
        PROJECTED_CONTEXT_INPUT: _tensor_record(payload.get(PROJECTED_CONTEXT_INPUT)),
        CROSS_KEY_INPUT: _tensor_record(payload.get(CROSS_KEY_INPUT)),
        CROSS_VALUE_INPUT: _tensor_record(payload.get(CROSS_VALUE_INPUT)),
        "modality_ids": _tensor_record(modality_ids),
        TEXT_SEQ_LEN_INPUT: _tensor_record(payload.get(TEXT_SEQ_LEN_INPUT)),
        "shared_batch_layout": bool(
            torch.equal(modality_ids, modality_ids[:1].expand_as(modality_ids))
        ),
        "segment_token_counts": {
            "video": int(((ids == 0) | (ids == 1)).sum().item()),
            "video_denoising": int((ids == 0).sum().item()),
            "video_conditioning": int((ids == 1).sum().item()),
            "state": int((ids == 2).sum().item()),
            "action": int((ids == 3).sum().item()),
        },
        "attention_mask_type": (
            type(packed.attention_mask).__name__ if packed.attention_mask is not None else None
        ),
        "attention_mask_pattern": "bidirectional_video_action_full",
        "all_tokens_valid": packed.token_valid_mask is None or bool(packed.token_valid_mask.all()),
        "pad": int(packed.pad),
    }


def _run_adapter(adapter: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> Any:
    with torch.inference_mode(), torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        output = adapter(*inputs)
    _sync()
    return output


def _output_record(output: Any) -> Any:
    if isinstance(output, tuple):
        return [_tensor_record(item) for item in output]
    return _tensor_record(output)


def _save_export_reference(
    path: Path,
    *,
    oracle: str,
    input_names: list[str],
    inputs: tuple[torch.Tensor, ...],
    output_names: list[str],
    output: Any,
    extras: dict[str, torch.Tensor] | None = None,
) -> None:
    outputs = output if isinstance(output, tuple) else (output,)
    if len(input_names) != len(inputs) or len(output_names) != len(outputs):
        raise ValueError("export reference names do not match the captured tensors")
    sample: dict[str, Any] = {"oracle": oracle}
    sample.update(
        {name: tensor.detach().cpu() for name, tensor in zip(input_names, inputs, strict=True)}
    )
    sample.update(
        {name: tensor.detach().cpu() for name, tensor in zip(output_names, outputs, strict=True)}
    )
    if extras:
        sample.update({name: tensor.detach().cpu() for name, tensor in extras.items()})
    torch.save(sample, path)


def _external_attribute_tensors(model: Any) -> list[tuple[Any, str]]:
    """Every node-attribute tensor of the top-level graph stored in an external file."""
    import onnx

    found = []
    for node in model.graph.node:
        for attribute in node.attribute:
            tensors = [attribute.t] if attribute.type == onnx.AttributeProto.TENSOR else []
            if attribute.type == onnx.AttributeProto.TENSORS:
                tensors = list(attribute.tensors)
            for tensor in tensors:
                if tensor.data_location != onnx.TensorProto.EXTERNAL:
                    continue
                location = {entry.key: entry.value for entry in tensor.external_data}.get(
                    "location"
                )
                if not location:
                    raise RuntimeError(
                        f"attribute tensor of {node.name!r} has no external location"
                    )
                found.append((tensor, location))
    return found


def _inline_attribute_tensors(onnx_path: Path) -> list[str]:
    """Move node-attribute tensors (Constant values) back into the protobuf.

    ``torch.onnx.export(external_data=True)`` writes every tensor of a large graph
    as one file named after its node, next to the model. The joint and action graphs
    of a bundle share node names, so the second export silently overwrites the first
    graph's constants (LayerNorm scale/bias) with its own values whenever their
    numbering differs. Initializers keep their per-tensor files: they are named after
    the parameter and carry identical content in every graph of the bundle. Returns
    the locations that were inlined; their files are removed unless an initializer
    of this graph also points at them.
    """
    import onnx
    from onnx import external_data_helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_locations = {
        entry.value
        for initializer in model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        for entry in initializer.external_data
        if entry.key == "location"
    }
    inlined: list[str] = []
    for tensor, location in _external_attribute_tensors(model):
        external_data_helper.load_external_data_for_tensor(tensor, str(onnx_path.parent))
        del tensor.external_data[:]
        tensor.data_location = onnx.TensorProto.DEFAULT
        inlined.append(location)
    if inlined:
        onnx.save(model, str(onnx_path))
        for location in dict.fromkeys(inlined):
            if location not in initializer_locations:
                (onnx_path.parent / location).unlink(missing_ok=True)
    return inlined


def _snapshot_external_files(directory: Path, *, skip: Path) -> dict[Path, tuple[int, int, str]]:
    """size, mtime_ns and sha256 of every file already in the bundle directory."""
    from groot.deployment.n2_trt.manifest import sha256_file

    snapshot = {}
    for path in directory.iterdir():
        if path.is_file() and path.resolve() != skip.resolve():
            stat = path.stat()
            snapshot[path] = (stat.st_size, stat.st_mtime_ns, sha256_file(path))
    return snapshot


def _assert_no_differing_overwrite(
    before: dict[Path, tuple[int, int, str]], directory: Path
) -> None:
    """Refuse a bundle in which an export replaced an existing file with other content."""
    from groot.deployment.n2_trt.manifest import sha256_file

    changed = []
    for path, (size, mtime_ns, digest) in before.items():
        if not path.exists():
            continue
        stat = path.stat()
        if stat.st_size == size and stat.st_mtime_ns == mtime_ns:
            continue
        if sha256_file(path) != digest:
            changed.append(path.name)
    if changed:
        raise RuntimeError(
            f"export into {directory} overwrote {len(changed)} existing file(s) with different "
            f"content ({changed[:5]}): two graphs of the bundle share an external-data file name"
        )


def _export_onnx(
    *,
    adapter: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = _snapshot_external_files(output_path.parent, skip=output_path)
    start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        torch.onnx.export(
            adapter,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
            external_data=True,
            do_constant_folding=False,
        )
    _sync()
    inlined = _inline_attribute_tensors(output_path)
    _assert_no_differing_overwrite(before, output_path.parent)
    elapsed_s = time.perf_counter() - start
    return {
        "onnx": str(output_path),
        "export_elapsed_s": elapsed_s,
        "onnx_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "onnx_summary": _onnx_summary(output_path),
        "inlined_attribute_tensors": len(inlined),
    }


def main() -> int:
    args = _parse_args()
    from groot.deployment.n2_trt.manifest import (
        TRANSFORMER_EXPORT_JOINT_COND_ORACLE,
        TRANSFORMER_EXPORT_JOINT_ORACLE,
        sha256_file,
        transformer_export_action_oracle,
    )

    action_export_oracle = transformer_export_action_oracle(args.action_context_mode)
    wanted_cond = bool(getattr(args, "cond_kv_freeze", False))
    projected_context = args.action_context_mode == "projected"
    cross_kv = args.action_context_mode == "kv"
    action_context_input_names = _action_context_input_names(args.action_context_mode)
    args.checkpoint = str(_prepare_checkpoint(args.checkpoint))
    contract = N2TensorRTContract.from_checkpoint(
        args.checkpoint,
        embodiment=args.embodiment_tag,
        video_denoising_steps=args.num_video_steps,
        action_denoising_steps=args.num_action_steps,
        joint_denoise=True,
        conditional_kv_cache=True,
    )

    real_text_keys = contract.text_keys == "real"
    if real_text_keys and cross_kv:
        raise ValueError("the 'kv' action context mode does not support text_keys='real'")
    text_input_names = [TEXT_SEQ_LEN_INPUT] if real_text_keys else []

    # Retain production-native samples for behavioral diagnosis. Engine
    # correctness is qualified separately against the exact SDPA export source.
    os.environ["ENABLE_TENSORRT"] = "False"
    production_sample_oracle = _production_eager_oracle()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[n2-transformer-export] output_dir={output_dir}")
    print(f"[n2-transformer-export] checkpoint={args.checkpoint}")
    print("[n2-transformer-export] running policy-shaped transformer call probe...")
    model, sample_paths, captured_samples = _run_policy_probe(args, output_dir)
    if getattr(model, "text_keys", "all") != contract.text_keys:
        raise RuntimeError(
            f"loaded model text keys {getattr(model, 'text_keys', 'all')!r} != contract "
            f"{contract.text_keys!r}"
        )
    for name, payload in captured_samples.items():
        if (payload.get(TEXT_SEQ_LEN_INPUT) is not None) != real_text_keys:
            raise RuntimeError(
                f"captured {name} call text padding mask does not match text_keys="
                f"{contract.text_keys!r}"
            )
    # The captured production outputs above remain immutable CPU samples.  Only
    # the adapter execution and ONNX trace below use the exportable SDPA path.
    os.environ["ENABLE_TENSORRT"] = "True"

    wanted_joint = args.mode in ("production", "joint_capture")
    wanted_action = args.mode in ("production", "action_cached")
    if wanted_joint and "joint_full_capture" not in captured_samples:
        raise RuntimeError("No joint_full_capture transformer call was captured")
    if wanted_action and "action_cached" not in captured_samples:
        raise RuntimeError("No action_cached transformer call was captured")
    if wanted_cond and not wanted_joint:
        raise RuntimeError("conditioning-cached export requires the joint export")
    if wanted_cond and "joint_cond_capture" not in captured_samples:
        raise RuntimeError("No joint_cond_capture transformer call was captured")

    original_flex_attention = None
    dense_mask = None
    segmented_layout = None
    coarse_segmented_layout = None
    token_merge = None
    token_merge_layers = None
    if wanted_joint:
        packed = captured_samples["joint_full_capture"]["packed"]
        if packed.attention_mask is None:
            raise RuntimeError("Captured joint_full_capture payload has no attention mask")
        token_merge = captured_samples["joint_full_capture"].get("token_merge")
        if model.config.token_merge_kernel is not None and token_merge is None:
            raise RuntimeError("token-merge checkpoint did not pass a plan to the joint call")
        if token_merge is not None:
            token_merge_layers = (token_merge.start_layer, token_merge.end_layer)
        if wanted_cond and token_merge is not None:
            raise RuntimeError("conditioning-cached export does not support token merge")
        dense_mask = _dense_token_mask_from_block_mask(packed.attention_mask)
        latent_shape = _model_latent_shape(
            model,
            conditioning_frames=contract.conditioning_frames,
            image_size=contract.image_size,
        )
        (
            original_flex_attention,
            segmented_layout,
            coarse_segmented_layout,
        ) = _install_segmented_sdpa_flex_patch(
            packed.modality_ids,
            dense_mask,
            contract=contract,
            total_camera_slot_count=model.max_n_cameras,
            action_horizon=int(model.config.action_cfg["action_horizon"]),
            latent_shape=latent_shape,
            total_latent_frames=contract.total_latent_frames,
            token_merge=token_merge,
            patch_size=contract.patch_size,
        )

    joint_adapter = None
    joint_inputs = None
    joint_output = None
    action_adapter = None
    action_inputs = None
    conditioning_stack = ExitStack()
    try:
        conditioning_metadata = conditioning_stack.enter_context(
            export_conditioning(
                model.transformer,
                torch.cat([call["sigmas"] for call in model._conditioning_probe_calls]),
            )
        )
        conditioning_metadata["observed_call_count"] = len(model._conditioning_probe_calls)
        conditioning_metadata["probe_calls"] = [
            {**call, "sigmas": call["sigmas"].float().tolist()}
            for call in model._conditioning_probe_calls
        ]
        if wanted_joint:
            joint_payload = captured_samples["joint_full_capture"]
            packed = joint_payload["packed"]
            video_token_indices, non_video_token_indices = _resolve_routing_indices(
                packed.modality_ids,
                getattr(packed, "video_token_indices", None),
                getattr(packed, "non_video_token_indices", None),
            )
            joint_adapter = N2TransformerJointCaptureExportAdapter(
                model.transformer,
                attention_mask=packed.attention_mask,
                pad=int(packed.pad),
                video_token_indices=video_token_indices,
                non_video_token_indices=non_video_token_indices,
                token_merge=token_merge,
            ).eval()
            joint_inputs = (*_joint_inputs(joint_payload), *_text_inputs(joint_payload))

        if wanted_action:
            action_payload = captured_samples["action_cached"]
            packed = action_payload["packed"]
            video_token_indices, non_video_token_indices = _resolve_routing_indices(
                packed.modality_ids,
                getattr(packed, "video_token_indices", None),
                getattr(packed, "non_video_token_indices", None),
            )
            if cross_kv and token_merge_layers is None:
                action_adapter = N2TransformerActionCachedCrossKVExportAdapter(
                    model.transformer, video_token_indices, non_video_token_indices
                ).eval()
            elif cross_kv:
                action_adapter = N2TransformerTokenMergeActionCachedCrossKVExportAdapter(
                    model.transformer,
                    video_token_indices,
                    non_video_token_indices,
                    token_merge_layers,
                ).eval()
            elif token_merge_layers is None:
                action_adapter = N2TransformerActionCachedExportAdapter(
                    model.transformer,
                    video_token_indices,
                    non_video_token_indices,
                    projected_context=projected_context,
                ).eval()
            else:
                action_adapter = N2TransformerTokenMergeActionCachedExportAdapter(
                    model.transformer,
                    video_token_indices,
                    non_video_token_indices,
                    token_merge_layers,
                    projected_context=projected_context,
                ).eval()

        manifests: dict[str, Any] = {}
        if joint_adapter is not None and joint_inputs is not None:
            joint_adapter.transformer = model.transformer
            print("[n2-transformer-export] exporting joint-capture ONNX...")
            joint_output = _run_adapter(joint_adapter, joint_inputs)
            joint_output_names = (
                ["output", "output2", "output3"]
                if token_merge_layers is None
                else [
                    "output",
                    "kv_cache_key_fine",
                    "kv_cache_value_fine",
                    "kv_cache_key_coarse",
                    "kv_cache_value_coarse",
                ]
            )
            joint_input_names = [
                "latents",
                "timesteps",
                "rope_cos",
                "rope_sin",
                "context",
                "modality_ids",
                *text_input_names,
            ]
            joint_reference_path = output_dir / "samples/joint_export_source.pt"
            _save_export_reference(
                joint_reference_path,
                oracle=TRANSFORMER_EXPORT_JOINT_ORACLE,
                input_names=joint_input_names,
                inputs=joint_inputs,
                output_names=joint_output_names,
                output=joint_output,
                extras=_joint_sample_extras(joint_payload),
            )
            sample_paths["joint_export_source"] = joint_reference_path
            export = _export_onnx(
                adapter=joint_adapter,
                inputs=joint_inputs,
                output_path=output_dir / "n2_transformer_joint.onnx",
                input_names=joint_input_names,
                output_names=joint_output_names,
                opset=args.opset,
            )
            manifests["joint_capture"] = {
                **export,
                "sample_path": str(joint_reference_path),
                "production_sample_path": str(sample_paths.get("joint_full_capture")),
                "input_records": [_tensor_record(item) for item in joint_inputs],
                "output": _output_record(joint_output),
                "attention_mask_type": type(
                    captured_samples["joint_full_capture"]["packed"].attention_mask
                ).__name__,
                "dense_mask": _tensor_record(dense_mask),
                "joint_mask_export": args.joint_mask_export,
                "segmented_layout": (segmented_layout.to_manifest() if segmented_layout else None),
                "coarse_segmented_layout": (
                    coarse_segmented_layout.to_manifest()
                    if coarse_segmented_layout is not None
                    else None
                ),
                "pad": int(captured_samples["joint_full_capture"]["packed"].pad),
            }
            _write_json(
                output_dir / "n2_transformer_joint_export.json",
                manifests["joint_capture"],
            )

        cond_manifest_extras: dict[str, Any] = {}
        if wanted_cond:
            if joint_output is None or segmented_layout is None:
                raise RuntimeError("conditioning-cached export requires the joint export output")
            cond_payload = captured_samples["joint_cond_capture"]
            cond_packed = cond_payload["packed"]
            prefix_layout = _validate_conditioning_prefix_layout(
                cond_packed.modality_ids, segmented_layout
            )
            _require_conditioning_prefix_count(
                prefix_layout,
                num_cameras=model.max_n_cameras,
                latent_shape=latent_shape,
                conditioning_latent_frames=contract.conditioning_latent_frames,
            )
            _install_conditioning_prefix_flex_patch(prefix_layout)
            cond_inputs = (
                *_joint_cond_inputs(cond_payload, joint_output),
                *_text_inputs(cond_payload),
            )
            cond_index, query_index = _conditioning_prefix_indices(cond_packed.modality_ids)
            video_token_indices, non_video_token_indices = _resolve_routing_indices(cond_inputs[5])
            cond_adapter = N2TransformerJointCondCachedExportAdapter(
                model.transformer,
                attention_mask=cond_packed.attention_mask,
                video_token_indices=video_token_indices,
                non_video_token_indices=non_video_token_indices,
            ).eval()
            print("[n2-transformer-export] exporting conditioning-cached joint ONNX...")
            cond_output = _run_adapter(cond_adapter, cond_inputs)
            cond_input_names = [
                "latents",
                "timesteps",
                "rope_cos",
                "rope_sin",
                "context",
                "modality_ids",
                "cond_key",
                "cond_value",
                *text_input_names,
            ]
            cond_output_names = ["output", "kv_key", "kv_value"]
            cond_reference_path = output_dir / "samples/joint_condcached_export_source.pt"
            _save_export_reference(
                cond_reference_path,
                oracle=TRANSFORMER_EXPORT_JOINT_COND_ORACLE,
                input_names=cond_input_names,
                inputs=cond_inputs,
                output_names=cond_output_names,
                output=cond_output,
                extras={"cond_index": cond_index, "query_index": query_index},
            )
            sample_paths["joint_condcached_export_source"] = cond_reference_path
            export = _export_onnx(
                adapter=cond_adapter,
                inputs=cond_inputs,
                output_path=output_dir / "n2_transformer_joint_condcached.onnx",
                input_names=cond_input_names,
                output_names=cond_output_names,
                opset=args.opset,
            )
            manifests["joint_condcached"] = {
                **export,
                "sample_path": str(cond_reference_path),
                "production_sample_path": str(sample_paths.get("joint_cond_capture")),
                "input_records": [_tensor_record(item) for item in cond_inputs],
                "output": _output_record(cond_output),
                "joint_mask_export": args.joint_mask_export,
                "conditioning_prefix_layout": prefix_layout.to_manifest(),
                "pad": 0,
            }
            _write_json(
                output_dir / "n2_transformer_joint_condcached_export.json",
                manifests["joint_condcached"],
            )
            cond_manifest_extras = {
                "engine_oracles": {"joint_condcached": TRANSFORMER_EXPORT_JOINT_COND_ORACLE},
                "validation_sample_sha256": {
                    "transformer_joint_condcached_export": sha256_file(cond_reference_path),
                    "transformer_joint_condcached_production": sha256_file(
                        sample_paths["joint_cond_capture"]
                    ),
                },
            }

        if action_adapter is not None:
            if joint_output is None:
                raise RuntimeError("action export requires the matched joint export output")
            action_payload = captured_samples["action_cached"]
            captured_action_inputs = _action_inputs(
                action_payload,
                token_merge_layers=token_merge_layers,
                action_context_mode=args.action_context_mode,
            )
            source_cache = _static_cache_outputs(
                joint_output,
                captured_samples["joint_full_capture"],
                token_merge_layers=token_merge_layers,
            )
            action_inputs = (
                *captured_action_inputs[: -len(source_cache)],
                *source_cache,
                *_text_inputs(action_payload),
            )
            action_adapter.transformer = model.transformer
            print("[n2-transformer-export] exporting action-cached ONNX...")
            output = _run_adapter(action_adapter, action_inputs)
            action_cache_input_names = (
                ["kv_cache_key", "kv_cache_value"]
                if token_merge_layers is None
                else [
                    "kv_cache_key_fine",
                    "kv_cache_value_fine",
                    "kv_cache_key_coarse",
                    "kv_cache_value_coarse",
                ]
            )
            action_input_names = [
                "latents",
                "timesteps",
                "rope_cos",
                "rope_sin",
                *action_context_input_names,
                "modality_ids",
                *action_cache_input_names,
                *text_input_names,
            ]
            action_reference_path = output_dir / "samples/action_export_source.pt"
            _save_export_reference(
                action_reference_path,
                oracle=action_export_oracle,
                input_names=action_input_names,
                inputs=action_inputs,
                output_names=["output1"],
                output=output,
            )
            sample_paths["action_export_source"] = action_reference_path
            export = _export_onnx(
                adapter=action_adapter,
                inputs=action_inputs,
                output_path=output_dir / "n2_transformer_action_cache.onnx",
                input_names=action_input_names,
                output_names=["output1"],
                opset=args.opset,
            )
            manifests["action_cached"] = {
                **export,
                "sample_path": str(action_reference_path),
                "production_sample_path": str(sample_paths.get("action_cached")),
                "input_records": [_tensor_record(item) for item in action_inputs],
                "output": _output_record(output),
                "action_context_mode": args.action_context_mode,
                "kv_cache_inputs": {
                    name: _tensor_record(value)
                    for name, value in zip(
                        action_input_names,
                        action_inputs,
                        strict=True,
                    )
                    if name in action_cache_input_names
                },
            }
            _write_json(
                output_dir / "n2_transformer_action_cache_export.json",
                manifests["action_cached"],
            )

        manifest = _runtime_manifest(args, contract=contract)
        manifest.update(
            {
                "output_dir": str(output_dir),
                "production_sample_oracle": production_sample_oracle,
                "conditioning": conditioning_metadata,
                "engine_oracles": {
                    "joint": TRANSFORMER_EXPORT_JOINT_ORACLE,
                    "action_cache": action_export_oracle,
                    **cond_manifest_extras.get("engine_oracles", {}),
                },
                "sample_paths": {key: str(value) for key, value in sample_paths.items()},
                "validation_sample_sha256": {
                    "transformer_joint_export": sha256_file(sample_paths["joint_export_source"]),
                    "transformer_action_export": sha256_file(sample_paths["action_export_source"]),
                    "transformer_joint_production": sha256_file(sample_paths["joint_full_capture"]),
                    "transformer_action_production": sha256_file(sample_paths["action_cached"]),
                    **cond_manifest_extras.get("validation_sample_sha256", {}),
                },
                "token_merge_kernel": (
                    list(model.config.token_merge_kernel)
                    if model.config.token_merge_kernel is not None
                    else None
                ),
                "token_merge_layers": (
                    list(token_merge_layers) if token_merge_layers is not None else None
                ),
                "exports": manifests,
                "captured_layouts": {
                    name: _captured_layout_record(payload)
                    for name, payload in captured_samples.items()
                },
                "notes": [
                    "Transformer weights and graph I/O are BF16.",
                    "Segmented attention executes in BF16; eager parity is qualified separately.",
                ],
            }
        )
        _write_json(output_dir / "n2_transformer_export.json", manifest)
    finally:
        conditioning_stack.close()
        _restore_flex_patch(original_flex_attention)

    print("[n2-transformer-export] done")
    print(f"[n2-transformer-export] manifest={output_dir / 'n2_transformer_export.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
