# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Cosmos Transformer 3D Model — self-contained port from diffusers.

Ported from diffusers CosmosTransformer3DModel (transformer_cosmos.py).
All diffusers dependencies replaced with inline implementations.
Weight naming matches HF checkpoints (ffn convention) for direct loading.

Stripped: ControlNet support, img_context, ONNX paths, GQA (unused for Cosmos 2B/14B).
"""

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from groot.core.training.models.dream_cosmos.attention_utils.flex_attention import flex_attention_op
from groot.core.training.models.n2.components.flash_attention import flash_attn_func_v23
from groot.core.training.models.n2.components.kv_cache import CondKVCache
from groot.core.training.models.n2.components.masked_attention import MaskedAttentionPlan
from groot.core.training.models.n2.components.text_pad_mask import TextPadMask

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _video_non_video_indices(
    modality_ids: torch.Tensor,
    device: torch.device,
    video_token_indices: torch.Tensor | None = None,
    non_video_token_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build multiview cross-attention indices.

    Prefer packer-provided routing indices over data-dependent nonzero()/any()
    so torch.compile sees stable shape logic. The routing indices are derived
    from the packed sequence layout, while modality_ids remains the semantic
    source of truth and the fallback for legacy/manual PackedSequence objects.
    """
    if video_token_indices is not None:
        video_idx = video_token_indices.to(device=device)
        if non_video_token_indices is not None:
            non_video_idx = non_video_token_indices.to(device=device)
            return video_idx, non_video_idx
        if modality_ids.shape[1] == video_idx.numel():
            return video_idx, None

    # Fallback for legacy/manual PackedSequence objects without layout metadata.
    video_mask = (modality_ids[0] == 0) | (modality_ids[0] == 1)
    video_mask = video_mask.to(device)
    video_idx = video_mask.nonzero(as_tuple=True)[0]
    non_video_mask = ~video_mask
    non_video_idx = non_video_mask.nonzero(as_tuple=True)[0] if non_video_mask.any() else None
    return video_idx, non_video_idx


class LayerwiseSequenceTransform(Protocol):
    """Generic sequence-layout change applied across a contiguous layer range."""

    start_layer: int
    end_layer: int
    reduced_token_indices: torch.Tensor
    reduced_attention_mask: BlockMask | None
    reduced_pad: int
    reduced_video_token_indices: torch.Tensor
    reduced_non_video_token_indices: torch.Tensor | None

    def reduce_sequence(self, latents: torch.Tensor) -> tuple[torch.Tensor, Any]: ...

    def restore_sequence(self, latents: torch.Tensor, state: Any) -> torch.Tensor: ...


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (matches diffusers implementation)."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.weight.dtype)
            x = x * self.weight
        else:
            x = x.to(input_dtype)
        return x.to(input_dtype)


class _GELU(nn.Module):
    """Linear + GELU activation, matching diffusers FeedForward naming: net.0.proj."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    """FeedForward matching diffusers key naming: ff.net.0.proj.weight, ff.net.2.weight."""

    def __init__(self, dim: int, mult: float = 4.0, bias: bool = False):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList(
            [
                _GELU(dim, inner_dim, bias=bias),
                nn.Identity(),  # dropout placeholder (0.0)
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x


def apply_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply rotary position embeddings (Cosmos convention: use_real_unbind_dim=-2).

    Args:
        x: (B, H, S, D) query or key tensor
        freqs: (cos, sin) each (S, D) from CosmosRotaryPosEmbed
    """
    cos, sin = freqs
    cos = cos[None, None, :, :].to(x.device)  # (1, 1, S, D)
    sin = sin[None, None, :, :].to(x.device)
    # Cosmos convention (use_real_unbind_dim=-2): split D into [2, D//2]
    x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # each (B, H, S, D//2)
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)  # (B, H, S, D)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


# ─── Attention ────────────────────────────────────────────────────────────────

# Attention.inference_backend "cudnn": cuDNN first; sdpa_kernel enables only the listed
# backends, so inputs cuDNN declines fall through to the default chain.
_CUDNN_FIRST_SDPA_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


class Attention(nn.Module):
    """Self-attention or cross-attention with QK-norm and RoPE support."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None,
        heads: int,
        dim_head: int,
        qk_norm: str = "rms_norm",
        out_bias: bool = False,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        cross_dim = cross_attention_dim or query_dim
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim, bias=out_bias), nn.Identity()])

        # Match diffusers Cosmos attention q/k RMSNorm epsilon.
        self.norm_q = RMSNorm(dim_head, eps=1e-5) if qk_norm == "rms_norm" else nn.Identity()
        self.norm_k = RMSNorm(dim_head, eps=1e-5) if qk_norm == "rms_norm" else nn.Identity()
        # The unmasked attention kernel in eval mode: "flash_attn" or "cudnn" (torch SDPA, cuDNN
        # first). GrootN2 sets "cudnn" only for a dit_text_keys "real" checkpoint whose config
        # names it, the kernel the branch that trains those runs at inference.
        self.inference_backend = "flash_attn"

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple | None = None,
        attention_mask: BlockMask | MaskedAttentionPlan | TextPadMask | None = None,
        pad: int = 0,
        compile_flex_attention: bool = True,
        kv_cache: CondKVCache | None = None,
        layer_idx: int = 0,
        store_kv: bool = True,
    ) -> torch.Tensor:
        """Self- or cross-attention forward pass.

        When ``kv_cache`` is provided, calls ``kv_cache.update()`` after
        QK-norm and RoPE. On the first call (capture), K/V are stored.
        On subsequent calls (splice), cached prefix K/V are prepended.
        Set ``store_kv=False`` during diffusion reuse to leave the cache
        unchanged across action denoising steps. A ``TextPadMask`` masks the
        cross-attention to each sample's real prompt tokens.
        """
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        B, S_q, _ = q.shape
        S_kv = k.shape[1]
        dim_head = q.shape[-1] // self.heads

        # (B, S, H*D) → (B, H, S, D) for RoPE
        q = q.view(B, S_q, self.heads, dim_head).transpose(1, 2)
        k = k.view(B, S_kv, self.heads, dim_head).transpose(1, 2)
        v = v.view(B, S_kv, self.heads, dim_head).transpose(1, 2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx, store=store_kv)

        if isinstance(attention_mask, TextPadMask):
            # (B, H, S, D) -> (B, S, H, D), the layout the mask takes
            out = attention_mask.cross_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
        elif isinstance(attention_mask, MaskedAttentionPlan):
            # a segment plan (video_action_attention, cuDNN); it takes (B, S, H, D)
            if kv_cache is not None:
                # load-bearing: a spliced cache lengthens K/V past the plan's permutation and the
                # gather would read wrong rows without any shape error
                raise RuntimeError("the masked attention segment plan does not support kv_cache")
            out = attention_mask.self_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
        elif attention_mask is not None:
            # flex_attention expects (B, S, H, D) input
            q = q.transpose(1, 2)  # (B, S, H, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flex_attention_op(
                q,
                k,
                v,
                attn_mask=attention_mask,
                pad=pad,
                flatten_heads=False,
                compile_attention=compile_flex_attention,
            )
        elif self.inference_backend == "cudnn" and not self.training:
            with sdpa_kernel(_CUDNN_FIRST_SDPA_BACKENDS, set_priority=True):
                out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2)  # (B, S_q, H, D)
        else:
            # (B, H, S, D) → (B, S, H, D) for flash attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func_v23(q, k, v, flatten=False)  # (B, S_q, H, D)

        out = out.reshape(B, S_q, -1)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class MultiViewCrossAttention(Attention):
    """Cross-attention where each view attends to its own text tokens independently.

    Matches Cosmos Predict 2.5 MultiViewCrossAttention:
    - Video tokens are split by view, each view cross-attends to its own text copy
    - Non-video tokens (action/state) cross-attend to a single text copy
    - encoder_hidden_states is (B, V*M, D) — text repeated V times
    - video_idx / non_video_idx are precomputed integer indices from modality_ids
    - text_pad_mask (dit_text_keys "real"): every query attends its sample's real prompt tokens
      only; None attends all M slots, pads included (Cosmos Predict 2.5)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple | None = None,
        n_cameras: int = 1,
        video_idx: torch.Tensor | None = None,
        non_video_idx: torch.Tensor | None = None,
        text_pad_mask: TextPadMask | None = None,
    ) -> torch.Tensor:
        if n_cameras <= 1 or encoder_hidden_states is None:
            return super().forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                attention_mask=text_pad_mask,
            )

        V = n_cameras
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1] // V

        # No video tokens (e.g. action-only forward with a KV cache carrying
        # the video/state prefix) — skip the per-view fan-out and cross-attend
        # non-video tokens to a single text copy.
        if video_idx is None or video_idx.numel() == 0:
            return super().forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states[:, :M],
                image_rotary_emb=image_rotary_emb,
                attention_mask=text_pad_mask,
            )

        # Video tokens: per-view cross-attention
        video_hs = hidden_states[:, video_idx]
        L_video = video_hs.shape[1] // V

        x = video_hs.view(B, V, L_video, -1).permute(1, 0, 2, 3).reshape(V * B, L_video, -1)
        ctx = encoder_hidden_states.view(B, V, M, -1).permute(1, 0, 2, 3).reshape(V * B, M, -1)
        x = super().forward(
            x,
            encoder_hidden_states=ctx,
            image_rotary_emb=None,
            attention_mask=None if text_pad_mask is None else text_pad_mask.repeat(V),
        )
        x = x.view(V, B, L_video, -1).permute(1, 0, 2, 3).reshape(B, V * L_video, -1)

        # Video-only: no scatter needed
        if non_video_idx is None:
            return x

        # Mixed modalities: scatter video + non-video into output
        out = torch.zeros(B, hidden_states.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
        out[:, video_idx] = x
        out[:, non_video_idx] = super().forward(
            hidden_states[:, non_video_idx],
            encoder_hidden_states=encoder_hidden_states[:, :M],
            image_rotary_emb=None,
            attention_mask=text_pad_mask,
        )

        return out


# ─── Cosmos Building Blocks ──────────────────────────────────────────────────


class CosmosPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], out_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        p_t, p_h, p_w = self.patch_size
        x = x.reshape(B, C, T // p_t, p_t, H // p_h, p_h, W // p_w, p_w)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)
        return self.proj(x)


class Timesteps(nn.Module):
    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(timesteps)))


class CosmosEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()
        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor):
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


# Consecutive segments of the token sequence, each ``groups`` groups of ``tokens`` tokens that
# share one sigma, e.g. ((frames, tokens_per_frame), (state_tokens, 1), (action_tokens, 1)).
TimestepSegments = tuple[tuple[int, int], ...]


def _per_token(t: torch.Tensor, segments: TimestepSegments) -> torch.Tensor:
    """(B, G, C) modulation rows -> (B, L, C), one row per group of ``segments``.

    Under torch.compile the expands are fused into the consumer and their backward is a plain
    sum over each group, so the per-token (B, L, C) modulation tensor never exists in memory.
    """
    B, _, C = t.shape
    pieces, g0 = [], 0
    for groups, tokens in segments:
        rows = t[:, g0 : g0 + groups, None, :]
        pieces.append(rows.expand(B, groups, tokens, C).reshape(B, groups * tokens, C))
        g0 += groups
    return torch.cat(pieces, dim=1)


class CosmosAdaLayerNorm(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.embedding_dim = in_features
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        segments: TimestepSegments | None = None,
    ):
        et = self.linear_2(self.linear_1(self.activation(embedded_timestep)))
        if temb is not None:
            et = et + temb[..., : 2 * self.embedding_dim]
        shift, scale = et.chunk(2, dim=-1)
        x = self.norm(hidden_states)
        if et.ndim == 2:
            shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        elif segments is not None:
            shift, scale = (_per_token(t, segments) for t in (shift, scale))
        return x * (1 + scale) + shift


class CosmosAdaLayerNormZero(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()
        self.linear_1 = (
            nn.Identity()
            if hidden_features is None
            else nn.Linear(in_features, hidden_features, bias=False)
        )
        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        segments: TimestepSegments | None = None,
    ):
        et = self.linear_2(self.linear_1(self.activation(embedded_timestep)))
        if temb is not None:
            et = et + temb
        shift, scale, gate = et.chunk(3, dim=-1)
        x = self.norm(hidden_states)
        if et.ndim == 2:
            shift, scale, gate = (t.unsqueeze(1) for t in (shift, scale, gate))
        elif segments is not None:
            shift, scale, gate = (_per_token(t, segments) for t in (shift, scale, gate))
        return x * (1 + scale) + shift, gate


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
    ):
        super().__init__()
        self.max_size = [s // p for s, p in zip(max_size, patch_size, strict=False)]
        self.patch_size = patch_size
        self.base_fps = base_fps

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(self, hidden_states: torch.Tensor, fps: int | None = None):
        B, C, T, H, W = hidden_states.shape
        pe = [T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]]
        device = hidden_states.device

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)
        h_freqs = 1.0 / (
            (10000.0 * self.h_ntk_factor)
            ** (
                torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[
                    : self.dim_h // 2
                ]
                / self.dim_h
            )
        )
        w_freqs = 1.0 / (
            (10000.0 * self.w_ntk_factor)
            ** (
                torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[
                    : self.dim_w // 2
                ]
                / self.dim_w
            )
        )
        t_freqs = 1.0 / (
            (10000.0 * self.t_ntk_factor)
            ** (
                torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[
                    : self.dim_t // 2
                ]
                / self.dim_t
            )
        )

        emb_h = torch.outer(seq[: pe[1]], h_freqs)[None, :, None, :].repeat(pe[0], 1, pe[2], 1)
        emb_w = torch.outer(seq[: pe[2]], w_freqs)[None, None, :, :].repeat(pe[0], pe[1], 1, 1)
        t_seq = seq[: pe[0]] if fps is None else seq[: pe[0]] / fps * self.base_fps
        emb_t = torch.outer(t_seq, t_freqs)[:, None, None, :].repeat(1, pe[1], pe[2], 1)

        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
        return torch.cos(freqs), torch.sin(freqs)


class MultiViewRoPE(CosmosRotaryPosEmbed):
    """Per-view RoPE: each view gets temporal positions [0, T/V) independently.

    Matches Cosmos Predict 2.5 MultiCameraVideoRopePosition3DEmb:
    computes single-view RoPE for T_per_view frames, then repeats V times.
    n_cameras is a runtime argument (supports cross-embodiment with varying view counts).
    """

    def forward(self, hidden_states: torch.Tensor, fps: int | None = None, n_cameras: int = 1):
        if n_cameras <= 1:
            return super().forward(hidden_states, fps=fps)

        B, C, VT, H, W = hidden_states.shape
        T_per_view = VT // n_cameras

        # Compute RoPE for a single view (T_per_view frames)
        single_view_input = hidden_states[:, :, :T_per_view]
        cos_single, sin_single = super().forward(single_view_input, fps=fps)

        # Each view gets the same [0, T_per_view) positions
        # cos_single shape: (T_per_view * H_patch * W_patch, D)
        # Repeat V times along the sequence dimension
        p_t, p_h, p_w = self.patch_size
        tokens_per_view = (T_per_view // p_t) * (H // p_h) * (W // p_w)
        assert (
            cos_single.shape[0] == tokens_per_view
        ), f"Expected {tokens_per_view} tokens per view, got {cos_single.shape[0]}"

        cos = cos_single.repeat(n_cameras, 1)
        sin = sin_single.repeat(n_cameras, 1)
        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.max_size = [s // p for s, p in zip(max_size, patch_size, strict=False)]
        self.patch_size = patch_size
        self.eps = eps
        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2], hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        pe = [T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]]
        emb = (
            self.pos_emb_t[: pe[0]][None, :, None, None, :].repeat(B, 1, pe[1], pe[2], 1)
            + self.pos_emb_h[: pe[1]][None, None, :, None, :].repeat(B, pe[0], 1, pe[2], 1)
            + self.pos_emb_w[: pe[2]][None, None, None, :, :].repeat(B, pe[0], pe[1], 1, 1)
        )
        emb = emb.flatten(1, 3)
        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


# ─── Transformer Block ───────────────────────────────────────────────────────


class CosmosTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        qk_norm: str = "rms_norm",
        out_bias: bool = False,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)
        self.attn1 = Attention(
            hidden_size, None, num_attention_heads, attention_head_dim, qk_norm, out_bias
        )

        self.norm2 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)
        self.attn2 = MultiViewCrossAttention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            out_bias=out_bias,
        )

        self.norm3 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: tuple | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        n_cameras: int = 1,
        video_idx: torch.Tensor | None = None,
        non_video_idx: torch.Tensor | None = None,
        attention_mask: BlockMask | MaskedAttentionPlan | None = None,
        pad: int = 0,
        compile_flex_attention: bool = True,
        kv_cache: CondKVCache | None = None,
        layer_idx: int = 0,
        store_kv: bool = True,
        text_pad_mask: TextPadMask | None = None,
        timestep_segments: TimestepSegments | None = None,
    ) -> torch.Tensor:
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        norm_hs, gate = self.norm1(hidden_states, embedded_timestep, temb, timestep_segments)
        hidden_states = hidden_states + gate * self.attn1(
            norm_hs,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            pad=pad,
            compile_flex_attention=compile_flex_attention,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            store_kv=store_kv,
        )

        norm_hs, gate = self.norm2(hidden_states, embedded_timestep, temb, timestep_segments)
        # Unmasked export adapters retain their original cross-attention call signature.
        text_mask_kwargs = {} if text_pad_mask is None else {"text_pad_mask": text_pad_mask}
        hidden_states = hidden_states + gate * self.attn2(
            norm_hs,
            encoder_hidden_states=encoder_hidden_states,
            n_cameras=n_cameras,
            video_idx=video_idx,
            non_video_idx=non_video_idx,
            **text_mask_kwargs,
        )

        norm_hs, gate = self.norm3(hidden_states, embedded_timestep, temb, timestep_segments)
        hidden_states = hidden_states + gate * self.ff(norm_hs)

        return hidden_states


# ─── Transformer Config ──────────────────────────────────────────────────────


@dataclass
class CosmosTransformerConfig:
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    num_layers: int = 28
    mlp_ratio: float = 4.0
    text_embed_dim: int = 1024
    adaln_lora_dim: int = 256
    max_size: tuple[int, int, int] = (128, 240, 240)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
    extra_pos_embed_type: str | None = "learnable"
    use_crossattn_projection: bool = False
    crossattn_proj_in_channels: int = 1024
    # Multiview
    max_n_cameras: int = 4
    view_embed_dim: int = 0  # 7 for official AV multiview
    per_view_text: bool = False


COSMOS_2B_CONFIG = CosmosTransformerConfig(
    in_channels=17,  # 16 latent + 1 cond_mask (pre-concatenated by pipeline)
    out_channels=16,
    num_attention_heads=16,
    attention_head_dim=128,
    num_layers=28,
    text_embed_dim=1024,
    adaln_lora_dim=256,
    rope_scale=(1.0, 3.0, 3.0),
    extra_pos_embed_type=None,  # 2B has no learnable pos embed
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
)

COSMOS_14B_CONFIG = CosmosTransformerConfig(
    in_channels=17,
    out_channels=16,
    num_attention_heads=24,
    attention_head_dim=256,
    num_layers=36,
    text_embed_dim=1024,
    adaln_lora_dim=256,
    rope_scale=(2.0, 1.0, 1.0),
    extra_pos_embed_type="learnable",
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
)


# ─── Main Transformer ────────────────────────────────────────────────────────


class CosmosTransformer3DModel(nn.Module):
    """Pure-sequence Cosmos DiT — modality-agnostic (B, L, D) -> (B, L, D).

    Takes pre-packed token sequences and returns processed tokens.
    The caller handles all modality-specific encoding (video patchify via
    VideoTokenizer, action encode, RoPE, packing) before calling forward().

    Weight naming matches HF diffusers checkpoints for direct loading.
    """

    def __init__(self, config: CosmosTransformerConfig):
        super().__init__()
        self.config = config
        self.n_cameras = config.max_n_cameras
        self.per_view_text = config.per_view_text
        hidden_size = config.num_attention_heads * config.attention_head_dim

        # 1. Time embedding (raw sigma -> AdaLN modulation)
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)

        # 2. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    config.num_attention_heads,
                    config.attention_head_dim,
                    config.text_embed_dim,
                    config.mlp_ratio,
                    config.adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 3. Output norm
        self.norm_out = CosmosAdaLayerNorm(hidden_size, config.adaln_lora_dim)

        # 4. Cross-attention projection (optional)
        if config.use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(
                    config.crossattn_proj_in_channels,
                    config.text_embed_dim,
                    bias=True,
                ),
                nn.GELU(),
            )

        self.gradient_checkpointing = False
        self.compile_flex_attention = True
        self.requires_sequence_transform = False
        # dit_text_keys "real" (GrootN2 sets it): a forward must carry the text mask, so no caller
        # attends the prompt's padding unnoticed
        self.require_text_pad_mask = False

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_freqs: tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: BlockMask | MaskedAttentionPlan | None = None,
        pad: int = 0,
        kv_cache: CondKVCache | None = None,
        store_kv: bool = True,
    ) -> torch.Tensor:
        """Pure-sequence transformer forward.

        Args:
            latents: (B, L, D) pre-packed token sequence
            timesteps: (B, L) raw sigma per token (fed to CosmosEmbedding)
            rope_freqs: (cos, sin) each (L, D_head) rotary position embeddings
            context: (B, N, D) text embeddings for cross-attention
            extra_pos_emb: (B, L, D) learnable positional embedding, or None.
                Added at the start of every block (matching Cosmos convention).
            modality_ids: (B, L) int per-token modality. Ignored by Dense blocks;
                used by MoTBlock (Phase 3).
            attention_mask: Optional flex-attention BlockMask.
            pad: Number of padding positions appended for flex BlockMask
                alignment. The transformer blocks consume this internally; no
                padded rows are stored in ``latents`` itself.
            kv_cache: unified KV cache. When empty (len 0), each block's
                self-attention stores its K/V into the cache (capture mode).
                When populated, each block prepends the cached K/V to its
                freshly computed K/V along the sequence axis (splice mode).
            store_kv: If True (default), update the cache with concatenated
                K/V. If False, return the concatenation but leave the cache
                unchanged — used for diffusion action denoising where action
                K/V changes every step but the static prefix is reused.

        Returns:
            (B, L, D) processed token sequence (after norm_out)
        """
        return self._forward_with_routing_indices(
            latents,
            timesteps,
            rope_freqs,
            context,
            modality_ids,
            extra_pos_emb=extra_pos_emb,
            attention_mask=attention_mask,
            pad=pad,
            kv_cache=kv_cache,
            store_kv=store_kv,
            video_token_indices=None,
            non_video_token_indices=None,
        )

    def _forward_with_routing_indices(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_freqs: tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: BlockMask | MaskedAttentionPlan | None = None,
        pad: int = 0,
        kv_cache: CondKVCache | None = None,
        store_kv: bool = True,
        video_token_indices: torch.Tensor | None = None,
        non_video_token_indices: torch.Tensor | None = None,
        sequence_transform: LayerwiseSequenceTransform | None = None,
        text_pad_mask: TextPadMask | None = None,
        timestep_segments: TimestepSegments | None = None,
    ) -> torch.Tensor:
        """Forward path with optional packer-derived routing indices.

        Public callers keep ``modality_ids`` as the source of truth. N2 packed
        training passes the derived indices here to avoid data-dependent
        nonzero()/any() in the compiled training graph. ``text_pad_mask``: the
        cross-attention to each sample's real prompt tokens (dit_text_keys "real").

        ``timestep_segments`` (GrootN2Config.dit_adaln_grouped): consecutive groups of tokens
        sharing one sigma, e.g. ((frames, tokens_per_frame), (state_tokens, 1), (action_tokens,
        1)); the time embedding and every AdaLN modulation are computed once per group and
        broadcast. Not combined with a sequence transform (token merging), whose bands index the
        conditioning per token.
        """
        B, L, D = latents.shape
        if text_pad_mask is not None and self.per_view_text:
            raise ValueError(
                "text_pad_mask masks one prompt per sample; per_view_text has one per view"
            )
        if self.require_text_pad_mask and text_pad_mask is None:
            raise ValueError(
                "dit_text_keys is 'real' but this forward carries no text mask: the caller would "
                "cross-attend the prompt's padding (pass text_pad_mask)"
            )

        # 1. Timestep embeddings from raw sigmas, one row per group of tokens sharing a sigma
        G = L
        if timestep_segments is not None:
            if sequence_transform is not None:
                raise ValueError("timestep_segments does not combine with a sequence transform")
            sizes = [
                groups * tokens for groups, tokens in timestep_segments
            ]  # split raises if != L
            rows = []
            for chunk, (groups, tokens) in zip(
                timesteps.split(sizes, dim=1), timestep_segments, strict=True
            ):
                grouped = chunk.reshape(B, groups, tokens)
                if not torch.compiler.is_compiling():  # eager steps check the contract
                    assert torch.equal(
                        grouped, grouped[:, :, :1].expand_as(grouped)
                    ), "timesteps vary inside a group of timestep_segments"
                rows.append(grouped[:, :, 0])
            timesteps = torch.cat(rows, dim=1)
            G = timesteps.shape[1]
        temb, embedded_timestep = self.time_embed(latents, timesteps.reshape(-1))
        temb = temb.view(B, G, -1)
        embedded_timestep = embedded_timestep.view(B, G, -1)

        # 2. Text projection. Unless text is view-specific, repeating the
        # identical context and splitting video queries per view is equivalent
        # to one cross-attention over all video queries. Avoiding that fan-out
        # removes redundant text K/V projections and small attention kernels.
        text_context = context
        if self.config.use_crossattn_projection:
            text_context = self.crossattn_proj(text_context)
        cross_attn_cameras = self.n_cameras if self.per_view_text else 1

        # 3. Precompute video/non-video indices for multiview cross-attention
        #    Same layout for every block — compute once, pass through.
        # Both MODALITY_VIDEO (0) and MODALITY_VIDEO_COND (1) are video tokens
        video_idx, non_video_idx = _video_non_video_indices(
            modality_ids,
            latents.device,
            video_token_indices,
            non_video_token_indices,
        )
        if (
            self.requires_sequence_transform
            and sequence_transform is None
            and video_idx.numel() > 0
        ):
            raise ValueError("token merge full-sequence forward requires sequence_transform")

        # 4. Transformer block loop
        if (
            kv_cache is not None
            and len(kv_cache) > 0
            and len(kv_cache) != len(self.transformer_blocks)
        ):
            raise ValueError(
                f"kv_cache has {len(kv_cache)} layers but transformer has "
                f"{len(self.transformer_blocks)}"
            )
        if sequence_transform is not None and kv_cache is not None and len(kv_cache) > 0:
            # Splice-mode forwards contain only the live suffix, so a transform
            # defined over the original full sequence cannot be applied safely.
            raise ValueError("sequence_transform cannot ride a cached (splice-mode) forward")
        transform_state = transform_saved = None
        for layer_idx, block in enumerate(self.transformer_blocks):
            if sequence_transform is not None and layer_idx == sequence_transform.end_layer:
                latents = sequence_transform.restore_sequence(latents, transform_state)
                (
                    rope_freqs,
                    temb,
                    embedded_timestep,
                    extra_pos_emb,
                    attention_mask,
                    pad,
                    video_idx,
                    non_video_idx,
                ) = transform_saved
            if sequence_transform is not None and layer_idx == sequence_transform.start_layer:
                latents, transform_state = sequence_transform.reduce_sequence(latents)
                transform_saved = (
                    rope_freqs,
                    temb,
                    embedded_timestep,
                    extra_pos_emb,
                    attention_mask,
                    pad,
                    video_idx,
                    non_video_idx,
                )
                idx = sequence_transform.reduced_token_indices
                rope_freqs = (rope_freqs[0][idx], rope_freqs[1][idx])
                temb = temb[:, idx]
                embedded_timestep = embedded_timestep[:, idx]
                if extra_pos_emb is not None:
                    extra_pos_emb = extra_pos_emb[:, idx]
                attention_mask = sequence_transform.reduced_attention_mask
                pad = sequence_transform.reduced_pad
                video_idx = sequence_transform.reduced_video_token_indices
                non_video_idx = sequence_transform.reduced_non_video_token_indices
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                latents = torch_checkpoint(
                    block,
                    latents,
                    text_context,
                    embedded_timestep,
                    temb,
                    rope_freqs,
                    extra_pos_emb,
                    cross_attn_cameras,
                    video_idx,
                    non_video_idx,
                    attention_mask,
                    pad,
                    self.compile_flex_attention,
                    text_pad_mask=text_pad_mask,
                    timestep_segments=timestep_segments,
                    use_reentrant=False,
                )
            else:
                latents = block(
                    latents,
                    text_context,
                    embedded_timestep,
                    temb,
                    rope_freqs,
                    extra_pos_emb,
                    cross_attn_cameras,
                    video_idx,
                    non_video_idx,
                    attention_mask,
                    pad,
                    self.compile_flex_attention,
                    kv_cache,
                    layer_idx,
                    store_kv,
                    text_pad_mask=text_pad_mask,
                    timestep_segments=timestep_segments,
                )

        if sequence_transform is not None and sequence_transform.end_layer == len(
            self.transformer_blocks
        ):
            # The in-loop restore never fires after the last block; norm_out
            # still requires the original sequence layout.
            latents = sequence_transform.restore_sequence(latents, transform_state)
            temb, embedded_timestep = transform_saved[1], transform_saved[2]

        # 5. Final norm (AdaLN applied to all tokens)
        latents = self.norm_out(latents, embedded_timestep, temb, timestep_segments)

        return latents

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
