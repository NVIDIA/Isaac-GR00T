# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

import torch

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_2_AVAILABLE = False

import warnings


def scaled_dot_product_attention_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Run export SDPA without widening the strongly typed BF16 graph.

    TensorRT parity is qualified against the production eager oracle, while
    decoded-action and closed-loop gates protect behavior at the serving
    boundary. Keeping Q/K/V in BF16 avoids introducing an FP32 attention island
    that materially regresses the promoted Turbo2 engine on RTX 5090.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        **kwargs,
    )


def flash_attn_func_v23(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    flatten: bool = True,
):
    """Run FlashAttention over packed or padded Q/K/V tensors.

    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        # torch.full on-device instead of torch.tensor(...).to(): the scalar
        # rides as a kernel argument, so no CPU->GPU copy -- which is also what
        # makes this legal under CUDA graph capture.
        q_lens = torch.full((b,), lq, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens, strict=False)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.full((b,), lk, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens, strict=False)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens, strict=False)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            "Flash attention 3 is not available, use flash attention 2 instead.",
            stacklevel=2,
        )

    # Check for TensorRT at runtime, not import time
    if os.getenv("ENABLE_TENSORRT", "False").lower() == "true":
        # use torch.nn.functional.scaled_dot_product_attention for tensorrt export

        # The input is (s, n, d), but sdpa needs (b, n, s, d).
        # We add a batch dimension and transpose.
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        # Fix for ONNX export: repeat k and v to match q's batch size in cross-attention
        if q.shape[0] != k.shape[0] and k.shape[0] == 1:
            k = k.repeat(q.shape[0], 1, 1, 1)
            v = v.repeat(q.shape[0], 1, 1, 1)

        attn_mask = None
        out = scaled_dot_product_attention_bf16(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        # Transpose back to (b, s, n, d) format.
        out = out.transpose(1, 2).contiguous()
        return out

    elif (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    if flatten:
        x = x.reshape(b, lq, -1)
    return x.type(out_dtype)
