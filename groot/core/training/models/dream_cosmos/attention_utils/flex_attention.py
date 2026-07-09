# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Sequence
import inspect
import math

from einops import rearrange
import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention as torch_flex_attention,
    or_masks,
)

# Check kernel_options support BEFORE compiling (compiled wrapper may not preserve signature).
_HAS_KERNEL_OPTIONS = "kernel_options" in inspect.signature(torch_flex_attention).parameters
compiled_flex_attention = torch.compile(torch_flex_attention)
create_block_mask = torch.compile(create_block_mask)
BLOCK_SIZE = 128

# Auto-detect GPU shared memory and set kernel_options for flex_attention.
# Desktop GPUs (RTX) have ~99KB shared memory per SM — the default Triton
# tile size (128) requires ~110KB and fails. Force 64x64 tiles on those GPUs.
_FLEX_KERNEL_OPTIONS: dict = {}
if torch.cuda.is_available() and _HAS_KERNEL_OPTIONS:
    try:
        _shmem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        if _shmem < 110_000:  # ~110KB threshold matches the observed failure
            _FLEX_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64}
    except AttributeError:
        pass


def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Scaled dot-product attention with optional mask.

    Inputs are shaped [B, S, H, D]. If flatten_heads=True, flattens heads to return [B, S, H*D].
    Otherwise returns [B, S, H, D].
    """
    q_B_H_S_D = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k_B_H_S_D = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v_B_H_S_D = rearrange(v_B_S_H_D, "b s h d -> b h s d")
    result_B_H_S_D = torch.nn.functional.scaled_dot_product_attention(
        q_B_H_S_D, k_B_H_S_D, v_B_H_S_D, attn_mask=attn_mask
    )
    if flatten_heads:
        return rearrange(result_B_H_S_D, "b h s d -> b s (h d)")
    else:
        return rearrange(result_B_H_S_D, "b h s d -> b s h d")


def flex_attention_op(
    q: torch.Tensor,  # (B, S, H, D)
    k: torch.Tensor,  # (B, S, H, D)
    v: torch.Tensor,  # (B, S, H, D)
    attn_mask: BlockMask | None = None,
    pad: int = 0,
    flatten_heads: bool = True,
    compile_attention: bool = True,
) -> torch.Tensor:
    # Rearrange to [B, H, S, D]
    q = rearrange(q, "b s h d -> b h s d")
    k = rearrange(k, "b s h d -> b h s d")
    v = rearrange(v, "b s h d -> b h s d")

    B, H, S, D = q.shape
    if pad > 0:
        pad_tensor = torch.zeros((B, H, pad, D), device=q.device, dtype=q.dtype)
        q = torch.cat([q, pad_tensor], dim=2)
        k = torch.cat([k, pad_tensor], dim=2)
        v = torch.cat([v, pad_tensor], dim=2)

    kwargs = {"query": q, "key": k, "value": v, "block_mask": attn_mask}
    if _FLEX_KERNEL_OPTIONS:
        kwargs["kernel_options"] = _FLEX_KERNEL_OPTIONS
    attention_fn = compiled_flex_attention if compile_attention else torch_flex_attention
    out = attention_fn(**kwargs)
    out = out[:, :, :S] if pad > 0 else out
    if flatten_heads:
        return rearrange(out, "b h s d -> b s (h d)")
    else:
        return rearrange(out, "b h s d -> b s h d")


def create_sparse_mask(split_lens, attn_modes, device):
    """create sparse attention mask for flex attention for block causal attention
    Inputs:
        split_lens: list[int] - sequence lengths of each split
        attn_modes: list[str] - attention modes for each split ("full", "noise", "zero")
            - "full" blocks can attend to all previous "full" and "noise" blocks (causal)
            - "noise" blocks can attend to all previous "full" blocks (causal)
            - "zero" blocks cannot attend to any block (including themselves)
        device: torch.device
    Returns:
        flex_attention.mask_mod
    Example.
    >>> split_lens = [2, 2, 2, 2, 2]
    >>> attn_modes = ["full", "noise", "full", "noise", "zero"]
    >>> mask = create_sparse_mask(split_lens, attn_modes, device)
    >>> print_mask(mask, sum(split_lens), sum(split_lens))
    >>> 1100000000
        1100000000
        1111000000
        1111000000
        1100110000
        1100110000
        1100111100
        1100111100
        0000000000
        0000000000
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (
            full_and_noise_seq_id[q_idx] >= 0
        )

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    def padding_mask(b, h, q_idx, kv_idx):
        return zero_id[q_idx] & zero_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []
    zero_tmp = []  # NEW

    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        # original logic for full/noise groupings
        value = i if mode in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)

        value_noise = i if mode == "noise" else -1
        noise_tmp.extend([value_noise] * length)

        # NEW: padding eligibility
        zero_tmp.extend([mode != "zero"] * length)

    full_and_noise_seq_id = torch.tensor(full_and_noise_tmp, dtype=torch.int32, device=device)
    noise_seq_id = torch.tensor(noise_tmp, dtype=torch.int32, device=device)
    zero_id = torch.tensor(zero_tmp, dtype=torch.bool, device=device)
    document_lens = [sum(split_lens)]
    document_id = torch.cat(
        [
            torch.full((seq_len,), i, dtype=torch.int32, device=device)
            for i, seq_len in enumerate(document_lens, start=1)
        ],
        dim=0,
    )

    base = and_masks(
        or_masks(causal_mask, full_and_noise_mask),
        remove_noise_mask,
        sample_mask,
    )

    return and_masks(base, padding_mask)


def print_mask(mask_mod, len_q, len_kv):
    """utility function to visualize the mask"""

    def _fn(q, len_kv):
        mask = [
            mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(k))
            for k in range(len_kv)
        ]
        mask = [int(m) for m in mask]
        return mask

    for q in range(len_q):
        row = _fn(q, len_kv)
        print(f"{''.join(map(str, row))}")


# @functools.lru_cache(maxsize=32)
def prepare_flex_attn_mask(
    frame_seqlen: int,
    n_frame_per_chunk: int,
    n_action_per_chunk: int,
    n_state_per_chunk: int,
    n_frames: int,
    device: torch.device | str,
    block_size: int = 128,
):
    split_lens = []
    attn_modes = []
    n_chunk = (n_frames - 1) // n_frame_per_chunk
    for i in range(n_chunk + 1):
        if i == 0:
            split_lens += [frame_seqlen, frame_seqlen]
            attn_modes += ["noise", "full"]
        else:
            split_lens += [
                n_state_per_chunk + n_action_per_chunk + frame_seqlen * n_frame_per_chunk,
                frame_seqlen * n_frame_per_chunk,
            ]
            attn_modes += ["noise", "full"]

    # first create mask mod with padding to ensure block size compatibility
    seqlen = sum(split_lens)
    pad = ((seqlen + block_size - 1) // block_size) * block_size - seqlen
    if pad > 0:
        split_lens += [pad]
        attn_modes += ["zero"]
    mask_mod = create_sparse_mask(split_lens, attn_modes, device)
    attn_mask = create_block_mask(
        mask_mod,
        B=1,
        H=1,
        Q_LEN=seqlen + pad,
        KV_LEN=seqlen + pad,
        BLOCK_SIZE=block_size,
        device=device,
    )
    attn_kwargs = {"attn_mask": attn_mask, "pad": pad}
    return attn_kwargs


def create_sparse_mask_tensor(
    split_lens: Sequence[int],
    attn_modes: Sequence[str],
    device: torch.device | str,
) -> torch.Tensor:
    if len(split_lens) != len(attn_modes):
        raise ValueError(
            f"len(split_lens) != len(attn_modes): {len(split_lens)} vs {len(attn_modes)}"
        )

    S_split = int(sum(split_lens))

    full_and_noise_tmp = []
    noise_tmp = []
    pad_ok_tmp = []

    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        full_and_noise_tmp.extend([i if mode in ["full", "noise"] else -1] * int(length))
        noise_tmp.extend([i if mode == "noise" else -1] * int(length))
        pad_ok_tmp.extend([mode != "zero"] * int(length))

    full_and_noise_seq_id = torch.tensor(
        full_and_noise_tmp, dtype=torch.int32, device=device
    )  # (S,)
    noise_seq_id = torch.tensor(noise_tmp, dtype=torch.int32, device=device)  # (S,)
    pad_ok = torch.tensor(pad_ok_tmp, dtype=torch.bool, device=device)  # (S,)

    document_id = torch.cat(
        [
            torch.full((int(seq_len),), i, dtype=torch.int32, device=device)
            for i, seq_len in enumerate([S_split], start=1)
        ],
        dim=0,
    )  # (S,)

    # ---- Vectorized mask construction (S, S) ----
    q = torch.arange(S_split, device=device)
    k = torch.arange(S_split, device=device)
    q_idx = q[:, None]  # (S,1)
    k_idx = k[None, :]  # (1,S)

    # causal_mask: q_idx >= kv_idx
    causal = q_idx >= k_idx

    # full_and_noise_mask:
    # (full_and_noise_seq_id[q] == full_and_noise_seq_id[k]) & (full_and_noise_seq_id[q] >= 0)
    fn_q = full_and_noise_seq_id[:, None]  # (S,1)
    fn_k = full_and_noise_seq_id[None, :]  # (1,S)
    full_and_noise = (fn_q == fn_k) & (fn_q >= 0)

    # remove_noise_mask:
    # ~( (noise_seq_id[k] >= 0) & (noise_seq_id[q] != noise_seq_id[k]) )
    nz_q = noise_seq_id[:, None]  # (S,1)
    nz_k = noise_seq_id[None, :]  # (1,S)
    remove_noise = ~((nz_k >= 0) & (nz_q != nz_k))

    # sample_mask: document_id[q] == document_id[k]
    doc_q = document_id[:, None]
    doc_k = document_id[None, :]
    same_doc = doc_q == doc_k

    # padding_mask: pad_ok[q] & pad_ok[k]
    pad_q = pad_ok[:, None]
    pad_k = pad_ok[None, :]
    padding = pad_q & pad_k

    mask = (causal | full_and_noise) & remove_noise & same_doc & padding  # (S,S), bool
    return mask


def eager_masked_attention_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Eager scaled dot-product attention with a boolean mask."""
    assert q.dim() == k.dim() == v.dim() == 4, "q/k/v must be (B, S, H, D)"
    B, S_q, H, D = q.shape
    Bk, S_k, Hk, Dk = k.shape
    Bv, S_v, Hv, Dv = v.shape
    assert B == Bk == Bv
    assert H == Hk == Hv
    assert D == Dk
    assert S_k == S_v
    assert Dv == D, "this implementation assumes v has same last dim as q/k"
    assert attn_mask.shape == (S_q, S_k)
    assert attn_mask.dtype == torch.bool

    # Compute logits: (B, H, S_q, S_k)
    q_bhsd = q.permute(0, 2, 1, 3)  # (B, H, S_q, D)
    k_bhsd = k.permute(0, 2, 1, 3)  # (B, H, S_k, D)
    v_bhsd = v.permute(0, 2, 1, 3)  # (B, H, S_k, D)

    logits = torch.matmul(q_bhsd, k_bhsd.transpose(-1, -2)) / math.sqrt(D)  # (B,H,S_q,S_k)

    # Apply mask (broadcast to B,H): disallowed positions -> -inf
    logits = logits.masked_fill(~attn_mask.view(1, 1, S_q, S_k), float("-inf"))

    attn_weights = torch.softmax(logits, dim=-1)  # (B,H,S_q,S_k)

    # Output: (B, H, S_q, D) -> (B, S_q, H, D)
    out_bhsd = torch.matmul(attn_weights, v_bhsd)  # (B,H,S_q,D)
    out = out_bhsd.permute(0, 2, 1, 3).contiguous()
    if flatten_heads:
        out = rearrange(out, "b s h d -> b s (h d)")  # (B, S_q, H*D)

    return out
