# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""cuDNN fused attention for the DiT's masked (video-action) attention, as custom ops.

``cudnn_prefix_sdpa``: the padded layout, q (B, Sq, H, D), k/v (B, Skv, H, D); each sample's
valid queries and keys are a prefix of its row and only the lengths vary. cuDNN takes the two
``(B,)`` int32 lengths as its padding mask, masks the padded key columns, zeroes the padded query
rows (output and dq) when ``seq_len_q`` is limited and skips fully masked tiles.
``F.scaled_dot_product_attention`` exposes no such mask, and its boolean ``attn_mask`` path
materializes an L x L bias. The kernel uses the ``nvidia-cudnn-frontend`` Python package (module
``cudnn``), which the training images ship and no extra declares. Validated with 1.11 (the H100
image) to 1.15; 1.31 segfaults in ``create_execution_plans`` with torch 2.7.1's cuDNN 9.7.1 on sm120.

Shapes are static; lengths are tensor data, so the op traces under torch.compile and replays
inside CUDA graphs. The cuDNN graphs are built once per shape.
"""

from __future__ import annotations

import torch

_HANDLES: dict[int, object] = {}


def _cudnn():
    try:
        import cudnn  # nvidia-cudnn-frontend; lazy so CPU-only imports stay cheap
    except ImportError as exc:  # the training image ships it; a bare venv may not
        raise ImportError(
            "dit_masked_attention='cudnn' needs the nvidia-cudnn-frontend package (module cudnn)"
        ) from exc
    return cudnn


def _handle(device: torch.device):
    cudnn = _cudnn()
    if device.index not in _HANDLES:
        with torch.cuda.device(device):
            _HANDLES[device.index] = cudnn.create_handle()
    handle = _HANDLES[device.index]
    # per call, not at creation: CUDA-graph capture runs the op on a side stream
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


# --- prefix (padded, per sample) ---------------------------------------------------------------

_PREFIX_GRAPHS: dict[tuple, tuple] = {}


def _bshd(h: int, s: int, d: int, batch_stride: int | None = None) -> tuple[int, int, int, int]:
    """Strides of a logical (B, H, S, D) tensor stored (B, S, H, D) with rows contiguous.

    ``batch_stride`` > S*H*D describes a sequence slice ``x[:, a:b]`` of a longer tensor, which
    cuDNN reads in place; the default is the contiguous tensor.
    """
    return (batch_stride if batch_stride is not None else s * h * d, d, h * d, 1)


def _row_major(x: torch.Tensor) -> bool:
    """(B, S, H, D) with contiguous rows: stride (>= S*H*D, H*D, D, 1)."""
    _, s, h, d = x.shape
    return x.stride() == (x.stride(0), h * d, d, 1) and x.stride(0) >= s * h * d


def _prefix_graphs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """(fwd graph, fwd tensors, fwd workspace bytes, bwd graph, bwd tensors, bwd workspace bytes).

    Workspaces are allocated per call, not cached: a buffer allocated during a CUDA-graph
    warmup would live in the graph pool without being a graph output (cudagraph trees
    reject that), and the caching allocator makes the per-call allocation free.
    """
    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    strides = (q.stride(0), k.stride(0), v.stride(0))
    key = (B, Sq, Skv, H, D, q.dtype, q.device.index, strides)
    if key in _PREFIX_GRAPHS:
        return _PREFIX_GRAPHS[key]
    cudnn = _cudnn()
    handle = _handle(q.device)
    cap = torch.cuda.get_device_capability(q.device)
    if q.dtype != torch.bfloat16:
        raise ValueError(f"cudnn_prefix_sdpa takes bf16 tensors, got {q.dtype}")
    io_type = cudnn.data_type.BFLOAT16
    scale = D**-0.5

    def new_graph():
        return cudnn.pygraph(
            io_data_type=io_type,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=handle,
            sm_version=cap[0] * 10 + cap[1],
        )

    batch_stride = {"q": strides[0], "k": strides[1], "v": strides[2]}

    def declare(g, names_lengths):
        out = {}
        for name, s in names_lengths:
            out[name] = g.tensor(
                name=name,
                dim=(B, H, s, D),
                stride=_bshd(H, s, D, batch_stride.get(name)),
                data_type=io_type,
            )
        for name in ("seq_len_q", "seq_len_kv"):
            out[name] = g.tensor(
                name=name, dim=(B, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32
            )
        return out

    def finish(g):
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()
        return max(g.get_workspace_size(), 1)  # torch.empty(0) has no data pointer to pass

    gf = new_graph()
    tf = declare(gf, (("q", Sq), ("k", Skv), ("v", Skv)))
    sdpa_args = {
        "name": "sdpa",
        "q": tf["q"],
        "k": tf["k"],
        "v": tf["v"],
        "attn_scale": scale,
        "use_padding_mask": True,
        "seq_len_q": tf["seq_len_q"],
        "seq_len_kv": tf["seq_len_kv"],
    }
    try:
        o, stats = gf.sdpa(generate_stats=True, **sdpa_args)
    except TypeError:
        # nvidia-cudnn-frontend releases before generate_stats (e.g. 1.11) name the same
        # switch is_inference=False
        o, stats = gf.sdpa(is_inference=False, **sdpa_args)
    o.set_output(True).set_dim((B, H, Sq, D)).set_stride(_bshd(H, Sq, D)).set_data_type(io_type)
    stats.set_output(True).set_dim((B, H, Sq, 1)).set_stride((H * Sq, Sq, 1, 1)).set_data_type(
        cudnn.data_type.FLOAT
    )
    tf["o"], tf["stats"] = o, stats
    wf = finish(gf)

    gb = new_graph()
    tb = declare(gb, (("q", Sq), ("k", Skv), ("v", Skv), ("o", Sq), ("dO", Sq)))
    tb["stats"] = gb.tensor(
        name="stats", dim=(B, H, Sq, 1), stride=(H * Sq, Sq, 1, 1), data_type=cudnn.data_type.FLOAT
    )
    dq, dk, dv = gb.sdpa_backward(
        name="sdpa_bwd",
        q=tb["q"],
        k=tb["k"],
        v=tb["v"],
        o=tb["o"],
        dO=tb["dO"],
        stats=tb["stats"],
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=tb["seq_len_q"],
        seq_len_kv=tb["seq_len_kv"],
    )
    for t, s in ((dq, Sq), (dk, Skv), (dv, Skv)):
        t.set_output(True).set_dim((B, H, s, D)).set_stride(_bshd(H, s, D)).set_data_type(io_type)
    tb["dq"], tb["dk"], tb["dv"] = dq, dk, dv
    wb = finish(gb)
    _PREFIX_GRAPHS[key] = (gf, tf, wf, gb, tb, wb)
    return _PREFIX_GRAPHS[key]


@torch.library.custom_op("groot::cudnn_prefix_sdpa_fwd", mutates_args=())
def cudnn_prefix_sdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix-masked attention; q (B, Sq, H, D), k/v (B, Skv, H, D) row-major, seq_len_* (B,) int32.

    Returns (o (B, Sq, H, D), stats (B, H, Sq, 1) fp32 logsumexp for the backward).
    """
    B, Sq, H, _ = q.shape
    gf, tf, workspace_bytes, _, _, _ = _prefix_graphs(q, k, v)
    handle = _handle(q.device)
    workspace = torch.empty(workspace_bytes, device=q.device, dtype=torch.uint8)
    o = torch.empty_like(q)
    stats = torch.empty(B, H, Sq, 1, device=q.device, dtype=torch.float32)
    gf.execute(
        {
            tf["q"]: q,
            tf["k"]: k,
            tf["v"]: v,
            tf["seq_len_q"]: seq_len_q.view(B, 1, 1, 1),
            tf["seq_len_kv"]: seq_len_kv.view(B, 1, 1, 1),
            tf["o"]: o,
            tf["stats"]: stats,
        },
        workspace,
        handle=handle,
    )
    return o, stats


@cudnn_prefix_sdpa_fwd.register_fake
def _prefix_fwd_fake(q, _k, _v, _seq_len_q, _seq_len_kv):
    B, Sq, H, _ = q.shape
    return torch.empty_like(q), q.new_empty((B, H, Sq, 1), dtype=torch.float32)


@torch.library.custom_op("groot::cudnn_prefix_sdpa_bwd", mutates_args=())
def cudnn_prefix_sdpa_bwd(  # the op schema lists the cuDNN graph's tensors
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_o: torch.Tensor,
    stats: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = q.shape[0]
    _, _, _, gb, tb, workspace_bytes = _prefix_graphs(q, k, v)
    handle = _handle(q.device)
    workspace = torch.empty(workspace_bytes, device=q.device, dtype=torch.uint8)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    gb.execute(
        {
            tb["q"]: q,
            tb["k"]: k,
            tb["v"]: v,
            tb["o"]: o,
            tb["dO"]: grad_o,
            tb["stats"]: stats,
            tb["seq_len_q"]: seq_len_q.view(B, 1, 1, 1),
            tb["seq_len_kv"]: seq_len_kv.view(B, 1, 1, 1),
            tb["dq"]: dq,
            tb["dk"]: dk,
            tb["dv"]: dv,
        },
        workspace,
        handle=handle,
    )
    return dq, dk, dv


@cudnn_prefix_sdpa_bwd.register_fake
def _prefix_bwd_fake(q, k, v, _o, _grad_o, _stats, _seq_len_q, _seq_len_kv):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _prefix_setup_context(ctx, inputs, output):
    q, k, v, seq_len_q, seq_len_kv = inputs
    o, stats = output
    ctx.save_for_backward(q, k, v, o, stats, seq_len_q, seq_len_kv)


def _prefix_backward(ctx, grad_o, _grad_stats):
    q, k, v, o, stats, seq_len_q, seq_len_kv = ctx.saved_tensors
    dq, dk, dv = torch.ops.groot.cudnn_prefix_sdpa_bwd(
        q, k, v, o, grad_o.contiguous(), stats, seq_len_q, seq_len_kv
    )
    return dq, dk, dv, None, None


cudnn_prefix_sdpa_fwd.register_autograd(_prefix_backward, setup_context=_prefix_setup_context)


def cudnn_prefix_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
) -> torch.Tensor:
    """Attention of the first ``seq_len_q[b]`` queries over the first ``seq_len_kv[b]`` keys.

    q (B, Sq, H, D), k/v (B, Skv, H, D) bf16, contiguous or sequence slices ``x[:, a:b]`` of a
    contiguous tensor (read in place); rows past the valid prefix come back as zeros and
    receive zero gradient.
    """
    # load-bearing: the cuDNN graph bakes in the strides of _bshd; a non-row-major tensor
    # would be read silently wrong, so it is copied. Never taken by the DiT's inputs.
    q, k, v = (t if _row_major(t) else t.contiguous() for t in (q, k, v))
    o, _ = torch.ops.groot.cudnn_prefix_sdpa_fwd(q, k, v, seq_len_q, seq_len_kv)
    return o
