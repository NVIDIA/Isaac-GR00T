# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Moonlight matrix updates over DeepSpeed ZeRO-2 FP32 optimizer partitions.

Ported from ``wj/0823_herorun_launch`` (2c0b519562b, 5d17966c854): the optimizer the
hero-run stage-2 Muon checkpoints were trained with (``muon_impl="moonlight_zero2"``).

The BF16 momentum gather follows Moonshot's distributed PoC (Megatron-LM
PR 1428, f432fbe). Each rank reconstructs only the matrices intersecting its
partition; DeepSpeed owns gradient reduction, clipping and parameter gathering.
This implementation supports data parallelism without tensor parallelism or
round-robin parameter reordering.

Optimizer state is saved and restored through DeepSpeed's non-elastic path
(torch's ``load_state_dict`` per group). The groups carry different state keys
(Adam: exp_avg/exp_avg_sq/step, Muon: momentum_buffer), which DeepSpeed's
``elastic_checkpoint=True`` restore does not handle; keep it off.
"""

from collections.abc import Iterable
import math
import os

import torch
import torch.distributed as dist

from groot.core.utils.muon import adam_update, zeropower_via_newtonschulz5

_newton_schulz = torch.compile(zeropower_via_newtonschulz5)


def moonlight_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    muon_param_ids: set[int],
    decay_parameter_names: Iterable[str],
    weight_decay: float,
) -> list[dict]:
    """Split trainable parameters into {AdamW, Muon} x {no decay, decay} groups.

    ``muon_param_ids`` holds ``id(p)`` of the 2-D matrices Muon updates; every other
    trainable parameter goes to AdamW. Empty groups are dropped.
    """
    trainable = [(name, p) for name, p in named_parameters if p.requires_grad]
    decay = set(decay_parameter_names)
    groups = []
    for use_muon in (False, True):
        for use_decay in (False, True):
            params = [
                p
                for name, p in trainable
                if (id(p) in muon_param_ids) == use_muon and (name in decay) == use_decay
            ]
            if params:
                groups.append(
                    {
                        "params": params,
                        "use_muon": use_muon,
                        "weight_decay": weight_decay if use_decay else 0.0,
                    }
                )
    return groups


class MoonlightZeroOptimizer(torch.optim.Optimizer):
    def __init__(self, param_groups, *, lr, betas, eps):
        self._original_groups = {}
        self._checked_layouts = set()
        # One Newton-Schulz per matrix shape instead of per matrix (GROOT_MUON_BATCHED_NS=0
        # restores per-matrix). Each rank orthogonalizes every matrix intersecting its partition;
        # with few ranks that is many matrices (70 on one GB300 node, ~165 ms of host time per step
        # in eager chains), and same-shape ones stack into one batched iteration.
        self._batched_ns = os.environ.get("GROOT_MUON_BATCHED_NS", "1") != "0"
        for index, group in enumerate(param_groups):
            group["moonlight_group"] = index
            group["moonlight_shapes"] = [tuple(p.shape) for p in group["params"]]
            self._original_groups[index] = list(group["params"])
            if group["use_muon"]:
                assert all(p.ndim == 2 for p in group["params"])
        # momentum 0.95: Moonlight's default (groot.core.utils.muon.muon_update uses the same).
        super().__init__(param_groups, {"lr": lr, "betas": betas, "eps": eps, "momentum": 0.95})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        world = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            index = group["moonlight_group"]
            shapes = group["moonlight_shapes"]
            assert len(group["params"]) == 1, "Expected one flattened ZeRO partition per group"
            param = group["params"][0]
            assert param.ndim == 1 and param.dtype == torch.float32
            size = param.numel()
            start = rank * size
            end = start + size
            if index not in self._checked_layouts:
                # DeepSpeed replaced group["params"] with the flat fp32 partition, so the
                # layout (original tensors contiguous and in order) can only be verified here.
                originals = self._original_groups[index]
                base_ptr = originals[0].data_ptr()
                offset = 0
                for original, shape in zip(originals, shapes, strict=True):
                    assert tuple(original.shape) == tuple(shape)
                    assert (
                        original.data_ptr() - base_ptr == offset * original.element_size()
                    ), "Moonlight requires contiguous, unreordered ZeRO groups"
                    offset += original.numel()
                # ZeRO pads each group to a multiple of 2 * world (nccl_start_alignment_factor = 2).
                expected = math.ceil(offset / (2 * world)) * 2
                assert size == expected, (size, expected)
                self._checked_layouts.add(index)
            grad = param.grad
            assert grad is not None, "ZeRO must provide the clipped gradient partition"
            state = self.state[param]
            param.mul_(1 - group["lr"] * group["weight_decay"])
            if not group["use_muon"]:
                if not state:
                    state.update(
                        exp_avg=torch.zeros_like(param), exp_avg_sq=torch.zeros_like(param), step=0
                    )
                state["step"] += 1
                update = adam_update(
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group["betas"],
                    group["eps"],
                )
                param.add_(update, alpha=-group["lr"])
                continue
            if not state:
                state["momentum_buffer"] = torch.zeros_like(param)
            buffer = state["momentum_buffer"]
            buffer.mul_(group["momentum"]).add_(grad)
            local_input = grad.add(buffer, alpha=group["momentum"]).bfloat16()
            full_input = torch.empty(size * world, device=param.device, dtype=torch.bfloat16)
            dist.all_gather_into_tensor(full_input, local_input)
            # Matrices intersecting this rank's partition [start, end): (offset, lo, hi) with the
            # matrix's global range [offset, offset + count); only the overlap is written back.
            mine: dict[tuple, list[tuple[int, int, int]]] = {}
            offset = 0
            for shape in shapes:
                count = math.prod(shape)
                lo, hi = max(start, offset), min(end, offset + count)
                if lo < hi:
                    mine.setdefault(tuple(shape), []).append((offset, lo, hi))
                offset += count
            for shape, items in mine.items():
                count = math.prod(shape)
                # Moonlight: scale the orthogonalized update so its RMS matches AdamW's (~0.2),
                # which lets one learning rate serve both parameter groups.
                scale = 0.2 * math.sqrt(max(shape))
                mats = [full_input[o : o + count].view(shape) for o, _, _ in items]
                if self._batched_ns and len(mats) > 1:
                    updates = _newton_schulz(torch.stack(mats), steps=5)
                else:  # a group of one stays 2D: bitwise the per-matrix path
                    updates = [_newton_schulz(m, steps=5) for m in mats]
                for (o, lo, hi), update in zip(items, updates, strict=True):
                    param[lo - start : hi - start].add_(
                        update.reshape(-1)[lo - o : hi - o], alpha=-group["lr"] * scale
                    )
        return loss
