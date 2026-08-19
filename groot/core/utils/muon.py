# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright (c) 2024 Keller Jordan

import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps: int):
    """Compute the zeroth power or orthogonalization of G with a Newton-Schulz iteration.

    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing
    the slope at zero even beyond the point where the iteration no longer converges all the way to
    one everywhere on the interval. This iteration therefore does not produce UV^T but rather
    something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not
    to hurt model performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, batched=False):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == (5 if batched else 4):  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                dist.get_world_size() - len(params) % dist.get_world_size()
            )
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon variant for usage in non-distributed settings."""

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Distributed Muon variant that can optimize all network parameters.

    It runs an internal AdamW for the parameters that are not compatible with Muon. The user must
    manually specify which parameters shall be optimized with Muon and which with Adam by passing in
    a list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/a949398b58045fb1660aff402e3c280109654839/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [
        dict(params=head_params, lr=0.22),
        dict(params=embed_params, lr=0.6),
        dict(params=scalar_params, lr=0.04),
    ]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad, state["momentum_buffer"], beta=group["momentum"]
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed variant of MuonWithAuxAdam."""

    def __init__(self, param_groups, use_batching=False):
        self.use_batching = use_batching
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
        super().__init__(param_groups, {})

    def _batched_muon_group_step(self, group):
        # Group parameters by their shape and dtype, ordered descending by numel
        from collections import defaultdict

        params_by_shape_dtype = defaultdict(lambda: {"params": [], "batched_momentum_buffer": None})
        for p in group["params"]:
            key = (tuple(p.shape), p.dtype)
            params_by_shape_dtype[key]["params"].append(p)

        if "params_by_shape_dtype" in group:
            # try to reused batched momenta if possible
            for key, params_info in params_by_shape_dtype.items():
                if key in group["params_by_shape_dtype"]:
                    params_info["batched_momentum_buffer"] = group["params_by_shape_dtype"][key][
                        "batched_momentum_buffer"
                    ]

        group["params_by_shape_dtype"] = dict(params_by_shape_dtype)

        for _key, params_info in params_by_shape_dtype.items():
            params = params_info["params"]
            batched_momenta = params_info["batched_momentum_buffer"]
            # go through batch momenta initialize them if not present yet

            needs_reinit = batched_momenta is None

            for p in params:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    needs_reinit = True

            if needs_reinit:
                batched_momenta = torch.stack([self.state[p]["momentum_buffer"] for p in params])
                for i, p in enumerate(params):
                    self.state[p]["momentum_buffer"] = batched_momenta[i]
                params_info["batched_momentum_buffer"] = batched_momenta

            # 1. form a batched tensor of params, momenta, and grads
            batched_grads = torch.stack([p.grad.detach() for p in params])
            batched_updates = muon_update(
                batched_grads, batched_momenta, beta=group["momentum"], batched=True
            )

            param_data_list = [p.data for p in params]
            torch._foreach_mul_(param_data_list, 1 - group["lr"] * group["weight_decay"])

            batched_updates_view = batched_updates.view(
                len(param_data_list), *param_data_list[0].shape
            )
            updates_list = [batched_updates_view[i] for i in range(len(param_data_list))]
            torch._foreach_add_(param_data_list, updates_list, alpha=-group["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                if self.use_batching:
                    self._batched_muon_group_step(group)
                    continue

                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class TorchMuonWithAuxAdam(torch.optim.Optimizer):
    """One Optimizer that wraps ``torch.optim.Muon`` + ``torch.optim.AdamW``.

    Official ``torch.optim.Muon`` (torch>=2.9) for the ``use_muon=True`` groups and
    ``torch.optim.AdamW`` for the rest. Same group contract as :class:`MuonWithAuxAdam`
    (each group carries a ``use_muon`` flag), so it is a drop-in ``muon_impl`` for a
    trainer that assumes a single ``self.optimizer`` with unified ``param_groups``
    (KL-adaptive lr edits, accelerate wrapping, checkpointing). It holds two real
    sub-optimizers and exposes their ``param_groups`` as one live view -- in-place
    ``group["lr"]`` edits propagate because the group dicts are shared references. This
    lets you A/B the official torch Muon (fused kernels, DTensor/distributed
    integration) against the vendored :class:`MuonWithAuxAdam` without touching the
    rest of the trainer.
    """

    def __init__(self, param_groups):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "torch.optim.Muon is unavailable (needs torch>=2.9). Use "
                "muon_impl=dist or single_batched for the vendored reference impl."
            )
        # Split into muon / adam sub-optimizers, remembering each group's original
        # position so param_groups re-exposes them in input order: the trainer tags
        # groups by zipping with a same-order role list, and the KL controller reads
        # each group's use_muon flag to hold the muon lr fixed.
        muon_groups, adam_groups = [], []
        self._group_order = []  # ("muon"|"adam", index within that sub-optimizer)
        for group in param_groups:
            assert "use_muon" in group, "each param group needs a use_muon flag"
            g = dict(group)
            if group["use_muon"]:
                self._group_order.append(("muon", len(muon_groups)))
                muon_groups.append(g)
            else:
                self._group_order.append(("adam", len(adam_groups)))
                adam_groups.append(g)

        # match_rms_adamw scaling (lr*0.2*sqrt(max(rows,cols))) lets an AdamW-scale
        # muon_lr be reused, matching how these groups are tuned.
        self._muon_opt = (
            torch.optim.Muon(muon_groups, adjust_lr_fn="match_rms_adamw") if muon_groups else None
        )
        self._adam_opt = torch.optim.AdamW(adam_groups) if adam_groups else None
        self.optimizers = [o for o in (self._muon_opt, self._adam_opt) if o is not None]
        if not self.optimizers:
            raise ValueError("TorchMuonWithAuxAdam received no param groups")

        # Register an EMPTY param group so torch.optim.Optimizer.__init__ builds the real
        # machinery (defaults + step/state hooks) for us: no params means nothing is
        # double-registered, and the base's writes to param_groups/state land on the
        # no-op setters below (the real groups/state live in the sub-optimizers).
        super().__init__([{"params": []}], self.optimizers[0].defaults)

    @property
    def param_groups(self):
        # Original input order (so the trainer's lr_role zip and in-place lr edits line
        # up with how the groups were built), with use_muon restored on each dict.
        return [
            (self._muon_opt if kind == "muon" else self._adam_opt).param_groups[idx]
            for kind, idx in self._group_order
        ]

    @param_groups.setter
    def param_groups(self, value):
        # Groups are owned by the sub-optimizers; in-place edits of the dicts the
        # getter returns still propagate, so external reassignment is a no-op.
        pass

    @property
    def state(self):
        merged = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    @state.setter
    def state(self, value):
        # Owned by the sub-optimizers; the base __init__ assigns an empty dict here,
        # which we ignore (the getter merges the real per-sub-optimizer state).
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"sub_optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["sub_optimizers"], strict=True):
            opt.load_state_dict(sd)

    def add_param_group(self, param_group):
        # torch.optim.Optimizer.__init__ calls this to register our empty bootstrap
        # group -> ignore it (param_groups delegates to the sub-optimizers). Any real
        # post-construction addition is unsupported.
        if param_group.get("params"):
            raise NotImplementedError("TorchMuonWithAuxAdam has fixed param groups")
