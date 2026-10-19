# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Exponential moving average (EMA) of the trainable weights.

After each optimizer step every trainable tensor's shadow is updated as

    theta_ema <- beta * theta_ema + (1 - beta) * theta,  beta = min(decay, (1 + n) / (10 + n))

where n counts the updates so far; the ramp keeps the average from dragging the
initialization along for the first 1/(1 - decay) steps. The shadow never feeds back into
training (no gradient touches it and it is not optimizer state), so
``training.ema_decay = 0`` leaves training byte-identical.

Scope. Only parameters with ``requires_grad`` are shadowed.

Rank 0 only. Under DeepSpeed ZeRO-1/2, DDP and single-GPU every rank holds the full,
identical parameters after ``optimizer.step()``, so rank 0 alone keeps the shadow: one
extra fp32 copy of the trainable weights on one GPU. ZeRO-3 / FSDP shard parameters and
are rejected at ``on_train_begin``.

fp32, not bf16: at beta = 0.999 each step moves the average by ~1e-3 of the parameter
magnitude, below bf16's ~4e-3 resolution, so a bf16 shadow would round most updates away.

File format. Every ``checkpoint-<step>/`` gets ``ema_model.safetensors``: the shadowed
tensors only, in fp32, keyed by their ``named_parameters()`` names, with the update count
in the ``ema_num_updates`` metadata. A resumed run continues the average from the file in
the checkpoint the trainer resumes from; a resumed checkpoint without it is an error unless
``GROOT_ALLOW_EMA_RESET_RESUME=1``. The file is not a model on its own: overwrite the
checkpoint's tensors of the same names with it (cast to their dtype) to get one.
"""

import logging
import os
from pathlib import Path
import time

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers import TrainerCallback

EMA_WEIGHTS_FILENAME = "ema_model.safetensors"

# Consent to restart the average when the resumed checkpoint has no EMA file (EMA switched
# on mid-run, or the EMA write of that checkpoint was interrupted); mirrors
# GROOT_ALLOW_OPTIMIZER_RESET_RESUME.
EMA_RESET_RESUME_ENV = "GROOT_ALLOW_EMA_RESET_RESUME"

# beta ramps as (1 + n) / (_WARMUP_OFFSET + n): the average is a plain running
# mean of the first few steps and reaches the target decay from below.
_WARMUP_OFFSET = 10.0


class EMACallback(TrainerCallback):
    """Track an EMA of the trainable weights and save it with each checkpoint."""

    def __init__(self, decay: float, resume_checkpoint: str | None = None):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = decay
        # The checkpoint trainer.train(resume_from_checkpoint=...) restores, None for a fresh
        # run (experiment._continuation_checkpoint): an explicit recovery source lives outside
        # output_dir, so the shadow cannot be looked up from args.output_dir.
        self._resume_checkpoint = resume_checkpoint
        self._names: list[str] = []
        self._params: list[torch.nn.Parameter] = []
        self._shadow: list[torch.Tensor] = []
        self._num_updates = 0
        # Only the saving rank maintains the shadow (see module docstring).
        self._active = False

    def on_train_begin(self, _args, state, _control, *, model, **_kwargs):
        self._active = state.is_world_process_zero
        if not self._active:
            return

        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("EMACallback enabled but the model has no trainable parameters")
        if any(p.numel() == 0 for _, p in trainable):
            raise RuntimeError(
                "EMACallback found zero-sized parameters, i.e. parameters are sharded "
                "(DeepSpeed ZeRO-3 / FSDP). EMA needs replicated parameters — use "
                "ZeRO-1/2, DDP or a single GPU, or disable training.ema_decay."
            )

        self._names = [name for name, _ in trainable]
        self._params = [p for _, p in trainable]
        self._shadow = [p.detach().float().clone() for p in self._params]
        self._num_updates = 0

        if state.global_step > 0:
            if self._resume_checkpoint is None:
                raise RuntimeError(
                    f"EMA: training resumes at step {state.global_step} but EMACallback was "
                    "given no resume checkpoint to restore the average from"
                )
            self._load(Path(self._resume_checkpoint))

        numel = sum(p.numel() for p in self._params)
        # print, not logging.info: the training workers run the root logger at
        # WARNING, so info-level records never reach the job log (which is why
        # the neighboring checkpoint callbacks print their [CP-SAVE] lines).
        print(  # noqa: T201
            f"[EMA] enabled: decay={self.decay}, shadowing {len(self._params)} tensors / "
            f"{numel / 1e9:.2f}B params ({numel * 4 / 2**30:.1f} GiB fp32 on rank 0), "
            f"resuming at {self._num_updates} updates"
        )

    @torch.no_grad()
    def on_step_end(self, _args, _state, _control, **_kwargs):
        if not self._active:
            return
        self._num_updates += 1
        beta = min(self.decay, (1.0 + self._num_updates) / (_WARMUP_OFFSET + self._num_updates))
        # foreach kernels over ~1k tensors; the bf16 params upcast into the fp32
        # shadow, which is the whole point of keeping the shadow in fp32.
        torch._foreach_mul_(self._shadow, beta)
        torch._foreach_add_(self._shadow, self._params, alpha=1.0 - beta)

    def on_save(self, args, state, _control, *, model, **_kwargs):
        if not self._active:
            return
        # The shadow set is snapshotted once at on_train_begin, so a module
        # that is unfrozen mid-run (UnfreezeBackboneCallback, staged LoRA)
        # would silently train outside the average. Fail loudly instead:
        # splicing the EMA back over a checkpoint is only exact because the
        # unshadowed weights never moved, so their online value is their average.
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        if trainable != set(self._names):
            raise RuntimeError(
                "EMA: the set of trainable parameters changed during training "
                f"({len(trainable)} now vs {len(self._names)} shadowed). Mid-run "
                "freeze/unfreeze is incompatible with training.ema_decay."
            )
        self._save(Path(args.output_dir) / f"checkpoint-{state.global_step}")

    @torch.no_grad()
    def _save(self, checkpoint_dir: Path):
        """Write the shadow into ``checkpoint_dir`` synchronously (rank 0 only).

        The upload callbacks registered after this one read the directory (see
        ``experiment.run``). Costs 4 bytes per trainable parameter every ``save_steps``.
        """
        t_start = time.time()
        tensors = {
            name: shadow.to("cpu") for name, shadow in zip(self._names, self._shadow, strict=True)
        }
        # Write-then-rename so an interrupted save cannot leave a truncated file
        # that the next resume would pick up.
        path = checkpoint_dir / EMA_WEIGHTS_FILENAME
        tmp_path = path.with_suffix(".safetensors.tmp")
        # The model shards are written by a background process, which may not have
        # created the directory yet when on_save fires.
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            str(tmp_path),
            metadata={
                "format": "pt",
                # Provenance only; nothing reads it back.
                "ema_decay": str(self.decay),
                "ema_num_updates": str(self._num_updates),
            },
        )
        os.replace(tmp_path, path)
        print(  # noqa: T201
            f"[CP-SAVE][callback=EMACallback] wrote {path} after {self._num_updates} "
            f"updates, time={time.time() - t_start:.2f}s"
        )

    @torch.no_grad()
    def _load(self, checkpoint_dir: Path):
        """Restore the shadow and its update count from the resumed checkpoint."""
        path = checkpoint_dir / EMA_WEIGHTS_FILENAME
        if not path.exists():
            if os.environ.get(EMA_RESET_RESUME_ENV, "").lower() not in {"1", "true", "yes"}:
                raise FileNotFoundError(
                    f"EMA resume: no {EMA_WEIGHTS_FILENAME} in {checkpoint_dir}, so the "
                    "average cannot be continued (its EMA write was interrupted, or EMA was "
                    "switched on mid-run). Resume from an earlier checkpoint that has the "
                    f"file, or set {EMA_RESET_RESUME_ENV}=1 to restart the average from the "
                    "resumed online weights."
                )
            logging.warning(
                "EMA resume: no %s in %s and %s is set — restarting the average from the "
                "resumed online weights.",
                EMA_WEIGHTS_FILENAME,
                checkpoint_dir,
                EMA_RESET_RESUME_ENV,
            )
            return

        with safe_open(str(path), framework="pt") as f:
            num_updates = int(f.metadata()["ema_num_updates"])
        loaded = load_file(str(path))
        missing = [name for name in self._names if name not in loaded]
        if missing:
            raise RuntimeError(
                f"EMA resume: {path} is missing {len(missing)} trainable tensors "
                f"(e.g. {missing[:3]}). The checkpoint does not match this model."
            )
        for name, shadow in zip(self._names, self._shadow, strict=True):
            shadow.copy_(loaded[name])
        self._num_updates = num_updates
