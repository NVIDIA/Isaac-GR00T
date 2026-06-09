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

"""torchrun entry point for the multi-GPU FSDP best-metric-save weight-parity test.

Reproduces the failure that motivated the sharded-aware save rewrite: under a
parameter-sharded backend each rank holds only ``1/world_size`` of every parameter,
so the old ``model.save_pretrained`` on rank-0 wrote that rank's shard with the rest
of the model left at near-zero placeholders. The corruption was silent — shapes
matched, the load path succeeded — and only surfaced at deploy time.

PyTorch FSDP is used as the sharded backend rather than DeepSpeed ZeRO-3 because the
two share the *exact same* callback path (`_broadcast_save_decision` → `dist.barrier`
→ `Trainer.save_model`; the gather is internal to `save_model`), and FSDP is part of
core torch whereas the DeepSpeed wheel is published only for x86_64 Linux
(`pyproject.toml`) and is therefore absent on the arm64 multi-GPU CI runner.

This harness deliberately does **not** go through ``experiment.run``: the sharded
training dataset hard-asserts ``eval_strategy == "no"`` (see
``gr00t/data/dataset/factory.py``), so ``on_evaluate`` can never fire on that path.
Instead we wrap a tiny model in a real HF ``Trainer`` + FSDP full-shard, run a single
no-op optimizer step (``learning_rate=0`` keeps the weights bit-stable so the saved
checkpoint can be compared against a pre-shard reference snapshot), then drive the
callback's ``on_evaluate`` exactly as the eval loop would — with ``metrics`` populated
on rank-0 only, the worst case for the cross-rank broadcast.

Parity contract checked on rank-0:
  * every saved parameter matches the pre-shard reference (generous tolerance — the
    point is to catch zeroed-out shards, where buggy values differ from the
    unit-normal reference by ~1.0, not to police fp32-vs-bf16 rounding);
  * no saved parameter tensor is entirely zero.

Exits non-zero with a diagnostic on any mismatch so the spawning pytest can surface it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist


_DIM = 256
_N_LAYERS = 5
_SEED = 1234


class _TinyModel(nn.Module):
    """A few stacked Linear layers — large enough that FSDP shards parameters
    across ranks, small enough to train in milliseconds."""

    def __init__(self, dim: int = _DIM, n_layers: int = _N_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.head = nn.Linear(dim, dim)

    def forward(self, x, labels=None):
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
        h = self.head(h)
        loss = ((h - labels) ** 2).mean() if labels is not None else h.sum()
        return {"loss": loss, "logits": h}


class _RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 16, dim: int = _DIM, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randn(n, dim, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "labels": self.y[i]}


def _collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _load_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load a saved checkpoint, tolerating either safetensors or torch.bin layout."""
    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        merged: dict[str, torch.Tensor] = {}
        for f in safetensor_files:
            merged.update(load_file(str(f)))
        return merged

    bin_files = sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    if bin_files:
        merged = {}
        for f in bin_files:
            merged.update(torch.load(str(f), map_location="cpu"))
        return merged

    raise FileNotFoundError(
        f"No weight files (*.safetensors / pytorch_model*.bin) in {checkpoint_dir}"
    )


def _verify_parity(best_dir: Path, reference: dict[str, torch.Tensor]) -> None:
    """Rank-0 check: saved weights equal the pre-shard reference and are non-zero."""
    saved = _load_state_dict(best_dir)

    missing = set(reference) - set(saved)
    if missing:
        raise AssertionError(
            f"Saved checkpoint is missing {len(missing)} params, e.g. {sorted(missing)[:3]}"
        )

    for name, ref_tensor in reference.items():
        got = saved[name].detach().to(torch.float32).cpu()
        ref = ref_tensor.detach().to(torch.float32).cpu()
        if got.shape != ref.shape:
            raise AssertionError(f"Shape mismatch for {name}: saved {got.shape} vs ref {ref.shape}")
        # Generous tolerance: zeroed shards (the bug) differ from the unit-normal
        # reference by ~1.0, far above this; legitimate fp32/bf16 rounding does not.
        if not torch.allclose(got, ref, atol=1e-2, rtol=1e-2):
            max_abs = (got - ref).abs().max().item()
            raise AssertionError(
                f"Weight mismatch for {name}: max|saved-ref|={max_abs:.4f}. "
                "A non-rank-0 shard was likely written un-consolidated (the original bug)."
            )
        if torch.count_nonzero(got) == 0:
            raise AssertionError(
                f"Saved parameter {name} is entirely zero — un-consolidated shard."
            )


def main() -> None:
    args = _parse_args()

    from gr00t.experiment.utils import BestMetricCheckpointCallback
    from transformers import Trainer, TrainingArguments

    # Identical init on every rank so the pre-shard snapshot is a valid cross-rank
    # reference for the consolidated save.
    torch.manual_seed(_SEED)
    model = _TinyModel()
    reference = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=2,
        max_steps=1,
        learning_rate=0.0,
        weight_decay=0.0,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 100,
            "state_dict_type": "FULL_STATE_DICT",
            "use_orig_params": True,
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_RandomDataset(seed=_SEED),
        data_collator=_collate,
    )

    # Initializes FSDP and shards parameters; lr=0 keeps weights bit-stable.
    trainer.train()

    callback = BestMetricCheckpointCallback(
        metric_name="eval_loss",
        trainer=trainer,
        greater_is_better=False,
    )

    # Worst case for the broadcast: only rank-0 has the metrics dict.
    metrics = {"eval_loss": 0.123} if trainer.state.is_world_process_zero else None
    callback.on_evaluate(
        args=trainer.args,
        state=trainer.state,
        control=trainer.control,
        metrics=metrics,
        model=trainer.model,
    )

    if trainer.state.is_world_process_zero:
        best_dirs = list(Path(args.output_dir).glob("checkpoint-*-best-eval_loss_*"))
        if len(best_dirs) != 1:
            raise AssertionError(f"Expected exactly one best checkpoint dir, found: {best_dirs}")
        _verify_parity(best_dirs[0], reference)
        print(f"[rank0] weight parity verified for {best_dirs[0].name}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface the diagnostic through the non-zero exit
        rank = dist.get_rank() if dist.is_initialized() else "?"
        print(f"[rank{rank}] FAILED: {exc}", flush=True)
        raise SystemExit(1) from exc
