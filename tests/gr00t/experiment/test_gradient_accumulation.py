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

"""Gradient-accumulation equivalence tests for Gr00tTrainer.

The low-VRAM fine-tuning recipe relies on one property: training with
micro-batch ``b`` and ``k`` accumulation steps must perform the same
optimization as micro-batch ``b*k`` with no accumulation. These tests pin that
property on the actual ``Gr00tTrainer`` + ``TrainingArguments`` path (mean-loss
scaling, sequential sample consumption, one optimizer step per ``k`` forwards)
using a deterministic model, so a transformers upgrade or trainer change that
breaks accumulation semantics fails loudly here.

The GR00T flow-matching loss draws fresh noise per forward call, so full-model
runs are only *statistically* equivalent across batching choices; the trainer
math itself, verified here, is exact.
"""

from __future__ import annotations

import copy

from gr00t.experiment.trainer import Gr00tTrainer
import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments


class _TinyRegressionConfig(PretrainedConfig):
    model_type = "TinyGradAccumRegression"

    def __init__(self, in_dim: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim


class _TinyRegressionModel(PreTrainedModel):
    """Deterministic MSE regressor with Gr00tN1d7's dict-style forward signature."""

    config_class = _TinyRegressionConfig

    def __init__(self, config: _TinyRegressionConfig):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.in_dim, 1)

    def forward(self, inputs: dict) -> dict:
        pred = self.linear(inputs["x"]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, inputs["y"])
        return {"loss": loss}


class _FixedDataset(torch.utils.data.Dataset):
    """Deterministic samples so both trainer configurations see identical data."""

    def __init__(self, num_samples: int, in_dim: int):
        generator = torch.Generator().manual_seed(1234)
        self.x = torch.randn(num_samples, in_dim, generator=generator)
        self.y = torch.randn(num_samples, generator=generator)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {"x": self.x[idx], "y": self.y[idx]}


def _collate(examples: list[dict]) -> dict:
    """Mirror the production collator shape: a single ``inputs`` dict."""
    return {
        "inputs": {
            "x": torch.stack([e["x"] for e in examples]),
            "y": torch.stack([e["y"] for e in examples]),
        }
    }


def _train(
    model: _TinyRegressionModel,
    dataset: _FixedDataset,
    tmp_path,
    *,
    per_device_batch: int,
    accumulation_steps: int,
    max_steps: int,
) -> _TinyRegressionModel:
    args = TrainingArguments(
        output_dir=str(tmp_path / f"out_b{per_device_batch}_k{accumulation_steps}"),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=accumulation_steps,
        max_steps=max_steps,
        learning_rate=0.1,
        lr_scheduler_type="constant",
        seed=0,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
    )
    trainer = Gr00tTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=_collate,
    )
    trainer.train()
    assert trainer.state.global_step == max_steps
    return model


@pytest.fixture
def dataset() -> _FixedDataset:
    return _FixedDataset(num_samples=64, in_dim=4)


@pytest.fixture
def initial_model() -> _TinyRegressionModel:
    torch.manual_seed(0)
    return _TinyRegressionModel(_TinyRegressionConfig())


def test_accumulated_training_matches_full_batch(tmp_path, dataset, initial_model):
    """micro-batch 4 x 2 accumulation == batch 8 x 1, parameter for parameter.

    Both runs start from identical weights and consume the same 8 sequential
    samples per optimizer step, so the resulting parameters must agree up to
    floating-point summation order.
    """
    model_full = copy.deepcopy(initial_model)
    model_accum = copy.deepcopy(initial_model)

    _train(model_full, dataset, tmp_path, per_device_batch=8, accumulation_steps=1, max_steps=4)
    _train(model_accum, dataset, tmp_path, per_device_batch=4, accumulation_steps=2, max_steps=4)

    full_sd = model_full.state_dict()
    accum_sd = model_accum.state_dict()
    assert full_sd.keys() == accum_sd.keys()
    for key in full_sd:
        assert torch.allclose(full_sd[key], accum_sd[key], rtol=1e-5, atol=1e-7), (
            f"parameter {key} diverged between accumulated and full-batch training:\n"
            f"full={full_sd[key]}\naccum={accum_sd[key]}"
        )

    # Guard against vacuous success: training must actually have moved the weights.
    for key, initial_value in initial_model.state_dict().items():
        assert not torch.allclose(full_sd[key], initial_value), (
            f"parameter {key} did not change during training; the equivalence "
            f"assertion above proved nothing"
        )


def test_accumulation_runs_k_forwards_per_optimizer_step(tmp_path, dataset, initial_model):
    """With k accumulation steps the trainer must run k forwards per optimizer
    step — i.e. accumulation is actually active, not silently ignored."""
    model = copy.deepcopy(initial_model)
    forward_calls = 0

    def _count_forward(module, args, kwargs):
        nonlocal forward_calls
        forward_calls += 1

    handle = model.register_forward_pre_hook(_count_forward, with_kwargs=True)
    try:
        _train(model, dataset, tmp_path, per_device_batch=4, accumulation_steps=2, max_steps=4)
    finally:
        handle.remove()

    assert forward_calls == 8, (
        f"expected 4 optimizer steps x 2 accumulation steps = 8 forwards, got {forward_calls}"
    )
