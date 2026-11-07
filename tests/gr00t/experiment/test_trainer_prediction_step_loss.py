# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T returns loss from forward without HF labels / return_loss.

Hugging Face ``prediction_step`` only computes loss when ``label_names`` are
present in the batch or when ``can_return_loss`` is true. Gr00tN1d7 has neither
(``find_labels`` is empty; forward has no ``return_loss``). Eval still runs the
forward pass, so ``eval_runtime`` appears, but ``eval_loss`` never does.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from gr00t.experiment.trainer import Gr00tTrainer


class _LossModel(torch.nn.Module):
    def forward(self, inputs=None, **kwargs):
        return {"loss": torch.tensor(2.5, requires_grad=False)}


def test_prediction_step_returns_loss_without_hf_labels():
    trainer = object.__new__(Gr00tTrainer)
    trainer.label_names = []
    trainer.can_return_loss = False
    trainer.model = _LossModel()
    trainer.args = SimpleNamespace(past_index=-1, average_tokens_across_devices=False, device="cpu")
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = nullcontext
    trainer._get_num_items_in_batch = MagicMock(return_value=None)
    trainer.compute_loss = lambda model, inputs, return_outputs=False, num_items_in_batch=None: (
        (torch.tensor(2.5), {"loss": torch.tensor(2.5)})
        if return_outputs
        else torch.tensor(2.5)
    )

    loss, logits, labels = trainer.prediction_step(
        trainer.model,
        {"inputs": {"action": torch.zeros(1, 1, 1)}},
        prediction_loss_only=True,
    )

    assert loss is not None
    assert torch.is_tensor(loss)
    assert float(loss) == 2.5
    assert logits is None
    assert labels is None
