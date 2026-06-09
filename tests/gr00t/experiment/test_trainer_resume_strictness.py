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

"""CPU regression test pinning ``Gr00tTrainer.train()`` resume-error semantics.

Before this MR, ``Gr00tTrainer.train()`` swallowed HF's ``ValueError("No valid
checkpoint found ...")`` and downgraded it to a single ``logging.warning``,
silently falling back to fresh training. That silencer is what hid the
``resume_from_checkpoint=True`` hardcoding (see the silent-corruption fix in
this same MR) for ~6 months: fresh runs printed one easy-to-miss WARNING and
otherwise looked normal.

This test ensures the strict-raise behavior never regresses: an explicit
``resume_from_checkpoint=True`` against a directory with no ``checkpoint-*``
must hard-fail.
"""

from pathlib import Path

from gr00t.experiment.trainer import Gr00tTrainer
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x, labels=None):
        out = self.fc(x)
        loss = ((out - labels) ** 2).mean() if labels is not None else None
        return {"loss": loss, "logits": out}


class _TinyDataset(Dataset):
    def __init__(self, n: int = 4):
        torch.manual_seed(0)
        self.x = torch.randn(n, 4)
        self.y = torch.randn(n, 4)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "labels": self.y[i]}


def _collate(batch):
    return {
        "x": torch.stack([r["x"] for r in batch]),
        "labels": torch.stack([r["labels"] for r in batch]),
    }


def _make_trainer(output_dir: Path) -> Gr00tTrainer:
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=1,
        per_device_train_batch_size=2,
        save_strategy="no",
        report_to="none",
        seed=0,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    return Gr00tTrainer(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDataset(),
        data_collator=_collate,
    )


def test_raises_when_resume_true_but_no_checkpoint(tmp_path):
    """``resume_from_checkpoint=True`` + empty output_dir must hard-fail.

    A silent fallback here is what allowed the hardcoded ``True`` in
    ``experiment.run()`` to go unnoticed for months: fresh runs trained
    successfully and only emitted one easily-missed WARNING. The override now
    propagates HF's ``ValueError`` instead of swallowing it.
    """
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="No valid checkpoint found"):
        trainer.train(resume_from_checkpoint=True)


def test_raises_when_resume_true_and_only_non_checkpoint_subdirs(tmp_path):
    """Sibling directories like ``processor/`` / ``experiment_cfg/`` written
    by ``experiment.run()`` before ``trainer.train()`` must not be mistaken
    for resumable checkpoints.
    """
    (tmp_path / "processor").mkdir()
    (tmp_path / "experiment_cfg").mkdir()
    (tmp_path / "wandb_config.json").write_text("{}")
    trainer = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="No valid checkpoint found"):
        trainer.train(resume_from_checkpoint=True)
