# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from gr00t.experiment.trainer import Gr00tTrainer


class _EvalDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield {"value": torch.tensor(1)}


def test_eval_dataloader_preserves_dataset_owned_distributed_sharding():
    trainer = object.__new__(Gr00tTrainer)
    trainer.eval_dataset = _EvalDataset()
    trainer.data_collator = lambda samples: samples
    trainer._get_collator_with_removed_columns = lambda collator, description: collator
    trainer.multiprocessing_context = "fork"
    trainer.accelerator = MagicMock()
    trainer.args = SimpleNamespace(
        eval_batch_size=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
    )

    dataloader = trainer.get_eval_dataloader()

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.dataset is trainer.eval_dataset
    trainer.accelerator.prepare.assert_not_called()
