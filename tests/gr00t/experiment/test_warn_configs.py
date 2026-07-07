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

"""CPU-only guards for the training-config invariants in ``experiment``:

* ``num_gpus`` must equal the launcher ``WORLD_SIZE`` (otherwise per-device
  batch math and the real data-parallel size disagree).
* ``warmup_steps`` is actually forwarded to HF ``TrainingArguments`` so its
  documented "overrides warmup_ratio" contract holds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


EXPERIMENT_SRC = Path(__file__).resolve().parents[3] / "gr00t" / "experiment" / "experiment.py"


@pytest.mark.parametrize(
    "world_size_env, num_gpus, expect_error",
    [
        (None, 8, False),  # no launcher → single process, unchecked
        ("8", 8, False),  # torchrun matches config
        ("8", 1, True),  # 8-rank launch, num_gpus=1 → 8x effective batch
        ("4", 8, True),  # config over-counts ranks
    ],
)
def test_num_gpus_must_match_world_size(monkeypatch, world_size_env, num_gpus, expect_error):
    from gr00t.experiment.experiment import _assert_num_gpus_matches_world_size

    if world_size_env is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size_env)

    if expect_error:
        with pytest.raises(ValueError, match="WORLD_SIZE"):
            _assert_num_gpus_matches_world_size(num_gpus)
    else:
        _assert_num_gpus_matches_world_size(num_gpus)


def _training_arguments_keywords() -> set[str]:
    """Keyword names passed to the ``TrainingArguments(...)`` call in ``run``."""
    tree = ast.parse(EXPERIMENT_SRC.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TrainingArguments"
        ):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError("TrainingArguments(...) call not found in experiment.py")


def test_warmup_steps_is_forwarded_to_training_arguments():
    # warn_configs warns that warmup_steps "will override warmup_ratio"; that is
    # only true if the field actually reaches the trainer.
    keywords = _training_arguments_keywords()
    assert "warmup_steps" in keywords
    assert "warmup_ratio" in keywords
