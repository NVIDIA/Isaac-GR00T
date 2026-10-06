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

"""Regression tests for :mod:`gr00t.utils.determinism`.

Covers the four contracts future refactors must preserve:

1. Opt-in: no arg and no ``GR00T_EVAL_SEED`` => no-op, no global flags flipped.
2. Seeding actually makes Python / NumPy / torch CPU RNGs reproducible.
3. ``GR00T_EVAL_SEED=N`` is equivalent to ``seed_everything(N)``.
4. A malformed ``GR00T_EVAL_SEED`` raises instead of being silently ignored.
"""

from __future__ import annotations

import random

from gr00t.utils.determinism import EVAL_SEED_ENV_VAR, get_eval_seed, seed_everything
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_deterministic_algorithms(monkeypatch):
    """Undo the one piece of global state that can break unrelated tests.

    ``use_deterministic_algorithms(True)`` makes legitimately-nondeterministic
    ops in later tests raise; cudnn flags and CUBLAS env var are harmless to
    leak. We also ensure the env var is not inherited from the outer shell.
    """
    monkeypatch.delenv(EVAL_SEED_ENV_VAR, raising=False)
    saved = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved, warn_only=True)


def _sample():
    return (random.random(), float(np.random.rand()), torch.rand(4).tolist())


def test_noop_when_unset():
    before = torch.are_deterministic_algorithms_enabled()
    assert seed_everything() is None
    assert torch.are_deterministic_algorithms_enabled() == before


def test_seed_is_reproducible():
    seed_everything(42)
    first = _sample()
    seed_everything(42)
    assert _sample() == first


def test_env_var_fallback_matches_explicit_arg(monkeypatch):
    monkeypatch.setenv(EVAL_SEED_ENV_VAR, "42")
    assert seed_everything() == 42
    via_env = _sample()
    assert seed_everything(42) == 42
    assert _sample() == via_env


def test_invalid_env_var_raises(monkeypatch):
    monkeypatch.setenv(EVAL_SEED_ENV_VAR, "not-an-int")
    with pytest.raises(ValueError, match=EVAL_SEED_ENV_VAR):
        get_eval_seed()


def test_cudnn_flags_flipped_when_seeded():
    seed_everything(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
