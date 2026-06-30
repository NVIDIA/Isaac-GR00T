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

"""Regression tests for episode-length accounting in policy rollouts.

The sim rollout runs under gymnasium 0.29.1 (inline autoreset), whose vector
env relocates a terminating step's info into ``final_info`` while the top-level
info describes the freshly reset env. Reading ``n_env_steps`` only from the
top-level info therefore undercounts the terminal macro-step to 0 env-steps,
collapsing ``episode_length`` to 0 and tripping the zero-length-episode
invariant. These tests pin the version-agnostic accounting helper.
"""

from gr00t.eval.rollout_policy import _macro_step_env_steps
import numpy as np
import pytest


def test_counts_top_level_n_env_steps():
    """gymnasium >=1.0 keeps the (terminal) step's info at the top level."""
    env_infos = {"n_env_steps": np.array([3])}
    assert _macro_step_env_steps(env_infos, 0) == 3


def test_counts_n_env_steps_relocated_into_final_info():
    """gymnasium 0.29.1 inline autoreset moves the terminal info into
    final_info; the top-level info reflects the reset env and lacks
    n_env_steps. The terminal macro-step must still be counted (>0)."""
    env_infos = {
        # Top-level mirrors the freshly reset env -> no usable n_env_steps.
        "_final_info": np.array([True]),
        "final_info": np.array([{"n_env_steps": 2, "success": [False]}], dtype=object),
    }
    assert _macro_step_env_steps(env_infos, 0) == 2


def test_final_info_takes_precedence_over_masked_top_level():
    """When both are present, final_info (the real terminal step) wins."""
    env_infos = {
        "n_env_steps": np.array([0]),
        "final_info": np.array([{"n_env_steps": 2}], dtype=object),
    }
    assert _macro_step_env_steps(env_infos, 0) == 2


def test_missing_n_env_steps_returns_zero():
    """Steps that carry no env-step count contribute nothing (e.g. a pure
    autoreset placeholder), but must not raise."""
    assert _macro_step_env_steps({}, 0) == 0
    assert _macro_step_env_steps({"final_info": np.array([None], dtype=object)}, 0) == 0


@pytest.mark.parametrize("env_idx", [0, 1])
def test_per_env_indexing_multi_env(env_idx):
    """In a multi-env batch, one env may terminate (final_info) while another
    keeps running (top-level), and each must read its own count."""
    env_infos = {
        "n_env_steps": np.ma.array([5, 7], mask=[True, False]),
        "final_info": np.array([{"n_env_steps": 4}, None], dtype=object),
    }
    expected = {0: 4, 1: 7}
    assert _macro_step_env_steps(env_infos, env_idx) == expected[env_idx]
