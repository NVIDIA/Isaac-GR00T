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

"""CPU-only tests for the policy-resolved horizon contract.

These are dependency-free: they exercise ``PolicyHorizonSpec`` against a tiny
fake modality config, with no gym / torch / model import.
"""

from dataclasses import dataclass

from gr00t.eval._horizon_contract import PolicyHorizonSpec
import numpy as np
import pytest


@dataclass
class _FakeModalityConfig:
    delta_indices: list


class _FakePolicy:
    def __init__(self, modality_config):
        self._mc = modality_config

    def get_modality_config(self):
        return self._mc


def _mc(action_n=16, video=(0,), state=(0,)):
    cfg = {
        "action": _FakeModalityConfig(delta_indices=list(range(action_n))),
        "video": _FakeModalityConfig(delta_indices=list(video)),
    }
    if state is not None:
        cfg["state"] = _FakeModalityConfig(delta_indices=list(state))
    return cfg


def test_full_chunk_default():
    c = PolicyHorizonSpec.from_modality_config(_mc(action_n=16))
    assert c.action_horizon == 16
    assert c.n_action_steps == 16  # full chunk when not overridden
    assert c.video_delta_indices == (0,)
    assert c.state_delta_indices == (0,)


def test_short_open_loop_is_declarable():
    # LIBERO 8/16: deliberate receding-horizon tuning is explicit + valid.
    c = PolicyHorizonSpec.from_modality_config(_mc(action_n=16), n_action_steps=8)
    assert c.n_action_steps == 8
    assert c.action_horizon == 16


def test_short_open_loop_equal_to_horizon():
    c = PolicyHorizonSpec.from_modality_config(_mc(action_n=40), n_action_steps=40)
    assert c.n_action_steps == 40


def test_n_action_steps_greater_than_horizon_raises():
    with pytest.raises(ValueError, match="IndexError"):
        PolicyHorizonSpec.from_modality_config(_mc(action_n=16), n_action_steps=17)


def test_n_action_steps_below_one_raises():
    with pytest.raises(ValueError, match="execute nothing"):
        PolicyHorizonSpec.from_modality_config(_mc(action_n=16), n_action_steps=0)


def test_non_contiguous_action_delta_raises():
    cfg = {
        "action": _FakeModalityConfig(delta_indices=[0, 4, 8, 12]),
        "video": _FakeModalityConfig(delta_indices=[0]),
        "state": _FakeModalityConfig(delta_indices=[0]),
    }
    with pytest.raises(ValueError, match="contiguous"):
        PolicyHorizonSpec.from_modality_config(cfg)


def test_video_state_sourced_from_policy():
    # DROID-like: video delta is [-15, 0], not the wrapper's old [0] default.
    c = PolicyHorizonSpec.from_modality_config(_mc(action_n=40, video=(-15, 0), state=(0,)))
    assert c.video_delta_indices == (-15, 0)
    assert c.state_delta_indices == (0,)
    # ndarray views match what MultiStepWrapper consumes.
    np.testing.assert_array_equal(c.video_delta_indices_array, np.array([-15, 0]))
    np.testing.assert_array_equal(c.state_delta_indices_array, np.array([0]))


def test_vision_only_policy_has_no_state():
    c = PolicyHorizonSpec.from_modality_config(_mc(action_n=8, state=None))
    assert c.state_delta_indices is None
    assert c.state_delta_indices_array is None


def test_missing_action_raises():
    with pytest.raises(ValueError, match="no 'action' entry"):
        PolicyHorizonSpec.from_modality_config({"video": _FakeModalityConfig(delta_indices=[0])})


def test_empty_action_raises():
    with pytest.raises(ValueError, match="empty action"):
        PolicyHorizonSpec.from_modality_config(
            {
                "action": _FakeModalityConfig(delta_indices=[]),
                "video": _FakeModalityConfig(delta_indices=[0]),
            }
        )


def test_from_policy_delegates():
    policy = _FakePolicy(_mc(action_n=16))
    c = PolicyHorizonSpec.from_policy(policy, n_action_steps=8)
    assert c.n_action_steps == 8
    assert c.action_horizon == 16


def test_contract_is_picklable():
    import pickle

    c = PolicyHorizonSpec.from_modality_config(_mc(action_n=16), n_action_steps=8)
    assert pickle.loads(pickle.dumps(c)) == c


def test_numpy_delta_indices_accepted():
    cfg = {
        "action": _FakeModalityConfig(delta_indices=np.arange(16)),
        "video": _FakeModalityConfig(delta_indices=np.array([-15, 0])),
        "state": _FakeModalityConfig(delta_indices=np.array([0])),
    }
    c = PolicyHorizonSpec.from_modality_config(cfg)
    assert c.action_horizon == 16
    assert c.video_delta_indices == (-15, 0)
    assert all(isinstance(v, int) for v in c.video_delta_indices)
