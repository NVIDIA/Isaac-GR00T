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

from types import SimpleNamespace

import numpy as np

from gr00t.data.types import ModalityConfig
from gr00t.policy.replay_policy import ReplayPolicy


def _make_policy_like(delta_indices, *, current_step=0):
    return SimpleNamespace(
        modality_configs={
            "action": ModalityConfig(delta_indices=delta_indices, modality_keys=["arm"]),
        },
        execution_horizon=1,
        current_step=current_step,
        episode_length=6,
        episode_index=0,
        actions={"arm": np.arange(6, dtype=np.float32).reshape(6, 1)},
    )


def test_replay_policy_honors_non_contiguous_action_delta_indices():
    policy = _make_policy_like([0, 2, 4])

    action, _ = ReplayPolicy._get_action(policy, observation=None, options={"batch_size": 2})

    np.testing.assert_array_equal(action["arm"][0, :, 0], np.array([0, 2, 4], dtype=np.float32))
    np.testing.assert_array_equal(action["arm"][1], action["arm"][0])
    ReplayPolicy.check_action(policy, action)


def test_replay_policy_clips_configured_offsets_at_episode_end():
    policy = _make_policy_like([0, 2, 4], current_step=4)

    action, _ = ReplayPolicy._get_action(policy, observation=None, options={"batch_size": 1})

    np.testing.assert_array_equal(action["arm"][0, :, 0], np.array([4, 5, 5], dtype=np.float32))


def test_replay_policy_defaults_batch_size_when_options_are_omitted():
    policy = _make_policy_like([0, 2, 4])

    action, _ = ReplayPolicy._get_action(policy, observation=None, options=None)

    assert action["arm"].shape == (1, 3, 1)
