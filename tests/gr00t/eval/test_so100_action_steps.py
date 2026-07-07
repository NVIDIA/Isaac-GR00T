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

"""SO100 real-robot eval must not silently execute fewer action steps than the
configured ``action_horizon``. ``_select_action_steps`` returns the requested
window when the policy supplies enough steps and raises otherwise.
"""

from __future__ import annotations

import pytest


eval_so100 = pytest.importorskip("gr00t.eval.real_robot.SO100.eval_so100")
_select_action_steps = eval_so100._select_action_steps


def _chunk(n: int) -> list:
    return [{"step": i} for i in range(n)]


def test_returns_requested_window_when_chunk_is_long_enough():
    actions = _chunk(40)
    selected = _select_action_steps(actions, 8)
    assert selected == actions[:8]


def test_returns_full_chunk_when_horizon_equals_length():
    actions = _chunk(8)
    assert _select_action_steps(actions, 8) == actions


def test_raises_when_horizon_exceeds_chunk():
    actions = _chunk(4)
    with pytest.raises(ValueError) as excinfo:
        _select_action_steps(actions, 8)
    message = str(excinfo.value)
    assert "action_horizon=8" in message
    assert "chunk length 4" in message
