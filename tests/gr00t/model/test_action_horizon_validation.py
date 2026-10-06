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

"""``validate_action_horizons`` must reject an embodiment whose action horizon
exceeds the model's ``max_action_horizon`` at processor construction, rather than
letting it surface deep in the first forward.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


processing = pytest.importorskip("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
validate_action_horizons = processing.validate_action_horizons


def _action(horizon: int) -> dict:
    return {"action": SimpleNamespace(delta_indices=list(range(horizon)))}


def test_passes_when_all_horizons_fit():
    configs = {"emb_a": _action(40), "emb_b": _action(8)}
    validate_action_horizons(configs, max_action_horizon=40)


def test_raises_when_horizon_exceeds_max():
    configs = {"g1": _action(50)}
    with pytest.raises(ValueError) as excinfo:
        validate_action_horizons(configs, max_action_horizon=40)
    message = str(excinfo.value)
    assert "g1=50" in message
    assert ">= 50" in message


def test_reports_the_largest_required_horizon():
    configs = {"g1": _action(50), "big": _action(64), "small": _action(8)}
    with pytest.raises(ValueError) as excinfo:
        validate_action_horizons(configs, max_action_horizon=40)
    assert ">= 64" in str(excinfo.value)


def test_ignores_embodiments_without_an_action_config():
    configs = {"vlm_only": {"video": SimpleNamespace(delta_indices=[0])}}
    validate_action_horizons(configs, max_action_horizon=40)
