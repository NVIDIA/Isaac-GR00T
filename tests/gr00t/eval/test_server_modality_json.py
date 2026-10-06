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

import json

from gr00t.data.types import ModalityConfig
from gr00t.eval.run_gr00t_server import _load_json_modality_configs
import pytest


def test_dataset_layout_json_raises_actionable(tmp_path):
    # A dataset's meta/modality.json (start/end layout), not ModalityConfig fields.
    p = tmp_path / "modality.json"
    p.write_text(json.dumps({"state": {"single_arm": {"start": 0, "end": 5}}}))

    with pytest.raises(ValueError) as exc:
        _load_json_modality_configs(p)

    msg = str(exc.value)
    assert "ModalityConfig" in msg
    assert ".py" in msg


def test_valid_modality_config_json_loads(tmp_path):
    payload = {"action": {"delta_indices": [0, 1], "modality_keys": ["x"]}}
    p = tmp_path / "mc.json"
    p.write_text(json.dumps(payload))

    configs = _load_json_modality_configs(p)

    assert set(configs) == set(payload)
    assert isinstance(configs["action"], ModalityConfig)
    assert configs["action"].delta_indices == payload["action"]["delta_indices"]
    assert configs["action"].modality_keys == payload["action"]["modality_keys"]
