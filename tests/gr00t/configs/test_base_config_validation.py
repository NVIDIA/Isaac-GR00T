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

import pytest

from gr00t.configs.base_config import Config
from gr00t.configs.data.data_config import SingleDatasetConfig
from gr00t.data.types import ModalityConfig


def test_missing_embodiment_tag_reports_dataset_paths():
    cfg = Config()
    cfg.data.datasets = [SingleDatasetConfig(dataset_paths=["dataset-a"], embodiment_tag=None)]

    with pytest.raises(ValueError, match="dataset-a"):
        cfg.validate()


def test_default_action_configs_are_distinct_objects():
    cfg = Config()
    cfg.data.datasets = [SingleDatasetConfig(dataset_paths=["dataset-a"], embodiment_tag="test_robot")]
    cfg.data.modality_configs = {
        "test_robot": {
            "action": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
        }
    }

    cfg.validate()

    action_configs = cfg.data.modality_configs["test_robot"]["action"].action_configs
    assert action_configs is not None
    assert len(action_configs) == 2
    assert action_configs[0] is not action_configs[1]

    action_configs[0].state_key = "arm_state"
    assert action_configs[1].state_key is None
