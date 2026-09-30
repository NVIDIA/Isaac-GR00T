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

from gr00t.configs.data.data_config import DataConfig
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS


def test_default_modality_configs_are_deeply_isolated():
    first = DataConfig()
    second = DataConfig()

    assert first.modality_configs is not second.modality_configs
    assert first.modality_configs is not MODALITY_CONFIGS

    embodiment = next(iter(MODALITY_CONFIGS))
    modality = next(iter(MODALITY_CONFIGS[embodiment]))
    global_config = MODALITY_CONFIGS[embodiment][modality]
    original_delta_indices = list(global_config.delta_indices)

    assert first.modality_configs[embodiment][modality] is not global_config
    first.modality_configs[embodiment][modality].delta_indices.append(999_999)

    assert second.modality_configs[embodiment][modality].delta_indices == original_delta_indices
    assert MODALITY_CONFIGS[embodiment][modality].delta_indices == original_delta_indices
