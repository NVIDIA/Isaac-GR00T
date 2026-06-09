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

"""``accumulated_batch_size`` must equal what HuggingFace ``Trainer`` consumes
per optimizer step (``per_device × num_gpus × gradient_accumulation_steps``)."""

from __future__ import annotations

import warnings

from gr00t.configs.training.training_config import TrainingConfig
import pytest


@pytest.mark.parametrize(
    "kwargs, per_device, num_gpus, grad_accum",
    [
        pytest.param(
            dict(global_batch_size=1024, num_gpus=8, gradient_accumulation_steps=4),
            128,
            8,
            4,
            id="global_path",
        ),
        pytest.param(
            dict(per_gpu_batch_size=32, num_gpus=8, gradient_accumulation_steps=4),
            32,
            8,
            4,
            id="per_gpu_path_regression_pin",
        ),
    ],
)
def test_training_config_accumulated_mirrors_hf(kwargs, per_device, num_gpus, grad_accum):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = TrainingConfig(**kwargs)
    assert cfg.accumulated_batch_size == per_device * num_gpus * grad_accum
