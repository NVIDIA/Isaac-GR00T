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

"""CPU regression test for the ``save_only_model`` × ``resume_from_checkpoint`` guard.

The guard lives in stdlib-only ``gr00t.configs.training.training_config`` so
this test runs without torch / transformers / wandb.
"""

from gr00t.configs.training.training_config import TrainingConfig, check_resume_compatibility
import pytest


def test_raises_on_save_only_model_and_resume():
    training = TrainingConfig(save_only_model=True, resume_from_checkpoint=True)
    with pytest.raises(ValueError, match="save_only_model=True is incompatible"):
        check_resume_compatibility(training)


@pytest.mark.parametrize(
    ("save_only_model", "resume_from_checkpoint"),
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_compatible_combinations_pass(save_only_model, resume_from_checkpoint):
    training = TrainingConfig(
        save_only_model=save_only_model,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    check_resume_compatibility(training)


def test_default_config_is_compatible():
    """Default TrainingConfig must not trigger the conflict (would break every fresh run)."""
    check_resume_compatibility(TrainingConfig())
