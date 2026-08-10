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
"""Standalone AutoResearch primitives."""

# Public imports define the small API other packages should depend on.
from skills.autoresearch.config import StudyConfig
from skills.autoresearch.study_state import (
    Objective,
    StageState,
    StudyState,
    Trial,
    make_trial_id,
)

# Keep exported names explicit so internal controller pieces remain private.
__all__ = [
    "Objective",
    "StageState",
    "StudyConfig",
    "StudyState",
    "Trial",
    "make_trial_id",
]
