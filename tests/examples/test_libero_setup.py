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

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "gr00t/eval/sim/LIBERO/setup_libero.sh"


def test_libero_setup_pins_mujoco_compatible_with_robosuite_1_4() -> None:
    """robosuite 1.4 uses the MuJoCo 2.x ``mj_fullM`` Python signature."""
    setup_script = SETUP_SCRIPT.read_text()

    assert "mujoco==2.3.7" in setup_script
