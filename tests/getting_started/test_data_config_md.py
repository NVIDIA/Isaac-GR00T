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

from __future__ import annotations

from test_support.readme import extract_code_blocks, find_block
from test_support.runtime import build_shared_runtime_env, get_root, run_subprocess_step


REPO_ROOT = get_root()
DATA_CONFIG_README = REPO_ROOT / "getting_started" / "data_config.md"

_IMPORTS = (
    "from gr00t.data.types import ModalityConfig, ActionConfig, ActionFormat, ActionRepresentation, ActionType\n"
    "from gr00t.data.embodiment_tags import EmbodimentTag\n"
    "from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config\n"
    "MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)\n"
)
_REGISTER = "register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)\n"


def test_complete_so100_config() -> None:
    """The complete SO-100 config example in data_config.md executes without error."""
    blocks = extract_code_blocks(DATA_CONFIG_README)
    so100 = find_block(blocks, "so100_config = {", language="python", occurrence=2)
    code = _IMPORTS + "\n" + so100.code + "\n" + _REGISTER
    env = build_shared_runtime_env("data_config")
    run_subprocess_step(
        ["uv", "run", "python", "-c", code],
        step="so100_config",
        cwd=REPO_ROOT,
        env=env,
    )
