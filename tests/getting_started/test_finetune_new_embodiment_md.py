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

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import build_shared_runtime_env, get_root, run_subprocess_step


REPO_ROOT = get_root()
FINETUNE_README = REPO_ROOT / "getting_started" / "finetune_new_embodiment.md"

_TRAINING_STEPS = 2
_CHECKPOINT = f"/tmp/so100/checkpoint-{_TRAINING_STEPS}"


# ---------------------------------------------------------------------------
# Step 2: modality configuration + registration
# ---------------------------------------------------------------------------


def test_modality_config_block() -> None:
    """The SO-100 modality config block in finetune_new_embodiment.md executes without error."""
    blocks = extract_code_blocks(FINETUNE_README)
    config_block = find_block(blocks, "register_modality_config", language="python")
    env = build_shared_runtime_env("finetune_new_embodiment")
    run_subprocess_step(
        ["uv", "run", "python", "-c", config_block.code],
        step="modality_config_block",
        cwd=REPO_ROOT,
        env=env,
    )


# ---------------------------------------------------------------------------
# Steps 3 + 4: fine-tune then open-loop eval
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_open_loop_eval() -> None:
    """Run Step 3 (finetune) then Step 4 (open-loop eval) from finetune_new_embodiment.md."""
    blocks = extract_code_blocks(FINETUNE_README)
    env = build_shared_runtime_env("finetune_new_embodiment")

    # Step 3: finetune with minimal steps to produce a NEW_EMBODIMENT checkpoint.
    finetune_cmd = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    replace_once(
                        find_block(blocks, "--base-model-path", language="bash").code,
                        "--save-steps 2000",
                        f"--save-steps {_TRAINING_STEPS}",
                    ),
                    "--max-steps 2000",
                    f"--max-steps {_TRAINING_STEPS}",
                ),
                "--use-wandb",
                "--no-use-wandb",
            ),
            "--global-batch-size 32",
            "--global-batch-size 2",
        ),
        "--dataloader-num-workers 4",
        "--dataloader-num-workers 0",
    )
    run_bash_blocks([finetune_cmd], cwd=REPO_ROOT, env=env)

    # Step 4: open-loop eval against the freshly produced checkpoint.
    eval_cmd = replace_once(
        replace_once(
            find_block(blocks, "open_loop_eval.py", language="bash").code,
            "/tmp/so100/checkpoint-2000",
            _CHECKPOINT,
        ),
        "--steps 400",
        "--steps 5",
    )
    run_bash_blocks([eval_cmd], cwd=REPO_ROOT, env=env)
