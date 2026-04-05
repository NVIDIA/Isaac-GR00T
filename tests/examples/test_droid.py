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

import os
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    assert_port_available,
    build_shared_runtime_env,
    get_root,
    start_server_process,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

README = REPO_ROOT / "examples/DROID/README.md"


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_droid_readme_server_starts() -> None:
    """Verify the DROID inference server starts and accepts connections."""

    env = build_shared_runtime_env("droid")
    blocks = extract_code_blocks(README)

    model_server_host = "127.0.0.1"
    model_server_port = 5557

    # Build server command — README uses --use_sim_policy_wrapper (underscore)
    server_code = replace_once(
        find_block(blocks, "run_gr00t_server.py", language="bash").code,
        "uv run python gr00t/eval/run_gr00t_server.py",
        "uv run --extra=dev python gr00t/eval/run_gr00t_server.py",
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    assert_port_available(model_server_host, model_server_port)
    model_server_proc, server_log = start_server_process(server_code, cwd=REPO_ROOT, env=env)
    try:
        wait_for_server_ready(
            proc=model_server_proc,
            host=model_server_host,
            port=model_server_port,
            timeout_s=float(
                os.getenv("DROID_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
            ),
            server_log=server_log,
        )
    finally:
        if model_server_proc.poll() is None:
            model_server_proc.terminate()
            try:
                model_server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                model_server_proc.kill()
                model_server_proc.wait(timeout=15)
