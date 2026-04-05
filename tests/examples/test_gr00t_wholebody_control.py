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
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    assert_port_available,
    build_shared_runtime_env,
    find_nvidia_egl_vendor_file,
    get_root,
    run_subprocess_step,
    start_server_process,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

README = REPO_ROOT / "examples/GR00T-WholeBodyControl/README.md"


@pytest.mark.gpu
@pytest.mark.timeout(1800)
@pytest.mark.skip(
    reason="UNITREE_G1 is a posttrain tag requiring a finetuned checkpoint; "
    "the base model's pretrain tags (e.g. REAL_G1) have incompatible video "
    "horizon with the G1 sim environment. Re-enable when an N1.7 G1 checkpoint is available."
)
def test_gr00t_wholebody_control_readme_eval_flow() -> None:
    """Run the G1 LocoManipulation README server+client eval using the remote checkpoint."""

    print(f"[egl] NVIDIA EGL vendor file: {find_nvidia_egl_vendor_file()}", flush=True)

    env = build_shared_runtime_env(
        "gr00t-wholebody-control", extra_env={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    )
    blocks = extract_code_blocks(README)

    # Step 1: Setup sim
    run_bash_blocks(
        [find_block(blocks, "setup_GR00T_WholeBodyControl.sh", language="bash")],
        cwd=REPO_ROOT,
        env=env,
        force_yes=True,
    )

    model_server_host = "127.0.0.1"
    model_server_port = 5554

    # Step 2: Server — use the base model with REAL_G1 pretrain tag (no finetuned
    # G1 checkpoint for N1.7 yet; UNITREE_G1 is a posttrain tag that requires one).
    server_code = replace_once(
        replace_once(
            replace_once(
                find_block(blocks, "nvidia/GR00T-N1.6-G1-PnPAppleToPlate", language="bash").code,
                "uv run python gr00t/eval/run_gr00t_server.py",
                "uv run --extra=dev python gr00t/eval/run_gr00t_server.py",
            ),
            "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            "nvidia/GR00T-N1.7-3B",
        ),
        "--embodiment-tag UNITREE_G1",
        "--embodiment-tag REAL_G1",
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    # Step 3: Rollout — README has no host/port flags; inject port and substitute test-safe values.
    # Also add --policy-client-host and --policy-client-port since the test server uses a non-default port.
    rollout_code = replace_once(
        replace_once(
            find_block(blocks, "GR00T-WholeBodyControl_uv/.venv/bin/python", language="bash").code,
            "--n-episodes 10",
            "--n-episodes 1",
        ),
        "--max-episode-steps 1440",
        "--max-episode-steps 2",
    )
    rollout_code += f" --policy-client-host {model_server_host} --policy-client-port {model_server_port} --n-envs 1"

    assert_port_available(model_server_host, model_server_port)
    model_server_proc, server_log = start_server_process(server_code, cwd=REPO_ROOT, env=env)
    wait_for_server_ready(
        proc=model_server_proc,
        host=model_server_host,
        port=model_server_port,
        timeout_s=float(
            os.getenv("G1_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
        ),
        server_log=server_log,
    )

    try:
        simulation_result, _ = run_subprocess_step(
            ["bash", "-c", rollout_code],
            step="g1_rollout",
            cwd=REPO_ROOT,
            env=env,
            log_prefix="gr00t-wholebody-control",
            failure_prefix="G1 LocoManipulation rollout failed",
            output_tail_chars=4000,
        )
        simulation_output = (simulation_result.stdout or "") + (simulation_result.stderr or "")
        assert "results:" in simulation_output, (
            "Simulation output did not include expected 'results:' marker.\n"
            f"output_tail=\n{simulation_output[-4000:]}"
        )
        assert "success rate:" in simulation_output, (
            "Simulation output did not include expected 'success rate:' marker.\n"
            f"output_tail=\n{simulation_output[-4000:]}"
        )
    finally:
        if model_server_proc.poll() is None:
            model_server_proc.terminate()
            try:
                model_server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                model_server_proc.kill()
                model_server_proc.wait(timeout=15)
