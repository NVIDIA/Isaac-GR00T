from __future__ import annotations

import os
import platform
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    assert_port_available,
    build_shared_runtime_env,
    get_root,
    has_rt_core_gpu,
    run_subprocess_step,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

README = REPO_ROOT / "examples/BEHAVIOR/README.md"

# pymeshlab (via omnigibson) and Isaac Sim have no aarch64 wheels.
pytestmark = pytest.mark.skipif(
    platform.machine() != "x86_64",
    reason="BEHAVIOR depends on omnigibson/Isaac Sim which have no aarch64 wheels",
)


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_behavior_readme_eval_flow() -> None:
    """Run the BEHAVIOR README server+client eval using the remote BEHAVIOR1k checkpoint."""

    env = build_shared_runtime_env("behavior")
    blocks = extract_code_blocks(README)

    # Step 1: Setup — clone BEHAVIOR-1K + run setup_uv.sh
    run_bash_blocks(
        [find_block(blocks, "BEHAVIOR-1K", language="bash")],
        cwd=REPO_ROOT,
        env=env,
    )

    # Step 2: Download test instances
    run_bash_blocks(
        [find_block(blocks, "prepare_test_instances.py", language="bash")],
        cwd=REPO_ROOT,
        env=env,
    )

    model_server_host = "127.0.0.1"
    model_server_port = 5558

    # Step 3: Server — the README block includes uv sync + uv pip install before the server cmd.
    # Inject --extra=dev and test-specific host/port.
    server_code = replace_once(
        find_block(blocks, "nvidia/GR00T-N1.6-BEHAVIOR1k", language="bash").code,
        "uv run gr00t/eval/run_gr00t_server.py",
        f"uv run --extra=dev python gr00t/eval/run_gr00t_server.py --host {model_server_host} --port {model_server_port}",
    )
    server_code += " --device cuda:0"

    # Step 4: Rollout — uses uv run python (no separate venv), substitute test-safe values
    rollout_code = replace_once(
        replace_once(
            replace_once(
                find_block(blocks, "sim_behavior_r1_pro/turning_on_radio", language="bash").code,
                "--n_episodes 10",
                "--n_episodes 1",
            ),
            "--policy_client_port 5555",
            f"--policy_client_port {model_server_port}",
        ),
        "--max_episode_steps=999999999",
        "--max_episode_steps=2",
    )

    assert_port_available(model_server_host, model_server_port)
    model_server_proc = subprocess.Popen(
        ["bash", "-c", server_code],
        cwd=REPO_ROOT,
        env=env,
    )
    wait_for_server_ready(
        proc=model_server_proc,
        host=model_server_host,
        port=model_server_port,
        timeout_s=float(
            os.getenv("BEHAVIOR_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
        ),
    )

    try:
        if has_rt_core_gpu():
            simulation_result, _ = run_subprocess_step(
                ["bash", "-c", rollout_code],
                step="behavior_rollout",
                cwd=REPO_ROOT,
                env=env,
                log_prefix="behavior",
                failure_prefix="BEHAVIOR rollout failed",
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
