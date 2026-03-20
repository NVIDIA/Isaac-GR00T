from __future__ import annotations

import os
import platform
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    assert_port_available,
    build_shared_runtime_env,
    get_root,
    has_rt_core_gpu,
    run_subprocess_step,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

README = REPO_ROOT / "examples/SimplerEnv/README.md"

# sapien==2.2.2 (required by ManiSkill2_real2sim) ships x86_64 wheels only.
pytestmark = pytest.mark.skipif(
    platform.machine() != "x86_64",
    reason="SimplerEnv depends on sapien which has no aarch64 wheels",
)

DEFAULT_SERVER_STARTUP_SECONDS = 180.0


def _run_simplerenv_eval(
    env: dict,
    blocks: list,
    server_model_key: str,
    client_env_name_old: str,
    client_env_name_new: str,
    server_startup_env_var: str,
) -> None:
    """Shared helper: setup sim, start server, run rollout, assert results."""
    # Step 1: Setup sim (shared across both benchmarks)
    run_bash_blocks(
        [find_block(blocks, "setup_SimplerEnv.sh", language="bash")],
        cwd=REPO_ROOT,
        env=env,
    )

    model_server_host = "127.0.0.1"
    model_server_port = 5559

    # Step 2: Server — inject test-specific flags
    server_code = replace_once(
        find_block(blocks, server_model_key, language="bash").code,
        "uv run python gr00t/eval/run_gr00t_server.py",
        "uv run --extra=dev python gr00t/eval/run_gr00t_server.py",
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    # Step 3: Rollout — substitute test-safe values
    rollout_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    replace_once(
                        find_block(blocks, client_env_name_old, language="bash").code,
                        "--n_episodes 10",
                        "--n_episodes 1",
                    ),
                    "--policy_client_port 5555",
                    f"--policy_client_port {model_server_port}",
                ),
                "--max_episode_steps=300",
                "--max_episode_steps=2",
            ),
            "--n_envs 5",
            "--n_envs 1",
        ),
        client_env_name_old,
        client_env_name_new,
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
        timeout_s=float(os.getenv(server_startup_env_var, str(DEFAULT_SERVER_STARTUP_SECONDS))),
    )

    try:
        if has_rt_core_gpu():
            simulation_result, _ = run_subprocess_step(
                ["bash", "-c", rollout_code],
                step="simplerenv_rollout",
                cwd=REPO_ROOT,
                env=env,
                log_prefix="simplerenv",
                failure_prefix="SimplerEnv rollout failed",
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


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_simplerenv_fractal_readme_eval_flow() -> None:
    """Run the SimplerEnv README server+client eval using the remote fractal (Google robot) checkpoint."""
    env = build_shared_runtime_env("simplerenv")
    blocks = extract_code_blocks(README)
    _run_simplerenv_eval(
        env=env,
        blocks=blocks,
        server_model_key="nvidia/GR00T-N1.6-fractal",
        client_env_name_old="simpler_env_google/google_robot_pick_coke_can",
        client_env_name_new="simpler_env_google/google_robot_pick_coke_can",
        server_startup_env_var="SIMPLERENV_SERVER_STARTUP_SECONDS",
    )


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_simplerenv_bridge_readme_eval_flow() -> None:
    """Run the SimplerEnv README server+client eval using the remote bridge (WidowX robot) checkpoint."""
    env = build_shared_runtime_env("simplerenv")
    blocks = extract_code_blocks(README)
    _run_simplerenv_eval(
        env=env,
        blocks=blocks,
        server_model_key="nvidia/GR00T-N1.6-bridge",
        client_env_name_old="simpler_env_widowx/widowx_spoon_on_towel",
        client_env_name_new="simpler_env_widowx/widowx_spoon_on_towel",
        server_startup_env_var="SIMPLERENV_SERVER_STARTUP_SECONDS",
    )
