import logging
import os
import pathlib
import shutil
import socket
import subprocess
import time

import pytest
from tests.examples.utils import build_shared_runtime_env, run_subprocess_step


LOGGER = logging.getLogger(__name__)


ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_SCRIPT = ROOT / "gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh"
MODEL_SERVER_SCRIPT = ROOT / "gr00t/eval/run_gr00t_server.py"
ROLLOUT_SCRIPT = ROOT / "gr00t/eval/rollout_policy.py"
ROBOCASA_PYTHON = ROOT / "gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
ROBOCASA_SUBMODULE_PATH = pathlib.Path("external_dependencies/robocasa-gr1-tabletop-tasks")
ROBOCASA_ASSETS_REPO_DIR = (
    ROOT / "external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/models/assets"
)

SHARED_DRIVE_ROOT = pathlib.Path("/shared")
ROBOCASA_ASSETS_SHARED_DIR = SHARED_DRIVE_ROOT / "robocasa-gr1-tabletop-tasks/assets"

REQUIRED_ASSET_DIRS = (
    "textures",
    "fixtures",
    "objects/objaverse",
    "generative_textures",
    "objects/lightwheel",
    "objects/sketchfab",
)
DEFAULT_SERVER_STARTUP_SECONDS = 180.0


def _shared_assets_ready() -> bool:
    """Return True when all required shared asset directories are populated."""
    return all((ROBOCASA_ASSETS_SHARED_DIR / rel).is_dir() for rel in REQUIRED_ASSET_DIRS)


def _assert_required_assets_present() -> None:
    """Raise if required RoboCasa asset directories are missing in the repo path."""
    missing_dirs = [
        str(ROBOCASA_ASSETS_REPO_DIR / rel)
        for rel in REQUIRED_ASSET_DIRS
        if not (ROBOCASA_ASSETS_REPO_DIR / rel).is_dir()
    ]
    if missing_dirs:
        missing = "\n".join(missing_dirs)
        raise RuntimeError(f"Missing required RoboCasa assets:\n{missing}")


def _ensure_robocasa_submodule() -> None:
    """Ensure the RoboCasa tabletop tasks submodule is initialized."""
    subprocess.run(
        ["git", "submodule", "update", "--init", str(ROBOCASA_SUBMODULE_PATH)],
        cwd=ROOT,
        check=True,
    )


def _point_repo_assets_to_shared() -> None:
    """Symlink heavy repo asset directories to their shared PVC counterparts."""
    ROBOCASA_ASSETS_SHARED_DIR.parent.mkdir(parents=True, exist_ok=True)
    ROBOCASA_ASSETS_REPO_DIR.mkdir(parents=True, exist_ok=True)

    # Keep static repository assets (e.g. arenas/*.xml) in place and remap only
    # large downloaded directories to the shared cache.
    for rel in REQUIRED_ASSET_DIRS:
        repo_dir = ROBOCASA_ASSETS_REPO_DIR / rel
        shared_dir = ROBOCASA_ASSETS_SHARED_DIR / rel
        if not shared_dir.is_dir():
            raise RuntimeError(f"Missing shared asset directory: {shared_dir}")

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.is_symlink():
            if repo_dir.resolve() == shared_dir.resolve():
                continue
            repo_dir.unlink()
        elif repo_dir.exists():
            shutil.rmtree(repo_dir)

        repo_dir.symlink_to(shared_dir, target_is_directory=True)


def _move_repo_assets_to_shared() -> None:
    """Move downloaded repo asset directories into the shared PVC cache."""
    ROBOCASA_ASSETS_SHARED_DIR.mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_ASSET_DIRS:
        src = ROBOCASA_ASSETS_REPO_DIR / rel
        dst = ROBOCASA_ASSETS_SHARED_DIR / rel
        if not src.is_dir():
            raise RuntimeError(f"Missing downloaded asset directory for move: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))


def _remove_dangling_repo_asset_symlinks() -> None:
    """Delete repo asset symlinks that point to missing targets."""
    for rel in REQUIRED_ASSET_DIRS:
        repo_dir = ROBOCASA_ASSETS_REPO_DIR / rel
        if repo_dir.is_symlink() and not repo_dir.exists():
            repo_dir.unlink()


def _build_runtime_env(
    skip_download_assets: str,
) -> dict[str, str]:
    """Build the runtime environment used by setup, model server, and rollout."""
    return build_shared_runtime_env(
        "robocasa-gr1-tabletop",
        extra_env={
            "SKIP_DOWNLOAD_ASSETS": skip_download_assets,
            # not needed in simulation since it doesn't run models.
            "INSTALL_FLASH_ATTN": "0",
        },
    )


def _wait_for_server_ready(
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout_s: float,
) -> None:
    """Wait until the server accepts connections or fail fast."""
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            raise AssertionError(f"Model server failed to start.\nreturncode={proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                elapsed = time.monotonic() - deadline + timeout_s
                print(f"Model server is ready to accept connections after {elapsed:.1f}s.")
                return
        except OSError:
            if time.monotonic() >= deadline:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=15)
                raise AssertionError(
                    "Model server did not become ready before timeout.\n"
                    f"timeout_seconds={timeout_s}\n"
                    "Set ROBOCASA_SERVER_STARTUP_SECONDS to override."
                )
            time.sleep(0.5)


# may need to increase timeout since first run may need to download assets
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_robocasa_gr1_tabletop_readme_eval_flow():
    """
    Tests the directions given in https://gitlab-master.nvidia.com/gr00t-release/Isaac-GR00T/-/blob/main/examples/robocasa-gr1-tabletop-tasks/README.md
    """

    _ensure_robocasa_submodule()

    # Environment setup:
    # 1) If assets already exist on shared PVC, reuse them by symlinking.
    # 2) Otherwise run setup with download enabled.
    shared_assets_ready = _shared_assets_ready()
    if not shared_assets_ready and not SHARED_DRIVE_ROOT.exists():
        pytest.skip(
            "Shared asset drive not available (/shared); "
            "this test requires either pre-cached assets or a CI environment with /shared mounted."
        )
    if shared_assets_ready:
        # Ensure setup sees required repo asset paths when downloads are skipped.
        _point_repo_assets_to_shared()
    else:
        _remove_dangling_repo_asset_symlinks()

    skip_download_assets = "1" if shared_assets_ready else "0"
    runtime_env = _build_runtime_env(
        skip_download_assets=skip_download_assets,
    )
    LOGGER.info("Running setup script")
    run_subprocess_step(
        ["bash", str(SETUP_SCRIPT)],
        step="setup_robocasa",
        cwd=ROOT,
        env=runtime_env,
        log_prefix="robocasa",
        failure_prefix="RoboCasa setup step failed",
        output_tail_chars=4000,
    )

    # When setup performs a fresh download, move those assets into shared PVC
    # so subsequent runs can skip download and reuse the cached shared copy.
    if not shared_assets_ready:
        _move_repo_assets_to_shared()
        _point_repo_assets_to_shared()

    _assert_required_assets_present()

    # command to run the model server, this is queried by the simulation.
    model_server_host = "127.0.0.1"
    model_server_port = "5551"
    model_server_cmd = [
        "uv",
        "run",
        "--extra=dev",
        "python",
        str(MODEL_SERVER_SCRIPT),
        "--model-path",
        "nvidia/GR00T-N1.6-3B",
        "--embodiment-tag",
        "GR1",
        "--use-sim-policy-wrapper",
        "--device",
        "cuda:0",
        "--host",
        model_server_host,
        "--port",
        model_server_port,
    ]

    # command to run the simulation
    # uses the venv created in SETUP_SCRIPT
    simulation_cmd = [
        str(ROBOCASA_PYTHON),
        str(ROLLOUT_SCRIPT),
        "--n_episodes",
        "1",
        "--policy_client_host",
        model_server_host,
        "--policy_client_port",
        model_server_port,
        "--max_episode_steps=2",
        "--env_name",
        "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
        "--n_action_steps",
        "8",
        "--n_envs",
        "1",
    ]

    LOGGER.info(
        "Starting model server process (UV_PROJECT_ENVIRONMENT=%s)",
        runtime_env.get("UV_PROJECT_ENVIRONMENT", "<unset>"),
    )
    model_server_proc = subprocess.Popen(
        model_server_cmd,
        cwd=ROOT,
        env=runtime_env,
    )
    _wait_for_server_ready(
        proc=model_server_proc,
        host=model_server_host,
        port=int(model_server_port),
        timeout_s=float(
            os.getenv("ROBOCASA_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
        ),
    )

    try:
        LOGGER.info("Starting simulation process")
        simulation_result, _ = run_subprocess_step(
            simulation_cmd,
            step="simulation_rollout",
            cwd=ROOT,
            env=runtime_env,
            log_prefix="robocasa",
            failure_prefix="Simulation rollout command failed",
            output_tail_chars=4000,
        )
        simulation_output = (simulation_result.stdout or "") + (simulation_result.stderr or "")
        assert simulation_result.returncode == 0, (
            "Simulation rollout command failed.\n"
            f"returncode={simulation_result.returncode}\n"
            f"output_tail=\n{simulation_output[-4000:]}"
        )
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
