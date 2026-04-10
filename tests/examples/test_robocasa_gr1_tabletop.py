import logging
import os
import pathlib
import shutil
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    SHARED_DRIVE_ROOT,
    assert_port_available,
    build_shared_runtime_env,
    get_root,
    run_subprocess_step,
    wait_for_server_ready,
)


REPO_ROOT = get_root()


LOGGER = logging.getLogger(__name__)

README = REPO_ROOT / "examples/robocasa-gr1-tabletop-tasks/README.md"
ROBOCASA_SUBMODULE_PATH = pathlib.Path("external_dependencies/robocasa-gr1-tabletop-tasks")
ROBOCASA_ASSETS_REPO_DIR = (
    REPO_ROOT / "external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/models/assets"
)

ROBOCASA_ASSETS_SHARED_DIR = SHARED_DRIVE_ROOT / "robocasa-gr1-tabletop-tasks/assets"

REQUIRED_ASSET_DIRS = (
    "textures",
    "fixtures",
    "objects/objaverse",
    "generative_textures",
    "objects/lightwheel",
    "objects/sketchfab",
)


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
        cwd=REPO_ROOT,
        check=True,
    )


def _point_repo_assets_to_shared() -> None:
    """Symlink heavy repo asset directories to their shared PVC counterparts.

    If the shared location contains symlinks back to the repo (from
    _move_repo_assets_to_shared), the repo paths already have the actual
    files and no symlink creation is needed.
    """
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
            # Check if shared_dir is a symlink pointing back to repo_dir.
            # If so, the repo already has the actual files and we should
            # keep them instead of creating circular symlinks.
            if shared_dir.is_symlink() and shared_dir.resolve() == repo_dir.resolve():
                # Shared points to repo, so repo already has the files.
                # No need to create symlinks.
                continue
            shutil.rmtree(repo_dir)

        repo_dir.symlink_to(shared_dir, target_is_directory=True)


def _move_repo_assets_to_shared() -> None:
    """Move downloaded repo asset directories into the shared PVC cache.

    Uses symlinks instead of copying to avoid cross-device copy overhead.
    The shared location will have symlinks pointing to the repo assets,
    which allows _shared_assets_ready() to detect that assets are available
    for subsequent runs.
    """
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
        # Use symlink instead of move to avoid cross-device copy overhead.
        # This creates a symlink from shared -> repo, allowing subsequent
        # runs to detect that assets are available without copying.
        dst.symlink_to(src, target_is_directory=True)


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


# may need to increase timeout since first run may need to download assets
@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_robocasa_gr1_tabletop_readme_eval_flow():
    """
    Tests the directions given in https://gitlab-master.nvidia.com/gr00t-release/Isaac-GR00T/-/blob/main/examples/robocasa-gr1-tabletop-tasks/README.md
    """

    _ensure_robocasa_submodule()

    # Environment setup:
    # 1) If assets already exist on shared PVC, reuse them by symlinking.
    # 2) Otherwise run setup with download enabled.
    shared_assets_ready = _shared_assets_ready()
    if shared_assets_ready:
        # Ensure setup sees required repo asset paths when downloads are skipped.
        _point_repo_assets_to_shared()
    else:
        _remove_dangling_repo_asset_symlinks()

    skip_download_assets = "1" if shared_assets_ready else "0"
    runtime_env = _build_runtime_env(
        skip_download_assets=skip_download_assets,
    )
    blocks = extract_code_blocks(README)

    LOGGER.info("Running setup script")
    run_bash_blocks(
        [find_block(blocks, "setup_RoboCasaGR1TabletopTasks.sh", language="bash")],
        cwd=REPO_ROOT,
        env=runtime_env,
    )

    # When setup performs a fresh download, move those assets into shared PVC
    # so subsequent runs can skip download and reuse the cached shared copy.
    if not shared_assets_ready:
        _move_repo_assets_to_shared()
        _point_repo_assets_to_shared()

    _assert_required_assets_present()

    model_server_host = "127.0.0.1"
    model_server_port = 5556

    # Build server command from README, injecting test-specific flags.
    server_code = replace_once(
        find_block(blocks, "run_gr00t_server.py", language="bash").code,
        "uv run python",
        "uv run --extra=dev python",
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    # Build rollout command from README, substituting test-safe values.
    rollout_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    find_block(blocks, "rollout_policy.py", language="bash").code,
                    "--n_episodes 10",
                    "--n_episodes 1",
                ),
                "--policy_client_port 5555",
                f"--policy_client_port {model_server_port}",
            ),
            "--max_episode_steps=720",
            "--max_episode_steps=2",
        ),
        "--n_envs 5",
        "--n_envs 1",
    )

    LOGGER.info(
        "Starting model server process (UV_PROJECT_ENVIRONMENT=%s)",
        runtime_env.get("UV_PROJECT_ENVIRONMENT", "<unset>"),
    )
    assert_port_available(model_server_host, model_server_port)
    model_server_proc = subprocess.Popen(
        ["bash", "-c", server_code],
        cwd=REPO_ROOT,
        env=runtime_env,
    )
    wait_for_server_ready(
        proc=model_server_proc,
        host=model_server_host,
        port=model_server_port,
        timeout_s=float(
            os.getenv("ROBOCASA_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
        ),
    )

    try:
        LOGGER.info("Starting simulation process")
        simulation_result, _ = run_subprocess_step(
            ["bash", "-c", rollout_code],
            step="simulation_rollout",
            cwd=REPO_ROOT,
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
