from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    TEST_CACHE_PATH,
    assert_port_available,
    build_shared_runtime_env,
    get_root,
    run_subprocess_step,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

LOGGER = logging.getLogger(__name__)

README = REPO_ROOT / "examples/robocasa/README.md"

ROBOCASA_SUBMODULE_PATH = REPO_ROOT / "external_dependencies/robocasa"
SHARED_ROBOCASA_REPO = TEST_CACHE_PATH / "repos/robocasa"

ROBOCASA_ASSETS_REPO_DIR = REPO_ROOT / "external_dependencies/robocasa/robocasa/models/assets"
ROBOCASA_ASSETS_SHARED_DIR = TEST_CACHE_PATH / "robocasa-assets"
# Version file written alongside cached assets to detect robocasa submodule updates.
_ASSETS_VERSION_FILE = ROBOCASA_ASSETS_SHARED_DIR / ".robocasa_commit"


def _robocasa_submodule_commit() -> str:
    """Return the robocasa submodule commit hash recorded in the main repo HEAD.

    Uses ``git ls-tree`` against the main repo so this works even before the
    submodule is initialized.
    """
    try:
        result = subprocess.run(
            ["git", "ls-tree", "HEAD", "external_dependencies/robocasa"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        # output: "160000 commit <hash>\texternal_dependencies/robocasa"
        parts = result.stdout.split()
        return parts[2] if len(parts) >= 3 else "unknown"
    except Exception:
        return "unknown"


def _shared_asset_dirs() -> list[pathlib.Path]:
    """Return all top-level subdirectories in the shared asset cache."""
    if not ROBOCASA_ASSETS_SHARED_DIR.is_dir():
        return []
    return [p for p in ROBOCASA_ASSETS_SHARED_DIR.iterdir() if p.is_dir()]


def _shared_assets_ready() -> bool:
    """Return True when the shared asset cache is present, non-empty, and matches
    the current robocasa submodule commit.

    A stale cache (e.g. from a previous robocasa version that lacked newer
    fixture files) is treated as not-ready so the assets are re-downloaded.
    """
    if not _ASSETS_VERSION_FILE.is_file():
        return False
    if _ASSETS_VERSION_FILE.read_text().strip() != _robocasa_submodule_commit():
        return False
    for d in _shared_asset_dirs():
        try:
            if next((f for f in d.rglob("*") if f.is_file()), None) is not None:
                return True
        except OSError:
            pass
    return False


def _assert_required_assets_present() -> None:
    """Raise if the repo asset directory is empty."""
    if not ROBOCASA_ASSETS_REPO_DIR.is_dir() or not any(ROBOCASA_ASSETS_REPO_DIR.iterdir()):
        raise RuntimeError(f"RoboCasa assets missing at {ROBOCASA_ASSETS_REPO_DIR}")


def _point_repo_assets_to_shared() -> None:
    """Symlink all shared asset subdirectories into the repo asset path."""
    ROBOCASA_ASSETS_REPO_DIR.mkdir(parents=True, exist_ok=True)
    for shared_dir in _shared_asset_dirs():
        repo_dir = ROBOCASA_ASSETS_REPO_DIR / shared_dir.name
        if repo_dir.is_symlink():
            if repo_dir.resolve() == shared_dir.resolve():
                continue
            repo_dir.unlink()
        elif repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.symlink_to(shared_dir, target_is_directory=True)


def _move_repo_assets_to_shared() -> None:
    """Move all downloaded repo asset subdirectories into the shared cache.

    Also writes a version file recording the current robocasa submodule commit
    so that stale caches are detected when the submodule is updated.
    """
    ROBOCASA_ASSETS_SHARED_DIR.mkdir(parents=True, exist_ok=True)
    if not ROBOCASA_ASSETS_REPO_DIR.is_dir():
        return
    for src in ROBOCASA_ASSETS_REPO_DIR.iterdir():
        if not src.is_dir() or src.is_symlink():
            continue
        dst = ROBOCASA_ASSETS_SHARED_DIR / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        # Use cp -r + rm -rf instead of shutil.move: the repo and the shared
        # PVC are on different filesystems, so shutil.move falls back to a slow
        # Python-level copytree that times out on large asset dirs (e.g.
        # generative_textures with thousands of PNGs).
        subprocess.run(["cp", "-r", str(src), str(dst)], check=True)
        shutil.rmtree(str(src))
    _ASSETS_VERSION_FILE.write_text(_robocasa_submodule_commit())


def _remove_dangling_repo_asset_symlinks() -> None:
    """Delete repo asset symlinks that point to missing targets."""
    if not ROBOCASA_ASSETS_REPO_DIR.is_dir():
        return
    for repo_dir in ROBOCASA_ASSETS_REPO_DIR.iterdir():
        if repo_dir.is_symlink() and not repo_dir.exists():
            repo_dir.unlink()


def _robocasa_submodule_initialized() -> bool:
    return (ROBOCASA_SUBMODULE_PATH / ".git").is_file()


def _git_modules_path(submodule_path: pathlib.Path) -> pathlib.Path | None:
    git_file = submodule_path / ".git"
    if not git_file.is_file():
        return None
    content = git_file.read_text().strip()
    if not content.startswith("gitdir:"):
        return None
    rel = content[len("gitdir:") :].strip()
    return (submodule_path / rel).resolve()


def _prepare_robocasa_repo(env: dict[str, str]) -> None:
    """Populate external_dependencies/robocasa from shared cache, or init and cache it."""
    if _robocasa_submodule_initialized():
        return

    wt_cache = SHARED_ROBOCASA_REPO / "wt"
    modules_cache = SHARED_ROBOCASA_REPO / "modules"

    if (wt_cache / ".git").is_file() and modules_cache.exists():
        print(f"[robocasa] restoring submodule from cache {wt_cache}", flush=True)
        shutil.copytree(wt_cache, ROBOCASA_SUBMODULE_PATH, dirs_exist_ok=True)
        modules_path = _git_modules_path(ROBOCASA_SUBMODULE_PATH)
        if modules_path is not None:
            modules_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_cache, modules_path, dirs_exist_ok=True)
        return

    # The directory may exist but be uninitialized (no .git file) due to CI
    # checkout strategies that populate submodule dirs without git-initializing them.
    # Remove it so git submodule update --init can clone cleanly.
    if ROBOCASA_SUBMODULE_PATH.exists() and not _robocasa_submodule_initialized():
        shutil.rmtree(ROBOCASA_SUBMODULE_PATH)

    run_subprocess_step(
        ["git", "submodule", "update", "--init", "external_dependencies/robocasa"],
        step="robocasa_repo_init",
        cwd=REPO_ROOT,
        env=env,
        log_prefix="robocasa",
    )
    if TEST_CACHE_PATH.exists():
        modules_path = _git_modules_path(ROBOCASA_SUBMODULE_PATH)
        print(f"[robocasa] caching submodule to {wt_cache}", flush=True)
        wt_cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROBOCASA_SUBMODULE_PATH, wt_cache, dirs_exist_ok=True)
        if modules_path is not None:
            modules_cache.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_path, modules_cache, dirs_exist_ok=True)


def _build_runtime_env(skip_download_assets: str) -> dict[str, str]:
    """Build the runtime environment used by setup, model server, and rollout."""
    return build_shared_runtime_env(
        "robocasa",
        extra_env={
            "SKIP_DOWNLOAD_ASSETS": skip_download_assets,
            "INSTALL_FLASH_ATTN": "0",
        },
    )


@pytest.mark.gpu
@pytest.mark.timeout(2700)
def test_robocasa_readme_eval_flow() -> None:
    """Run the RoboCasa README server+client eval using the remote GR00T-N1.6-3B checkpoint."""

    # Environment setup:
    # 1) If assets already exist on shared PVC, reuse them by symlinking.
    # 2) Otherwise run setup with download enabled.
    shared_assets_ready = _shared_assets_ready()
    if shared_assets_ready:
        _point_repo_assets_to_shared()
    else:
        _remove_dangling_repo_asset_symlinks()

    skip_download_assets = "1" if shared_assets_ready else "0"
    env = _build_runtime_env(skip_download_assets=skip_download_assets)

    # Ensure submodule is git-initialized from cache before setup script runs.
    _prepare_robocasa_repo(env)

    blocks = extract_code_blocks(README)

    # Step 1: Setup sim
    LOGGER.info("Running setup script")
    run_bash_blocks(
        [find_block(blocks, "setup_RoboCasa.sh", language="bash")],
        cwd=REPO_ROOT,
        env=env,
        force_yes=True,
    )

    # When setup performs a fresh download, move those assets into shared PVC
    # so subsequent runs can skip download and reuse the cached shared copy.
    if not shared_assets_ready:
        _move_repo_assets_to_shared()
        _point_repo_assets_to_shared()

    _assert_required_assets_present()

    model_server_host = "127.0.0.1"
    model_server_port = 5551

    # Step 2: Server — inject test-specific flags
    server_code = replace_once(
        find_block(blocks, "ROBOCASA_PANDA_OMRON", language="bash").code,
        "uv run python gr00t/eval/run_gr00t_server.py",
        "uv run --extra=dev python gr00t/eval/run_gr00t_server.py",
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    # Step 3: Rollout — substitute test-safe values
    rollout_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    find_block(blocks, "robocasa_uv/.venv/bin/python", language="bash").code,
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
        env.get("UV_PROJECT_ENVIRONMENT", "<unset>"),
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
            os.getenv("ROBOCASA_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
        ),
    )

    try:
        simulation_result, _ = run_subprocess_step(
            ["bash", "-c", rollout_code],
            step="robocasa_rollout",
            cwd=REPO_ROOT,
            env=env,
            log_prefix="robocasa",
            failure_prefix="RoboCasa rollout failed",
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
