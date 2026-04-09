from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    SHARED_DRIVE_ROOT,
    assert_port_available,
    build_shared_runtime_env,
    find_nvidia_egl_vendor_file,
    get_root,
    run_subprocess_step,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

TRAINING_STEPS = 2

README = REPO_ROOT / "examples/LIBERO/README.md"

DATASET_REL_PATH = pathlib.Path("examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot")
DATASET_ROOT = REPO_ROOT / DATASET_REL_PATH
SHARED_DATASETS_ROOT = SHARED_DRIVE_ROOT / "datasets"

SHARED_DATASET_ROOT = SHARED_DATASETS_ROOT / DATASET_REL_PATH
MODEL_CHECKPOINT = pathlib.Path(f"/tmp/libero_spatial/checkpoint-{TRAINING_STEPS}")

LIBERO_REPO_PATH = REPO_ROOT / "external_dependencies/LIBERO"
SHARED_LIBERO_REPO = SHARED_DRIVE_ROOT / "repos/LIBERO"

DEFAULT_SERVER_STARTUP_SECONDS = 180.0


def _libero_submodule_initialized() -> bool:
    """Return True when the LIBERO submodule is properly git-initialized."""
    return (LIBERO_REPO_PATH / ".git").is_file()


def _git_modules_path(submodule_path: pathlib.Path) -> pathlib.Path | None:
    """Resolve the .git/modules/<name> path from a submodule's .git file."""
    git_file = submodule_path / ".git"
    if not git_file.is_file():
        return None
    content = git_file.read_text().strip()
    if not content.startswith("gitdir:"):
        return None
    rel = content[len("gitdir:") :].strip()
    return (submodule_path / rel).resolve()


def _prepare_libero_repo(env: dict[str, str]) -> None:
    """Populate external_dependencies/LIBERO, reusing shared cache when available.

    The cache stores both the working tree (which includes the .git pointer file)
    and the git modules directory, so that after restore git sees a fully
    initialized submodule and ``git submodule update --init`` is a fast no-op.
    """
    if _libero_submodule_initialized():
        return

    wt_cache = SHARED_LIBERO_REPO / "wt"
    modules_cache = SHARED_LIBERO_REPO / "modules"

    if (wt_cache / ".git").is_file() and modules_cache.exists():
        # Fast path: restore working tree and git modules from cache.
        print(f"[libero] restoring submodule from cache {wt_cache}", flush=True)
        shutil.copytree(wt_cache, LIBERO_REPO_PATH, dirs_exist_ok=True)
        modules_path = _git_modules_path(LIBERO_REPO_PATH)
        if modules_path is not None:
            modules_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_cache, modules_path, dirs_exist_ok=True)
        return

    # Slow path: git submodule init, then populate cache.
    run_subprocess_step(
        ["git", "submodule", "update", "--init", "external_dependencies/LIBERO"],
        step="libero_repo_init",
        cwd=REPO_ROOT,
        env=env,
        log_prefix="libero",
    )
    if SHARED_DRIVE_ROOT.exists():
        modules_path = _git_modules_path(LIBERO_REPO_PATH)
        print(f"[libero] caching submodule to {wt_cache}", flush=True)
        wt_cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(LIBERO_REPO_PATH, wt_cache, dirs_exist_ok=True)
        if modules_path is not None:
            modules_cache.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_path, modules_cache, dirs_exist_ok=True)


def _dataset_ready(dataset_root: pathlib.Path) -> bool:
    """Return True when the LIBERO dataset looks complete enough to reuse."""
    modality_path = dataset_root / "meta/modality.json"
    videos_dir = dataset_root / "videos"
    if not modality_path.is_file() or not videos_dir.is_dir():
        return False
    return next(videos_dir.rglob("*.mp4"), None) is not None


def _point_repo_dataset_to_shared() -> None:
    """Point the repo-local dataset path at the shared cached dataset."""
    if DATASET_ROOT.is_symlink():
        if DATASET_ROOT.resolve() == SHARED_DATASET_ROOT.resolve():
            return
        DATASET_ROOT.unlink()
    elif DATASET_ROOT.exists():
        # Keep an existing real local dataset intact rather than replacing it.
        return

    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.symlink_to(SHARED_DATASET_ROOT, target_is_directory=True)


def _prepare_libero_dataset(blocks: list, env: dict[str, str]) -> None:
    """Populate the LIBERO spatial dataset once on shared storage and reuse it."""
    if _dataset_ready(DATASET_ROOT):
        return

    if _dataset_ready(SHARED_DATASET_ROOT):
        _point_repo_dataset_to_shared()
        return

    download_code = find_block(
        blocks, "libero_spatial_no_noops_1.0.0_lerobot", language="bash"
    ).code
    if SHARED_DRIVE_ROOT.exists():
        download_code = download_code.replace(
            "examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/",
            f"{SHARED_DATASET_ROOT}/",
        )

    run_bash_blocks([download_code], cwd=REPO_ROOT, env=env)

    if _dataset_ready(SHARED_DATASET_ROOT):
        _point_repo_dataset_to_shared()
        return

    assert _dataset_ready(DATASET_ROOT), f"Expected LIBERO dataset at {DATASET_ROOT}"


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_libero_readme_workflow_executes_via_subprocess() -> None:
    """Run the LIBERO README finetune (libero_spatial) then server+client eval."""

    print(f"[egl] NVIDIA EGL vendor file: {find_nvidia_egl_vendor_file()}", flush=True)

    env = build_shared_runtime_env(
        "libero", extra_env={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    )
    blocks = extract_code_blocks(README)

    # Step 1: Download + copy modality once, preferring the shared mounted dataset cache.
    _prepare_libero_dataset(blocks, env)

    # Step 2: Finetune — inline README values are replaced to keep the run short
    finetune_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    find_block(blocks, "--output-dir /tmp/libero_spatial", language="bash").code,
                    "NUM_GPUS=8",
                    "NUM_GPUS=1",
                ),
                "MAX_STEPS=20000",
                f"MAX_STEPS={TRAINING_STEPS}",
            ),
            "SAVE_STEPS=1000",
            f"SAVE_STEPS={TRAINING_STEPS}",
        ),
        "GLOBAL_BATCH_SIZE=640",
        "GLOBAL_BATCH_SIZE=2",
    )
    run_bash_blocks(
        [finetune_code],
        cwd=REPO_ROOT,
        env={
            **env,
            "USE_WANDB": "0",
            "DATALOADER_NUM_WORKERS": "0",
            "SHARD_SIZE": "64",
            "NUM_SHARDS_PER_EPOCH": "1",
            # Limit to one GPU so HuggingFace Trainer uses plain single-device
            # mode instead of DataParallel, which breaks the model's device property.
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )
    assert MODEL_CHECKPOINT.exists(), (
        f"Expected model checkpoint after finetune: {MODEL_CHECKPOINT}"
    )

    # Step 3: Setup sim — populate LIBERO repo from shared cache if available
    _prepare_libero_repo(env)
    run_bash_blocks(
        [find_block(blocks, "setup_libero.sh", language="bash")],
        cwd=REPO_ROOT,
        env=env,
    )

    model_server_host = "127.0.0.1"
    model_server_port = 5552

    # Step 4: Server — inject test-specific flags and replace checkpoint path
    server_code = replace_once(
        replace_once(
            find_block(blocks, "/tmp/libero_spatial/checkpoint-20000", language="bash").code,
            "uv run python gr00t/eval/run_gr00t_server.py",
            "uv run --extra=dev python gr00t/eval/run_gr00t_server.py",
        ),
        "/tmp/libero_spatial/checkpoint-20000/",
        str(MODEL_CHECKPOINT),
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    # Step 5: Rollout — substitute test-safe values
    rollout_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    find_block(blocks, "libero_uv/.venv/bin/python", language="bash").code,
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
            os.getenv("LIBERO_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
        ),
    )

    try:
        simulation_result, _ = run_subprocess_step(
            ["bash", "-c", rollout_code],
            step="libero_rollout",
            cwd=REPO_ROOT,
            env=env,
            log_prefix="libero",
            failure_prefix="LIBERO rollout failed",
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
