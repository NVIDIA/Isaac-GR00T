"""
Standalone inference smoke test.

Runs standalone_inference_script.py in PyTorch mode on the bundled
libero_demo dataset for a few steps.  Verifies that the full inference
pipeline (model loading, data processing, action prediction, plot saving)
completes without errors.

Environment variables (optional):
  INFERENCE_TEST_MODEL_PATH   – checkpoint path (default: checkpoints/GR00T-N1.7-LIBERO/libero_10)
  INFERENCE_TEST_DATASET_PATH – dataset path    (default: demo_data/libero_demo)
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

from tests.examples.utils import build_uv_runtime_env, run_subprocess_step


ROOT = pathlib.Path(__file__).resolve().parents[3]

_HF_REPO_ID = "nvidia/GR00T-N1.7-LIBERO"
_HF_SUBDIR = "libero_10"
SHARED_DRIVE_ROOT = pathlib.Path("/shared")
SHARED_LIBERO_MODEL = SHARED_DRIVE_ROOT / "models/GR00T-N1.7-LIBERO/libero_10"


def _model_complete(path: pathlib.Path) -> bool:
    """Return True if the checkpoint directory has config.json and all shard files."""
    if not (path / "config.json").is_file():
        return False
    index_file = path / "model.safetensors.index.json"
    if not index_file.is_file():
        return True  # single-shard model
    shards = set(json.loads(index_file.read_text()).get("weight_map", {}).values())
    return all((path / shard).is_file() for shard in shards)


def _find_libero_checkpoint() -> str:
    """Resolve checkpoint path: env var > local checkpoints/ > git toplevel > shared drive.

    On CI the shared drive at /shared is available and the checkpoint is
    downloaded there automatically (with HF_TOKEN). Locally, users can set
    INFERENCE_TEST_MODEL_PATH or place the checkpoint under checkpoints/.
    """
    # 1. Explicit env var
    env_path = os.getenv("INFERENCE_TEST_MODEL_PATH", "")
    if env_path:
        return env_path

    # 2. Local checkpoints/ relative to repo root
    local = ROOT / "checkpoints/GR00T-N1.7-LIBERO/libero_10"
    if _model_complete(local):
        return str(local)

    # 3. Git toplevel (for worktrees)
    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(ROOT),
            text=True,
        ).strip()
        git_root = pathlib.Path(toplevel) / "checkpoints/GR00T-N1.7-LIBERO/libero_10"
        if _model_complete(git_root):
            return str(git_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 4. Shared drive (CI)
    if _model_complete(SHARED_LIBERO_MODEL):
        return str(SHARED_LIBERO_MODEL)

    # Return shared path as default — _prepare_model() will download if HF_TOKEN is set
    return str(SHARED_LIBERO_MODEL)


def _prepare_model() -> str:
    """Ensure the LIBERO checkpoint is available, downloading to shared storage if needed."""
    path = pathlib.Path(_find_libero_checkpoint())
    if _model_complete(path):
        return str(path)

    # Try downloading to shared storage
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return str(path)  # will trigger pytest.skip in the test

    SHARED_LIBERO_MODEL.mkdir(parents=True, exist_ok=True)
    env = build_uv_runtime_env()
    run_subprocess_step(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={_HF_REPO_ID!r}, "
            f"allow_patterns={_HF_SUBDIR + '/*'!r}, "
            f"local_dir={str(SHARED_LIBERO_MODEL.parent)!r}, token={token!r})",
        ],
        step="libero_model_download",
        cwd=ROOT,
        env=env,
        log_prefix="inference",
        failure_prefix="Failed to download LIBERO checkpoint",
    )
    return str(SHARED_LIBERO_MODEL)


DEFAULT_MODEL_PATH = _find_libero_checkpoint()

DEFAULT_DATASET_PATH = os.getenv(
    "INFERENCE_TEST_DATASET_PATH",
    str(ROOT / "demo_data/libero_demo"),
)


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_standalone_inference_pytorch() -> None:
    """Run standalone inference in PyTorch mode for 5 steps on 1 trajectory."""

    model_path = _prepare_model()
    dataset_path = DEFAULT_DATASET_PATH

    if not _model_complete(pathlib.Path(model_path)):
        pytest.skip(f"Model checkpoint not found at {model_path}")
    if not pathlib.Path(dataset_path).exists():
        pytest.skip(f"Dataset not found at {dataset_path}")

    env = build_uv_runtime_env()

    run_subprocess_step(
        [
            "python",
            "scripts/deployment/standalone_inference_script.py",
            "--model-path",
            model_path,
            "--dataset-path",
            dataset_path,
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--traj-ids",
            "0",
            "--inference-mode",
            "pytorch",
            "--action-horizon",
            "8",
            "--steps",
            "20",
        ],
        step="standalone_inference_pytorch",
        cwd=ROOT,
        env=env,
        log_prefix="inference",
        failure_prefix="Standalone inference (PyTorch) failed",
        output_tail_chars=8000,
    )


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_standalone_inference_invalid_traj_id() -> None:
    """Passing an out-of-range --traj-ids should raise ValueError, not UnboundLocalError."""

    model_path = _prepare_model()
    dataset_path = DEFAULT_DATASET_PATH

    if not _model_complete(pathlib.Path(model_path)):
        pytest.skip(f"Model checkpoint not found at {model_path}")
    if not pathlib.Path(dataset_path).exists():
        pytest.skip(f"Dataset not found at {dataset_path}")

    env = build_uv_runtime_env()

    result = subprocess.run(
        [
            "python",
            "scripts/deployment/standalone_inference_script.py",
            "--model-path",
            model_path,
            "--dataset-path",
            dataset_path,
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--traj-ids",
            "999",
            "--inference-mode",
            "pytorch",
            "--action-horizon",
            "8",
            "--steps",
            "5",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "Expected non-zero exit for out-of-range traj_id"
    combined = result.stdout + result.stderr
    assert "out of range" in combined.lower(), (
        f"Expected 'out of range' in output, got:\n{combined[-2000:]}"
    )
    assert "UnboundLocalError" not in combined, "Should raise ValueError, not UnboundLocalError"


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_open_loop_eval_with_checkpoint() -> None:
    """Run open_loop_eval.py directly with a model checkpoint."""

    model_path = _prepare_model()
    dataset_path = DEFAULT_DATASET_PATH

    if not _model_complete(pathlib.Path(model_path)):
        pytest.skip(f"Model checkpoint not found at {model_path}")
    if not pathlib.Path(dataset_path).exists():
        pytest.skip(f"Dataset not found at {dataset_path}")

    env = build_uv_runtime_env()

    run_subprocess_step(
        [
            "python",
            "gr00t/eval/open_loop_eval.py",
            "--dataset-path",
            dataset_path,
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--model-path",
            model_path,
            "--traj-ids",
            "0",
            "--action-horizon",
            "8",
            "--steps",
            "5",
        ],
        step="open_loop_eval",
        cwd=ROOT,
        env=env,
        log_prefix="inference",
        failure_prefix="Open-loop eval failed",
        output_tail_chars=8000,
    )
