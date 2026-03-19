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

import os
import pathlib

import pytest
from tests.examples.utils import build_uv_runtime_env, run_subprocess_step


ROOT = pathlib.Path(__file__).resolve().parents[3]


# Resolve checkpoint path: env var > local checkpoints/ > git toplevel checkpoints/
def _find_libero_checkpoint() -> str:
    env_path = os.getenv("INFERENCE_TEST_MODEL_PATH", "")
    if env_path:
        return env_path
    # Check relative to this test file's repo root
    local = ROOT / "checkpoints/GR00T-N1.7-LIBERO/libero_10"
    if local.exists():
        return str(local)
    # For worktrees: walk up to find the git toplevel (may differ from ROOT)
    import subprocess

    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(ROOT),
            text=True,
        ).strip()
        git_root = pathlib.Path(toplevel) / "checkpoints/GR00T-N1.7-LIBERO/libero_10"
        if git_root.exists():
            return str(git_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return str(local)  # will trigger skip


DEFAULT_MODEL_PATH = _find_libero_checkpoint()

DEFAULT_DATASET_PATH = os.getenv(
    "INFERENCE_TEST_DATASET_PATH",
    str(ROOT / "demo_data/libero_demo"),
)


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_standalone_inference_pytorch() -> None:
    """Run standalone inference in PyTorch mode for 5 steps on 1 trajectory."""

    model_path = DEFAULT_MODEL_PATH
    dataset_path = DEFAULT_DATASET_PATH

    if not pathlib.Path(model_path).exists():
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
def test_open_loop_eval_with_checkpoint() -> None:
    """Run open_loop_eval.py directly with a model checkpoint."""

    model_path = DEFAULT_MODEL_PATH
    dataset_path = DEFAULT_DATASET_PATH

    if not pathlib.Path(model_path).exists():
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
