"""
LIBERO example smoke test.

Runs a 2-step finetune on the bundled libero_demo dataset, then verifies the
checkpoint with a short open-loop eval.  This catches model-loading,
data-pipeline, and training-loop regressions for the LIBERO embodiment tag.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest
from tests.examples.utils import build_uv_runtime_env, run_subprocess_step


ROOT = pathlib.Path(__file__).resolve().parents[3]

TRAINING_STEPS = 2
OUTPUT_DIR = pathlib.Path("/tmp/libero_ci_finetune")
MODEL_CHECKPOINT = OUTPUT_DIR / f"checkpoint-{TRAINING_STEPS}"

# Use the small bundled demo dataset (ships with the repo).
DATASET_PATH = ROOT / "demo_data/libero_demo"


@pytest.mark.gpu
@pytest.mark.timeout(600)
def test_libero_finetune_and_open_loop_eval() -> None:
    """Finetune for 2 steps on libero_demo, then run open-loop eval on the checkpoint."""

    if not DATASET_PATH.exists():
        pytest.skip(f"Demo dataset not found at {DATASET_PATH}")

    env = build_uv_runtime_env(
        extra_env={
            # Pin to a single GPU to avoid DataParallel which triggers
            # StopIteration when the model's .device property is called on
            # an empty parameter iterator inside DP replicas.
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    try:
        # ── Step 1: Finetune (2 training steps, 1 GPU) ─────────────────
        run_subprocess_step(
            [
                "python",
                "gr00t/experiment/launch_finetune.py",
                "--base_model_path",
                "nvidia/GR00T-N1.7-3B",
                "--dataset_path",
                str(DATASET_PATH),
                "--embodiment_tag",
                "LIBERO_PANDA",
                "--num_gpus",
                "1",
                "--output_dir",
                str(OUTPUT_DIR),
                "--save_steps",
                str(TRAINING_STEPS),
                "--save_total_limit",
                "1",
                "--max_steps",
                str(TRAINING_STEPS),
                "--warmup_ratio",
                "0.0",
                "--weight_decay",
                "0",
                "--learning_rate",
                "1e-4",
                "--global_batch_size",
                "2",
                "--dataloader_num_workers",
                "0",
                "--shard_size",
                "64",
                "--num_shards_per_epoch",
                "1",
                "--episode_sampling_rate",
                "0.02",
            ],
            step="finetune_libero",
            cwd=ROOT,
            env=env,
            log_prefix="libero",
            failure_prefix="LIBERO finetune failed",
            output_tail_chars=8000,
        )

        assert MODEL_CHECKPOINT.exists(), f"Expected checkpoint after finetune: {MODEL_CHECKPOINT}"

        # ── Step 2: Open-loop eval on the finetuned checkpoint ──────────
        run_subprocess_step(
            [
                "python",
                "gr00t/eval/open_loop_eval.py",
                "--dataset-path",
                str(DATASET_PATH),
                "--embodiment-tag",
                "LIBERO_PANDA",
                "--model-path",
                str(MODEL_CHECKPOINT),
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
            log_prefix="libero",
            failure_prefix="LIBERO open-loop eval failed",
            output_tail_chars=8000,
        )

    finally:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
