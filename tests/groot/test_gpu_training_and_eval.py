# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional OSS train/eval GPU smoke tests."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import pytest

_RUN_ENV_VAR = "GROOT_OSS_GPU_FUNCTIONAL_TESTS"
_EXPERIMENT = "n2_yam_simulation_stack_bowls"
_DEFAULT_PRETRAIN_CHECKPOINT = (
    "/mnt/amlfs-07/shared/checkpoints/ruijiez/"
    "n2_yam_pretrain-bidrectional-full-200k-source/checkpoint-200000"
)
_DEFAULT_DATASET_PATH = "/mnt/amlfs-01/shared/datasets/yam_workstation_stack_bowls_20260430_170700"
_DEFAULT_COSMOS_MODEL_PATH = "/mnt/amlfs-02/shared/ckpts/cosmos25/cosmos_reason1/"
_DEFAULT_DIFFUSION_TRANSFORMER_PATH = (
    "/mnt/amlfs-02/shared/ckpts/cosmos25/Cosmos-Predict2.5-2B-diffusers/transformer"
)
_DEFAULT_VAE_PATH = "/mnt/amlfs-02/shared/ckpts/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"

# TODO(cuonguyen): Re-enable after OSS model loading moves from S3/mounted
# artifact paths to Hugging Face so this job can run without internal mounts.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skip(
        reason="TODO: re-enable when OSS model loading uses Hugging Face artifacts",
    ),
    pytest.mark.skipif(
        os.environ.get(_RUN_ENV_VAR) != "1",
        reason=f"set {_RUN_ENV_VAR}=1 to run functional OSS GPU smoke tests",
    ),
]


@pytest.mark.gpu
def test_oss_training_gpu_smoke(tmp_path: Path) -> None:
    """Run a small real OSS training job on the GPU runner."""
    package_root = _package_root()
    output_dir = tmp_path / "train-output"
    env = _base_env(output_dir=output_dir)
    _require_gpu_runtime()
    _require_paths(
        env,
        "PRETRAIN_CHECKPOINT",
        "LFS_CACHE_PATH",
        "COSMOS_MODEL_PATH",
        "DIFFUSION_TRANSFORMER_PATH",
        "VAE_PATH",
    )

    max_steps = env["OSS_GPU_TRAIN_MAX_STEPS"]
    _start_ray(env, package_root)
    try:
        _run(
            [
                sys.executable,
                "-m",
                "groot.infra.oss",
                "train",
                env["OSS_GPU_EXPERIMENT"],
                "--compute-type",
                env["OSS_GPU_COMPUTE_TYPE"],
                "--max-steps",
                max_steps,
                "--num-gpus",
                env["OSS_GPU_TRAIN_NUM_GPUS"],
                "--batch-size",
                env["OSS_GPU_TRAIN_BATCH_SIZE"],
                "--output-dir",
                str(output_dir),
                "--set",
                f"training.save_steps={max_steps}",
            ],
            cwd=package_root,
            env=env,
            timeout=int(env["OSS_GPU_TRAIN_TIMEOUT_SECONDS"]),
        )
    finally:
        _run(["ray", "stop", "--force"], cwd=package_root, env=env, check=False)

    _assert_training_checkpoint(output_dir, max_steps)


@pytest.mark.gpu
def test_oss_open_loop_eval_gpu_smoke(tmp_path: Path) -> None:
    """Run the real OSS policy-server open-loop evaluator on the GPU runner."""
    package_root = _package_root()
    env = _base_env(output_dir=tmp_path / "unused-train-output")
    _require_gpu_runtime()
    checkpoint = _eval_checkpoint(env)
    dataset_path = _eval_dataset_path(env)
    _require_paths(env, "COSMOS_MODEL_PATH", "DIFFUSION_TRANSFORMER_PATH", "VAE_PATH")
    _require_path("OSS eval checkpoint", checkpoint)
    _require_path("OSS eval dataset", dataset_path)

    output_json = tmp_path / "oss_policy_server_eval.json"
    _run(
        [
            sys.executable,
            "-m",
            "groot.infra.oss",
            "eval",
            env["OSS_GPU_EXPERIMENT"],
            "--checkpoint",
            checkpoint,
            "--dataset-path",
            dataset_path,
            "--traj-ids",
            env["OSS_GPU_EVAL_TRAJ_IDS"],
            "--steps",
            env["OSS_GPU_EVAL_STEPS"],
            "--action-horizon",
            env["OSS_GPU_EVAL_ACTION_HORIZON"],
            "--denoising-steps",
            env["OSS_GPU_EVAL_DENOISING_STEPS"],
            "--server-parity-tolerance",
            env["OSS_GPU_EVAL_SERVER_PARITY_TOLERANCE"],
            "--output-json",
            str(output_json),
        ],
        cwd=package_root,
        env=env,
        timeout=int(env["OSS_GPU_EVAL_TIMEOUT_SECONDS"]),
    )
    assert output_json.exists(), f"open-loop eval did not write {output_json}"


def _base_env(*, output_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OSS_GPU_EXPERIMENT", _EXPERIMENT)
    env.setdefault("OSS_GPU_COMPUTE_TYPE", env.get("CLUSTER_TYPE", "gb300"))
    env.setdefault("OSS_GPU_TRAIN_NUM_GPUS", env.get("NUM_GPUS", "1"))
    env.setdefault("OSS_GPU_TRAIN_BATCH_SIZE", env.get("BATCH_SIZE", "1"))
    env.setdefault("OSS_GPU_TRAIN_MAX_STEPS", env.get("MAX_STEPS", "2"))
    env.setdefault("OSS_GPU_TRAIN_TIMEOUT_SECONDS", "5400")
    env.setdefault("OSS_GPU_EVAL_TRAJ_IDS", os.environ.get("OPEN_LOOP_EVAL_TRAJ_IDS", "0"))
    env.setdefault("OSS_GPU_EVAL_STEPS", os.environ.get("OPEN_LOOP_EVAL_STEPS", "30"))
    env.setdefault(
        "OSS_GPU_EVAL_ACTION_HORIZON",
        os.environ.get("OPEN_LOOP_EVAL_ACTION_HORIZON", "30"),
    )
    env.setdefault(
        "OSS_GPU_EVAL_DENOISING_STEPS",
        os.environ.get("OPEN_LOOP_EVAL_DENOISING_STEPS", "4"),
    )
    env.setdefault(
        "OSS_GPU_EVAL_SERVER_PARITY_TOLERANCE",
        os.environ.get("OPEN_LOOP_EVAL_SERVER_PARITY_TOLERANCE", "0.02"),
    )
    env.setdefault("OSS_GPU_EVAL_TIMEOUT_SECONDS", "5400")
    env.setdefault("CLUSTER_TYPE", env["OSS_GPU_COMPUTE_TYPE"])
    env.setdefault("NUM_NODES", "1")
    env["NUM_GPUS"] = env["OSS_GPU_TRAIN_NUM_GPUS"]
    env["BATCH_SIZE"] = env["OSS_GPU_TRAIN_BATCH_SIZE"]
    env["MAX_STEPS"] = env["OSS_GPU_TRAIN_MAX_STEPS"]
    env.setdefault("DATALOADER_NUM_WORKERS", "2")
    env.setdefault("JOB_NAME", f"oss_gpu_smoke_{os.environ.get('CI_COMMIT_SHORT_SHA', 'manual')}")
    env.setdefault("USER_SLUG", "ci")
    env.setdefault("TRAINING_OUTPUT_DIR", str(output_dir))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    env.setdefault("OTEL_SDK_DISABLED", "true")
    env.setdefault("WANDB_API_KEY", "disabled")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("NCCL_DEBUG", "WARNING")
    env.setdefault("NCCL_TIMEOUT", "600")
    env.setdefault("DECORD_EOF_RETRY_MAX", "20480")
    env.setdefault("AWS_ENDPOINT_URL", "https://pdx.s8k.io")
    env.setdefault("S3_ENDPOINT_URL", "https://pdx.s8k.io")
    env.setdefault("PRETRAIN_CHECKPOINT", _DEFAULT_PRETRAIN_CHECKPOINT)
    env.setdefault("LFS_CACHE_PATH", "/mnt/amlfs-01/shared")
    env.setdefault("COSMOS_MODEL_PATH", _DEFAULT_COSMOS_MODEL_PATH)
    env.setdefault("DIFFUSION_TRANSFORMER_PATH", _DEFAULT_DIFFUSION_TRANSFORMER_PATH)
    env.setdefault("VAE_PATH", _DEFAULT_VAE_PATH)
    return env


def _eval_checkpoint(env: dict[str, str]) -> str:
    return (
        env.get("OSS_GPU_EVAL_CHECKPOINT")
        or env.get("OPEN_LOOP_EVAL_MODEL_PATH")
        or env.get("FINAL_CHECKPOINT_PATH")
        or env["PRETRAIN_CHECKPOINT"]
    )


def _eval_dataset_path(env: dict[str, str]) -> str:
    return (
        env.get("OSS_GPU_EVAL_DATASET_PATH")
        or env.get("OPEN_LOOP_EVAL_DATASET_PATH")
        or _DEFAULT_DATASET_PATH
    )


def _start_ray(env: dict[str, str], cwd: Path) -> None:
    if shutil.which("ray") is None:
        pytest.fail("ray executable is required for OSS GPU training smoke")
    _run(["ray", "stop", "--force"], cwd=cwd, env=env, check=False)
    _run(
        [
            "ray",
            "start",
            "--head",
            f"--num-gpus={env['OSS_GPU_TRAIN_NUM_GPUS']}",
            f"--num-cpus={env.get('OSS_GPU_RAY_NUM_CPUS', str(os.cpu_count() or 1))}",
            "--disable-usage-stats",
            "--include-dashboard=False",
        ],
        cwd=cwd,
        env=env,
        timeout=120,
    )
    env["RAY_ADDRESS"] = "auto"
    time.sleep(5)


def _require_gpu_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.fail("functional OSS GPU smoke requires torch.cuda.is_available()")


def _require_paths(env: dict[str, str], *names: str) -> None:
    missing = [
        f"{name}={env.get(name, '')}"
        for name in names
        if not env.get(name) or not Path(env[name]).expanduser().exists()
    ]
    if missing:
        pytest.fail("missing required OSS GPU smoke path(s): " + ", ".join(missing))


def _require_path(label: str, value: str) -> None:
    if not Path(value).expanduser().exists():
        pytest.fail(f"{label} does not exist: {value}")


def _assert_training_checkpoint(output_dir: Path, max_steps: str) -> None:
    checkpoint = output_dir / f"checkpoint-{max_steps}"
    assert output_dir.is_dir(), f"training did not create output directory: {output_dir}"
    assert checkpoint.is_dir(), f"training did not create final checkpoint: {checkpoint}"
    assert any(path.is_file() for path in checkpoint.rglob("*")), (
        f"training checkpoint is empty: {checkpoint}"
    )


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"[oss-gpu-smoke] $ {shlex.join(args)}", flush=True)
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        timeout=timeout,
    )


def _package_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )
