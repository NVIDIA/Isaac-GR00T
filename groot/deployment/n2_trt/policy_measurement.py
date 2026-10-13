# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Checkpoint-flat eager/TensorRT action measurement for automatic bundles."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from groot.core.policy.benchmark_contract import (
    CHECKPOINT_FLAT_SYNTHETIC_INPUT_ID,
    action_equivalence,
    expected_checkpoint_flat_actions,
    gate_action_equivalence,
    load_checkpoint_flat_workload,
    make_checkpoint_flat_observation,
    summarize_ms,
    validate_action,
    write_json_atomic,
)
from groot.deployment.n2_trt.contract import N2TensorRTContract
from groot.deployment.n2_trt.manifest import (
    MANIFEST_FILENAME,
    canonical_sha256,
    checkpoint_file_hashes,
    sha256_file,
)

POLICY_MEASUREMENT_FILENAME = "n2_trt_policy_measurement.json"
POLICY_MEASUREMENT_SCHEMA = "groot_n2_trt_policy_measurement_v1"
DEFAULT_INSTRUCTION = "pick up the object"
DEFAULT_WARMUP_REQUESTS = 1
DEFAULT_MEASURED_REQUESTS = 3
logger = logging.getLogger(__name__)


def _embodiment_tag(embodiment: str) -> Any:
    from groot.core.data import schema

    try:
        return schema.EmbodimentTag[embodiment.upper()]
    except KeyError:
        return schema.EmbodimentTag(embodiment.lower())


def _measure_backend(
    *,
    backend: str,
    checkpoint: Path,
    embodiment: str,
    engine_dir: Path,
    codec: str,
    modality: dict[str, Any],
    statistics: dict[str, Any],
    workload: dict[str, Any],
    action_shapes: dict[str, tuple[int, int]],
    warmup: int,
    measured: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
    from groot.deployment.n2_trt.benchmark_adapter import build_benchmark_policy

    policy = build_benchmark_policy(
        backend=backend,
        checkpoint=str(checkpoint),
        embodiment_tag=_embodiment_tag(embodiment),
        codec=codec,
        device=f"cuda:{torch.cuda.current_device()}",
        engine_dir=str(engine_dir),
        num_video_steps=2,
        num_action_steps=4,
    )
    wrapper = Gr00tFlatPolicyWrapper(policy, strict=True)
    latencies_ms: list[float] = []
    samples: dict[str, list[np.ndarray]] = {key: [] for key in action_shapes}
    try:
        for index in range(warmup + measured):
            observation = make_checkpoint_flat_observation(
                modality,
                statistics,
                workload,
                instruction=DEFAULT_INSTRUCTION,
                seed=index,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                action, _ = wrapper.get_action(observation)
            torch.cuda.synchronize()
            validate_action(action, action_shapes)
            if index < warmup:
                continue
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            for key, value in action.items():
                samples[key].append(np.asarray(value))
        runtime_contract = dict(policy.benchmark_runtime_contract)
    finally:
        del wrapper, policy
        gc.collect()
        torch.cuda.empty_cache()
    return (
        {key: np.stack(values) for key, values in samples.items()},
        {
            "runtime_contract": runtime_contract,
            "latency_ms": summarize_ms(latencies_ms),
        },
    )


def measure_policy_actions(
    *,
    checkpoint: Path,
    embodiment: str,
    engine_dir: Path,
    codec: str,
    warmup: int = DEFAULT_WARMUP_REQUESTS,
    measured: int = DEFAULT_MEASURED_REQUESTS,
) -> dict[str, Any]:
    """Measure decoded eager/TRT actions using the shared benchmark contract."""
    if warmup < 1 or measured < 1:
        raise ValueError("policy measurement warmup and measured counts must be positive")
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    engine_dir = engine_dir.expanduser().resolve(strict=True)
    contract = N2TensorRTContract.from_checkpoint(
        checkpoint,
        embodiment=embodiment,
        video_denoising_steps=2,
        action_denoising_steps=4,
        joint_denoise=True,
        conditional_kv_cache=True,
    )
    modality, statistics, workload = load_checkpoint_flat_workload(checkpoint, embodiment)
    action_shapes = expected_checkpoint_flat_actions(modality, statistics, workload)
    eager, eager_metadata = _measure_backend(
        backend="eager",
        checkpoint=checkpoint,
        embodiment=embodiment,
        engine_dir=engine_dir,
        codec=codec,
        modality=modality,
        statistics=statistics,
        workload=workload,
        action_shapes=action_shapes,
        warmup=warmup,
        measured=measured,
    )
    trt, trt_metadata = _measure_backend(
        backend="trt",
        checkpoint=checkpoint,
        embodiment=embodiment,
        engine_dir=engine_dir,
        codec=codec,
        modality=modality,
        statistics=statistics,
        workload=workload,
        action_shapes=action_shapes,
        warmup=warmup,
        measured=measured,
    )
    equivalence = action_equivalence(eager, trt)
    exact_breaches = gate_action_equivalence(
        equivalence,
        rtol=0.0,
        max_abs=0.0,
        label="trt",
    )
    manifest_path = engine_dir / MANIFEST_FILENAME
    return {
        "schema": POLICY_MEASUREMENT_SCHEMA,
        "status": "complete_with_drift" if exact_breaches else "complete",
        "contract_sha256": canonical_sha256(contract.to_dict()),
        "checkpoint_sha256": canonical_sha256(checkpoint_file_hashes(checkpoint)),
        "manifest_sha256": sha256_file(manifest_path),
        "synthetic_input_id": CHECKPOINT_FLAT_SYNTHETIC_INPUT_ID,
        "instruction": DEFAULT_INSTRUCTION,
        "warmup_requests": warmup,
        "measured_requests": measured,
        "observation_seed_schedule": {"warmup_start": 0, "measured_start": warmup, "step": 1},
        "action_shapes": {key: list(shape) for key, shape in action_shapes.items()},
        "equivalence": equivalence,
        "exact_equivalence_breaches": exact_breaches,
        "backends": {"eager": eager_metadata, "trt": trt_metadata},
    }


def validate_policy_measurement(
    path: Path,
    *,
    contract: N2TensorRTContract,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate the retained automatic post-build measurement receipt."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != POLICY_MEASUREMENT_SCHEMA:
        raise ValueError(f"unsupported policy measurement schema: {payload.get('schema')!r}")
    if payload.get("status") not in {"complete", "complete_with_drift"}:
        raise RuntimeError(f"policy measurement is incomplete: {payload.get('status')!r}")
    if payload.get("contract_sha256") != canonical_sha256(contract.to_dict()):
        raise RuntimeError("policy measurement contract does not match the checkpoint")
    if payload.get("checkpoint_sha256") != canonical_sha256(
        checkpoint_file_hashes(contract.checkpoint)
    ):
        raise RuntimeError("policy measurement checkpoint identity does not match")
    if payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("policy measurement manifest identity does not match")
    if payload.get("measured_requests", 0) < 1 or not isinstance(payload.get("equivalence"), dict):
        raise RuntimeError("policy measurement omits decoded-action evidence")
    return payload


def report_policy_measurement(engine_dir: Path) -> dict[str, Any]:
    """Surface retained eager/TRT decoded-action drift to the caller."""
    receipt = engine_dir / POLICY_MEASUREMENT_FILENAME
    payload = json.loads(receipt.read_text())
    if payload.get("exact_equivalence_breaches"):
        drift = ", ".join(
            f"{key}: max_abs={metrics['max_abs']:.6g}, rel_rmse={metrics['rel_rmse']:.6g}"
            for key, metrics in sorted(payload["equivalence"].items())
        )
        logger.warning(
            "TensorRT decoded-action drift from eager (%s); full receipt: %s",
            drift,
            receipt,
        )
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "mp4v", "none"), required=True)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_REQUESTS)
    parser.add_argument("--measured", type=int, default=DEFAULT_MEASURED_REQUESTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = measure_policy_actions(
        checkpoint=args.checkpoint,
        embodiment=args.embodiment,
        engine_dir=args.engine_dir,
        codec=args.codec,
        warmup=args.warmup,
        measured=args.measured,
    )
    write_json_atomic(args.out, payload)
    logger.info("[n2-trt-policy-measurement] receipt=%s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
