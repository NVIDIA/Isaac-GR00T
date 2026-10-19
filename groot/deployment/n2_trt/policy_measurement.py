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
from contextlib import contextmanager
import gc
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from groot.core.policy.benchmark_contract import (
    CHECKPOINT_FLAT_SYNTHETIC_INPUT_ID,
    action_equivalence,
    checkpoint_identity,
    cuda_graph_lifecycle_snapshot,
    expected_checkpoint_flat_actions,
    gate_action_equivalence,
    load_checkpoint_flat_workload,
    make_checkpoint_flat_observation,
    summarize_ms,
    validate_action,
    validate_runtime_artifacts,
    write_json_atomic,
)
from groot.core.policy.env_utils import env_bool_optional
from groot.deployment.n2_trt.contract import N2TensorRTContract
from groot.deployment.n2_trt.manifest import (
    BUILD_FILENAME,
    MANIFEST_FILENAME,
    canonical_sha256,
    checkpoint_file_hashes,
    declared_transformer_precisions,
    declared_vae_precision,
    sha256_file,
)

POLICY_MEASUREMENT_FILENAME = "n2_trt_policy_measurement.json"
POLICY_MEASUREMENT_SCHEMA = "groot_n2_trt_policy_measurement_v1"
DEFAULT_INSTRUCTION = "pick up the object"
DEFAULT_WARMUP_REQUESTS = 1
DEFAULT_MEASURED_REQUESTS = 3
# The receipt compares a reference leg with a candidate leg, in this order.
DEFAULT_BACKENDS = ("eager", "trt")
COND_KV_FREEZE_ENV = "GROOT_N2_COND_KV_FREEZE"
# Both legs denoise from fixed noise (every request reuses the policy seed), the regime every
# retained receipt was measured in.
FIXED_SEED_ENV = "GROOT_N2_FIXED_SEED"
_TENSORRT_BACKENDS = frozenset(("trt", "trt_graph"))
logger = logging.getLogger(__name__)


def _embodiment_tag(embodiment: str) -> Any:
    from groot.core.data import schema

    try:
        return schema.EmbodimentTag[embodiment.upper()]
    except KeyError:
        return schema.EmbodimentTag(embodiment.lower())


@contextmanager
def _served_policy_env(cond_kv_freeze: bool):
    """Select the served conditioning-cache mode and fixed noise for ONE policy construction."""
    selected = {COND_KV_FREEZE_ENV: "1" if cond_kv_freeze else "0", FIXED_SEED_ENV: "1"}
    previous = {name: os.environ.get(name) for name in selected}
    os.environ.update(selected)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _engine_settings(engine_dir: Path, *, bare_engines: bool) -> dict[str, Any]:
    """The build settings of the engines under ``engine_dir`` (manifest, or bare build receipt)."""
    if bare_engines:
        settings = json.loads((engine_dir / BUILD_FILENAME).read_text()).get("settings")
    else:
        settings = json.loads((engine_dir / MANIFEST_FILENAME).read_text()).get("build_settings")
    if not isinstance(settings, dict):
        raise TypeError(f"the engines under {engine_dir} declare no build settings")
    return settings


def _selected_gpu() -> dict[str, Any]:
    import torch

    device = torch.cuda.current_device()
    return {
        "torch_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {mode: int(count) - int(before.get(mode, 0)) for mode, count in after.items()}


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
    served_cond_kv_freeze: bool,
    bare_engines: bool = False,
    checkpoint_record: dict[str, Any] | None = None,
    gpu: dict[str, Any] | None = None,
    save_actions: Path | None = None,
    num_video_steps: int = 2,
    num_action_steps: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
    from groot.deployment.n2_trt.benchmark_adapter import build_benchmark_policy

    uses_tensorrt = backend in _TENSORRT_BACKENDS
    with _served_policy_env(served_cond_kv_freeze):
        policy = build_benchmark_policy(
            backend=backend,
            checkpoint=str(checkpoint),
            embodiment_tag=_embodiment_tag(embodiment),
            codec=codec,
            device=f"cuda:{torch.cuda.current_device()}",
            engine_dir=str(engine_dir),
            num_video_steps=num_video_steps,
            num_action_steps=num_action_steps,
            bare_engines=bare_engines,
        )
    wrapper = Gr00tFlatPolicyWrapper(policy, strict=True)
    runtime = getattr(policy.model, "n2_trt_runtime", None)
    if uses_tensorrt != (runtime is not None):
        raise RuntimeError(
            f"{backend} leg {'has no' if uses_tensorrt else 'installed a'} TensorRT runtime"
        )
    # The policy updates this dict in place (graph lifecycle counters) as requests run.
    live_contract = policy.benchmark_runtime_contract

    def engine_calls() -> dict[str, int] | None:
        return dict(runtime.engine_calls) if runtime is not None else None

    def graph_lifecycle() -> dict[str, Any] | None:
        if backend != "trt_graph":
            return None
        return cuda_graph_lifecycle_snapshot(live_contract)

    latencies_ms: list[float] = []
    samples: dict[str, list[np.ndarray]] = {key: [] for key in action_shapes}
    calls_at_start = engine_calls()
    calls_after_warmup = lifecycle_after_warmup = None
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
                if index == warmup - 1:
                    calls_after_warmup = engine_calls()
                    lifecycle_after_warmup = graph_lifecycle()
                continue
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            for key, value in action.items():
                samples[key].append(np.asarray(value))
        calls_after_measured = engine_calls()
        lifecycle_after_measured = graph_lifecycle()
        runtime_contract = dict(live_contract)
        installed_contract = runtime.contract.to_dict() if runtime is not None else None
    finally:
        del wrapper, policy
        gc.collect()
        torch.cuda.empty_cache()
    if bool(runtime_contract.get("cond_kv_freeze", False)) != served_cond_kv_freeze:
        raise RuntimeError(
            f"{backend} leg served cond_kv_freeze={runtime_contract.get('cond_kv_freeze')!r}, "
            f"selected {served_cond_kv_freeze}"
        )
    if runtime_contract.get("noise_mode") != "fixed":
        raise RuntimeError(
            f"{backend} leg served noise_mode={runtime_contract.get('noise_mode')!r}, not fixed"
        )
    stacked = {key: np.stack(values) for key, values in samples.items()}
    metadata: dict[str, Any] = {
        "runtime_contract": runtime_contract,
        "latency_ms": summarize_ms(latencies_ms),
        "served_cond_kv_freeze": served_cond_kv_freeze,
    }
    if runtime is not None:
        metadata["installed_contract"] = installed_contract
        metadata["engine_calls"] = {
            "warmup": _counter_delta(calls_at_start, calls_after_warmup),
            "measured": _counter_delta(calls_after_warmup, calls_after_measured),
        }
        metadata["engine_source"] = "bare" if bare_engines else "admitted"
        if bare_engines:
            bare = runtime_contract.get("bare_engines")
            if not isinstance(bare, dict) or bare.get("admitted") is not False:
                raise RuntimeError(f"{backend} leg served bare engines without their provenance")
            metadata["bare_engines"] = bare
        else:
            if checkpoint_record is None or gpu is None:
                raise RuntimeError(
                    "authenticating an admitted bundle needs the checkpoint and GPU identity"
                )
            validated = validate_runtime_artifacts(
                runtime_contract=runtime_contract, checkpoint=checkpoint_record, gpu=gpu
            )
            if not isinstance(validated, dict):
                raise RuntimeError(f"{backend} leg runtime artifacts were not authenticated")
            metadata["runtime_artifacts_validation"] = validated
    if backend == "trt_graph":
        captured = runtime_contract.get("cuda_graph_captured_engine_calls")
        if not isinstance(captured, dict):
            raise RuntimeError("trt_graph leg published no captured engine schedule")
        metadata["cuda_graph_lifecycle"] = {
            "after_warmup": lifecycle_after_warmup,
            "after_measured": lifecycle_after_measured,
        }
        metadata["cuda_graph_captured_engine_calls"] = dict(captured)
    if save_actions is not None:
        save_actions.mkdir(parents=True, exist_ok=True)
        actions_path = save_actions / f"{backend}.npz"
        np.savez(actions_path, **stacked)
        metadata["actions_path"] = str(actions_path)
    return stacked, metadata


def measure_policy_actions(
    *,
    checkpoint: Path,
    embodiment: str,
    engine_dir: Path,
    codec: str,
    warmup: int = DEFAULT_WARMUP_REQUESTS,
    measured: int = DEFAULT_MEASURED_REQUESTS,
    backends: tuple[str, ...] = DEFAULT_BACKENDS,
    served_cond_kv_freeze: bool | None = None,
    bare_engines: bool = False,
    save_actions: Path | None = None,
    num_video_steps: int = 2,
    num_action_steps: int = 4,
) -> dict[str, Any]:
    """Measure decoded actions of a reference and a candidate leg with the shared contract.

    One conditioning-cache mode is served by BOTH legs (the eager leg emulates it):
    ``served_cond_kv_freeze``, else ``GROOT_N2_COND_KV_FREEZE``, else the engines' default
    (on exactly when they carry the conditioning-cached joint engine). ``bare_engines``
    serves the engines of a build receipt without a manifest; such a receipt is never
    an admission.
    """
    if warmup < 1 or measured < 1:
        raise ValueError("policy measurement warmup and measured counts must be positive")
    backends = tuple(backends)
    if (
        len(backends) != 2
        or len(set(backends)) != 2
        or not set(backends) <= _TENSORRT_BACKENDS | {"eager"}
        or not set(backends) & _TENSORRT_BACKENDS
    ):
        raise ValueError(
            "policy measurement compares one reference leg with one candidate leg among "
            f"eager/trt/trt_graph, at least one of them TensorRT; got {list(backends)}"
        )
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    engine_dir = engine_dir.expanduser().resolve(strict=True)
    settings = _engine_settings(engine_dir, bare_engines=bare_engines)
    bundle_cond_kv_freeze = bool(settings.get("cond_kv_freeze", False))
    if served_cond_kv_freeze is None:
        served_cond_kv_freeze = env_bool_optional(COND_KV_FREEZE_ENV)
    if served_cond_kv_freeze is None:
        served_cond_kv_freeze = bundle_cond_kv_freeze
    served_cond_kv_freeze = bool(served_cond_kv_freeze)
    if served_cond_kv_freeze and not bundle_cond_kv_freeze:
        raise RuntimeError(
            "served cond_kv_freeze requires the conditioning-cached joint engine; the engines "
            f"under {engine_dir} do not carry it"
        )
    modality, statistics, workload = load_checkpoint_flat_workload(checkpoint, embodiment)
    action_shapes = expected_checkpoint_flat_actions(modality, statistics, workload)
    checkpoint_record = gpu = None
    if not bare_engines:
        checkpoint_record = checkpoint_identity(checkpoint)
        gpu = _selected_gpu()
    actions: dict[str, dict[str, np.ndarray]] = {}
    legs: dict[str, dict[str, Any]] = {}
    for backend in backends:
        actions[backend], legs[backend] = _measure_backend(
            backend=backend,
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
            served_cond_kv_freeze=served_cond_kv_freeze,
            bare_engines=bare_engines,
            checkpoint_record=checkpoint_record,
            gpu=gpu,
            save_actions=save_actions,
            num_video_steps=num_video_steps,
            num_action_steps=num_action_steps,
        )
    reference, candidate = backends
    equivalence = action_equivalence(actions[reference], actions[candidate])
    exact_breaches = gate_action_equivalence(
        equivalence,
        rtol=0.0,
        max_abs=0.0,
        label=candidate,
    )
    tensorrt_legs = [legs[backend] for backend in backends if backend in _TENSORRT_BACKENDS]
    installed = tensorrt_legs[0]["installed_contract"]
    if any(leg["installed_contract"] != installed for leg in tensorrt_legs):
        raise RuntimeError("the TensorRT legs installed different runtime contracts")
    payload: dict[str, Any] = {
        "schema": POLICY_MEASUREMENT_SCHEMA,
        "status": "complete_with_drift" if exact_breaches else "complete",
        # The installed runtime's contract: the bundle's capability, resolved on the checkpoint.
        "contract_sha256": canonical_sha256(installed),
        "checkpoint_sha256": canonical_sha256(checkpoint_file_hashes(checkpoint)),
        "engine_source": "bare" if bare_engines else "admitted",
        "codec": codec,
        "served_cond_kv_freeze": served_cond_kv_freeze,
        "bundle_cond_kv_freeze": bundle_cond_kv_freeze,
        "synthetic_input_id": CHECKPOINT_FLAT_SYNTHETIC_INPUT_ID,
        "instruction": DEFAULT_INSTRUCTION,
        "warmup_requests": warmup,
        "measured_requests": measured,
        "observation_seed_schedule": {"warmup_start": 0, "measured_start": warmup, "step": 1},
        "action_shapes": {key: list(shape) for key, shape in action_shapes.items()},
        "backend_order": list(backends),
        "equivalence": equivalence,
        "exact_equivalence_breaches": exact_breaches,
        "backends": legs,
    }
    if bare_engines:
        blocks = [
            {key: value for key, value in leg["bare_engines"].items() if key != "backend"}
            for leg in tensorrt_legs
        ]
        if any(block != blocks[0] for block in blocks):
            raise RuntimeError("the TensorRT legs served different bare engines")
        payload["bare_engines"] = blocks[0]
    else:
        payload["manifest_sha256"] = sha256_file(engine_dir / MANIFEST_FILENAME)
    return payload


def _bare_engine_expectations(engine_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict]:
    """(build receipt identity, per-role live engine identities, receipt) under ``engine_dir``."""
    build_path = (engine_dir / BUILD_FILENAME).resolve(strict=True)
    receipt = json.loads(build_path.read_text())
    build_identity = {
        "path": str(build_path),
        "size_bytes": build_path.stat().st_size,
        "sha256": sha256_file(build_path),
        "quantize_mode": receipt.get("quantize_mode"),
        "settings": receipt.get("settings"),
    }
    engines: dict[str, Any] = {}
    for artifact in receipt.get("artifacts") or ():
        path = (engine_dir / artifact["engine"]).resolve(strict=True)
        engines[artifact["role"]] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return build_identity, engines, receipt


def validate_policy_measurement(
    path: Path,
    *,
    contract: N2TensorRTContract,
    manifest_path: Path | None = None,
    engine_dir: Path | None = None,
    codec: str | None = None,
    served_cond_kv_freeze: bool | None = None,
) -> dict[str, Any]:
    """Validate the retained post-build measurement receipt against its engines.

    Exactly one of ``manifest_path`` (an admitted bundle) or ``engine_dir`` (bare engines
    beside their build receipt) binds the receipt to the engines it measured; both are
    re-read here. ``served_cond_kv_freeze`` is the expected served mode (``None`` = the
    engines' default).
    """
    if (manifest_path is None) == (engine_dir is None):
        raise ValueError(
            "pass exactly one of manifest_path (admitted bundle) or engine_dir (bare engines)"
        )
    payload = json.loads(path.read_text())
    if payload.get("schema") != POLICY_MEASUREMENT_SCHEMA:
        raise ValueError(f"unsupported policy measurement schema: {payload.get('schema')!r}")
    if payload.get("status") not in {"complete", "complete_with_drift"}:
        raise RuntimeError(f"policy measurement is incomplete: {payload.get('status')!r}")
    expected_source = "admitted" if manifest_path is not None else "bare"
    recorded_source = payload.get("engine_source", "admitted")
    if recorded_source != expected_source:
        raise RuntimeError(
            f"policy measurement engine source {recorded_source!r} does not match the "
            f"expected {expected_source!r}"
        )
    if manifest_path is not None:
        manifest_payload = json.loads(Path(manifest_path).read_text())
        settings = manifest_payload.get("build_settings") or {}
        declared = manifest_payload.get("contract") or {}
        joint, action = declared_transformer_precisions(declared)
        vae = declared_vae_precision(declared)
    else:
        engine_dir = Path(engine_dir).expanduser().resolve(strict=True)
        build_identity, live_engines, receipt = _bare_engine_expectations(engine_dir)
        settings = receipt.get("settings") or {}
        formats = {
            artifact["role"]: str(artifact["numeric_format"])
            for artifact in receipt.get("artifacts") or ()
        }
        joint = formats.get("transformer_joint", "bf16")
        action = formats.get("transformer_action_cache", joint)
        vae = formats.get("vae_encoder", "bf16")
    bundle_cond_kv_freeze = bool(settings.get("cond_kv_freeze", False))
    if bool(payload.get("bundle_cond_kv_freeze", False)) != bundle_cond_kv_freeze:
        raise RuntimeError("policy measurement bundle capability does not match the engines")
    expected_served = (
        bundle_cond_kv_freeze if served_cond_kv_freeze is None else bool(served_cond_kv_freeze)
    )
    if bool(payload.get("served_cond_kv_freeze", False)) != expected_served:
        raise RuntimeError(
            f"policy measurement served cond_kv_freeze={payload.get('served_cond_kv_freeze')!r} "
            f"does not match the expected {expected_served}"
        )
    resolved = (
        contract.with_transformer_precisions(joint, action)
        .with_vae_precision(vae)
        .with_cond_kv_freeze(bundle_cond_kv_freeze)
    )
    if payload.get("contract_sha256") != canonical_sha256(resolved.to_dict()):
        raise RuntimeError("policy measurement contract does not match the checkpoint")
    if payload.get("checkpoint_sha256") != canonical_sha256(
        checkpoint_file_hashes(contract.checkpoint)
    ):
        raise RuntimeError("policy measurement checkpoint identity does not match")
    if codec is not None and payload.get("codec") != codec:
        raise RuntimeError(
            f"policy measurement codec {payload.get('codec')!r} does not match {codec!r}"
        )
    if manifest_path is not None:
        if payload.get("manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("policy measurement manifest identity does not match")
    else:
        recorded = payload.get("bare_engines")
        if not isinstance(recorded, dict) or recorded.get("admitted") is not False:
            raise RuntimeError("policy measurement carries no bare engine provenance")
        if recorded.get("build_receipt") != build_identity:
            raise RuntimeError(
                "policy measurement build receipt does not match the engine directory"
            )
        if recorded.get("engines") != live_engines:
            raise RuntimeError("policy measurement engines do not match the live engine files")
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
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help="reference,candidate legs among eager/trt/trt_graph",
    )
    parser.add_argument(
        "--served-cond-kv-freeze",
        choices=("0", "1"),
        default=None,
        help="served conditioning-cache mode for both legs (default: the engines' default)",
    )
    parser.add_argument(
        "--bare-engines",
        action="store_true",
        help="serve the engines of the build receipt under --engine-dir without a manifest",
    )
    parser.add_argument(
        "--save-actions",
        type=Path,
        default=None,
        help="directory receiving the stacked measured actions per leg (<backend>.npz)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument("--num-action-steps", type=int, default=4)
    args = parser.parse_args()
    payload = measure_policy_actions(
        checkpoint=args.checkpoint,
        embodiment=args.embodiment,
        engine_dir=args.engine_dir,
        codec=args.codec,
        warmup=args.warmup,
        measured=args.measured,
        backends=tuple(part.strip() for part in args.backends.split(",") if part.strip()),
        served_cond_kv_freeze=(
            None if args.served_cond_kv_freeze is None else args.served_cond_kv_freeze == "1"
        ),
        bare_engines=args.bare_engines,
        save_actions=args.save_actions,
        num_video_steps=args.num_video_steps,
        num_action_steps=args.num_action_steps,
    )
    write_json_atomic(args.out, payload)
    logger.info("[n2-trt-policy-measurement] receipt=%s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
