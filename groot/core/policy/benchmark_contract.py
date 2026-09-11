# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared measurement and correctness contract for policy benchmarks.

Fixture adapters own input construction and policy invocation. This module owns
the parts that must not drift between YAM and G1: metric names, summaries,
nested-stage derivation, action validation, equivalence gates, and retained
artifact identities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

import numpy as np

RESULT_SCHEMA_VERSION = 4

FORMAL_WARMUP_REQUESTS = 20
FORMAL_MEASURED_REQUESTS = 500
MEASUREMENT_VALIDATOR_SCHEMA_VERSION = "n2-measurement-validator-v1"
REFERENCE_APPROVAL_SCHEMA_VERSION = "n2-reference-approval-v1"
RUNTIME_ARTIFACT_SCHEMA_VERSION = "n2-runtime-artifacts-v1"
CUDA_GRAPH_LIFECYCLE_SCHEMA_VERSION = "n2-cuda-graph-lifecycle-v1"
REQUIRED_TRT_ENGINE_ROLES = (
    "transformer_joint_capture",
    "transformer_action_cached",
    "vae_encode",
)

PROFILE_WORKLOAD_FIELDS = (
    "embodiment",
    "camera_count",
    "camera_keys",
    "history_frames_per_camera",
    "history_delta_indices",
    "image_target_size",
    "action_horizon",
    "model_type",
    "model_dtype",
    "model_size",
    "vae_type",
    "token_merge_enabled",
    "token_merge_kernel",
    "token_merge_layers",
)

PROFILE_RUNTIME_FIELDS = (
    "precision",
    "codec",
    "num_video_steps",
    "num_action_steps",
    "shift",
    "joint_denoise",
    "kv_cache_cond",
    "video_guidance_scale",
    "action_guidance_scale",
    "seed",
)

MEASUREMENT_RUN_ROLES = ("reference", "candidate", "diagnostic")

NESTED_TIMING_FIELDS = (
    "client_wrapper_ms",
    "client_roundtrip_ms",
    "server_handler_ms",
    "policy_ms",
    "codec_ms",
    "model_predict_ms",
    "decode_ms",
)

DERIVED_TIMING_FIELDS = (
    "client_mapping_ms",
    "rpc_overhead_ms",
    "server_adapter_ms",
    "preprocess_other_ms",
    "non_model_ms",
)

CHECKPOINT_IDENTITY_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "model.safetensors.index.json",
)

G1_TARGET_PROFILES: dict[str, dict[str, Any]] = {
    "G1-H80-2V5F-THOR": {
        "embodiment": "unitree_g1_whole_body_teleop_latent_gripper",
        "camera_count": 2,
        "camera_keys": ["ego_view", "right_wrist"],
        "history_frames_per_camera": 5,
        "history_delta_indices": [-40, -30, -20, -10, 0],
        "image_target_size": [224, 320],
        "action_horizon": 80,
        "model_type": "GrootN2",
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "vae_type": "wan22",
        "token_merge_enabled": False,
        "token_merge_kernel": None,
        "token_merge_layers": None,
        "precision": "bf16",
        "codec": "mp4v",
        "num_video_steps": 2,
        "num_action_steps": 4,
        "shift": 5.0,
        "joint_denoise": True,
        "kv_cache_cond": True,
        "video_guidance_scale": 1.0,
        "action_guidance_scale": 1.0,
        "seed": 42,
        "gpu_name_contains": "Thor",
        "hardware_family": "thor",
    },
    "G1-H80-3V5F-THOR": {
        "embodiment": "unitree_g1_whole_body_teleop_latent_gripper",
        "camera_count": 3,
        "camera_keys": ["left_wrist", "ego_view", "right_wrist"],
        "history_frames_per_camera": 5,
        "history_delta_indices": [-40, -30, -20, -10, 0],
        "image_target_size": [224, 320],
        "action_horizon": 80,
        "model_type": "GrootN2",
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "vae_type": "wan22",
        "token_merge_enabled": False,
        "token_merge_kernel": None,
        "token_merge_layers": None,
        "precision": "bf16",
        "codec": "mp4v",
        "num_video_steps": 2,
        "num_action_steps": 4,
        "shift": 5.0,
        "joint_denoise": True,
        "kv_cache_cond": True,
        "video_guidance_scale": 1.0,
        "action_guidance_scale": 1.0,
        "seed": 42,
        "gpu_name_contains": "Thor",
        "hardware_family": "thor",
    },
    "G1-H80-3V5F-5090": {
        "embodiment": "unitree_g1_whole_body_teleop_latent_gripper",
        "camera_count": 3,
        "camera_keys": ["left_wrist", "ego_view", "right_wrist"],
        "history_frames_per_camera": 5,
        "history_delta_indices": [-40, -30, -20, -10, 0],
        "image_target_size": [224, 320],
        "action_horizon": 80,
        "model_type": "GrootN2",
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "vae_type": "wan22",
        "token_merge_enabled": False,
        "token_merge_kernel": None,
        "token_merge_layers": None,
        "precision": "bf16",
        "codec": "mp4v",
        "num_video_steps": 2,
        "num_action_steps": 4,
        "shift": 5.0,
        "joint_denoise": True,
        "kv_cache_cond": True,
        "video_guidance_scale": 1.0,
        "action_guidance_scale": 1.0,
        "seed": 42,
        "gpu_name_contains": "RTX 5090",
        "hardware_family": "rtx5090",
    },
}

YAM_TARGET_PROFILES: dict[str, dict[str, Any]] = {
    "YAM-H48-4V5F-5090": {
        "embodiment": "xdof_relative_eef_relative_joint_umi_keys_4views",
        "camera_count": 4,
        "camera_keys": [
            "top_camera-images-rgb_320_240",
            "left_camera-images-rgb_320_240",
            "right_camera-images-rgb_320_240",
            "bottom_camera-images-rgb_320_240",
        ],
        "history_frames_per_camera": 5,
        "history_delta_indices": [-24, -18, -12, -6, 0],
        "image_target_size": [224, 320],
        "action_horizon": 48,
        "model_type": "GrootN2",
        "model_dtype": "bfloat16",
        "model_size": "2B",
        "vae_type": "wan22",
        "token_merge_enabled": False,
        "token_merge_kernel": None,
        "token_merge_layers": None,
        "precision": "bf16",
        "codec": "mp4v",
        "num_video_steps": 2,
        "num_action_steps": 4,
        "shift": 5.0,
        "joint_denoise": True,
        "kv_cache_cond": True,
        "video_guidance_scale": 1.0,
        "action_guidance_scale": 1.0,
        "seed": 42,
        "gpu_name_contains": "RTX 5090",
        "hardware_family": "rtx5090",
    },
}


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize non-empty millisecond samples without discarding raw data."""
    if not values:
        raise ValueError("cannot summarize an empty timing series")
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise ValueError("timing series contains non-finite values")
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 6),
        "stdev": round(float(array.std()), 6),
        "min": round(float(array.min()), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "p99": round(float(np.percentile(array, 99)), 6),
        "max": round(float(array.max()), 6),
    }


def derive_nested_timing_samples(
    raw_ms: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Derive non-overlapping stages from aligned, nested per-request samples.

    The required timestamp order is::

        client_wrapper >= client_roundtrip >= server_handler >= policy
        policy >= codec + model_predict + decode

    Small negative derived values are retained instead of clamped so clock or
    instrumentation errors remain visible in the receipt.
    """
    missing = [name for name in NESTED_TIMING_FIELDS if name not in raw_ms]
    if missing:
        raise ValueError(f"missing nested timing series: {missing}")
    lengths = {name: len(raw_ms[name]) for name in NESTED_TIMING_FIELDS}
    if not lengths[NESTED_TIMING_FIELDS[0]]:
        raise ValueError("cannot derive timing from empty series")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"timing sample lengths do not match: {lengths}")

    def difference(left: str, *right: str) -> list[float]:
        return [
            float(left_value) - sum(float(raw_ms[name][index]) for name in right)
            for index, left_value in enumerate(raw_ms[left])
        ]

    return {
        "client_mapping_ms": difference("client_wrapper_ms", "client_roundtrip_ms"),
        "rpc_overhead_ms": difference("client_roundtrip_ms", "server_handler_ms"),
        "server_adapter_ms": difference("server_handler_ms", "policy_ms"),
        "preprocess_other_ms": difference("policy_ms", "codec_ms", "model_predict_ms", "decode_ms"),
        "non_model_ms": difference("client_wrapper_ms", "model_predict_ms"),
    }


def summarize_timing_samples(
    raw_ms: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int]]:
    """Validate aligned timing series and summarize each metric."""
    if not raw_ms:
        raise ValueError("cannot summarize empty timing samples")
    lengths = {name: len(values) for name, values in raw_ms.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"timing sample lengths do not match: {lengths}")
    return {name: summarize_ms(values) for name, values in raw_ms.items()}


def validate_action(
    action: Mapping[str, Any],
    expected: Mapping[str, tuple[int, ...]],
) -> None:
    """Require exact action keys, shapes, float32 dtype, and finite values."""
    if set(action) != set(expected):
        raise AssertionError(f"Expected action keys {sorted(expected)}, got {sorted(action)}")
    for key, shape in expected.items():
        value = action[key]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float32:
            raise AssertionError(
                f"{key} must be float32 with shape {shape}; got "
                f"{getattr(value, 'dtype', type(value))} {getattr(value, 'shape', None)}"
            )
        if not bool(np.isfinite(value).all()):
            raise AssertionError(f"{key} contains non-finite values")


def action_equivalence(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Return per-key decoded-action drift in physical units."""
    if set(reference) != set(candidate):
        raise ValueError(
            f"action keys do not match: reference={sorted(reference)}, candidate={sorted(candidate)}"
        )
    equivalence = {}
    for key, reference_value in reference.items():
        candidate_value = candidate[key]
        if candidate_value.shape != reference_value.shape:
            raise ValueError(
                f"action shape does not match for {key}: "
                f"reference={reference_value.shape}, candidate={candidate_value.shape}"
            )
        reference64 = reference_value.astype(np.float64)
        candidate64 = candidate_value.astype(np.float64)
        if not bool(np.isfinite(reference64).all()):
            raise ValueError(f"reference action {key} contains non-finite values")
        if not bool(np.isfinite(candidate64).all()):
            raise ValueError(f"candidate action {key} contains non-finite values")
        difference = candidate64 - reference64
        denominator = float(np.sqrt(np.mean(reference64**2))) + 1e-12
        equivalence[key] = {
            # Retain full float64 precision for the correctness gate. Formatting
            # belongs at the presentation boundary; rounding here can turn a
            # real drift into zero and incorrectly pass an exact-output gate.
            "max_abs": float(np.abs(difference).max()),
            "rel_rmse": float(np.sqrt(np.mean(difference**2)) / denominator),
        }
    return equivalence


def gate_action_equivalence(
    equivalence: Mapping[str, Mapping[str, float]],
    *,
    rtol: float,
    max_abs: float,
    label: str = "candidate",
) -> list[str]:
    """Return breaches against aggregate relative and localized absolute bounds."""
    for name, value in (("rtol", rtol), ("max_abs", max_abs)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    breaches = []
    for key, stats in equivalence.items():
        observed_max_abs = stats["max_abs"]
        observed_rel_rmse = stats["rel_rmse"]
        if not math.isfinite(observed_max_abs) or not math.isfinite(observed_rel_rmse):
            raise ValueError(f"equivalence statistics for {key!r} must be finite")
        if observed_max_abs > max_abs:
            breaches.append(f"{label}:{key} max_abs={observed_max_abs:g} > max_abs {max_abs:g}")
        if observed_rel_rmse > rtol:
            breaches.append(f"{label}:{key} rel_rmse={observed_rel_rmse:g} > rtol {rtol:g}")
    return breaches


def validate_runtime_contract(
    contract: Any,
    *,
    backend: str,
    precision: str,
    require_server_process: bool = False,
) -> dict[str, Any]:
    """Validate server-reported backend state and return a JSON-safe copy."""
    if not isinstance(contract, dict):
        raise TypeError("runtime contract must be a dict")
    required = ("backend", "precision", "trt_engines", "cuda_graph", "torch_compile")
    missing = [name for name in required if name not in contract]
    if missing:
        raise RuntimeError(f"runtime contract is missing fields: {missing}")
    if contract["backend"] != backend:
        raise RuntimeError(
            f"runtime contract backend {contract['backend']!r} does not match {backend!r}"
        )
    if str(contract["precision"]).lower() != precision.lower():
        raise RuntimeError(
            f"runtime contract precision {contract['precision']!r} does not match {precision!r}"
        )
    expected_state = {
        "trt_engines": backend.startswith("trt"),
        "cuda_graph": "graph" in backend,
        "torch_compile": backend.startswith("compile"),
    }
    for field, expected in expected_state.items():
        actual = contract[field]
        if type(actual) is not bool:
            raise TypeError(f"runtime contract {field} must be a bool, got {type(actual).__name__}")
        if actual is not expected:
            raise RuntimeError(
                f"runtime contract for {backend!r} requires {field}={expected}, got {actual}"
            )
    if require_server_process:
        server_process = contract.get("server_process")
        if not isinstance(server_process, dict):
            raise RuntimeError("runtime contract is missing live server_process identity")
        required_identity = ("runtime_source", "checkpoint", "gpu")
        missing_identity = [
            name for name in required_identity if not isinstance(server_process.get(name), dict)
        ]
        if missing_identity:
            raise RuntimeError(
                f"runtime contract server_process is missing identity: {missing_identity}"
            )
        if not server_process["gpu"].get("uuid"):
            raise RuntimeError("runtime contract server_process has no GPU UUID")
    try:
        json.dumps(contract)
    except TypeError as exc:
        raise TypeError("runtime contract must be JSON serializable") from exc
    return dict(contract)


def resolve_policy_acceleration_state(
    policy: Any,
    *,
    use_torch_compile: bool,
) -> dict[str, str | bool]:
    """Resolve the policy-reported execution state or the mainline fallback."""
    state = getattr(policy, "inference_acceleration_state", None)
    if state is None:
        state = {
            "backend": "compile" if use_torch_compile else "eager",
            "trt_engines": False,
            "cuda_graph": False,
            "torch_compile": use_torch_compile,
        }
    if not isinstance(state, dict):
        raise TypeError("policy inference_acceleration_state must be a dict")
    names = ("backend", "trt_engines", "cuda_graph", "torch_compile")
    resolved = {name: state[name] for name in names if name in state}
    validated = validate_runtime_contract(
        {**resolved, "precision": "bf16"},
        backend=str(resolved.get("backend", "")),
        precision="bf16",
    )
    return {name: validated[name] for name in names}


def _validate_target_profile(
    target_profile: str,
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    workload: Mapping[str, Any],
    gpu: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if target_profile not in profiles:
        raise ValueError(f"unknown target profile: {target_profile!r}")
    profile = profiles[target_profile]
    for field in PROFILE_WORKLOAD_FIELDS:
        if workload.get(field) != profile[field]:
            raise RuntimeError(
                f"target profile {target_profile!r} requires {field}={profile[field]}, "
                f"got {workload.get(field)!r}"
            )
    for field in PROFILE_RUNTIME_FIELDS:
        if runtime_contract.get(field) != profile[field]:
            raise RuntimeError(
                f"target profile {target_profile!r} requires {field}={profile[field]!r}, "
                f"got {runtime_contract.get(field)!r}"
            )
    gpu_name = str(gpu.get("torch_name", ""))
    expected_name = str(profile["gpu_name_contains"])
    if expected_name.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"target profile {target_profile!r} requires GPU name containing "
            f"{expected_name!r}, got {gpu_name!r}"
        )
    return dict(profile)


def validate_g1_target_profile(
    target_profile: str,
    *,
    workload: Mapping[str, Any],
    server_process: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a formal G1 run to one hardware and workload profile."""
    gpu = server_process.get("gpu")
    if not isinstance(gpu, Mapping):
        raise TypeError("target profile admission requires live server GPU identity")
    return _validate_target_profile(
        target_profile,
        profiles=G1_TARGET_PROFILES,
        workload=workload,
        gpu=gpu,
        runtime_contract=runtime_contract,
    )


def validate_yam_target_profile(
    target_profile: str,
    *,
    workload: Mapping[str, Any],
    gpu: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a formal YAM run to one hardware and workload profile."""
    return _validate_target_profile(
        target_profile,
        profiles=YAM_TARGET_PROFILES,
        workload=workload,
        gpu=gpu,
        runtime_contract=runtime_contract,
    )


def validate_formal_target_options(
    target_profile: str | None,
    *,
    allow_unprofiled_target: bool,
    diagnostic_options: Mapping[str, bool],
    fixture: str,
) -> None:
    """Keep diagnostic bypasses out of named-profile promotion receipts."""
    if target_profile is None:
        if not allow_unprofiled_target:
            raise ValueError(
                f"formal {fixture} benchmarks require --target-profile; "
                "use --allow-unprofiled-target only for diagnostics"
            )
        return
    if allow_unprofiled_target:
        raise ValueError("--target-profile cannot be combined with --allow-unprofiled-target")
    enabled = sorted(name for name, value in diagnostic_options.items() if value)
    if enabled:
        raise ValueError(
            f"formal target profile {target_profile!r} cannot use diagnostic options: {enabled}"
        )


def _command_state(command: Sequence[str]) -> dict[str, Any]:
    """Capture one target-state command without placing it in a timed span."""
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"hardware-state executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"hardware-state command timed out after {exc.timeout}s: {' '.join(command)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"hardware-state command exited {exc.returncode}: {' '.join(command)}; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        ) from exc
    return {
        "command": list(command),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _nvidia_smi_gpu_selector(gpu_uuid: Any) -> str:
    """Convert PyTorch's bare GPU UUID to an ``nvidia-smi`` selector."""
    selector = str(gpu_uuid).strip()
    if not selector:
        raise ValueError("nvidia-smi hardware-state capture requires a GPU UUID")
    if selector.startswith(("GPU-", "MIG-")):
        return selector
    return f"GPU-{selector}"


def _sysfs_thermal_state(root: Path = Path("/sys/class/thermal")) -> dict[str, Any]:
    """Capture raw thermal-zone and cooling-device state for Thor evidence."""

    def read_required(path: Path) -> str:
        try:
            return path.read_text().strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"required Thor thermal attribute is missing: {path}") from exc

    def read_optional(path: Path) -> str | None:
        try:
            return path.read_text().strip()
        except FileNotFoundError:
            return None

    thermal_zones = []
    for zone in sorted(root.glob("thermal_zone*")):
        thermal_zones.append(
            {
                "name": zone.name,
                "type": read_required(zone / "type"),
                "temperature_millicelsius": read_required(zone / "temp"),
                # The policy attribute is not exported by every supported
                # kernel; absence is explicit optional evidence rather than a
                # substitute for a required temperature reading.
                "policy": read_optional(zone / "policy"),
            }
        )
    cooling_devices = []
    for device in sorted(root.glob("cooling_device*")):
        cooling_devices.append(
            {
                "name": device.name,
                "type": read_required(device / "type"),
                "current_state": read_required(device / "cur_state"),
                "maximum_state": read_required(device / "max_state"),
            }
        )
    if not thermal_zones:
        raise RuntimeError(f"Thor hardware-state capture found no thermal zones under {root}")
    if not cooling_devices:
        raise RuntimeError(f"Thor hardware-state capture found no cooling devices under {root}")
    return {
        "thermal_zones": thermal_zones,
        "cooling_devices": cooling_devices,
        "optional_fields": ["thermal_zones[].policy"],
    }


def capture_g1_hardware_state(
    target_profile: str,
    *,
    server_process: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture profile-specific operating state before or after a formal G1 run."""
    if target_profile not in G1_TARGET_PROFILES:
        raise ValueError(f"unknown G1 target profile: {target_profile!r}")
    server_host = str(server_process.get("host", ""))
    current_host = socket.gethostname()
    if not server_host or server_host != current_host:
        raise RuntimeError(
            "hardware-state capture must run on the live server host: "
            f"server={server_host!r}, client={current_host!r}"
        )
    profile = G1_TARGET_PROFILES[target_profile]
    software = {
        "platform": server_process.get("platform"),
        "python": server_process.get("python"),
        "torch": server_process.get("torch"),
        "torch_cuda": server_process.get("torch_cuda"),
    }
    hardware_family = profile["hardware_family"]
    if hardware_family == "thor":
        driver_path = Path("/proc/driver/nvidia/version")
        try:
            software["driver"] = driver_path.read_text().strip()
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"Thor driver identity is unavailable: {driver_path}") from exc
        operating_state = {
            "power_mode": _command_state(("nvpmodel", "-q")),
            "clocks": _command_state(("jetson_clocks", "--show")),
            "temperature_and_load": _command_state(
                ("tegrastats", "--interval", "100", "--count", "1")
            ),
            "thermal_and_throttling": _sysfs_thermal_state(),
        }
    elif hardware_family == "rtx5090":
        gpu = server_process["gpu"]
        operating_state = {
            "nvidia_smi": _command_state(
                (
                    "nvidia-smi",
                    f"--id={_nvidia_smi_gpu_selector(gpu['uuid'])}",
                    "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,"
                    "clocks.current.graphics,clocks.current.memory,power.draw,power.limit,"
                    "clocks_throttle_reasons.active",
                    "--format=csv,noheader,nounits",
                )
            )
        }
    else:  # pragma: no cover - every registered profile is covered above.
        raise AssertionError(f"unsupported hardware family: {hardware_family!r}")
    return {
        "schema_version": "n2-g1-hardware-state-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "target_profile": target_profile,
        "host": current_host,
        "hardware_family": hardware_family,
        "software": software,
        "gpu": dict(server_process["gpu"]),
        "operating_state": operating_state,
    }


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    """Return a content identity for a retained benchmark artifact."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"benchmark artifact is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_artifact_record(record: Any, *, label: str) -> dict[str, Any]:
    """Require a retained artifact record to match the file currently on disk."""
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} identity must be a mapping")
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} identity has no path")
    actual = artifact_identity(Path(path))
    expected = {name: record.get(name) for name in ("path", "size_bytes", "sha256")}
    if actual != expected:
        raise ValueError(f"{label} content does not match its retained identity")
    return actual


def reference_approval_identity(
    path: Path,
    *,
    fixture: str,
    target_profile: str,
) -> dict[str, Any]:
    """Load and validate a content-addressed PIC approval declaration."""
    identity = artifact_identity(path)
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise TypeError("reference approval manifest must contain a JSON object")
    expected = {
        "schema_version": REFERENCE_APPROVAL_SCHEMA_VERSION,
        "fixture": fixture,
        "target_profile": target_profile,
    }
    mismatches = {
        name: {"expected": value, "actual": payload.get(name)}
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"reference approval manifest does not match the run: {mismatches}")
    approved_sha = payload.get("approved_runtime_sha")
    if not _is_git_sha(approved_sha):
        raise ValueError("reference approval manifest has no valid approved_runtime_sha")
    for field in ("approved_by", "approved_at"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"reference approval manifest has no {field}")
    _reference_gate_bounds(payload)
    return {**identity, "contents": payload}


def _reference_gate_bounds(contents: Mapping[str, Any]) -> dict[str, float]:
    """Return the PIC-declared maximum candidate tolerances."""
    bounds = contents.get("candidate_gate_bounds")
    if not isinstance(bounds, Mapping):
        raise TypeError("reference approval manifest has no candidate_gate_bounds")
    validated = {}
    for field in ("rtol", "max_abs"):
        value = bounds.get(field)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"reference approval candidate_gate_bounds.{field} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"reference approval candidate_gate_bounds.{field} must be finite "
                "and non-negative"
            )
        validated[field] = value
    return validated


def cuda_graph_lifecycle_snapshot(runtime_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one live CUDA-graph lifecycle counter snapshot."""
    lifecycle = runtime_contract.get("cuda_graph_lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise TypeError(
            "formal CUDA-graph candidates must report cuda_graph_lifecycle "
            "in the live runtime contract"
        )
    if lifecycle.get("schema_version") != CUDA_GRAPH_LIFECYCLE_SCHEMA_VERSION:
        raise ValueError(
            "cuda_graph_lifecycle uses an unsupported schema; expected "
            f"{CUDA_GRAPH_LIFECYCLE_SCHEMA_VERSION!r}"
        )
    snapshot: dict[str, int | str] = {
        "schema_version": CUDA_GRAPH_LIFECYCLE_SCHEMA_VERSION,
    }
    for field in ("capture_count", "recapture_count", "replay_count"):
        value = lifecycle.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"cuda_graph_lifecycle.{field} must be an integer")
        if value < 0:
            raise ValueError(f"cuda_graph_lifecycle.{field} must be non-negative")
        snapshot[field] = value
    return snapshot


def validate_cuda_graph_measurement_window(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    measured_requests: int,
) -> dict[str, Any]:
    """Prove a formal measured window replayed one already-captured graph."""
    if not isinstance(measured_requests, int) or isinstance(measured_requests, bool):
        raise TypeError("measured_requests must be an integer")
    if measured_requests <= 0:
        raise ValueError("measured_requests must be positive")
    before_snapshot = cuda_graph_lifecycle_snapshot({"cuda_graph_lifecycle": before})
    after_snapshot = cuda_graph_lifecycle_snapshot({"cuda_graph_lifecycle": after})
    if before_snapshot["capture_count"] < 1:
        raise RuntimeError("CUDA graph was not captured before the measured window")

    deltas = {
        field: after_snapshot[field] - before_snapshot[field]
        for field in ("capture_count", "recapture_count", "replay_count")
    }
    if deltas["capture_count"] != 0 or deltas["recapture_count"] != 0:
        raise RuntimeError(
            f"CUDA graph capture or recapture occurred during the measured window: {deltas}"
        )
    if deltas["replay_count"] != measured_requests:
        raise RuntimeError(
            "CUDA graph replay count does not match the measured request count: "
            f"expected {measured_requests}, got {deltas['replay_count']}"
        )
    return {
        "schema_version": CUDA_GRAPH_LIFECYCLE_SCHEMA_VERSION,
        "measured_requests": measured_requests,
        "before": before_snapshot,
        "after": after_snapshot,
        "delta": deltas,
        "capture_stable_during_measurement": True,
    }


def validate_candidate_runtime_artifacts(
    *,
    role: str,
    target_profile: str | None,
    runtime_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Authenticate the engine bundle used by a formal accelerated candidate."""
    if role != "candidate" or target_profile is None or runtime_contract["trt_engines"] is False:
        return None
    artifacts = runtime_contract.get("runtime_artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError(
            "formal TensorRT candidates must report runtime_artifacts in the live runtime contract"
        )
    if artifacts.get("schema_version") != RUNTIME_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "runtime_artifacts uses an unsupported schema; expected "
            f"{RUNTIME_ARTIFACT_SCHEMA_VERSION!r}"
        )
    for field in ("backend", "precision", "quantize_mode", "tensorrt_version"):
        if not isinstance(artifacts.get(field), str) or not artifacts[field]:
            raise ValueError(f"runtime_artifacts has no {field}")
    if artifacts["backend"] != runtime_contract["backend"]:
        raise ValueError("runtime_artifacts backend does not match the live runtime contract")
    if artifacts["precision"].lower() != str(runtime_contract["precision"]).lower():
        raise ValueError("runtime_artifacts precision does not match the live runtime contract")

    recorded_checkpoint = artifacts.get("checkpoint")
    if not isinstance(recorded_checkpoint, Mapping):
        raise TypeError("runtime_artifacts has no checkpoint identity")
    recorded_checkpoint_content = {
        "files": recorded_checkpoint.get("files"),
        "weight_shards": recorded_checkpoint.get("weight_shards"),
    }
    checkpoint_content = {
        "files": checkpoint.get("files"),
        "weight_shards": checkpoint.get("weight_shards"),
    }
    if recorded_checkpoint_content != checkpoint_content:
        raise ValueError("runtime_artifacts checkpoint does not match the measured checkpoint")

    target_gpu = artifacts.get("target_gpu")
    if not isinstance(target_gpu, Mapping):
        raise TypeError("runtime_artifacts has no target_gpu identity")
    actual_name = str(gpu.get("torch_name", ""))
    if target_gpu.get("torch_name") != actual_name:
        raise ValueError("runtime_artifacts target GPU name does not match the selected GPU")
    actual_capability = gpu.get("capability", gpu.get("compute_capability"))
    if target_gpu.get("compute_capability") != actual_capability:
        raise ValueError("runtime_artifacts compute capability does not match the selected GPU")

    build_manifest = _validate_artifact_record(
        artifacts.get("build_manifest"),
        label="TensorRT build manifest",
    )
    build_contents = json.loads(Path(build_manifest["path"]).read_text())
    if not isinstance(build_contents, Mapping):
        raise TypeError("TensorRT build manifest must contain a JSON object")
    if build_contents.get("quantize_mode") != artifacts["quantize_mode"]:
        raise ValueError("TensorRT build manifest quantize_mode does not match runtime_artifacts")
    if build_contents.get("tensorrt") != artifacts["tensorrt_version"]:
        raise ValueError("TensorRT build manifest version does not match runtime_artifacts")
    if (
        str(runtime_contract["precision"]).lower() == "bf16"
        and artifacts["quantize_mode"] != "none"
    ):
        raise ValueError("formal BF16 TensorRT candidates require quantize_mode='none'")

    engine_records = artifacts.get("loaded_engines")
    if not isinstance(engine_records, Mapping):
        raise TypeError("runtime_artifacts must identify the loaded engine bundle by role")
    if set(engine_records) != set(REQUIRED_TRT_ENGINE_ROLES):
        raise ValueError(
            "runtime_artifacts loaded_engines must contain exactly "
            f"{list(REQUIRED_TRT_ENGINE_ROLES)}"
        )
    loaded_engines = {
        role: _validate_artifact_record(engine_records[role], label=f"loaded {role} engine")
        for role in REQUIRED_TRT_ENGINE_ROLES
    }

    manifest_engines = build_contents.get("engines")
    if not isinstance(manifest_engines, list):
        raise TypeError("TensorRT build manifest has no engine list")
    if len(manifest_engines) != len(REQUIRED_TRT_ENGINE_ROLES):
        raise ValueError("TensorRT build manifest does not describe the complete fixed bundle")
    manifest_by_path = {}
    for index, record in enumerate(manifest_engines):
        if not isinstance(record, Mapping):
            raise TypeError(f"TensorRT build manifest engine {index} is not an object")
        path = record.get("engine")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"TensorRT build manifest engine {index} has no absolute path")
        resolved_path = str(Path(path).resolve())
        if resolved_path in manifest_by_path:
            raise ValueError("TensorRT build manifest contains duplicate engine paths")
        if record.get("size_bytes") != Path(path).stat().st_size:
            raise ValueError(f"TensorRT build manifest engine size is stale: {path}")
        precision = str(record.get("precision", "")).lower()
        if not precision.startswith("bf16") or "fp8" in precision:
            raise ValueError(f"TensorRT build manifest engine is not BF16-only: {path}")
        manifest_by_path[resolved_path] = record

    loaded_paths = {record["path"] for record in loaded_engines.values()}
    if set(manifest_by_path) != loaded_paths:
        raise ValueError(
            "live runtime loaded-engine identities do not match the complete build-manifest bundle"
        )
    return {
        **dict(artifacts),
        "build_manifest": {**build_manifest, "contents": dict(build_contents)},
        "loaded_engines": loaded_engines,
    }


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Hash checkpoint metadata, the weight index, and every indexed shard."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    files = {}
    for name in CHECKPOINT_IDENTITY_FILES:
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint identity file is missing: {path}")
        files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint weight index has no weight_map: {index_path}")

    shard_values = list(weight_map.values())
    if not all(isinstance(name, str) and name for name in shard_values):
        raise ValueError(f"checkpoint weight index contains invalid shard names: {index_path}")
    shard_names = sorted(set(shard_values))
    weight_shards = {}
    for name in shard_names:
        path = (checkpoint / name).resolve()
        try:
            path.relative_to(checkpoint)
        except ValueError as exc:
            raise ValueError(f"checkpoint shard escapes checkpoint directory: {name}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint weight shard is missing: {path}")
        weight_shards[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "path": str(checkpoint),
        "files": files,
        "weight_shards": weight_shards,
    }


def reference_action_identity(actions_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Bind external reference actions to the receipt that produced them."""
    actions = artifact_identity(actions_path)
    receipt = artifact_identity(receipt_path)
    payload = json.loads(Path(receipt_path).expanduser().resolve().read_text())
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "reference action source receipt uses an unsupported measurement schema; "
            "regenerate the reference with this harness"
        )
    if payload.get("status") != "complete":
        raise ValueError("reference action source receipt is not complete")
    recorded = payload.get("artifacts", {}).get("actions")
    if not isinstance(recorded, dict):
        raise TypeError("reference action source receipt does not identify its actions")
    if (
        recorded.get("sha256") != actions["sha256"]
        or recorded.get("size_bytes") != actions["size_bytes"]
    ):
        raise ValueError("reference action artifact does not match its source receipt")

    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise TypeError("reference action source receipt has no manifest")
    measurement_source = manifest.get("benchmark_source", manifest.get("harness_source"))
    required = {
        "run": payload.get("run"),
        "workload": payload.get("workload"),
        "measurement_source": measurement_source,
        "measurement_validator": manifest.get("measurement_validator"),
        "runtime_source": manifest.get("runtime_source"),
        "checkpoint": manifest.get("checkpoint"),
        "reference_approval": manifest.get("reference_approval"),
    }
    missing = [name for name, value in required.items() if not isinstance(value, dict)]
    if missing:
        raise ValueError(f"reference action source receipt is missing provenance: {missing}")
    if not isinstance(required["run"].get("run_id"), str):
        raise TypeError("reference action source receipt has no run_id")
    for name in ("measurement_source", "runtime_source"):
        source = required[name]
        if (
            not isinstance(source.get("sha"), str)
            or len(source["sha"]) != 40
            or not isinstance(source.get("dirty"), bool)
        ):
            raise ValueError(f"reference action source receipt has invalid {name}")
    return {
        "actions": actions,
        "receipt": receipt,
        "origin": {
            "schema_version": payload.get("schema_version"),
            **required,
        },
    }


def measurement_validator_identity(
    measurement_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Identify the trusted contract module independently of the runtime tree."""
    source_sha = measurement_source.get("sha")
    source_dirty = measurement_source.get("dirty")
    if not _is_git_sha(source_sha) or not isinstance(source_dirty, bool):
        raise ValueError("measurement source has no valid Git identity")
    return {
        "schema_version": MEASUREMENT_VALIDATOR_SCHEMA_VERSION,
        "measurement_source_sha": source_sha,
        "measurement_source_dirty": source_dirty,
        "contract": artifact_identity(Path(__file__)),
    }


def validate_reference_measurement_validator(
    reference: Mapping[str, Any],
    *,
    measurement_validator: Mapping[str, Any],
) -> None:
    """Require reference and candidate to use the same trusted validator bytes."""
    origin = reference.get("origin")
    recorded = origin.get("measurement_validator") if isinstance(origin, Mapping) else None
    if not isinstance(recorded, Mapping):
        raise TypeError("external reference has no measurement-validator identity")

    def comparable(identity: Mapping[str, Any]) -> dict[str, Any]:
        contract = identity.get("contract")
        if not isinstance(contract, Mapping):
            raise TypeError("measurement-validator identity has no contract artifact")
        return {
            "schema_version": identity.get("schema_version"),
            "measurement_source_sha": identity.get("measurement_source_sha"),
            "measurement_source_dirty": identity.get("measurement_source_dirty"),
            "contract": {
                "size_bytes": contract.get("size_bytes"),
                "sha256": contract.get("sha256"),
            },
        }

    if comparable(recorded) != comparable(measurement_validator):
        raise ValueError(
            "external reference measurement validator does not match the candidate validator"
        )


def validate_reference_origin(
    reference: Mapping[str, Any],
    *,
    workload: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    """Require an external reference to represent the same fixed workload."""
    origin = reference.get("origin")
    if not isinstance(origin, Mapping):
        raise TypeError("external reference has no source origin")
    if origin.get("workload") != dict(workload):
        raise ValueError("external reference workload does not match the candidate workload")

    reference_checkpoint = origin.get("checkpoint")
    if not isinstance(reference_checkpoint, Mapping):
        raise TypeError("external reference has no checkpoint identity")
    reference_content = {
        "files": reference_checkpoint.get("files"),
        "weight_shards": reference_checkpoint.get("weight_shards"),
    }
    candidate_content = {
        "files": checkpoint.get("files"),
        "weight_shards": checkpoint.get("weight_shards"),
    }
    if reference_content != candidate_content:
        raise ValueError("external reference checkpoint does not match the candidate checkpoint")


def validate_reference_run(
    reference: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    """Require selected run fields to match the candidate fixture contract."""
    origin = reference.get("origin")
    run = origin.get("run") if isinstance(origin, Mapping) else None
    if not isinstance(run, Mapping):
        raise TypeError("external reference has no source run")
    mismatches = {
        name: {"reference": run.get(name), "candidate": value}
        for name, value in expected.items()
        if run.get(name) != value
    }
    if mismatches:
        raise ValueError(f"external reference run does not match candidate fields: {mismatches}")


def validate_external_reference_args(
    *,
    ref_actions: Path | None,
    ref_receipt: Path | None,
    gate_rtol: float | None,
    gate_max_abs: float | None,
) -> None:
    """Require reference actions, their receipt, and both bounds as one bundle."""
    supplied = {
        "--ref-actions": ref_actions is not None,
        "--ref-receipt": ref_receipt is not None,
        "--gate-rtol": gate_rtol is not None,
        "--gate-max-abs": gate_max_abs is not None,
    }
    if any(supplied.values()) and not all(supplied.values()):
        missing = sorted(name for name, present in supplied.items() if not present)
        raise ValueError(
            "correctness gate requires --ref-actions, --ref-receipt, --gate-rtol, and "
            "--gate-max-abs "
            f"together; missing {missing}"
        )
    for name, value in (("--gate-rtol", gate_rtol), ("--gate-max-abs", gate_max_abs)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_measurement_role_args(
    *,
    role: str,
    target_profile: str | None,
    backends: Sequence[str],
    ref_actions: Path | None,
    ref_receipt: Path | None,
    gate_rtol: float | None,
    gate_max_abs: float | None,
    save_actions: Path | None,
    reference_approval_manifest: Path | None,
    warmup: int,
    measured: int,
    uses_custom_runtime: bool = False,
) -> None:
    """Keep reference, candidate, and diagnostic evidence roles unambiguous."""
    if role not in MEASUREMENT_RUN_ROLES:
        raise ValueError(f"unknown measurement run role: {role!r}")
    validate_external_reference_args(
        ref_actions=ref_actions,
        ref_receipt=ref_receipt,
        gate_rtol=gate_rtol,
        gate_max_abs=gate_max_abs,
    )

    has_gate = ref_actions is not None
    if role == "reference":
        if target_profile is None:
            raise ValueError("reference role requires a formal --target-profile")
        if tuple(backends) != ("eager",):
            raise ValueError("reference role requires exactly one eager backend")
        if uses_custom_runtime:
            raise ValueError("reference role cannot use a custom policy runtime")
        if has_gate:
            raise ValueError("reference role cannot consume an external correctness gate")
        if save_actions is None:
            raise ValueError("reference role requires --save-actions")
        if reference_approval_manifest is None:
            raise ValueError("reference role requires --reference-approval-manifest")

    elif reference_approval_manifest is not None:
        raise ValueError("--reference-approval-manifest is valid only for reference role")
    if role == "candidate" and target_profile is None:
        raise ValueError("candidate role requires a formal --target-profile")
    if role == "candidate" and not has_gate:
        raise ValueError(
            "candidate role requires --ref-actions, --ref-receipt, --gate-rtol, and "
            "--gate-max-abs"
        )
    if role == "diagnostic" and target_profile is not None:
        raise ValueError("diagnostic role cannot use a formal --target-profile")
    if role in ("reference", "candidate") and (
        warmup != FORMAL_WARMUP_REQUESTS or measured != FORMAL_MEASURED_REQUESTS
    ):
        raise ValueError(
            "formal reference and candidate runs require exactly "
            f"{FORMAL_WARMUP_REQUESTS} warmups and {FORMAL_MEASURED_REQUESTS} "
            f"measured requests; got {warmup}/{measured}"
        )


def validate_reference_runtime_source(
    *,
    role: str,
    approved_reference_runtime_sha: str | None,
    runtime_source: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind an ungated reference to the clean runtime SHA approved by its PIC."""
    if role != "reference":
        return None
    runtime_root = runtime_source.get("root")
    if not isinstance(runtime_root, str) or not runtime_root:
        raise RuntimeError("reference runtime source has no repository root")
    if not _is_git_sha(approved_reference_runtime_sha):
        raise ValueError("reference approval has no valid approved runtime SHA")
    actual_sha = runtime_source.get("sha")
    if actual_sha != approved_reference_runtime_sha:
        raise RuntimeError(
            "reference runtime SHA does not match the content-addressed approval declaration: "
            f"expected {approved_reference_runtime_sha!r}, got {actual_sha!r}"
        )
    if runtime_source.get("dirty") is not False:
        raise RuntimeError("reference role requires a clean approved runtime source")
    try:
        current_main = subprocess.run(
            ("git", "rev-parse", "origin/main"),
            cwd=runtime_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reachability = subprocess.run(
            ("git", "merge-base", "--is-ancestor", approved_reference_runtime_sha, "origin/main"),
            cwd=runtime_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to validate reference approval reachability") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("cannot resolve origin/main for reference approval validation") from exc
    if reachability.returncode == 1:
        raise RuntimeError(
            "approved reference runtime is not reachable from the locally fetched origin/main: "
            f"{approved_reference_runtime_sha}"
        )
    if reachability.returncode != 0:
        diagnostics = reachability.stderr.strip() or reachability.stdout.strip()
        detail = f": {diagnostics}" if diagnostics else ""
        raise RuntimeError(
            "git failed while validating approved-reference reachability"
            f" (exit {reachability.returncode}){detail}"
        )
    return {
        "approved_runtime_sha": approved_reference_runtime_sha,
        "validated_origin_main_sha": current_main,
        "approved_sha_reachable_from_origin_main": True,
    }


def validate_formal_measurement_source(
    *,
    role: str,
    measurement_source: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require formal validator code to be a clean commit reachable from main."""
    if role == "diagnostic":
        return None
    root = measurement_source.get("root")
    sha = measurement_source.get("sha")
    if not isinstance(root, str) or not root:
        raise RuntimeError("formal measurement source has no repository root")
    if not _is_git_sha(sha):
        raise ValueError("formal measurement source has no valid Git SHA")
    if measurement_source.get("dirty") is not False:
        raise RuntimeError("formal measurement source must be clean")
    try:
        current_main = subprocess.run(
            ("git", "rev-parse", "origin/main"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reachability = subprocess.run(
            ("git", "merge-base", "--is-ancestor", sha, "origin/main"),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to validate the measurement source") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("cannot resolve origin/main for measurement-source validation") from exc
    if reachability.returncode == 1:
        raise RuntimeError(
            "formal measurement source is not reachable from the locally fetched origin/main: "
            f"{sha}"
        )
    if reachability.returncode != 0:
        diagnostics = reachability.stderr.strip() or reachability.stdout.strip()
        detail = f": {diagnostics}" if diagnostics else ""
        raise RuntimeError(
            "git failed while validating measurement-source reachability "
            f"(exit {reachability.returncode}){detail}"
        )
    return {
        "measurement_source_sha": sha,
        "validated_origin_main_sha": current_main,
        "measurement_sha_reachable_from_origin_main": True,
    }


def validate_reference_approval(
    reference: Mapping[str, Any],
    *,
    gate_rtol: float,
    gate_max_abs: float,
) -> None:
    """Require external actions and thresholds to match the PIC declaration."""
    origin = reference.get("origin")
    run = origin.get("run") if isinstance(origin, Mapping) else None
    runtime_source = origin.get("runtime_source") if isinstance(origin, Mapping) else None
    approval = origin.get("reference_approval") if isinstance(origin, Mapping) else None
    if (
        not isinstance(run, Mapping)
        or not isinstance(runtime_source, Mapping)
        or not isinstance(approval, Mapping)
    ):
        raise TypeError("external reference has no approval provenance")
    if run.get("role") != "reference":
        raise ValueError("external actions were not produced by a reference-role run")
    approved_sha = run.get("approved_reference_runtime_sha")
    if not _is_git_sha(approved_sha):
        raise ValueError("external reference has no approved runtime SHA")
    if runtime_source.get("sha") != approved_sha or runtime_source.get("dirty") is not False:
        raise ValueError("external reference runtime does not match its approved clean SHA")
    contents = approval.get("contents")
    if not isinstance(contents, Mapping):
        raise TypeError("external reference has no retained approval-manifest contents")
    expected_approval = {
        "schema_version": REFERENCE_APPROVAL_SCHEMA_VERSION,
        "fixture": run.get("fixture"),
        "target_profile": run.get("target_profile"),
        "approved_runtime_sha": approved_sha,
    }
    mismatches = {
        name: {"expected": value, "actual": contents.get(name)}
        for name, value in expected_approval.items()
        if contents.get(name) != value
    }
    if mismatches:
        raise ValueError(f"external reference approval manifest is inconsistent: {mismatches}")
    for field in ("approved_by", "approved_at"):
        if not isinstance(contents.get(field), str) or not contents[field].strip():
            raise ValueError(f"external reference approval manifest has no {field}")
    approved_bounds = _reference_gate_bounds(contents)
    selected_bounds = {"rtol": gate_rtol, "max_abs": gate_max_abs}
    for field, selected in selected_bounds.items():
        if not math.isfinite(selected) or selected < 0:
            raise ValueError(f"candidate {field} must be finite and non-negative")
        if selected > approved_bounds[field]:
            raise ValueError(
                f"candidate {field}={selected:g} exceeds the PIC-declared maximum "
                f"{approved_bounds[field]:g}"
            )
    recorded = {name: approval.get(name) for name in ("path", "size_bytes", "sha256")}
    if not all(recorded.values()):
        raise ValueError("external reference has no approval-manifest artifact identity")


def gate_receipt(
    *,
    rtol: float | None,
    max_abs: float | None,
    breaches: Sequence[str],
    evaluated: bool,
) -> dict[str, Any]:
    """Describe whether a correctness gate ran, passed, or was not evaluated."""
    if evaluated and (rtol is None or max_abs is None):
        raise ValueError("an evaluated gate requires rtol and max_abs")
    for name, value in (("rtol", rtol), ("max_abs", max_abs)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"gate {name} must be finite and non-negative")
    return {
        "evaluated": evaluated,
        "rtol": rtol,
        "max_abs": max_abs,
        "passed": not breaches if evaluated else None,
        "breaches": list(breaches),
    }


def server_process_identity(
    *,
    runtime_root: Path,
    checkpoint: Path,
    device: str,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Capture runtime, checkpoint, and physical GPU identity in the live server."""
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    if not torch_module.cuda.is_available():
        raise RuntimeError("formal N2 server identity requires CUDA")
    parsed_device = torch_module.device(device)
    if parsed_device.type != "cuda":
        raise ValueError(f"formal N2 server identity requires a CUDA device, got {device!r}")
    logical_index = (
        torch_module.cuda.current_device() if parsed_device.index is None else parsed_device.index
    )
    properties = torch_module.cuda.get_device_properties(logical_index)
    gpu_uuid = str(getattr(properties, "uuid", ""))
    if not gpu_uuid:
        raise RuntimeError("CUDA device properties do not expose a GPU UUID")
    return {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": str(torch_module.__version__),
        "torch_cuda": str(torch_module.version.cuda),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_source": git_identity(Path(runtime_root)),
        "checkpoint": checkpoint_identity(Path(checkpoint)),
        "gpu": {
            "logical_device_index": int(logical_index),
            "torch_name": str(properties.name),
            "uuid": gpu_uuid,
            "capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        },
    }


def checkpoint_workload(checkpoint: Path, embodiment: str) -> dict[str, Any]:
    """Return the shape and architecture fields that define an N2 workload."""
    model_config = json.loads((checkpoint / "config.json").read_text())
    processor = json.loads((checkpoint / "processor_config.json").read_text())
    processor_kwargs = processor["processor_kwargs"]
    modality_configs = processor_kwargs["modality_configs"]
    keys = (embodiment, embodiment.lower())
    key = next((candidate for candidate in keys if candidate in modality_configs), None)
    if key is None:
        raise ValueError(f"checkpoint does not contain embodiment {embodiment!r}")
    modality = modality_configs[key]
    video_keys = tuple(modality["video"]["modality_keys"])
    history_deltas = tuple(value for value in modality["video"]["delta_indices"] if value <= 0)
    action_deltas = tuple(modality["action"]["delta_indices"])
    if len(video_keys) != len(set(video_keys)):
        raise ValueError(f"camera keys are not unique: {video_keys}")
    if model_config.get("max_n_cameras") != len(video_keys):
        raise ValueError(
            "model camera count does not match processor workload: "
            f"model={model_config.get('max_n_cameras')}, processor={len(video_keys)}"
        )
    if not history_deltas or history_deltas[-1] != 0:
        raise ValueError(f"conditioning history must end at delta 0: {history_deltas}")
    if action_deltas != tuple(range(len(action_deltas))):
        raise ValueError(f"action delta indices are not contiguous: {action_deltas}")

    token_merge_kernel = model_config.get("token_merge_kernel")
    source_vae_type = model_config.get("vae_type")
    return {
        "embodiment": key,
        "camera_count": len(video_keys),
        "camera_keys": list(video_keys),
        "history_frames_per_camera": len(history_deltas),
        "history_delta_indices": list(history_deltas),
        "image_target_size": list(processor_kwargs["image_target_size"]),
        "action_horizon": len(action_deltas),
        "model_type": model_config.get("model_type"),
        "model_dtype": model_config.get("model_dtype"),
        "model_size": model_config.get("model_size"),
        "source_vae_type_present": "vae_type" in model_config,
        "source_vae_type": source_vae_type,
        # GrootN2Config's compatibility default for checkpoints written before
        # vae_type became an explicit serialized field.
        "vae_type": source_vae_type or "wan21",
        "token_merge_enabled": token_merge_kernel is not None,
        "token_merge_kernel": token_merge_kernel,
        "token_merge_layers": model_config.get("token_merge_layers"),
        "tiled": model_config.get("tiled"),
    }


def git_identity(repo: Path) -> dict[str, str | bool]:
    """Return a source identity suitable for a benchmark receipt."""

    def run(*command: str) -> str:
        completed = subprocess.run(
            command,
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "root": str(repo.resolve()),
        "sha": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current") or "DETACHED",
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON receipt atomically."""
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered)
    os.replace(temporary, path)
