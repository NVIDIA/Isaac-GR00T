# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Runtime-owned construction seam for the shared N2 measurement harness."""

import json
import os
from pathlib import Path
from typing import Any

_SUPPORTED_BACKENDS = frozenset(("eager", "trt", "trt_graph"))
_REQUIRED_RECEIPT_ROLES = ("transformer_joint", "transformer_action_cache", "vae_encoder")


def _bare_engine_runtime_factory(engine_dir: Path, *, codec: str):
    """Serve the engines a build receipt describes without a manifest (never admitted).

    An admitted bundle is served as a bundle, so a directory that carries a manifest is
    refused here. The receipt's per-role engines, numeric formats and settings become the
    runtime constructor's arguments exactly as for a bundle; the factory receives the
    checkpoint contract and the served conditioning-cache selector at install time.
    """
    from groot.deployment.n2_trt.manifest import BUILD_FILENAME, MANIFEST_FILENAME
    from groot.deployment.n2_trt.runtime import ENGINE_ROLE_RUNTIME_KEYS

    if (engine_dir / MANIFEST_FILENAME).is_file():
        raise ValueError(
            f"{engine_dir} carries {MANIFEST_FILENAME}: an admitted bundle is served as a "
            "bundle, not as bare engines"
        )
    receipt_path = engine_dir / BUILD_FILENAME
    receipt = json.loads(receipt_path.read_text())
    settings = receipt.get("settings")
    artifacts = receipt.get("artifacts")
    if not isinstance(settings, dict) or not isinstance(artifacts, list):
        raise TypeError(f"malformed TensorRT build receipt: {receipt_path}")
    if settings.get("codec") != codec:
        raise ValueError(
            "N2 TensorRT engines were built for another serving codec: "
            f"engines={settings.get('codec')!r}, policy={codec!r}"
        )
    paths: dict[str, str] = {}
    formats: dict[str, str] = {}
    for artifact in artifacts:
        role = artifact["role"]
        paths[ENGINE_ROLE_RUNTIME_KEYS[role]] = str(
            (engine_dir / artifact["engine"]).resolve(strict=True)
        )
        formats[role] = str(artifact["numeric_format"])
    missing = [role for role in _REQUIRED_RECEIPT_ROLES if role not in formats]
    if missing:
        raise ValueError(f"TensorRT build receipt {receipt_path} omits engine roles {missing}")
    cond_kv_freeze = bool(settings.get("cond_kv_freeze", False))
    if cond_kv_freeze != ("joint_cond" in paths):
        raise ValueError(
            f"TensorRT build receipt {receipt_path} declares cond_kv_freeze={cond_kv_freeze} "
            "but its engine roles disagree"
        )
    action_context_mode = str(settings.get("action_context_mode", "raw"))

    def factory(contract, served_cond_kv_freeze):
        from groot.deployment.n2_trt.runtime import N2TensorRTRuntime

        resolved = (
            contract.with_transformer_precisions(
                formats["transformer_joint"], formats["transformer_action_cache"]
            )
            .with_vae_precision(formats["vae_encoder"])
            .with_cond_kv_freeze(cond_kv_freeze)
        )
        return N2TensorRTRuntime(
            dict(paths),
            contract=resolved,
            action_context_mode=action_context_mode,
            served_cond_kv_freeze=served_cond_kv_freeze,
        )

    return factory


def build_benchmark_policy(
    *,
    backend: str,
    checkpoint: str,
    embodiment_tag: Any,
    codec: str,
    device: str,
    engine_dir: str | None,
    num_video_steps: int,
    num_action_steps: int,
    bare_engines: bool = False,
):
    """Construct one admitted backend without putting runtime logic in the harness.

    The shared harness owns workload generation, timing, gates, and receipts. This
    adapter owns only the runtime-specific mapping from its stable backend names to
    the production ``Gr00tN2Policy`` constructor. ``bare_engines`` serves the engines
    described by the build receipt under ``engine_dir`` without a manifest (a search
    candidate that was never admitted); the default serves the admitted bundle.
    """
    if os.environ.get("ENABLE_TENSORRT", "false").strip().lower() == "true":
        raise RuntimeError(
            "ENABLE_TENSORRT=true selects an export-only SDPA attention path and cannot "
            "be used for formal eager/TensorRT runtime measurements"
        )
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported N2 benchmark backend {backend!r}; expected one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    if num_video_steps <= 0 or num_action_steps <= 0:
        raise ValueError("denoising step counts must be positive")

    uses_tensorrt = backend in ("trt", "trt_graph")
    if uses_tensorrt and engine_dir is None:
        raise ValueError(f"backend {backend!r} requires engine_dir")

    configured_video_steps = os.environ.get("GROOT_N2_NUM_VIDEO_STEPS")
    if configured_video_steps is not None and int(configured_video_steps) != num_video_steps:
        raise ValueError(
            "GROOT_N2_NUM_VIDEO_STEPS conflicts with the benchmark workload: "
            f"environment={configured_video_steps}, requested={num_video_steps}"
        )
    os.environ["GROOT_N2_NUM_VIDEO_STEPS"] = str(num_video_steps)

    policy_kwargs: dict[str, Any] = {}
    if uses_tensorrt and bare_engines:
        policy_kwargs["trt_runtime_factory"] = _bare_engine_runtime_factory(
            Path(engine_dir).expanduser(), codec=codec
        )

    from groot.core.policy.n2_policy_factory import create_gr00t_n2_policy

    return create_gr00t_n2_policy(
        model_path=checkpoint,
        embodiment_tag=embodiment_tag,
        num_inference_timesteps=num_action_steps,
        device=device,
        trt_engine_dir=(str(Path(engine_dir).expanduser()) if uses_tensorrt else None),
        use_cuda_graph=backend == "trt_graph",
        use_torch_compile=False,
        codec_roundtrip=codec,
        **policy_kwargs,
    )


__all__ = ["build_benchmark_policy"]
