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

import os
from pathlib import Path
from typing import Any

_SUPPORTED_BACKENDS = frozenset(("eager", "trt", "trt_graph"))


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
):
    """Construct one admitted backend without putting runtime logic in the harness.

    The shared harness owns workload generation, timing, gates, and receipts. This
    adapter owns only the runtime-specific mapping from its stable backend names to
    the production ``Gr00tN2Policy`` constructor.
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
    )


__all__ = ["build_benchmark_policy"]
