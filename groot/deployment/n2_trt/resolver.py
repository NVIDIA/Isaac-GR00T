# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed automatic TensorRT resolution shared by inference entrypoints."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_n2_trt_schedule(denoising_steps: int | None) -> tuple[int, int]:
    """Resolve and validate the cached schedule before build or worker setup."""
    action_steps = (
        int(os.environ.get("GROOT_N2_NUM_ACTION_STEPS", "4"))
        if denoising_steps is None
        else denoising_steps
    )
    video_steps = int(os.environ.get("GROOT_N2_NUM_VIDEO_STEPS", "2"))
    if (
        video_steps <= 0
        or not isinstance(action_steps, int)
        or isinstance(action_steps, bool)
        or action_steps <= 0
        or action_steps % video_steps != 0
    ):
        raise ValueError(
            f"N2 TRT schedule {video_steps}v{action_steps}a requires positive integer "
            "video/action steps with action steps an exact multiple of video steps"
        )
    return video_steps, action_steps


def resolve_n2_trt_device(device: str | None) -> str:
    """Return the explicit indexed CUDA device shared by build and inference."""
    import torch

    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        raise ValueError(f"automatic N2 TensorRT bundles require a CUDA device, got {device!r}")
    index = torch.cuda.current_device() if target.index is None else target.index
    return f"cuda:{index}"


def resolve_n2_trt_bundle(
    *,
    checkpoint: str | Path,
    embodiment: str,
    denoising_steps: int | None,
    cache_root: str | Path | None,
    rebuild: bool,
    device: str | None,
    codec: str,
    rebuild_token: str | None = None,
) -> Path:
    """Resolve a bundle exported for the requested inference schedule."""
    video_steps, action_steps = resolve_n2_trt_schedule(denoising_steps)

    from groot.deployment.n2_trt.auto_builder import resolve_or_build_engine_bundle

    bundle = resolve_or_build_engine_bundle(
        checkpoint=checkpoint,
        embodiment=embodiment,
        cache_root=cache_root,
        rebuild=rebuild,
        device=resolve_n2_trt_device(device),
        rebuild_token=rebuild_token,
        codec=codec,
        num_video_steps=video_steps,
        num_action_steps=action_steps,
    )
    from groot.deployment.n2_trt.policy_measurement import report_policy_measurement

    report_policy_measurement(bundle)
    return bundle
