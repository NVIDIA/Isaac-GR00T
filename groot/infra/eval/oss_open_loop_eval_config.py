# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Configuration for OSS post-training open-loop evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from groot.core.utils.storage.msc import DEFAULT_HF_ARTIFACT_CACHE_ROOT

DEFAULT_CACHE_ROOT = DEFAULT_HF_ARTIFACT_CACHE_ROOT
DEFAULT_CHECKPOINT_NAME = "checkpoint-100"
DEFAULT_RESULT_PATH = "/tmp/groot_oss_open_loop_eval.json"  # noqa: S108
DEFAULT_SERVER_LOG_PATH = "/tmp/groot_oss_policy_server.log"  # noqa: S108


@dataclass
class OssOpenLoopEvalConfig:
    """Configuration for OSS post-training checkpoint evaluation."""

    checkpoint: str | None = None
    """Local checkpoint directory or hf://model/<repo>@<revision>/<checkpoint> URI."""

    dataset_path: str | None = None
    """Local LeRobot dataset directory. If set, no dataset artifact is downloaded."""

    dataset_repo_id: str | None = None
    """HF dataset repo override. Defaults to GOLDEN_RUN_DATASET_REPO_ID."""

    dataset_revision: str | None = None
    """HF dataset revision override. Defaults to GOLDEN_RUN_DATASET_REVISION or main."""

    dataset_local_name: str | None = None
    """Dataset directory name inside the downloaded dataset artifact."""

    checkpoint_name: str | None = None
    """Checkpoint subdirectory for GOLDEN_RUN_FINAL_MODEL_* defaults."""

    cache_root: str | None = None
    """HF artifact cache root. Defaults to GROOT_HF_ARTIFACT_CACHE_ROOT or /tmp/groot_hf_artifacts."""

    output_json: str = DEFAULT_RESULT_PATH
    """Where to write the JSON evaluation report."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """Trajectory IDs to evaluate."""

    steps: int = 200
    """Maximum steps per trajectory window."""

    action_horizon: int = 30
    """Action horizon used by the policy and open-loop evaluator."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend for LeRobotEpisodeLoader."""

    embodiment_tag: str = "xdof_relative_eef_relative_joint"
    """Embodiment tag name or value."""

    denoising_steps: int = 4
    """N2 inference denoising steps."""

    trt_engine_dir: str | None = None
    """TensorRT engine bundle directory. When unset, use eager inference."""

    use_cuda_graph: bool = False
    """Replay policy prediction through a CUDA graph."""

    server_parity_tolerance: float = 0.01
    """Maximum allowed absolute aggregate metric delta between local and server eval."""

    max_mse: float | None = None
    """Optional maximum average MSE threshold for each evaluated mode."""

    local_only: bool = False
    """Run only local policy evaluation and skip the policy server."""

    device: str | None = None
    """Policy device. Defaults to cuda when available, otherwise cpu."""

    server_host: str = "127.0.0.1"
    """Host used by the PolicyClient."""

    server_bind_host: str = "127.0.0.1"
    """Host interface used by the PolicyServer subprocess."""

    server_port: int = 5555
    """Policy server port."""

    server_timeout_ms: int = 300_000
    """PolicyClient request timeout in milliseconds."""

    server_start_timeout_seconds: float = 600.0
    """How long to wait for the policy server ping endpoint."""

    server_log_path: str = DEFAULT_SERVER_LOG_PATH
    """File that receives policy-server stdout/stderr."""

    save_plot_dir: str | None = None
    """Optional directory for open-loop trajectory plots."""
