# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""Minimal Groot policy inference server.

Loads a ``Gr00tPolicy`` (or ``Gr00tN2Policy``) from a model checkpoint and
serves it over ZMQ via ``PolicyServer``. The server expects clients to send
observations in the nested ``Gr00tPolicy`` format (see
``Gr00tPolicy.check_observation``); no wrapper layers are applied beyond the
flat-key adapter. The policy class is detected from ``config.json`` in the
checkpoint directory.

Example:
    uv run python -m groot.core.policy.run_gr00t_server \\
        --model-path /path/to/checkpoint-10000 \\
        --embodiment-tag GR1_FULL_BODY \\
        --port 5555

    uv run python -m groot.core.policy.run_gr00t_server \\
        --model-path /path/to/n2-checkpoint \\
        --embodiment-tag XDOF_RELATIVE_EEF_RELATIVE_JOINT
"""

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
import tyro

from groot.core.data import schema
from groot.core.policy import base
from groot.core.policy.cot import cot_wrapper, planner_client
from groot.core.policy.debug_logging_wrapper import DebugLoggingPolicy
from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper, Gr00tPolicy
from groot.core.policy.n2_policy_factory import (
    authenticate_server_task_config,
    create_gr00t_n2_policy,
)
from groot.core.policy.server_client import PolicyServer

DEFAULT_MODEL_SERVER_PORT = 5555
N2_MODEL_TYPES = {"GrootN2", "GrootN2RL", "GrootN2TTT"}
# "gr00t" covers the migrated N1 series (GrootN1d6 / GrootN1d7) and any other
# non-N2 core model — they load through the generic Gr00tPolicy branch below.
# "groot_n2" is GrootN2 and may install an admitted N2 TensorRT bundle through
# the normal policy seam. N1.5 (legacy VLA), gr00t_ttt and dream0 are not hosted
# by this server (use the omni server for those).
PolicyType = Literal["gr00t", "groot_n2"]

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def detect_policy_type_from_checkpoint(model_path: str) -> PolicyType:
    """Auto-detect the policy implementation from the checkpoint config."""
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Checkpoint is missing config.json; cannot detect policy type: {config_path}"
        )

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    if model_type in N2_MODEL_TYPES:
        return "groot_n2"
    return "gr00t"


@dataclass
class ServerConfig:
    """Configuration for running the Groot inference server."""

    embodiment_tag: schema.EmbodimentTag
    """Embodiment tag"""

    model_path: str
    """Path to the model checkpoint directory"""

    device: str = "cuda"
    """Device to run the model on"""

    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    num_inference_timesteps: int | None = None
    """Number of inference (denoising) timesteps; for Gr00tN2Policy, None uses
    GROOT_N2_NUM_ACTION_STEPS and then the default of 4."""

    # Flow-matching timestep shift (groot_n2 only) is read by Gr00tN2Policy from
    # the GROOT_N2_SHIFT env var, not a CLI flag — keeps the launch uniform for N1
    # (Gr00tPolicy has no shift). Set GROOT_N2_SHIFT=<float> in the environment.

    use_torch_compile: bool = False
    """torch.compile the model (groot_n2 only). YAM RTX 5090 stations should
    leave this False and set TORCHDYNAMO_DISABLE=1 (triton SMEM limit)."""

    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    """torch.compile mode (groot_n2 only)."""

    trt_engine_dir: str | None = None
    """Target-local N2 TensorRT engine bundle. None keeps eager execution."""

    use_cuda_graph: bool = False
    """Capture and replay the fixed-shape N2 predict path. Explicit opt-in."""

    save_video_dir: str | None = None
    """If set, Gr00tN2Policy buffers generated video latents and writes them here."""

    enable_n2_pred_comparison: bool = False
    """For groot_n2: run video prediction and return decoded frames for pred-vs-real
    comparison PNGs even when --save-video-dir is unset (no traj_*.mp4 on server)."""

    benchmark_timing: bool = False
    """For groot_n2 benchmarks, return synchronized policy and model-predict
    timing in response metadata. Disabled by default because synchronization
    changes production scheduling."""

    task_config: str | None = None
    """Optional strict rl100_n2.task_config.v2 YAML, resolved together with the
    RL100_N2_TASK_CONFIG environment override (a conflicting pair fails closed).
    Authenticated at the START of main(), before the checkpoint is inspected and
    before any dependency preparation; the authenticated binding object is then
    handed to the N2 policy factory. None with no env override is the exact
    legacy behavior."""

    debug_save_subdir: str | None = None
    """If set, dump every received observation to
    /home/gear/data/debug_umi/<debug_save_subdir>/step_NNNNNN/ as .npy + meta.json.
    Use a different subdir per client stack (e.g. 'umi' vs 'old_control') for diffing."""

    # --- Chain-of-thought (System-2) planning ---------------------------------
    # When enabled, the loaded policy is wrapped in a CoT policy wrapper before
    # being served, so a remote PolicyClient (umi/g1 env) transparently gets CoT
    # subtask translation. The wrapper queries a remote CoT planner server and
    # rewrites the observation's instruction before System-1 inference.
    cot_mode: Literal["off", "start", "sync", "async"] = "off"
    """CoT wrapper to apply server-side: 'off' (none), 'start' (re-infer on
    instruction change), 'sync' (time-based), 'async' (background, freshness-bounded)."""

    cot_planner_url: str = "http://localhost:8100"
    """Base URL of the remote CoT planner server."""

    cot_max_drift: float = 2.0
    """Max seconds between CoT inferences (cot_mode 'sync'/'async' only)."""

    cot_memory_type: str | None = None
    """CoT server-side memory type (None, 'image', 'text', 'multiturn', 'hybrid_multiturn')."""

    cot_task_type: str | None = None
    """CoT task type for subtask candidates (e.g. 'i3', 'i4')."""

    cot_instruction_key: str = "annotation.task"
    """Observation key the CoT wrapper reads as the goal and overwrites with the
    planner output. Must be the language key the client populates."""

    cot_use_all_cameras: bool = False
    """Send every video camera to the planner (primary-first). Default sends
    only the first camera (legacy single-camera wire path)."""

    cot_max_tokens: int = 4096
    """Max tokens for CoT planner generation."""

    cot_temperature: float = 0.0
    """Sampling temperature for CoT planner generation."""

    cot_top_p: float = 1.0
    """Top-p for CoT planner generation."""

    cot_fallback_response: str = "do nothing"
    """CoT response used when the planner output can't be parsed."""


def _make_gr00t_flat_frame_extractor(
    policy: base.BasePolicy, use_all_cameras: bool
) -> Callable[[dict[str, Any]], dict[str, list[Image.Image]]]:
    """Build a CoT frame extractor for flat (``video.<cam>``) observations.

    The CoT wrapper sits outside ``Gr00tFlatPolicyWrapper``, so it receives the
    client's flat observation. Camera keys are derived from the policy's video
    modality config (primary = first). Frame values may be raw ``(T, H, W, C)``
    uint8 arrays or JPEG-byte lists (real-robot clients), so reuse the flat
    wrapper's decoder and take the most recent frame.
    """
    video_keys = list(policy.get_modality_config()["video"].modality_keys)
    flat_keys = [f"video.{k}" for k in video_keys]
    selected = flat_keys if use_all_cameras else flat_keys[:1]

    def extractor(observation: dict[str, Any]) -> dict[str, list[Image.Image]]:
        frames: dict[str, list[Image.Image]] = {}
        for flat in selected:
            if flat not in observation:
                continue
            arr = np.asarray(Gr00tFlatPolicyWrapper._decode_serialized_video(observation[flat]))
            frame = arr[-1] if arr.ndim == 4 else arr  # (T, H, W, C) -> latest frame
            frames[flat] = [Image.fromarray(frame.astype(np.uint8))]
        if not frames:
            raise KeyError(
                f"CoT frame extractor found none of the video keys {selected} in the observation"
            )
        return frames

    return extractor


def _wrap_with_cot(policy: base.BasePolicy, config: ServerConfig) -> base.BasePolicy:
    """Wrap ``policy`` in the configured CoT policy wrapper (cot_mode != 'off')."""
    planner_client.check_cot_endpoint(config.cot_planner_url)
    frame_extractor = _make_gr00t_flat_frame_extractor(policy, config.cot_use_all_cameras)
    common = dict(
        max_tokens=config.cot_max_tokens,
        temperature=config.cot_temperature,
        top_p=config.cot_top_p,
        fallback_response=config.cot_fallback_response,
        frame_extractor=frame_extractor,
        instruction_key=config.cot_instruction_key,
        planner_base_url=config.cot_planner_url,
        memory_type=config.cot_memory_type,
        task_type=config.cot_task_type,
        strict=config.strict,
    )
    if config.cot_mode == "start":
        return cot_wrapper.CoTStartPolicyWrapper(policy, **common)
    if config.cot_mode == "sync":
        return cot_wrapper.CoTSyncPolicyWrapper(policy, max_drift=config.cot_max_drift, **common)
    if config.cot_mode == "async":
        return cot_wrapper.CoTAsyncPolicyWrapper(policy, max_drift=config.cot_max_drift, **common)
    raise ValueError(f"Unknown cot_mode: {config.cot_mode!r}")


def _validate_server_config(config: ServerConfig) -> None:
    """Reject server combinations outside the qualified graph contract."""
    if os.environ.get("ENABLE_TENSORRT", "false").strip().lower() == "true":
        raise ValueError(
            "ENABLE_TENSORRT=true selects an export-only SDPA attention path and cannot "
            "be used by the production policy server"
        )
    if config.use_cuda_graph and config.cot_mode != "off":
        raise ValueError(
            "--use-cuda-graph has not been qualified with CoT and cannot be "
            f"combined with --cot-mode {config.cot_mode!r}; disable CoT or CUDA graph"
        )
    if config.trt_engine_dir is not None and config.use_torch_compile:
        raise ValueError("--trt-engine-dir cannot be combined with --use-torch-compile")


def main(config: ServerConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    _validate_server_config(config)

    # Authenticate the strict task config FIRST — before the checkpoint is
    # inspected, before the N2 policy implementation is imported, and before
    # ensure_cosmos_model_deps() can download or mutate a dependency cache.
    strict_binding = authenticate_server_task_config(config.task_config, logger=logger)

    policy_type = detect_policy_type_from_checkpoint(config.model_path)
    if policy_type != "groot_n2" and strict_binding is not None:
        raise ValueError(
            "a strict task config is a Gr00tN2Policy contract; the generic "
            f"{policy_type!r} policy cannot enforce it — refusing to serve"
        )

    logger.info("Starting Groot inference server...")
    logger.info(f"  Policy type:              {policy_type}")
    logger.info(f"  Embodiment tag:           {config.embodiment_tag}")
    logger.info(f"  Model path:               {config.model_path}")
    logger.info(f"  Device:                   {config.device}")
    logger.info(f"  Host:                     {config.host}")
    logger.info(f"  Port:                     {config.port}")
    logger.info(f"  Num inference timesteps:  {config.num_inference_timesteps}")
    logger.info(f"  use_torch_compile:        {config.use_torch_compile}")
    logger.info(f"  TensorRT engine dir:      {config.trt_engine_dir}")
    logger.info(f"  use_cuda_graph:           {config.use_cuda_graph}")
    logger.info(f"  enable_n2_pred_comparison: {config.enable_n2_pred_comparison}")

    if policy_type == "groot_n2":
        # The factory owns what used to be inline here: one final live-byte
        # check of the authenticated binding, the Cosmos DiT / text encoder /
        # Wan VAE download + path rewrite, and the choice of policy
        # implementation. It still imports zero groot.vla.omni, and it treats
        # groot.core.rl as optional.
        policy = create_gr00t_n2_policy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
            num_inference_timesteps=config.num_inference_timesteps,
            use_torch_compile=config.use_torch_compile,
            torch_compile_mode=config.torch_compile_mode,
            trt_engine_dir=config.trt_engine_dir,
            use_cuda_graph=config.use_cuda_graph,
            save_video_dir=config.save_video_dir,
            strict_task_binding=strict_binding,
            enable_n2_pred_comparison=config.enable_n2_pred_comparison,
        )
    else:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
            num_inference_timesteps=config.num_inference_timesteps,
        )

    if config.benchmark_timing:
        if policy_type != "groot_n2":
            raise ValueError("--benchmark-timing is supported only for groot_n2")
        import groot
        from groot.core.policy.benchmark_contract import server_process_identity
        from groot.core.policy.benchmark_timing import instrument_policy_timing

        server_process = server_process_identity(
            runtime_root=Path(groot.__file__).resolve().parents[1],
            checkpoint=Path(config.model_path),
            device=config.device,
        )
        instrument_policy_timing(policy, server_process=server_process)
        logger.info("  Benchmark timing:         synchronized policy and model spans")
        logger.info("  Benchmark server identity: live runtime/checkpoint/GPU authenticated")

    policy = Gr00tFlatPolicyWrapper(policy, strict=config.strict)

    # CoT (System-2) wraps the flat adapter so it sees the client's flat
    # observation, rewrites the instruction, then delegates to System-1.
    if config.cot_mode != "off":
        logger.info(
            f"  CoT mode:                 {config.cot_mode} (planner {config.cot_planner_url})"
        )
        policy = _wrap_with_cot(policy, config)

    if config.debug_save_subdir:
        save_dir = f"{config.debug_save_subdir}"
        policy = DebugLoggingPolicy(policy, save_dir)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        logger.info(
            "Server is listening on %s:%s. CUDA-graph serving requires a real-observation "
            "warmup before the robot is armed. Press Ctrl+C to stop.",
            config.host,
            config.port,
        )
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        try:
            policy.reset()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Policy reset during shutdown failed: %s", exc)


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
