# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# These are pre-existing exceptions in this legacy standalone evaluation CLI.
# ruff: noqa: C408, D417, PGH003, S108, SLF001
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
import time
from typing import Any, Literal
import warnings

from matplotlib import pyplot as plt
import numpy as np
import polars as pl
import tyro

from groot.core.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from groot.core.data.dataset.sharded_single_step_dataset import extract_step_data
from groot.core.data.schema import EmbodimentTag
from groot.core.policy.base import BasePolicy
from groot.core.policy.gr00t_policy import Gr00tPolicy
from groot.core.policy.server_client import PolicyClient

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def prepare_save_path(path: str, traj_id: int, extension: str) -> str:
    """Prepare the save path for a given trajectory.

    If the path has the specified extension, it is treated as a file path.
    Otherwise, it is treated as a directory, and a filename is generated using the trajectory ID.
    Ensures that the parent directory (or the directory itself) exists.
    """
    path_obj = Path(path)
    if path.endswith(f".{extension}"):
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return str(path_obj)
    else:
        path_obj.mkdir(parents=True, exist_ok=True)
        return str(path_obj / f"traj_{traj_id}.{extension}")


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID if input is a directory
    if isinstance(save_plot_path, str):
        save_plot_path = prepare_save_path(save_plot_path, traj_id, "png")
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    save_plot: bool = True,
    control_freq: int = 20,
    use_rtc_wrapper: bool = False,
    save_video_path: str | None = None,
    start_step: int = 0,
    enable_frechet: bool = True,
    enable_reverse_kl: bool = False,
    rkl_num_samples: int = 4,
    pooled_accumulator: list[tuple[np.ndarray, np.ndarray]] | None = None,
    compute_video_metrics: bool = False,
    video_lpips_net: Any = None,
    video_full_deltas: Any = None,
    projection_fn: Any = None,
    log_video: bool = False,
    video_embedder: Any = None,
    video_embed_out: dict | None = None,
) -> dict[str, float]:
    """Evaluate one window of a trajectory and return a dict of metrics.

    The window is ``[start_step, start_step + steps)``, capped by the trajectory
    length. ``start_step=0`` reproduces the legacy "roll from the prefix" behaviour;
    a non-zero ``start_step`` lets the in-process callback sample windows IID
    across the dataset distribution rather than always hitting the prefix.

    Returns a ``dict[str, float]`` with at least ``"mse"`` and ``"mae"``, plus
    optionally ``"frechet"`` (when ``enable_frechet=True``) and ``"reverse_kl"``
    (when ``enable_reverse_kl=True`` and the policy is stochastic enough for
    ``rkl_num_samples >= 2`` independent forward passes to differ — for
    diffusion / flow-matching action heads this is automatic).

    Args:
        enable_frechet: If True, compute the per-window 2-Wasserstein
            (Bures / Fréchet) distance between the empirical Gaussians of
            the predicted and ground-truth action sequences. Effectively
            free since it post-processes the existing rollout.
        enable_reverse_kl: If True, run ``rkl_num_samples`` independent
            policy forward passes per chunk (instead of 1) and report the
            per-window reverse-KL of the resulting empirical sample
            distribution against ``N(gt, I)``. Multiplies eval cost by
            roughly ``rkl_num_samples``.
        rkl_num_samples: Number of stochastic policy samples per chunk
            when ``enable_reverse_kl`` is on. Must be >= 2 for the
            reverse-KL estimator to be defined.
        pooled_accumulator: If provided, this window's
            ``(pred_action_across_time, gt_action_across_time)`` arrays are
            appended so the caller can pool them across the whole eval set
            (e.g. the dataset-level per-dim Wasserstein). No effect on the
            returned per-window dict.
    """
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    if start_step < 0:
        raise ValueError(f"start_step must be >= 0, got {start_step}")
    if start_step >= traj_length:
        raise ValueError(
            f"start_step={start_step} is beyond trajectory length {traj_length} for traj_id={traj_id}"
        )
    actual_steps = min(steps, traj_length - start_step)
    logging.info(
        f"Using {actual_steps} steps starting at {start_step} "
        f"(requested: {steps}, trajectory length: {traj_length})"
    )

    # When RKL is on we need >=2 independent samples per chunk; otherwise
    # one forward pass per chunk (legacy behaviour) is enough for
    # MSE/MAE/Fréchet.
    if enable_reverse_kl and rkl_num_samples < 2:
        raise ValueError(f"reverse_kl requires rkl_num_samples >= 2, got {rkl_num_samples}")
    num_samples = rkl_num_samples if enable_reverse_kl else 1

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    # pred_samples_across_time[m] is the per-sample list of per-step actions.
    # When num_samples == 1 (legacy path) this is just one list.
    pred_samples_across_time: list[list[np.ndarray]] = [[] for _ in range(num_samples)]

    policy.reset()  # Reset policy state (including any state or video buffers)
    first_chunk_action = None  # predicted action of the first window (for projection)
    first_chunk_gt_action = None  # GT action of the first window (for GT-column projection)
    for step_count in range(start_step, start_step + actual_steps, action_horizon):
        data_point = extract_step_data(
            traj, step_count, modality_configs, embodiment_tag, allow_padding=True
        )
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)

        # ``num_samples`` independent forward passes per chunk. For
        # diffusion / flow-matching action heads each call samples a
        # fresh noise vector, so the M outputs are independent draws
        # from the policy at this observation.
        for m in range(num_samples):
            _action_chunk, _ = policy.get_action(parsed_obs)
            action_chunk = parse_action_gr00t(_action_chunk)
            if first_chunk_action is None:
                first_chunk_action = action_chunk
                # GT action for the same window: slice each action key from the traj.
                # Same key format as first_chunk_action (action.<key> -> (horizon, dim)).
                if log_video:
                    first_chunk_gt_action = {
                        f"action.{k}": np.vstack(list(traj[f"action.{k}"]))[
                            start_step : start_step + action_horizon
                        ]
                        for k in action_keys
                        if f"action.{k}" in traj.columns
                    }
            for j in range(action_horizon):
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [
                        np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                        for key in action_keys
                    ],
                    axis=0,
                )
                pred_samples_across_time[m].append(concat_pred_action)
                if use_rtc_wrapper:
                    time.sleep(1 / control_freq)

    def extract_state_joints(traj: pl.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack(list(traj[column]))
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        start_step : start_step + actual_steps
    ]

    # Stack samples to (M, actual_steps, D). Use sample 0 as the canonical
    # prediction for MSE / MAE / Fréchet — this matches the legacy
    # ``num_samples == 1`` path exactly.
    pred_samples_array = np.array(pred_samples_across_time)[:, :actual_steps]
    pred_action_across_time = pred_samples_array[0]
    assert (
        gt_action_across_time.shape == pred_action_across_time.shape
    ), f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"

    # Stash this window's (pred, gt) so the caller can pool them across the eval
    # set for dataset-level metrics (the pooled per-dim Wasserstein).
    if pooled_accumulator is not None:
        pooled_accumulator.append((pred_action_across_time, gt_action_across_time))

    # calc MSE and MAE across time
    metrics: dict[str, float] = {}
    metrics["mse"] = float(np.mean((gt_action_across_time - pred_action_across_time) ** 2))
    metrics["mae"] = float(np.mean(np.abs(gt_action_across_time - pred_action_across_time)))
    logging.info(f"Unnormalized Action MSE across single traj: {metrics['mse']}")
    logging.info(f"Unnormalized Action MAE across single traj: {metrics['mae']}")

    if enable_frechet:
        try:
            # Lazy import: torch is heavy and we don't want to pay the
            # import cost on metric-disabled CLI runs.
            import torch

            from groot.core.training.open_loop.metrics import (
                frechet_gaussian_over_time,
            )

            pred_t = torch.from_numpy(pred_action_across_time).unsqueeze(0)  # (1, T, D)
            gt_t = torch.from_numpy(gt_action_across_time).unsqueeze(0)
            fd = frechet_gaussian_over_time(pred_t, gt_t)
            metrics["frechet"] = float(fd.item())
            logging.info(f"Frechet Gaussian over time: {metrics['frechet']}")
        except Exception:
            logging.exception("frechet_gaussian_over_time failed; skipping metric")

    if enable_reverse_kl:
        try:
            import torch

            from groot.core.training.open_loop.metrics import (
                reverse_kl_from_samples,
            )

            # pred_samples_array: (M, T, D) → (M, 1, T, D); gt: (1, T, D).
            samples_t = torch.from_numpy(pred_samples_array).unsqueeze(1)
            gt_t = torch.from_numpy(gt_action_across_time).unsqueeze(0)
            rkl = reverse_kl_from_samples(samples_t, gt_t)
            metrics["reverse_kl"] = float(rkl.item())
            logging.info(f"Reverse KL from {num_samples} samples: {metrics['reverse_kl']}")
        except Exception:
            logging.exception("reverse_kl_from_samples failed; skipping metric")

    # ── Video reconstruction metrics (world-model policies that predict video) ──
    # Scored over the WHOLE rollout path, in the model's OWN frame: the policy
    # buffers a predicted video chunk at every action hop, so we stitch all chunks
    # (mirroring how the action metric re-queries each hop) and aggregate
    # psnr/ssim/lpips over them — not just the first window. When the model trained
    # on a crop (crop_fraction < 1, e.g. 0.95) the GT is cropped the same way, else
    # it is naively resized. See groot/core/training/open_loop/video_metrics.py.
    if (
        compute_video_metrics
        and getattr(policy, "model", None) is not None
        and getattr(policy, "video_latents_buffer", None) is not None
    ):
        try:
            from groot.core.training.open_loop.video_metrics import (
                center_crop_resize,
                decode_video_latents,
                naive_resize,
                video_frame_metrics,
                whole_path_video_chunks,
            )

            # Decode the policy's buffered video latents via the model's EXISTING
            # decode_latents (no get_predicted_frames method needed on Gr00tN2Policy).
            view_frames = decode_video_latents(policy)
            video_cfg = loader.modality_configs.get("video")
            # Prefer the full (untruncated) deltas read from processor_config; the loader's
            # are truncated to cond-only by the policy, so they carry no predicted frames.
            raw_deltas = video_full_deltas or (
                list(video_cfg.delta_indices) if video_cfg is not None else []
            )
            if view_frames and video_cfg is not None and raw_deltas:
                deltas = sorted(int(d) for d in raw_deltas)
                n_total = len(deltas)
                pred_deltas = [d for d in deltas if d > 0]
                n_cond = n_total - len(pred_deltas)
                ego_key = video_cfg.modality_keys[0]
                ego_full = view_frames[0]  # (n_total*chunks, H, W, C)
                pred_frames = ego_full[n_cond:n_total]  # predicted (>0) frames, window 0
                gt_positions = np.array(
                    [min(max(start_step + d, 0), traj_length - 1) for d in pred_deltas]
                )
                gt_dict = loader._load_video_data(traj_id, gt_positions)
                gt_raw = gt_dict.get(ego_key)
                if gt_raw is None:
                    gt_raw = next(iter(gt_dict.values()))
                proc = getattr(policy, "processor", None)
                crop_fraction = float(
                    getattr(proc, "image_crop_fraction", None)
                    or getattr(proc, "crop_fraction", None)
                    or 0.95
                )
                # Truly-correct "model frame" GT: run the GT frame through the processor's
                # OWN eval image transform (the exact crop+resize the model saw for its
                # input), not a generic center-crop. Falls back to the generic crop inside
                # video_frame_metrics if the processor exposes no such transform.
                crop_transform = None
                _eit = getattr(proc, "eval_image_transform", None)
                if callable(_eit):

                    def crop_transform(f, _t=_eit):
                        return np.asarray(_t(image=f)["image"])

                # Whole-path: ``ego_full`` already holds every buffered chunk, so
                # score each chunk against its GT-anchored window and aggregate —
                # the video analogue of the action metric's per-hop re-query.
                chunks, n_clamped = whole_path_video_chunks(
                    len(ego_full),
                    n_cond=n_cond,
                    n_total=n_total,
                    pred_deltas=pred_deltas,
                    start_step=start_step,
                    action_horizon=action_horizon,
                    traj_length=traj_length,
                )
                per_chunk_vms: list[dict[str, float]] = []
                embed_pred_pf: list[np.ndarray] = []
                embed_gt_pf: list[np.ndarray] = []
                for pred_lo, pred_hi, gt_pos in chunks:
                    c_pred = np.asarray(ego_full[pred_lo:pred_hi])
                    c_gt_dict = loader._load_video_data(traj_id, np.array(gt_pos))
                    c_gt = c_gt_dict.get(ego_key)
                    if c_gt is None:
                        c_gt = next(iter(c_gt_dict.values()))
                    c_gt = np.asarray(c_gt)
                    per_chunk_vms.append(
                        video_frame_metrics(
                            c_pred,
                            c_gt,
                            crop_fraction,
                            video_lpips_net,
                            crop_transform=crop_transform,
                        )
                    )
                    # Distributional: per-frame embeddings (unrolled, matching the
                    # action metric), GT pushed into the model's own frame EXACTLY as
                    # video_frame_metrics does (processor transform → crop → resize) so
                    # the pred/GT clouds share the same FOV (crop-GT, not raw).
                    if video_embedder is not None and video_embed_out is not None:
                        hp, wp = c_pred.shape[1:3]
                        if crop_transform is not None:
                            c_gt_model = np.stack([np.asarray(crop_transform(f)) for f in c_gt])
                        elif crop_fraction is not None and float(crop_fraction) < 1.0:
                            c_gt_model = np.stack(
                                [center_crop_resize(f, float(crop_fraction), hp, wp) for f in c_gt]
                            )
                        else:
                            c_gt_model = np.stack([naive_resize(f, hp, wp) for f in c_gt])
                        embed_pred_pf.append(video_embedder.embed(c_pred, pool=False))
                        embed_gt_pf.append(video_embedder.embed(c_gt_model, pool=False))
                # Aggregate over chunks (mean) — replaces the window-0 scalars.
                vm = {k: float(np.mean([cv[k] for cv in per_chunk_vms])) for k in per_chunk_vms[0]}
                for k, v in vm.items():
                    metrics[f"video_{k}"] = v
                metrics["video_n_chunks"] = float(len(chunks))
                metrics["video_n_clamped"] = float(n_clamped)
                logging.info(
                    "Whole-path video metrics (%d chunks @stride %d from step %d, "
                    "%d GT clamped, crop=%.2f): %s",
                    len(chunks),
                    action_horizon,
                    start_step,
                    n_clamped,
                    crop_fraction,
                    ", ".join(f"{k}={v:.3f}" for k, v in vm.items()),
                )
                if video_embedder is not None and video_embed_out is not None and embed_pred_pf:
                    # Per-frame cloud across ALL chunks (unrolled — matches theirs).
                    video_embed_out["pred"] = np.concatenate(embed_pred_pf, axis=0)
                    video_embed_out["gt"] = np.concatenate(embed_gt_pf, axis=0)
                # Optional projection of the predicted action onto the predicted frames
                # (embodiment-specific; supplied by the caller). Reports a wrist
                # reprojection-error metric and saves overlays.
                if projection_fn is not None:
                    try:
                        proj_metrics = projection_fn(
                            loader=loader,
                            traj_id=traj_id,
                            traj=traj,
                            start_step=start_step,
                            pred_frames=pred_frames,
                            pred_deltas=pred_deltas,
                            pred_action=first_chunk_action,
                            save_video_path=save_video_path,
                            gt_frames=np.asarray(gt_raw) if log_video else None,
                            gt_action=first_chunk_gt_action if log_video else None,
                            return_frames=log_video,
                        )
                        for k, v in (proj_metrics or {}).items():
                            if k.startswith("_"):  # frame arrays — store, don't log as float
                                metrics[k] = v
                            else:
                                metrics[k] = v
                    except Exception:
                        logging.exception("projection_fn failed; skipping projection overlays")
                # When no projection fn but video logging requested: store raw frames.
                if log_video and "_pred_frames" not in metrics:
                    metrics["_pred_frames"] = np.asarray(pred_frames)
                    metrics["_gt_frames"] = np.asarray(gt_raw)
        except Exception:
            logging.exception("video metrics failed; skipping (action metrics unaffected)")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results (skippable — the eval-set runner scores thousands of
    # segments and doesn't want a matplotlib figure per segment).
    if save_plot:
        plot_trajectory_results(
            state_joints_across_time=state_joints_across_time,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=action_keys,
            action_horizon=action_horizon,
            save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
        )

    if save_video_path is not None:
        if hasattr(policy, "save_video"):
            save_video_path = prepare_save_path(save_video_path, traj_id, "mp4")
            policy.save_video(save_video_path, traj_id)  # type: ignore
        else:
            logging.warning("Policy does not support save_video method. Skipping saving of video.")

    return metrics


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    save_video_path: str | None = None
    """Path to save the video to."""

    control_freq: int = 20
    """Control frequency to use."""

    use_rtc_wrapper: bool = False
    """Whether to use the RTC wrapper."""

    policy_type: Literal["gr00t", "gr00TTT", "dream0", "gr00t_n2"] = "gr00t"
    """Policy type to use."""

    use_torch_compile: bool = False
    """Whether to use torch.compile."""

    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    """Torch compile mode to use."""

    enable_frechet: bool = True
    """Compute the per-window Fréchet (Bures / 2-Wasserstein) distance between
    the empirical Gaussians of predicted vs ground-truth action sequences.
    Effectively free since it post-processes the existing rollout."""

    enable_reverse_kl: bool = False
    """Compute the per-window reverse-KL of the empirical sample distribution
    against ``N(gt, I)``. Requires ``rkl_num_samples`` independent forward
    passes per chunk, multiplying eval cost by roughly that factor."""

    rkl_num_samples: int = 4
    """Number of stochastic policy samples per chunk when ``enable_reverse_kl``
    is on. Must be >= 2 for the estimator to be defined."""

    enable_wasserstein: bool = True
    """Compute the dataset-level pooled per-dimension Wasserstein distance
    between predicted and ground-truth action values, aggregated over every
    evaluated step across all trajectories (W1 = 1-Wasserstein, MAE-flavored;
    W2 = 2-Wasserstein, rooted/RMSE-scaled). Effectively free — post-processes the rollouts."""

    video_guidance_scale: float = 1.0
    """Video CFG scale (gr00t_n2 only). >1.0 enables classifier-free guidance on video."""

    action_guidance_scale: float = 1.0
    """Action CFG scale (gr00t_n2 only). >1.0 enables classifier-free guidance on action."""

    compute_video_metrics: bool = False
    """Compute world-model video reconstruction PSNR/SSIM/LPIPS vs GT, BOTH full-frame
    (deflated, historical) and crop_fraction (correct, model frame). Requires
    policy_type=gr00t_n2 + a set save_video_path (so the policy predicts video)."""

    start_step: int = 0
    """Trajectory frame to start the rollout / first video-prediction window at. For
    video metrics this must be >= -min(video delta) so the conditioning frames are
    in-bounds (e.g. 16 for the midtrain ckpt whose min video delta is -15)."""

    projection: Literal["none", "batch7"] = "none"
    """Project the predicted action onto the predicted video frames (needs a per-frame
    camera pose). 'batch7' uses the Sharpa-HA4 FK + observation.head_eef adapter; it
    reports ``wrist_reproj_px`` (pred-action vs GT-action wrist, px) and writes overlay
    mp4s under <save_video_path>/projection. Requires --compute-video-metrics."""

    projection_module_path: str | None = None
    """Optional directory containing the ``ol_projection`` module used by the Batch7
    projection. The module must otherwise be installed or available on ``PYTHONPATH``."""


def compute_pooled_wasserstein(
    pooled: list[tuple[np.ndarray, np.ndarray]],
    max_samples: int | None = None,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Dataset-level pooled Wasserstein over all evaluated steps — every variant.

    ``pooled`` is the list of per-window ``(pred, gt)`` arrays (each ``(T, D)``)
    collected during the eval loop. Concatenates them into ``(N, D)`` pred/gt
    pools and returns the full report — per-dim marginal (``wasserstein_{1,2}_perdim``)
    and multivariate exact-assignment (``wasserstein_{1,2}_joint``) W1/W2.

    ``max_samples`` caps **only** the joint exact-assignment OT, which is
    ``O(N²)`` memory + ``~O(N³)`` time (``linear_sum_assignment``); the per-dim
    marginals always use the full pool. Leave ``None`` for the exact joint value
    at modest ``N`` (a handful of episodes); set it (e.g. 4096) when pooling many
    episodes so the assignment stays tractable — otherwise a full-dataset eval
    would OOM/hang at this step. The minibatch-OT estimate it yields is the same
    one the in-process training callback uses by default.
    """
    from groot.core.training.open_loop.metrics import (
        pooled_wasserstein_report,
    )

    pred_pool = np.concatenate([p for p, _ in pooled], axis=0)
    gt_pool = np.concatenate([g for _, g in pooled], axis=0)
    return pooled_wasserstein_report(pred_pool, gt_pool, max_samples=max_samples, n_jobs=n_jobs)


def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        if args.policy_type == "gr00t":
            policy_cls = Gr00tPolicy
        elif args.policy_type == "gr00TTT":
            # Gr00TTTPolicy has not been migrated to groot.core yet — lazy import from vla.omni.
            from groot.vla.omni.policy import Gr00TTTPolicy

            policy_cls = Gr00TTTPolicy
        elif args.policy_type == "dream0":
            from groot.core.policy.dream0_policy import DreamZeroPolicy

            policy_cls = DreamZeroPolicy
        elif args.policy_type == "gr00t_n2":
            from groot.core.policy.n2_policy import Gr00tN2Policy

            policy_cls = Gr00tN2Policy
        else:
            raise ValueError(f"Unknown policy type: {args.policy_type}")

        policy_kwargs = dict(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_inference_timesteps=args.denoising_steps,
            use_torch_compile=args.use_torch_compile,
            torch_compile_mode=args.torch_compile_mode,
        )
        # Gr00tN2Policy enables video prediction only when save_video_dir is set.
        if args.policy_type == "gr00t_n2":
            if args.save_video_path is not None:
                policy_kwargs["save_video_dir"] = args.save_video_path
            policy_kwargs["video_guidance_scale"] = args.video_guidance_scale
            policy_kwargs["action_guidance_scale"] = args.action_guidance_scale
        policy = policy_cls(**policy_kwargs)
    else:
        policy = PolicyClient(address=f"tcp://{args.host}:{args.port}")

    # Apply RTC wrapper if needed
    if args.use_rtc_wrapper:
        # Gr00tRTCPolicyWrapper has not been migrated to groot.core yet.
        from groot.vla.omni.policy.rtc_wrapper import Gr00tRTCPolicyWrapper

        policy = Gr00tRTCPolicyWrapper(
            policy,
            args.control_freq,
            action_horizon=args.action_horizon,  # NOTE: this should be the max action horizon of the model
            denoising_steps=args.denoising_steps,
        )

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    per_metric: dict[str, list[float]] = {}
    # Per-window (pred, gt) arrays, concatenated after the loop for the
    # dataset-level pooled Wasserstein.
    pooled: list[tuple[np.ndarray, np.ndarray]] | None = [] if args.enable_wasserstein else None

    # Build LPIPS once (shared across trajectories) when video metrics are requested.
    video_lpips_net = None
    if args.compute_video_metrics:
        try:
            import lpips as _lpips
            import torch as _torch

            _dev = "cuda" if _torch.cuda.is_available() else "cpu"
            video_lpips_net = _lpips.LPIPS(net="alex").to(_dev).eval()
        except Exception:
            logging.exception("LPIPS unavailable; video LPIPS will be skipped")

    # The policy truncates the video modality to the cond (<=0) deltas for the
    # OBSERVATION; the model still PREDICTS the positive-delta frames. Read the full,
    # untruncated video delta_indices from the ckpt's processor_config so the video
    # metric can map predicted frames → the GT frames they should match.
    video_full_deltas = None
    if args.compute_video_metrics and local_model_path:
        try:
            import json as _json

            _pc = _json.loads((Path(local_model_path) / "processor_config.json").read_text())
            _mc = _pc["processor_kwargs"]["modality_configs"]
            _emb = getattr(args.embodiment_tag, "value", str(args.embodiment_tag))
            _vc = _mc.get(_emb) or _mc[next(iter(_mc))]
            video_full_deltas = [int(d) for d in _vc["video"]["delta_indices"]]
            logging.info(f"Full video deltas from processor_config: {video_full_deltas}")
        except Exception:
            logging.exception(
                "Could not read full video deltas from processor_config; "
                "falling back to the (truncated) loader modality config"
            )

    # Optional predicted-action → predicted-video projection hook (embodiment-specific).
    projection_fn = None
    if args.projection == "batch7":
        try:
            import sys as _sys

            if args.projection_module_path:
                _sys.path.insert(0, args.projection_module_path)
            from ol_projection import make_batch7_projection_fn

            _proj_dir = (args.save_video_path or "/tmp/open_loop_eval") + "/projection"
            projection_fn = make_batch7_projection_fn(_proj_dir)
            logging.info(f"batch7 projection overlays -> {_proj_dir}")
        except Exception:
            logging.exception("Could not build batch7 projection_fn; skipping projection")

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        metrics = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
            control_freq=args.control_freq,
            use_rtc_wrapper=args.use_rtc_wrapper,
            save_video_path=args.save_video_path,
            enable_frechet=args.enable_frechet,
            enable_reverse_kl=args.enable_reverse_kl,
            rkl_num_samples=args.rkl_num_samples,
            pooled_accumulator=pooled,
            start_step=args.start_step,
            compute_video_metrics=args.compute_video_metrics,
            video_lpips_net=video_lpips_net,
            video_full_deltas=video_full_deltas,
            projection_fn=projection_fn,
        )
        logging.info(
            "Trajectory %d metrics: %s",
            traj_id,
            ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
        )
        for k, v in metrics.items():
            per_metric.setdefault(k, []).append(v)

    if per_metric:
        for k, vals in per_metric.items():
            logging.info(f"Average {k} across all trajs: {float(np.mean(vals))}")
    else:
        logging.info("No valid trajectories were evaluated.")

    if pooled:
        for k, v in compute_pooled_wasserstein(pooled).items():
            logging.info(f"Pooled {k} across all trajs (per-dim mean): {v}")

    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
