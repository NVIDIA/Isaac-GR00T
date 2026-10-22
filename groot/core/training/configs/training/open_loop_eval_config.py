# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Nested configs for the open-loop evaluation features.

Two flavours:

* ``OpenLoopEvalConfig``         — legacy on-checkpoint-save callback in
  ``groot/core/training/experiment/utils.py::OpenLoopEvalCallback``.
  Lazy-imports ``groot.control``; rank-0 only.
* ``OpenLoopEvalInProcessConfig`` — distributed in-process callback in
  ``groot/core/training/open_loop/in_process_eval_callback.py``.
  Pure ``groot.core``; runs on every rank.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OpenLoopEvalConfig:
    """Legacy on-checkpoint-save open-loop evaluation."""

    enabled: bool = False
    """Master switch."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """Trajectory IDs to evaluate."""

    steps_per_traj: int = 100
    """Number of steps to evaluate per trajectory."""

    plot_indices: list[int] | None = None
    """List of action indices to plot. None plots all."""

    policy_type: str = "gr00t"
    """'gr00t' | 'dream0'. Selects the eval-time policy adapter in groot.control."""

    dataset_index: int | None = None
    """0-based index of the dataset to evaluate. None = all matching."""


@dataclass
class OpenLoopEvalInProcessConfig:
    """In-process distributed open-loop evaluation (groot.core only)."""

    enabled: bool = False
    """Master switch."""

    interval_steps: int = 1000
    """Run the in-process callback every N training steps.

    Named ``interval_steps`` rather than ``eval_steps`` to avoid confusion
    with HF Trainer's existing ``TrainingConfig.eval_steps`` field (the
    val-loss eval cadence). Even though the nested path
    ``open_loop_eval_in_process.eval_steps`` would not collide *as an
    attribute path*, the YAML/CLI surface ("which `eval_steps` did you
    mean?") would be ambiguous."""

    eval_at_start: bool = False
    """If True, also run a baseline eval on the first on_step_end."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """Trajectory IDs (used by 'fixed' sample mode)."""

    steps_per_traj: int = 100
    """Rollout window length (capped by trajectory length)."""

    action_horizon: int | None = None
    """Override action horizon. None -> config.model.action_horizon."""

    save_plot_dir: str | None = None
    """Directory to save trajectory plots. None disables plotting."""

    num_inference_timesteps: int | None = None
    """Optional override for action-head inference timesteps during eval.
    Plumbed for both N1.x (action_head.num_inference_timesteps) and
    N2 (policy.num_inference_steps)."""

    use_train_path: bool = False
    """If True, evaluate on each dataset's train path in addition to val."""

    dataset_indices: list[int] | None = None
    """Optional whitelist of data.datasets indices. None = all."""

    sample_mode: str = "fixed"
    """'fixed' | 'random_trajs' | 'random_samples'.
       - 'fixed': eval the explicit ``traj_ids`` list, each rolled out from
         step 0 (legacy behaviour).
       - 'random_trajs': per dataset, randomly sample ``num_random_samples``
         *unique* trajectory ids and roll them from step 0.
       - 'random_samples': per dataset, randomly sample ``num_random_samples``
         ``(traj_id, start_step)`` windows (with replacement)."""

    num_random_samples: int = 32
    """Per-dataset sample count for random_* modes."""

    random_seed: int = 0
    """Base seed; mixed with global_step at each eval call so different
    eval calls hit different samples while every rank within a single
    eval call sees the same seed."""

    enable_frechet: bool = True
    """Per-window 2-Wasserstein (Bures) distance between empirical Gaussians
    of predicted vs ground-truth action sequences. Effectively free."""

    enable_reverse_kl: bool = False
    """Per-window M-sample empirical KDE -> N(gt, I) reverse-KL.
    Multiplies eval cost by roughly ``rkl_num_samples``."""

    rkl_num_samples: int = 4
    """M for reverse_kl. Must be >= 2 when enable_reverse_kl is True."""

    def __post_init__(self) -> None:
        if self.sample_mode not in ("fixed", "random_trajs", "random_samples"):
            raise ValueError(
                f"sample_mode must be one of fixed/random_trajs/random_samples, "
                f"got {self.sample_mode!r}"
            )
        if self.enable_reverse_kl and self.rkl_num_samples < 2:
            raise ValueError(
                f"enable_reverse_kl requires rkl_num_samples >= 2, "
                f"got {self.rkl_num_samples}"
            )
