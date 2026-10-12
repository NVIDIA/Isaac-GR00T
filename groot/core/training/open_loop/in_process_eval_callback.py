# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""In-process, *distributed* open-loop evaluation callback for ``groot.core``.

Runs at fixed step intervals during training. Each rank evaluates a
round-robin slice of work units against the *live* training model +
processor (no checkpoint reload, no policy server). After local eval,
per-spec metrics are gathered across all ranks via
``dist.all_gather_object`` and rank 0 logs the aggregate.

Each work unit is a triple ``(spec_idx, traj_id, start_fraction)``:

* ``spec_idx`` indexes into ``self.eval_datasets``.
* ``traj_id`` is the trajectory index within the dataset.
* ``start_fraction`` is in ``[0, 1]`` and resolves to a real
  ``start_step = int(start_fraction * max(0, traj_length - steps_per_traj))``
  so the evaluator can roll any window of the trajectory rather than
  always rolling from step 0.

Three sampling modes are supported:

* ``"fixed"`` — ``traj_ids`` is a deterministic list, all rolled from
  step 0. Reproduces the original behaviour.
* ``"random_trajs"`` — for each dataset, sample ``num_random_samples``
  *unique* traj ids; each rolled from step 0. Broader trajectory
  coverage at the same per-traj cost.
* ``"random_samples"`` — for each dataset, sample ``num_random_samples``
  ``(traj_id, start_fraction)`` windows (with replacement). True IID
  estimator of the dataset-wide error.

Eval datasets are taken from ``DataConfig.datasets[*].val_dataset_path``
(and optionally each dataset's training path), so adding a new eval
target is a one-line change in the data config.

Compatible with DDP and DeepSpeed ZeRO-1/2 (full params on each rank).
ZeRO-3 would require gathering parameters around the inference call —
not handled here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import timedelta
import logging
import os
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from groot.core.data.schema import EmbodimentTag
from groot.core.training.open_loop.live_policy import (
    LivePolicyAdapter,
    build_live_policy,
)

# Sample modes for the in-process callback. Kept as plain strings (not an
# Enum) so they're trivially CLI-passable via tyro / hydra.
SAMPLE_MODE_FIXED = "fixed"
SAMPLE_MODE_RANDOM_TRAJS = "random_trajs"
SAMPLE_MODE_RANDOM_SAMPLES = "random_samples"
_VALID_SAMPLE_MODES = (
    SAMPLE_MODE_FIXED,
    SAMPLE_MODE_RANDOM_TRAJS,
    SAMPLE_MODE_RANDOM_SAMPLES,
)

if TYPE_CHECKING:
    # deferred: stubbed in tests via sys.modules injection (matplotlib weight,
    # torchcodec cold-start cost). evaluate_single_trajectory is also deferred
    # for the same reason — see _run_eval.
    from groot.core.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalDatasetSpec:
    """One eval target: a dataset path + the embodiment to evaluate against."""

    name: str
    dataset_path: str
    embodiment_tag: str
    eval_type: str  # "val" or "train" (for log labelling)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / DeepSpeed wrappers so we can read ``action_head`` etc.

    DeepSpeed engines and DDP both expose the inner ``nn.Module`` via
    ``.module``. We only need this for sniffing attributes; the inference
    call itself goes through the wrapped object so wrapper hooks still fire.
    """
    inner = model
    for _ in range(4):
        if hasattr(inner, "module") and isinstance(inner.module, torch.nn.Module):
            inner = inner.module
        else:
            break
    return inner


class InProcessOpenLoopEvalCallback(TrainerCallback):
    """Distributed in-process open-loop eval at fixed step intervals.

    Args:
        eval_datasets: list of ``EvalDatasetSpec`` describing what to eval.
            Typically built from ``config.data.datasets`` in
            ``experiment.py``.
        processor: the training processor (``BaseProcessor``).
        traj_ids: trajectory ids to evaluate per dataset.
        steps_per_traj: max trajectory steps to roll out per traj id.
        action_horizon: action horizon for chunked inference.
        eval_steps: run eval every ``eval_steps`` training steps.
        eval_at_start: also run a baseline eval at step 0.
        save_plot_dir: directory to save trajectory plots into. ``None``
            disables plotting (faster).
        video_backend: video backend for ``LeRobotEpisodeLoader``.
        num_inference_timesteps: optional override for action-head
            inference timesteps during eval.
        barrier_timeout_s: timeout (seconds) for the post-log
            ``monitored_barrier`` that lockstep-syncs all ranks after
            rank-0 finishes ``trainer.log()``. Default 1800 (30 min) —
            generous enough for worst-case wandb retry storms with many
            metric keys, but bounded so a genuinely deadlocked rank
            doesn't wedge the cluster forever. The barrier runs over a
            side gloo process group (NOT NCCL) so it is NOT subject to
            the NCCL watchdog timeout (typically 60s–25min) that the
            previous default-PG ``dist.barrier()`` was hitting. See the
            long comment in ``_run_eval`` for why a NCCL barrier was
            insufficient.
    """

    def __init__(
        self,
        eval_datasets: list[EvalDatasetSpec],
        processor: Any,
        traj_ids: list[int],
        steps_per_traj: int,
        action_horizon: int,
        eval_steps: int,
        eval_at_start: bool = False,
        save_plot_dir: str | None = None,
        video_backend: str = "torchcodec",
        num_inference_timesteps: int | None = None,
        sample_mode: str = SAMPLE_MODE_FIXED,
        num_random_samples: int = 32,
        random_seed: int = 0,
        enable_frechet: bool = True,
        enable_reverse_kl: bool = False,
        rkl_num_samples: int = 4,
        barrier_timeout_s: int = 1800,
    ) -> None:
        if sample_mode not in _VALID_SAMPLE_MODES:
            raise ValueError(
                f"sample_mode must be one of {_VALID_SAMPLE_MODES!r}, got {sample_mode!r}"
            )
        self.eval_datasets = list(eval_datasets)
        self.processor = processor
        self.traj_ids = list(traj_ids)
        self.steps_per_traj = int(steps_per_traj)
        self.action_horizon = int(action_horizon)
        self.eval_steps = int(eval_steps)
        self.eval_at_start = bool(eval_at_start)
        self.save_plot_dir = save_plot_dir
        self.video_backend = video_backend
        self.num_inference_timesteps = num_inference_timesteps
        self.sample_mode = sample_mode
        self.num_random_samples = int(num_random_samples)
        self.random_seed = int(random_seed)
        self.enable_frechet = bool(enable_frechet)
        self.enable_reverse_kl = bool(enable_reverse_kl)
        self.rkl_num_samples = int(rkl_num_samples)
        if self.enable_reverse_kl and self.rkl_num_samples < 2:
            raise ValueError(
                f"enable_reverse_kl requires rkl_num_samples >= 2, got {self.rkl_num_samples}"
            )
        if barrier_timeout_s <= 0:
            raise ValueError(f"barrier_timeout_s must be positive, got {barrier_timeout_s!r}")
        self._barrier_timeout = timedelta(seconds=int(barrier_timeout_s))
        # Side gloo process group used for the post-log monitored_barrier.
        # Created lazily on first eval call (rather than __init__) because
        # dist may not be initialized yet when the callback is constructed.
        self._gloo_pg = None

        self.trainer = None  # set after construction (callback.trainer = trainer)
        # Loaders are cached across eval calls, but only on ranks that
        # actually need them (lazy build for `fixed`; eager for the
        # random sampling modes since every rank needs `len(loader)`).
        self._loader_cache: dict[tuple[str, str], LeRobotEpisodeLoader] = {}
        # Per-spec dataset length cache keyed by spec_idx — populated by
        # the random-sampling pre-load.
        self._spec_length_cache: dict[int, int] = {}
        self._has_run_at_start = False

    # ------------------------------------------------------------------
    # Side gloo process group for the post-log monitored_barrier
    # ------------------------------------------------------------------

    def _get_gloo_pg(self):
        """Return a side gloo process group, lazily creating it on first use.

        We use a CPU/gloo PG (not the default NCCL PG) for the post-log
        barrier because:

        1. ``dist.barrier()`` over NCCL is itself a small ALLREDUCE
           collective and is therefore subject to the NCCL watchdog
           timeout. If rank 0's ``trainer.log()`` (e.g. wandb.log with
           network retries) takes longer than that timeout, the watchdog
           SIGABRTs the whole job — which is exactly the bug the
           callback is trying to prevent.
        2. Gloo has no equivalent watchdog. Ranks waiting at a gloo
           barrier wait quietly until either the participating rank
           shows up or the per-call ``monitored_barrier`` timeout fires
           with a clean Python exception (not SIGABRT).

        Returns ``None`` if torch.distributed is not initialized or
        world size is 1 — callers should fall back to a no-op in those
        cases.
        """
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return None
        if self._gloo_pg is not None:
            return self._gloo_pg
        try:
            self._gloo_pg = dist.new_group(
                backend="gloo",
                timeout=self._barrier_timeout,
            )
        except Exception:
            # Some deployments (e.g. NCCL-only torch builds) may fail to
            # create a gloo PG. Fall back to None so callers degrade
            # gracefully to a plain (NCCL) barrier — strictly worse than
            # gloo for this use case, but better than crashing.
            logger.exception(
                "[in-process open-loop eval] failed to create gloo "
                "process group for post-log barrier; falling back to "
                "default (NCCL) PG. Slow rank-0 trainer.log may still "
                "trip the NCCL watchdog."
            )
            self._gloo_pg = None
        return self._gloo_pg

    # ------------------------------------------------------------------
    # Trajectory assignment (round-robin across ranks)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_all_pairs(num_specs: int, traj_ids: list[int]) -> list[tuple[int, int]]:
        """Flatten ``(spec_idx, traj_id)`` cross product in a deterministic order.

        Legacy helper kept for backwards compatibility / unit tests; the
        current ``_run_eval`` path uses :meth:`_build_work_units` which
        also handles random sampling and start-step fractions.
        """
        return [(s, t) for s in range(num_specs) for t in traj_ids]

    @staticmethod
    def _build_work_units(
        num_specs: int,
        sample_mode: str,
        traj_ids: list[int],
        num_random_samples: int,
        dataset_lengths: list[int] | None,
        seed: int,
    ) -> list[tuple[int, int, float]]:
        """Build the deterministic ``(spec_idx, traj_id, start_fraction)`` work list.

        The output is identical on every rank (provided ``seed`` is the
        same), and stable for a given seed. ``_partition_pairs`` then
        slices it round-robin across ranks. Determinism is what makes
        the gather + aggregate step yield the same metric regardless of
        ``world_size``.

        ``start_fraction`` is in ``[0, 1]`` and resolved per-rank when
        the trajectory is loaded:
        ``start_step = int(start_fraction * max(0, traj_length - steps_per_traj))``.

        Mode semantics:
          * ``fixed``: cross product of ``range(num_specs)`` × ``traj_ids``
            with ``start_fraction = 0``. ``dataset_lengths`` is ignored.
          * ``random_trajs``: per spec, sample ``min(num_random_samples,
            dataset_length)`` *unique* traj ids; ``start_fraction = 0``.
          * ``random_samples``: per spec, sample ``num_random_samples``
            ``(traj_id, start_fraction)`` pairs *with* replacement.

        ``random_*`` modes require one int per spec in ``dataset_lengths``;
        any spec whose length is ``<= 0`` contributes no work.
        """
        if sample_mode == SAMPLE_MODE_FIXED:
            return [(s, t, 0.0) for s in range(num_specs) for t in traj_ids]

        if dataset_lengths is None or len(dataset_lengths) != num_specs:
            raise ValueError(
                f"dataset_lengths must be a length-{num_specs} list for sample_mode={sample_mode!r}"
            )
        if num_random_samples <= 0:
            return []

        units: list[tuple[int, int, float]] = []
        for s in range(num_specs):
            spec_len = int(dataset_lengths[s])
            if spec_len <= 0:
                continue
            # Per-spec rng so that swapping eval_datasets order does not
            # reshuffle the samples drawn for each individual spec.
            spec_seed = (seed * 1_000_003) ^ ((s + 1) * 2_654_435_761)
            rng = random.Random(spec_seed & 0xFFFFFFFF)

            if sample_mode == SAMPLE_MODE_RANDOM_TRAJS:
                n = min(num_random_samples, spec_len)
                sampled = sorted(rng.sample(range(spec_len), n))
                units.extend((s, t, 0.0) for t in sampled)

            elif sample_mode == SAMPLE_MODE_RANDOM_SAMPLES:
                for _ in range(num_random_samples):
                    t = rng.randrange(spec_len)
                    f = rng.random()
                    units.append((s, t, f))

            else:  # pragma: no cover — guarded by ctor validation
                raise ValueError(f"Unknown sample_mode: {sample_mode!r}")

        # Stable order across ranks. The rng draws above are already
        # deterministic, but sorting also makes the diff-friendly
        # rank=0 logging order consistent.
        units.sort()
        return units

    @staticmethod
    def _resolve_start_step(start_fraction: float, traj_length: int, steps_per_traj: int) -> int:
        """Map ``start_fraction`` ∈ [0, 1] to a real ``start_step`` ∈ [0, traj_length).

        ``start_fraction == 0`` always resolves to ``0`` so that the
        ``fixed`` mode matches the legacy "roll from the prefix" behaviour
        bit-for-bit. For non-zero fractions we anchor the window such
        that ``start_step + steps_per_traj <= traj_length`` whenever
        possible — short trajectories simply roll a shorter window
        (handled inside ``evaluate_single_trajectory``).
        """
        if start_fraction <= 0.0 or traj_length <= 0:
            return 0
        max_start = max(0, traj_length - int(steps_per_traj))
        if max_start == 0:
            return 0
        # `random.random()` returns in [0, 1); int() truncates toward 0,
        # so the resolved start is in [0, max_start] inclusive.
        return min(max_start, int(start_fraction * (max_start + 1)))

    @staticmethod
    def _partition_pairs(all_pairs: list, rank: int, world_size: int) -> list:
        """Return the round-robin slice of ``all_pairs`` belonging to ``rank``.

        Round-robin partitioning is used (rather than chunked) so work
        items are spread as evenly as possible across ranks even when
        ``len(all_pairs) % world_size != 0``. Element type is unconstrained
        — works for both ``(spec_idx, traj_id)`` pairs and
        ``(spec_idx, traj_id, start_fraction)`` triples.
        """
        return [p for i, p in enumerate(all_pairs) if i % world_size == rank]

    @staticmethod
    def _merge_local_results(
        gathered: list[dict[int, list[dict[str, float]]] | None],
    ) -> dict[int, list[dict[str, float]]]:
        """Merge per-rank ``local_results`` dicts into a single dict.

        Each per-window entry is a ``dict[str, float]`` of metric name →
        scalar value (``mse``, ``mae``, optionally ``frechet``,
        ``reverse_kl``). Order of extension follows ``gathered`` (i.e. rank
        order). The *set* of per-window dicts per spec is invariant across
        ``world_size`` — floating-point order in downstream ``np.mean``
        may differ slightly but the resulting aggregate per-spec value is
        bit-stable for reasonable magnitudes (see
        ``test_in_process_eval_distributed.py``).
        """
        merged: dict[int, list[dict[str, float]]] = {}
        for rank_results in gathered:
            if not rank_results:
                continue
            for spec_idx, items in rank_results.items():
                merged.setdefault(spec_idx, []).extend(items)
        return merged

    @staticmethod
    def _aggregate_metrics(
        merged: dict[int, list[dict[str, float]]],
    ) -> dict[int, dict[str, float]]:
        """Per-spec mean of every metric key seen + trajectory count.

        Metrics that aren't reported on every window (e.g. ``reverse_kl``
        when the policy is deterministic on some windows) are averaged
        only over the windows where they *are* present, so a single
        spurious failure doesn't drop the whole metric.
        """
        aggregate: dict[int, dict[str, float]] = {}
        for spec_idx, items in merged.items():
            if not items:
                continue
            metric_keys: set[str] = set()
            for it in items:
                metric_keys.update(it.keys())
            spec_agg: dict[str, float] = {
                "num_trajectories": float(len(items)),
            }
            for key in metric_keys:
                vals = [it[key] for it in items if key in it]
                if vals:
                    spec_agg[key] = float(np.mean(vals))
            aggregate[spec_idx] = spec_agg
        return aggregate

    def _rank_world_size(self) -> tuple[int, int]:
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _seed_for_call(self, global_step: int) -> int:
        """Mix ``random_seed`` with ``global_step`` for each eval call.

        Sample different work units (an unbiased estimator of dataset
        error over training time), while every rank inside a single eval
        call sees the *same* seed (preserves the IID partition + gather
        invariant).
        """
        return (int(self.random_seed) ^ int(global_step)) & 0xFFFFFFFF

    # ------------------------------------------------------------------
    # HF Trainer hook
    # ------------------------------------------------------------------

    def on_step_end(
        self,
        args: TrainingArguments,  # noqa: ARG002 - HF callback keyword contract
        state: TrainerState,
        _control: TrainerControl,
        model=None,
        **kwargs,  # noqa: ARG002 - HF callback keyword contract
    ) -> None:
        if not self.eval_datasets:
            return

        global_step = state.global_step
        is_eval_step = global_step > 0 and (global_step % self.eval_steps == 0)
        is_baseline_step = self.eval_at_start and not self._has_run_at_start
        if not (is_eval_step or is_baseline_step):
            return
        if is_baseline_step:
            self._has_run_at_start = True

        self._run_eval(global_step=global_step, model=model)

    # ------------------------------------------------------------------
    # Core eval driver (runs on every rank)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-spec policy / loader builders (used by both pre-load and eval loop)
    # ------------------------------------------------------------------

    def _build_policy(
        self,
        *,
        model,
        spec: EvalDatasetSpec,
        policy_cache: dict[str, LivePolicyAdapter],
        rank: int,
    ) -> tuple[LivePolicyAdapter | None, EmbodimentTag | None]:
        """Return (policy, embodiment_tag) — None on unknown embodiment.

        ``build_live_policy`` raising ``TypeError`` / ``RuntimeError``
        propagates: aborts the eval and the training step rather than
        silently logging "failed traj N" forever. The previous wrapping
        try/except around the build site was hiding exactly the bug
        Issue Y is filed against, so it's gone.
        """
        try:
            embodiment_tag = EmbodimentTag(spec.embodiment_tag)
        except ValueError:
            if rank == 0:
                logger.warning(
                    "[in-process open-loop eval] unknown embodiment_tag %r — skipping %s",
                    spec.embodiment_tag,
                    spec.name,
                )
            return None, None

        policy = policy_cache.get(spec.embodiment_tag)
        if policy is None:
            policy = build_live_policy(
                model=model,
                processor=self.processor,
                embodiment_tag=embodiment_tag,
                num_inference_timesteps=self.num_inference_timesteps,
            )
            policy_cache[spec.embodiment_tag] = policy
        return policy, embodiment_tag

    def _build_loader(
        self,
        *,
        spec: EvalDatasetSpec,
        policy: LivePolicyAdapter,
        rank: int,
    ) -> LeRobotEpisodeLoader | None:
        from groot.core.data.dataset.lerobot_episode_loader import (
            LeRobotEpisodeLoader,
        )

        loader_key = (spec.dataset_path, spec.embodiment_tag)
        loader = self._loader_cache.get(loader_key)
        if loader is None:
            try:
                loader = LeRobotEpisodeLoader(
                    dataset_path=spec.dataset_path,
                    modality_configs=policy.get_modality_config(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=None,
                )
                self._loader_cache[loader_key] = loader
            except Exception:
                logger.exception(
                    "[in-process open-loop eval][rank=%d] failed to load %s (%s)",
                    rank,
                    spec.dataset_path,
                    spec.embodiment_tag,
                )
                return None
        return loader

    def _compute_dataset_lengths(
        self,
        *,
        model,
        policy_cache: dict[str, LivePolicyAdapter],
        rank: int,
    ) -> list[int]:
        """Eagerly build policies + loaders and return per-spec dataset lengths.

        Return ``0`` for any spec whose loader could
        not be built, signalling "no work" to ``_build_work_units``.

        Required by the random-sampling modes since every rank must
        agree on dataset sizes to draw the *same* random sample set.
        """
        lengths: list[int] = []
        for spec_idx, spec in enumerate(self.eval_datasets):
            cached = self._spec_length_cache.get(spec_idx)
            if cached is not None:
                lengths.append(cached)
                continue
            policy, _ = self._build_policy(
                model=model, spec=spec, policy_cache=policy_cache, rank=rank
            )
            if policy is None:
                lengths.append(0)
                self._spec_length_cache[spec_idx] = 0
                continue
            loader = self._build_loader(spec=spec, policy=policy, rank=rank)
            if loader is None:
                lengths.append(0)
                self._spec_length_cache[spec_idx] = 0
                continue
            n = len(loader)
            lengths.append(n)
            self._spec_length_cache[spec_idx] = n
        return lengths

    # ------------------------------------------------------------------
    # Core eval driver (runs on every rank)
    # ------------------------------------------------------------------

    def _run_eval(self, *, global_step: int, model) -> None:
        from groot.core.training.open_loop.open_loop_eval import (
            evaluate_single_trajectory,
        )

        rank, world_size = self._rank_world_size()

        was_training = bool(model.training) if model is not None else True
        was_processor_training = bool(getattr(self.processor, "training", False))
        if model is not None:
            model.eval()
        try:
            self.processor.eval()
        except Exception:  # noqa: BLE001, S110 - preserve optional processor-hook behavior
            pass

        # Per-spec local results: spec_idx -> list of metric dicts.
        # Each dict has at least {"mse", "mae"}, optionally "frechet" and
        # "reverse_kl" depending on what the callback was constructed with.
        local_results: dict[int, list[dict[str, float]]] = {}
        # Per-embodiment policy cache (one LivePolicyAdapter per embodiment).
        policy_cache: dict[str, LivePolicyAdapter] = {}

        t0 = time.time()
        try:
            # Random sampling modes need dataset_lengths on every rank
            # so every rank draws the same sample set.
            dataset_lengths: list[int] | None = None
            if self.sample_mode != SAMPLE_MODE_FIXED:
                dataset_lengths = self._compute_dataset_lengths(
                    model=model, policy_cache=policy_cache, rank=rank
                )

            seed = self._seed_for_call(global_step)
            all_units = self._build_work_units(
                num_specs=len(self.eval_datasets),
                sample_mode=self.sample_mode,
                traj_ids=self.traj_ids,
                num_random_samples=self.num_random_samples,
                dataset_lengths=dataset_lengths,
                seed=seed,
            )
            my_units = self._partition_pairs(all_units, rank, world_size)

            if rank == 0:
                logger.info(
                    "[in-process open-loop eval] step=%d world_size=%d "
                    "specs=%d sample_mode=%s seed=%d total_units=%d",
                    global_step,
                    world_size,
                    len(self.eval_datasets),
                    self.sample_mode,
                    seed,
                    len(all_units),
                )

            for spec_idx, traj_id, start_fraction in my_units:
                spec = self.eval_datasets[spec_idx]

                policy, embodiment_tag = self._build_policy(
                    model=model, spec=spec, policy_cache=policy_cache, rank=rank
                )
                if policy is None:
                    continue

                loader = self._build_loader(spec=spec, policy=policy, rank=rank)
                if loader is None:
                    continue

                if traj_id >= len(loader):
                    if rank == 0:
                        logger.info(
                            "[in-process open-loop eval] traj_id %d out of range "
                            "for %s (len=%d) — skipping",
                            traj_id,
                            spec.name,
                            len(loader),
                        )
                    continue

                # Resolve start_step from start_fraction. We need
                # `traj_length` for this; `loader[traj_id]` gives a
                # pandas DataFrame and `len()` is O(1).
                try:
                    traj_length = len(loader[traj_id])
                except Exception:
                    logger.exception(
                        "[in-process open-loop eval][rank=%d] failed to read "
                        "traj %d length from %s",
                        rank,
                        traj_id,
                        spec.name,
                    )
                    continue
                start_step = self._resolve_start_step(
                    start_fraction, traj_length, self.steps_per_traj
                )

                # Resolve plot path (None disables plot save). Include
                # start_step in the filename so multiple windows from
                # the same traj don't collide.
                save_plot_path = None
                if self.save_plot_dir:
                    plot_target_dir = os.path.join(
                        self.save_plot_dir,
                        f"step_{global_step}",
                        f"{spec.embodiment_tag}__{spec.eval_type}__{spec.name}",
                    )
                    os.makedirs(plot_target_dir, exist_ok=True)
                    plot_basename = (
                        f"traj_{traj_id}.png"
                        if start_step == 0
                        else f"traj_{traj_id}__start_{start_step}.png"
                    )
                    save_plot_path = os.path.join(plot_target_dir, plot_basename)

                try:
                    metrics = evaluate_single_trajectory(
                        policy=policy,
                        loader=loader,
                        traj_id=traj_id,
                        embodiment_tag=embodiment_tag,
                        steps=self.steps_per_traj,
                        action_horizon=self.action_horizon,
                        save_plot_path=save_plot_path,
                        start_step=start_step,
                        enable_frechet=self.enable_frechet,
                        enable_reverse_kl=self.enable_reverse_kl,
                        rkl_num_samples=self.rkl_num_samples,
                    )
                except Exception:
                    logger.exception(
                        "[in-process open-loop eval][rank=%d] failed traj %d (start=%d) on %s",
                        rank,
                        traj_id,
                        start_step,
                        spec.name,
                    )
                    continue

                # Coerce all metric values to plain Python floats so the
                # gather collective only ships small picklable scalars.
                metrics_float = {k: float(v) for k, v in metrics.items()}
                local_results.setdefault(spec_idx, []).append(metrics_float)
                logger.info(
                    "[in-process open-loop eval][rank=%d] %s/%s/traj_%d/start_%d: %s",
                    rank,
                    spec.embodiment_tag,
                    spec.eval_type,
                    traj_id,
                    start_step,
                    " ".join(f"{k.upper()}={v:.6f}" for k, v in metrics_float.items()),
                )
        finally:
            for _emb, p in policy_cache.items():
                try:
                    p.close()
                except Exception:  # noqa: BLE001, S110 - best-effort adapter cleanup
                    pass
            if model is not None and was_training:
                model.train()
            try:
                if was_processor_training:
                    self.processor.train()
            except Exception:  # noqa: BLE001, S110 - preserve optional processor-hook behavior
                pass

        # Gather per-spec (mse, mae) lists across all ranks. We pickle-gather
        # because results are tiny.
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Round-robin assignment leaves a *subset* of ranks doing CUDA
            # inference (e.g. 24 work units over 64 ranks → 40 ranks did
            # zero forward work, 24 ranks have pending model-forward kernels
            # on the default CUDA stream). Without this synchronize, the
            # NCCL allgather kernel queues behind those pending eval kernels
            # on the work-ranks but on an empty stream on the no-work ranks
            # — the eval gather itself still completes, but the *next*
            # collective HF Trainer issues (`_nested_gather(acc_tensor)` in
            # `GR00TTrainer.compute_loss`, an ALLGATHER NumelIn=1 NumelOut=N)
            # has been observed to hit a 25-min NCCL watchdog timeout
            # because the work-rank stream is still draining when no-work
            # ranks have already moved on. Forcing a sync on every rank
            # drains any pending eval CUDA work before we touch NCCL.
            # Cheap (no-op) when nothing is pending.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_results)
            # Lockstep all ranks at the same NCCL SeqNum before returning
            # to HF Trainer's training loop. Defence-in-depth alongside
            # the synchronize above: guarantees no rank races ahead into
            # the next training-step collective while another is still
            # finishing the gather.
            dist.barrier()
        else:
            gathered = [local_results]

        local_elapsed = time.time() - t0

        # Only rank 0 builds the log dict and reports — but ALL ranks must
        # wait for it to finish before returning to the training loop.
        # Previously this branch did ``if rank != 0: return``, so non-zero
        # ranks returned to HF Trainer immediately after the post-gather
        # barrier. They'd then hit the next training-step NCCL collective
        # (``_nested_gather(acc_tensor)`` for loss aggregation in
        # ``GR00TTrainer.compute_loss``, an ALLGATHER NumelIn=1
        # NumelOut=world_size) and stall waiting for rank 0 to join. Rank
        # 0 was still busy doing aggregation + ``trainer.log()`` /
        # ``wandb.log()`` — which can take many seconds-to-minutes under
        # network pressure (wandb backoff/retry on flaky connection, many
        # metric keys, etc.). If rank 0's logging exceeded the NCCL
        # collective timeout (1500s default in DeepSpeed), the watchdog
        # SIGABRT'd every worker.
        #
        # Reproduced on the mecka mini-chinchilla canaries:
        # ``WorkNCCL(SeqNum=2003, OpType=ALLGATHER, NumelIn=1,
        # NumelOut=64) ran for 1500015ms before timing out`` —
        # SeqNum=2003 is the next training-step collective after the
        # eval-step's all_gather_object. All workers aborted at the same
        # Unix epoch second once the watchdog tripped.
        #
        # Fix: restructure so non-zero ranks no longer return early. Rank
        # 0 still does all the logging, but every rank reaches the final
        # ``dist.barrier()`` at the bottom of ``_run_eval``. The barrier
        # blocks all ranks (including rank 0) until rank 0 has finished
        # ``trainer.log() / wandb.log()``. Only then do ranks return to
        # training — in true lockstep — so the next training collective
        # starts with every rank participating from the same instant.
        if rank == 0:
            merged = self._merge_local_results(gathered)
            aggregate = self._aggregate_metrics(merged)

            log_dict: dict[str, float] = {"open_loop/eval_step": float(global_step)}
            # Per-metric accumulator across specs for the avg_* aggregates.
            per_metric_avg: dict[str, list[float]] = {}
            # ``num_trajectories`` is a count, not something we average across
            # the open_loop/avg_* surface — pull it out separately.
            for spec_idx, ds_metrics in aggregate.items():
                spec = self.eval_datasets[spec_idx]
                label = f"open_loop/{spec.embodiment_tag}/{spec.eval_type}/{spec.name}"
                log_dict[f"{label}/num_trajectories"] = ds_metrics.get("num_trajectories", 0.0)
                for metric_key, value in ds_metrics.items():
                    if metric_key == "num_trajectories":
                        continue
                    log_dict[f"{label}/{metric_key}"] = value
                    per_metric_avg.setdefault(metric_key, []).append(value)
                logger.info(
                    "[in-process open-loop eval] %s mean over %d traj: %s",
                    label,
                    int(ds_metrics.get("num_trajectories", 0)),
                    " ".join(
                        f"{k.upper()}={v:.6f}"
                        for k, v in ds_metrics.items()
                        if k != "num_trajectories"
                    ),
                )

            for metric_key, vals in per_metric_avg.items():
                if vals:
                    log_dict[f"open_loop/avg_{metric_key}"] = float(np.mean(vals))
            log_dict["open_loop/eval_wall_time_s"] = float(local_elapsed)

            logger.info(
                "[in-process open-loop eval] done step=%d in %.1fs (%d datasets reported)",
                global_step,
                local_elapsed,
                len(aggregate),
            )

            # Route through Trainer.log so HF's WandbCallback / TensorBoard
            # / etc. all pick it up.
            try:
                if self.trainer is not None and len(log_dict) > 1:
                    # HF on_log clears should_log. Rank-zero-only eval logging
                    # must not consume training's all-rank logging decision.
                    training_control = self.trainer.control
                    self.trainer.control = copy.copy(training_control)
                    try:
                        self.trainer.log(log_dict)
                    finally:
                        self.trainer.control = training_control
                else:
                    try:
                        import wandb as _wandb

                        if getattr(_wandb, "run", None) is not None:
                            _wandb.log(log_dict, step=global_step)
                    except Exception:  # noqa: BLE001, S110 - optional standalone metric sink
                        pass
            except Exception:
                logger.exception("[in-process open-loop eval] failed to log metrics")

        # Final lockstep — every rank waits here until rank 0 has finished
        # aggregation + trainer.log() / wandb.log(). See the long comment
        # above for the NCCL-watchdog-timeout cascade this prevents.
        #
        # CRITICAL: this barrier MUST run over a gloo (CPU) process group,
        # not the default NCCL PG. ``dist.barrier()`` over NCCL is itself
        # an ALLREDUCE collective and is therefore subject to the NCCL
        # watchdog timeout — so a slow rank-0 trainer.log() that exceeds
        # the watchdog timeout still SIGABRTs every rank (just at this
        # barrier instead of the next training-step collective). Earlier
        # iterations of this fix used a plain NCCL barrier here and
        # therefore did not actually fix the bug — they just relocated
        # where the watchdog tripped.
        #
        # Using ``monitored_barrier(group=gloo_pg, ..., wait_all_ranks=True)``
        # gives us:
        #   - gloo backend → no NCCL watchdog → rank 0's slow log won't
        #     SIGABRT the job
        #   - per-call timeout (``self._barrier_timeout``, default 30 min)
        #     → if a rank truly deadlocks (vs just slow), we get a clean
        #     Python exception with rank info, not a wedged cluster
        #   - ``wait_all_ranks=True`` → error message names the missing
        #     ranks, useful for diagnosing real hangs
        gloo_pg = self._get_gloo_pg()
        if gloo_pg is not None:
            try:
                dist.monitored_barrier(
                    group=gloo_pg,
                    timeout=self._barrier_timeout,
                    wait_all_ranks=True,
                )
            except Exception:
                # monitored_barrier raises if the per-call timeout fires
                # OR if any rank failed to participate. Both are real
                # bugs the operator needs to see, but we don't want to
                # mask the underlying training error if there is one.
                logger.exception(
                    "[in-process open-loop eval] post-log monitored_barrier failed (timeout=%s)",
                    self._barrier_timeout,
                )
                raise
        elif dist.is_initialized() and dist.get_world_size() > 1:
            # Gloo PG creation failed (logged in _get_gloo_pg). Fall
            # back to plain NCCL barrier — strictly worse but at least
            # preserves the lockstep semantics in well-behaved cases.
            dist.barrier()
