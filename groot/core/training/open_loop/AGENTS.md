# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# In-Process Open-Loop Evaluation — Agent Guide

This module provides distributed, in-process open-loop evaluation for
`groot.core` training (N1.6 / N1.7 / N2). It is the lightweight,
checkpoint-reload-free counterpart to the `OpenLoopEvalCallback` in
`groot/core/training/experiment/utils.py` (which lazily imports
`groot.control` and only runs on rank 0 by spinning up a ZMQ policy
server).

> Prefer this module over the legacy on-save callback when you are
> running pure `groot.core` training (no `groot.control`,
> no `groot.vla.omni`) and want eval signal at fixed step intervals
> rather than on every checkpoint.

---

## 1. What it does

At fixed training-step intervals, every rank:

1. Wraps the **live training model + processor** in a `LivePolicyAdapter`
   (no checkpoint reload, no policy server). `build_live_policy` auto-detects
   the model family (N1.x via `action_head`; N2 via `predict + has_action_head`)
   and dispatches to the right concrete `Gr00tPolicy` / `Gr00tN2Policy` accordingly.
2. Evaluates a round-robin slice of `(eval_dataset, trajectory_id)`
   pairs with `evaluate_single_trajectory` from
   `groot/core/training/open_loop/open_loop_eval.py`. The adapter wraps the
   forward pass in `torch.amp.autocast(dtype=torch.bfloat16)` so it works
   on fp32-loaded live training models too (HF Trainer's autocast covers
   `training_step` only, not `on_step_end` callbacks).
3. Reports per-trajectory MSE / MAE / Fréchet (optionally Reverse-KL).
4. All ranks `dist.all_gather_object` their local results.
5. Rank 0 averages and routes through `Trainer.log` (so wandb /
   tensorboard pick it up).

After eval, every rank restores `model.train()` / processor training
state.

---

## 2. Files

```
groot/core/training/open_loop/
├── live_policy.py                     # LivePolicyAdapter + build_live_policy + N1/N2 factories
├── in_process_eval_callback.py        # InProcessOpenLoopEvalCallback
├── open_loop_eval.py                  # evaluate_single_trajectory (CLI script)
├── metrics.py                         # frechet_gaussian_over_time, reverse_kl_from_samples
└── AGENTS.md                          # this file

tests/groot/core/training/open_loop/
├── test_in_process_eval_distributed.py   # CPU-only IID test for the static helpers
├── test_in_process_eval_e2e.py           # Spawned-Gloo IID test of the full callback
├── test_live_policy_adapter.py           # CPU unit tests for the N1/N2 dispatch + close() lifecycle
└── test_metrics.py                       # Unit tests for the Fréchet / RKL math
```

| File                             | What it is                                               |
| -------------------------------- | -------------------------------------------------------- |
| `live_policy.py`                 | `LivePolicyAdapter(BasePolicy)` + `build_live_policy(...)` factory. The adapter holds a concrete policy (`Gr00tPolicy` / `Gr00tN2Policy`) instantiated against the live training model + processor via `__new__` bypass. The factory dispatches on the unwrapped model's surface (`predict + has_action_head` → N2, otherwise N1 — including RL-wrapped). Composition isolates eval from upstream `Gr00tPolicy` refactors. The adapter wraps the forward in `torch.amp.autocast(bfloat16)` so eval works whether the live model is bf16 or fp32. |
| `in_process_eval_callback.py`    | `InProcessOpenLoopEvalCallback(TrainerCallback)` — the distributed callback. Plus `EvalDatasetSpec` dataclass and `_unwrap_model` helper. |
| `open_loop_eval.py`              | Existing standalone CLI eval (`tyro`). Reused by the callback for the per-trajectory loop and the plotting helper. |
| `metrics.py`                     | Pure-math distribution metrics (`frechet_gaussian_over_time`, `reverse_kl_from_samples`) used by `evaluate_single_trajectory`. |
| `tests/.../test_in_process_eval_distributed.py` | Unit tests of `_partition_pairs` / `_merge_local_results` / `_aggregate_metrics`. |
| `tests/.../test_in_process_eval_e2e.py`         | Spawns Gloo workers, injects fakes for `LivePolicyAdapter` / `LeRobotEpisodeLoader` / `evaluate_single_trajectory`, and verifies IID end-to-end. |
| `tests/.../test_live_policy_adapter.py`         | Unit tests for `_classify_model` (N1 vs N2 dispatch, RL-wrapped fallback), `_make_n2_concrete` attribute population + env reads, `close()` lifecycle (restorers + processor train mode), `reset()` semantics. |
| `tests/.../test_metrics.py`                     | Unit tests for the Fréchet / RKL functions. |

Wired in:

| File                                                            | What changes                                                                                   |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `groot/core/training/configs/training/open_loop_eval_config.py` | `OpenLoopEvalConfig` (legacy) + `OpenLoopEvalInProcessConfig` (in-process) nested dataclasses. |
| `groot/core/training/configs/training/training_config.py`       | Two nested fields: `open_loop_eval` and `open_loop_eval_in_process`, with `__post_init__` that coerces YAML `dict` → dataclass. |
| `groot/core/training/experiment/experiment.py`                  | Builds `EvalDatasetSpec`s from `config.data.datasets[*].val_dataset_path`, attaches callback.  |
| `groot/core/training/experiment/launch_train.py`                | Wraps `run(config)` in `try/finally` and calls `dist.destroy_process_group()` on exit so `torch.distributed.run` doesn't hang after the eval prints metrics. |

---

## 3. Distributed semantics (IMPORTANT — invariant)

The callback is **embarrassingly parallel** over `(spec_idx, traj_id)`
pairs. The invariant that all changes to this module must preserve:

> For a given training step (i.e. fixed model state) and a given set of
> `(spec_idx, traj_id)` pairs, the aggregate per-spec MSE and MAE must
> be **identical regardless of `world_size`** (1 vs 4 vs 8 GPUs).

This holds because:

* `evaluate_single_trajectory(policy, loader, traj_id, ...)` is
  deterministic given a fixed model — the inference call is wrapped in
  `torch.inference_mode()` and the action head's denoising uses fixed
  schedulers / no extra noise.
* Round-robin assignment (`i % world_size == rank`) partitions the
  pairs disjointly across ranks; `dist.all_gather_object` re-collects
  them losslessly.
* Aggregation is `np.mean` over the same set of per-trajectory
  `(mse, mae)` values for every world size (set, not multiset — every
  pair is evaluated exactly once).

Floating-point order does shift slightly because the merge order across
ranks differs by `world_size`, but the difference is well below
`atol=1e-7` for the per-spec means we report.

A unit test that asserts this invariant lives in
`tests/groot/core/training/open_loop/test_in_process_eval_distributed.py`.

> **DO NOT** introduce per-rank state into the eval (e.g. random
> sampling of `traj_ids`, rank-conditioned plotting that affects
> metrics, etc.) without re-deriving this invariant.

---

## 4. Configuration

### 4a. Per-dataset eval target (data config)

Eval targets come from `DataConfig.datasets`. Each `SingleDatasetConfig`
already has a `val_dataset_path` field — set it and the callback will
pick that dataset up automatically:

```yaml
data:
  datasets:
    - dataset_paths:
        - /lustre/.../gb300_dex04_train
      embodiment_tag: share_id
      val_dataset_path: /lustre/.../gb300_dex04_val   # ← used for eval
    - dataset_paths:
        - /lustre/.../gb300_dex05_train
      embodiment_tag: share_id
      val_dataset_path: /lustre/.../gb300_dex05_val
```

For each entry the callback emits one `EvalDatasetSpec` with
`eval_type="val"`. If `val_dataset_path` is missing **or**
`open_loop_eval_in_process.use_train_path: true`, a second spec is
emitted with `eval_type="train"` against the dataset's training path.

### 4b. Callback knobs (`TrainingConfig`)

The two open-loop eval feature surfaces (legacy on-checkpoint-save callback
and the in-process callback) live under their own nested dataclasses on
`TrainingConfig`. Set them via YAML or `--training.<dotted>.<field>` CLI
flags. `TrainingConfig.__post_init__` coerces `dict` → dataclass for both
fields, so YAML works out of the box.

```yaml
training:
  open_loop_eval_in_process:
    enabled: true
    interval_steps: 500           # eval every N steps (was: open_loop_eval_in_process_steps)
    eval_at_start: true           # baseline on first on_step_end (post step-1 optim update)
    traj_ids: [0, 5, 10]          # which trajs (only used by 'fixed' mode)
    steps_per_traj: 200           # rollout window length
    action_horizon: null          # null → config.model.action_horizon
    save_plot_dir: /lustre/.../runs/openloop_plots
    num_inference_timesteps: null # optional override
    use_train_path: false
    dataset_indices: null         # optional whitelist of data.datasets indices
    # Random sampling (for broader / IID coverage):
    sample_mode: fixed            # 'fixed' | 'random_trajs' | 'random_samples'
    num_random_samples: 32        # per-dataset count for random_* modes
    random_seed: 0                # mixed with global_step at each eval
    # Distribution metrics (always-on are MSE / MAE):
    enable_frechet: true          # Bures / 2-Wasserstein per window (cheap, default on)
    enable_reverse_kl: false      # M-sample reverse-KL per window (~Mx cost)
    rkl_num_samples: 4            # M for reverse_kl; ignored when off
```

CLI form (each YAML key is reachable via `--training.open_loop_eval_in_process.<field>`):

```bash
--training.open_loop_eval_in_process.enabled \
--training.open_loop_eval_in_process.interval_steps 500 \
--training.open_loop_eval_in_process.eval_at_start \
--training.open_loop_eval_in_process.sample_mode random_samples \
...
```

| Field                          | Default | Meaning                                                                 |
| ------------------------------ | ------- | ----------------------------------------------------------------------- |
| `enabled`                      | `false` | Master switch.                                                          |
| `interval_steps`               | `1000`  | Run eval every N training steps (`on_step_end`). Named `interval_steps` rather than `eval_steps` to avoid confusion with HF Trainer's `TrainingConfig.eval_steps`. |
| `eval_at_start`                | `false` | Run a baseline eval on the *first* `on_step_end` call (i.e. after step 1's optimizer update, not at literal step 0 — the callback only subscribes to `on_step_end`). Use this to get a baseline before the regular cadence kicks in. |
| `traj_ids`                     | `[0]`   | Trajectory ids to evaluate per dataset (only used in `fixed` mode).     |
| `steps_per_traj`               | `100`   | Rollout window length (capped by trajectory length).                    |
| `action_horizon`               | `None`  | Override action horizon. `None` → `config.model.action_horizon`.        |
| `save_plot_dir`                | `None`  | Where to drop trajectory plots. `None` skips plotting (faster).         |
| `num_inference_timesteps`      | `None`  | Override action head's inference timesteps during eval.                 |
| `use_train_path`               | `false` | Also evaluate on each dataset's train path.                             |
| `dataset_indices`              | `None`  | Whitelist of `data.datasets` indices. `None` = all.                     |
| `sample_mode`                  | `'fixed'` | Sampling strategy: `'fixed'` / `'random_trajs'` / `'random_samples'`. |
| `num_random_samples`           | `32`    | Per-dataset sample count used by `random_*` modes.                      |
| `random_seed`                  | `0`     | Base seed; mixed with `global_step` so different evals draw different samples. |
| `enable_frechet`               | `true`  | Per-window Bures / 2-Wasserstein distance between empirical Gaussians of pred vs GT action sequences. Effectively free. |
| `enable_reverse_kl`            | `false` | Per-window reverse-KL of an M-sample empirical policy distribution vs `N(gt, I)`. **Multiplies eval cost by `rkl_num_samples`** since each chunk is forwarded that many times. |
| `rkl_num_samples`              | `4`     | Number of stochastic policy forward passes per chunk when `enable_reverse_kl` is on. Must be >= 2. |

**Per-trajectory reset performance.** The adapter overrides `reset()` to skip
the prompt-cache invalidation that `Gr00tN2Policy.reset` does. This is the
right call for in-process eval where multiple trajectories share a prompt —
but if you ever need to flush the prompt cache between trajs (e.g. for a
debug eval that varies the prompt), do so through the wrapped policy
directly: `adapter._policy._cached_prompt_embs = None`.

### 4.1 Sampling modes

The callback supports three sampling strategies for choosing what to evaluate.
All three drive the same partition + gather + aggregate path; the only thing
that varies is *which* `(traj_id, start_fraction)` work-units rank 0 enumerates.

* **`fixed`** (default, legacy behaviour). `traj_ids = [0, 5, 10]` × every spec,
  every roll starts at step 0. Same trajectories every eval call → low-noise
  delta between adjacent evals, but biased toward the prefix of a small,
  hand-picked subset.

* **`random_trajs`**. Per dataset, sample `num_random_samples` *unique* trajectory
  ids (without replacement) and roll each from step 0. Broader coverage of the
  dataset at the same per-traj rollout cost. `traj_ids` is ignored.

* **`random_samples`**. Per dataset, sample `num_random_samples` ``(traj_id,
  start_fraction)`` windows *with* replacement. Each window is `steps_per_traj`
  long, anchored anywhere in the trajectory (`start_step = int(start_fraction *
  max(0, traj_length - steps_per_traj))`). True IID estimator of dataset-wide
  MSE/MAE. Recommended for monitoring training progress as a near-unbiased
  proxy for offline test-set error.

**Determinism guarantee.** For all modes the effective seed at each eval call
is `random_seed XOR global_step`, so:

* every rank inside a single eval call sees the *same* seed and draws the
  *same* sample list (this is what makes the partition + gather IID-equivalent
  to single-rank eval); and
* successive eval calls draw *different* sample lists (so the metric is an
  unbiased estimator over training time, not just a noisy point estimate at
  fixed coordinates).

If you want a single fixed sample set across the run for low-noise diffing,
set `random_seed` to a fixed value and disable seed-mixing by patching
`_seed_for_call` (or set `eval_steps` such that all your runs hit the same
`global_step`).

> **Performance note.** The `random_*` modes pre-load every eval dataset
> on every rank at the start of `_run_eval` (to compute `len(loader)`) — the
> first eval call therefore loads all loaders on all ranks, while `fixed`
> only loads the ones a rank is actually assigned. Subsequent calls reuse
> `_loader_cache` and `_spec_length_cache`, so the cost is paid once.

### 4.2 Distribution metrics (Fréchet + Reverse KL)

In addition to the always-on per-step `mse` / `mae`, two trajectory-distribution
metrics can be enabled per-spec:

* **Fréchet (Bures / 2-Wasserstein) over time** — the 2-Wasserstein distance
  between the empirical Gaussian fits of the predicted vs ground-truth
  ``(T, D)`` action sequences for each window. Captures structural drift
  beyond pointwise residual: a phase-shifted but otherwise valid trajectory
  has high MSE but low Fréchet, while a pointwise-correct prediction with
  the wrong covariance structure has low MSE but high Fréchet. Implemented
  in :func:`groot.core.training.open_loop.metrics.frechet_gaussian_over_time`.
  *Cost*: post-process of the existing rollout — effectively free. Default on.

* **Reverse KL (M-sample empirical KDE → ``N(gt, I)``)** — runs ``M``
  independent stochastic policy forward passes per chunk (instead of 1),
  treats the resulting ``(M, T, D)`` tensor as a Gaussian-KDE estimate of
  the policy distribution at that observation, and reports its reverse KL
  against ``N(gt, I)``. Penalises both bias and over-confidence; useful as
  a sanity check that the policy retains diversity at evaluation steps
  rather than collapsing to a point estimate. Implemented in
  :func:`groot.core.training.open_loop.metrics.reverse_kl_from_samples`.
  *Cost*: ``rkl_num_samples × `` the eval forward time, since each chunk
  is forwarded that many times. Default off; opt in when you specifically
  need a calibration / mode-coverage signal.

Both are reported per-window, then aggregated per-spec by simple mean (same
gather + aggregate path as MSE / MAE), and logged under
``open_loop/.../frechet`` and ``open_loop/.../reverse_kl`` (plus the
cross-spec means as ``open_loop/avg_frechet`` / ``open_loop/avg_reverse_kl``).
Failures of either metric on a single window (e.g. an ``eigh`` blow-up on a
degenerate covariance) are caught and skipped — the rest of the window's
metrics still report.

**Reverse-KL determinism caveat.** Because RKL requires ``M`` *stochastic*
forward passes, its per-window numbers are noisy by construction (the IID
property only holds when every rank draws from the same ``random_seed XOR
global_step`` work-unit list — the *within-window* stochasticity is not
seeded across ranks). The IID test runbook (Tier 3) exercises only MSE /
MAE / Fréchet for this reason. If you need IID-stable reverse-KL across
world sizes, you'll need to also seed the model-side RNG per (rank,
work-unit) — out of scope for this callback today.

---

## 5. wandb / log format

Per-spec keys (one per evaluated dataset). The `mse` / `mae` keys are always
emitted; `frechet` is emitted when `enable_frechet=true` (default); `reverse_kl`
is emitted when `enable_reverse_kl=true`.

```
open_loop/{embodiment_tag}/{eval_type}/{spec_name}/mse
open_loop/{embodiment_tag}/{eval_type}/{spec_name}/mae
open_loop/{embodiment_tag}/{eval_type}/{spec_name}/frechet      (if enable_frechet)
open_loop/{embodiment_tag}/{eval_type}/{spec_name}/reverse_kl   (if enable_reverse_kl)
open_loop/{embodiment_tag}/{eval_type}/{spec_name}/num_trajectories
```

Aggregate keys (mean over specs). The set of `avg_*` keys mirrors the per-spec
keys actually present, so toggling `enable_frechet` / `enable_reverse_kl`
adds / removes both surfaces at once:

```
open_loop/avg_mse
open_loop/avg_mae
open_loop/avg_frechet      (if enable_frechet)
open_loop/avg_reverse_kl   (if enable_reverse_kl)
open_loop/eval_step
open_loop/eval_wall_time_s
```

`num_trajectories` reflects the actual number of evaluated work-units, so:

* `fixed` → `len(traj_ids)`
* `random_trajs` → `min(num_random_samples, len(loader))`
* `random_samples` → `num_random_samples` (with replacement)

Routed through `Trainer.log`, so any HF integration callback (wandb,
tensorboard, mlflow, ...) that's attached to the trainer picks them up
automatically.

**Plot files.** When `save_plot_dir` is set, each window's plot is written to
`<dir>/step_<global_step>/<embodiment>__<eval_type>__<spec>/traj_<id>.png`
for windows starting at step 0, or `traj_<id>__start_<step>.png` for non-zero
start steps (so multiple windows of the same traj from `random_samples` mode
don't clobber each other).

---

## 6. Compatibility matrix

| Setup                    | Supported | Notes                                                                                  |
| ------------------------ | --------- | -------------------------------------------------------------------------------------- |
| Single-GPU               | ✅        | `world_size=1` path; same code, no collectives.                                        |
| DDP                      | ✅        | Round-robin partition + `all_gather_object`. **Tier-3 verified**: n=2/4/8 produce bit-identical metrics on N1.7. |
| DeepSpeed ZeRO-1 / ZeRO-2| ✅        | Each rank holds full params; inference works as-is.                                    |
| DeepSpeed ZeRO-3         | ⚠️         | Params are sharded; need to wrap inference in `deepspeed.zero.GatheredParameters(...)` (not currently implemented). |
| FSDP                     | ⚠️         | Same caveat as ZeRO-3 for sharded modes.                                               |
| `torch.compile`          | ✅        | `LivePolicyAdapter` reuses the live model — compilation already done in training.      |
| Live model in fp32       | ✅        | Adapter wraps forward in `torch.amp.autocast(bfloat16)` so FlashAttention works regardless of `--model.load_bf16`. |

| Model              | Forward call           | Action head?                                  | Concrete policy class |
|--------------------|------------------------|-----------------------------------------------|-----------------------|
| N1.6, N1.7         | `model.get_action(...)`| yes (`action_head` attr)                      | `Gr00tPolicy`         |
| N1.6 RL-wrapped    | `model.get_action(...)`| yes via `wrapped_model.action_head`           | `Gr00tPolicy`         |
| N2                 | `model.predict(...)`   | yes (`has_action_head=True` when `action_cfg` is set in `config`) | `Gr00tN2Policy` |

The adapter auto-detects via `_classify_model` in `live_policy.py`. The N2 row's
"has action head" field is True because `Gr00tN2.has_action_head` is set to True
when `config.action_cfg is not None` (`groot/core/training/models/n2/model.py`).
The discriminator requires *both* `predict` and `has_action_head=True` for N2,
so a future N2-with-no-head variant would classify as N1, which is the correct
fallback (no action head = nothing to dispatch on).

---

## 7. Running the tests

There are three tiers of tests, in increasing fidelity / cost:

| Tier | File / runbook                                                 | What it tests                                                                            | Runtime  | Needs GPU |
| ---- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------- | --------- |
| 1    | `test_in_process_eval_distributed.py`                          | Pure-CPU partition + merge + aggregate math (the static helpers).                        | ~30 s    | no        |
| 1    | `test_metrics.py`                                              | Pure-math unit tests for Fréchet / Reverse-KL.                                           | ~5 s     | no        |
| 1    | `test_live_policy_adapter.py`                                  | CPU unit tests for `_classify_model` / N1+N2 dispatch / `close()` / `reset()` lifecycle. | ~25 s    | no        |
| 1    | `test_open_loop_eval_config.py`                                | Nested-dataclass coercion + validation + hard-break on legacy flat names.                | ~20 s    | no        |
| 1    | `test_experiment_open_loop_wiring.py`                          | `experiment.py` reads from nested config paths without `AttributeError`.                 | ~20 s    | no        |
| 2    | `test_in_process_eval_e2e.py`                                  | Full callback under real Gloo collectives, with the model + dataset stubbed out.         | ~45 s    | no        |
| 3a   | `debug_n17_backpack_openloop_smoke.sh` + IID sweep             | Full N1.7 pipeline; verifies world-size IID across n=2/4/8.                              | ~10 min  | yes       |
| 3b   | `debug_n2_backpack_openloop_smoke.sh`                          | Full N2 pipeline; verifies `build_live_policy` auto-routes to `Gr00tN2Policy._get_action`.| ~5 min   | yes       |

Run tiers 1 + 2 in CI; run tier 3 manually before merging changes that touch
the model forward path or the collective code.

Quick single-command tier 1+2 sweep:

```bash
cd /mnt/amlfs-07/shared/rpunamiya/gr00t
uv run pytest \
  tests/groot/core/training/open_loop/ \
  tests/groot/core/training/configs/test_open_loop_eval_config.py \
  tests/groot/core/training/experiment/test_experiment_open_loop_wiring.py \
  -v
```

Reference run on `gr00t-dev`: 110 tests, ~85 s total.

### Tier 1 — Unit test: IID invariant for the static helpers (CPU-only)

```bash
cd /mnt/amlfs-07/shared/rpunamiya/gr00t
PYTHONPATH=. pytest \
  tests/groot/core/training/open_loop/test_in_process_eval_distributed.py -v
```

What it checks:

* `_get_assignments` partitions every `(spec_idx, traj_id)` pair to
  exactly one rank, and the union over ranks equals the full set.
* For a deterministic eval function `eval_fn(spec_idx, traj_id) -> (mse, mae)`,
  simulating the per-rank partition + merge for `world_size ∈ {1, 2, 4, 8}`
  produces **bit-equivalent** per-spec MSE / MAE within `atol=1e-9`.
* Empty-rank case: `world_size > total_pairs` produces the same
  aggregate as `world_size = 1`.

### Tier 2 — End-to-end integration: IID across real Gloo workers (CPU-only, no model)

```bash
cd /mnt/amlfs-07/shared/rpunamiya/gr00t
PYTHONPATH=. pytest \
  tests/groot/core/training/open_loop/test_in_process_eval_e2e.py -v
```

What it does:

* Spawns `world_size ∈ {1, 2, 4}` Python subprocesses with
  `multiprocessing.get_context("spawn")`.
* Each subprocess initializes a real Gloo CPU process group on
  `127.0.0.1:<free_port>`.
* Each subprocess injects three fake modules into `sys.modules` *before*
  `_run_eval` runs:
  - `groot.core.training.open_loop.live_policy.LiveGr00tPolicy` →
    no-op stub.
  - `groot.core.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader` →
    stub that encodes `spec_idx` in `dataset_path`.
  - `groot.core.training.open_loop.open_loop_eval.evaluate_single_trajectory` →
    deterministic closed-form `(mse, mae)` from `(spec_idx, traj_id)`.
* Drives the *real* `InProcessOpenLoopEvalCallback.on_step_end` →
  `_run_eval`, including `model.eval()` / `model.train()` toggling, the
  `dist.all_gather_object` collective, and the rank-0 `Trainer.log` capture.
* Asserts that the per-spec `mse` / `mae` / `num_trajectories` and the
  aggregate `avg_mse` / `avg_mae` captured on rank 0 are **identical**
  (`atol=1e-9`) for `world_size ∈ {1, 2, 4}`.

What it does **not** test:

* The actual model forward pass (model is a `torch.nn.Linear(2, 2)` placeholder).
* `LeRobotEpisodeLoader` data loading (replaced by the stub).
* `decode_action` / `evaluate_single_trajectory` math (replaced by the stub).

If this tier fails but tier 1 passes, the bug is in the *orchestration*
layer of `_run_eval` — i.e. one of: `model.eval()` toggling, the gather
collective, the rank-0 log path, or the lazy import wiring.

### Tier 3 — Real multi-GPU runbook: verify IID on actual hardware

The unit tests above prove the *partition + merge* math is IID. The
runbook below verifies that the *full pipeline* (model forward +
collectives + log) also produces the same metrics across world sizes,
which catches issues like:

* hidden non-determinism in `model.get_action` / `model.predict`,
* dropout / batchnorm leaking when `model.eval()` isn't fully toggled,
* `processor` training mode flipping per rank,
* `pt_elastic` cleanup hangs after the eval prints (now fixed by
  `dist.destroy_process_group()` in `launch_train.py`).

There are two ready-made smoke scripts that pin all the eval-relevant
flags and only vary `NUM_GPUS` — one per model family:

```
groot/core/training/scripts/n17/debug/debug_n17_backpack_openloop_smoke.sh
groot/core/training/scripts/n2/debug/debug_n2_backpack_openloop_smoke.sh
```

Both wrap `launch_train.py` against the local `amlfs-*` lustre layout
on the gr00t-dev box, enable `open_loop_eval_in_process`, set
`eval_at_start=true` and `use_train_path=true` so the dataset's training
path doubles as the eval target (no yaml edit needed). Both scripts wrap
the python invocation in `timeout ${SMOKE_TIMEOUT:-600}` so a stuck run
won't block CI.

#### Tier 3a — N1.7 IID sweep across world sizes

```bash
cd /mnt/amlfs-07/shared/rpunamiya/gr00t
OUT_BASE=/tmp/openloop_iid_$(date +%s)
mkdir -p $OUT_BASE

for n in 1 2 4 8; do
  NUM_GPUS=$n \
  OUT=$OUT_BASE/n${n} \
  MAX_STEPS=1 \
  SAMPLE_MODE=fixed \
    bash groot/core/training/scripts/n17/debug/debug_n17_backpack_openloop_smoke.sh \
    2>&1 | tee $OUT_BASE/n${n}.log
done

# Compare the per-spec metrics emitted by the in-process callback.
# The callback routes through Trainer.log; the captured dict shows up as
# a `{'open_loop/.../mse': X, 'open_loop/.../mae': Y, ...}` line in the log.
# Pull `open_loop/avg_*` out of each run and diff:
python3 - <<'PY'
import ast, re, sys
runs = {}
for n in (1, 2, 4, 8):
    text = open(f"$OUT_BASE/n{n}.log").read()
    # last `open_loop/avg_mse` block in the log is the eval-at-start metric
    m = re.findall(r"\{[^}]*open_loop/avg_mse[^}]*\}", text)
    if not m:
        print(f"n={n}: no metrics found"); continue
    d = ast.literal_eval(m[-1])
    runs[n] = {k: d[k] for k in d if k.startswith("open_loop/avg")}
for n, d in runs.items():
    print(f"n={n}: {d}")
PY
```

**Pass criterion (Tier 3a):**

* `NUM_GPUS ∈ {2, 4, 8}` produce **bit-identical** `open_loop/avg_*` values
  (verified to 10 decimal places in the reference run on `gr00t-dev`).
* `NUM_GPUS=1` is allowed to differ from `NUM_GPUS≥2` because num_gpus=1
  doesn't initialize DDP / `DistributedSampler` — the optimizer step at
  step 1 sees a different data ordering, so the model state at eval time
  isn't directly comparable. The CPU tier-1 / tier-2 tests already pin
  the IID property of the eval pipeline at *fixed model state*; this
  Tier-3 sweep stress-tests the wiring under DDP.

Reference numbers for the `n15_thirdstage_checkpoint-200000_patched`
checkpoint, `MAX_STEPS=1`, `SAMPLE_MODE=fixed`, `traj_ids=[0]`:

| `NUM_GPUS` | `avg_mse`       | `avg_mae`       | `avg_frechet`   |
|-----------|-----------------|-----------------|-----------------|
| 1         | 0.0461563902    | 0.1365263726    | 0.8911988139    |
| 2, 4, 8   | 0.0459027194    | 0.1365584573    | 0.8905599713    |

If you see drift among `NUM_GPUS ∈ {2, 4, 8}` larger than `1e-7`:

| Symptom                                                                                    | Likely cause                                                                                       |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `n2/n4/n8` differ from each other in a structured way                                      | `model.eval()` not toggled — dropout active during eval.                                           |
| `n4` and `n8` agree but differ from `n2`                                                   | Round-robin or merge order is dependent on `world_size`. Check `_get_assignments`.                 |
| Random differences run-to-run on the *same* world size                                     | A non-deterministic op in the model (e.g. `torch.use_deterministic_algorithms` not set).           |
| `n=N` log has fewer trajectories than expected                                             | A rank is silently failing in `_run_eval` — check per-rank `[rank=N]` log lines.                   |
| `RuntimeError: FlashAttention only support fp16 and bf16`                                  | The autocast wrap in `LivePolicyAdapter._get_action` was removed or is bypassed. Restore it.       |
| `pt_elastic` hangs for minutes after metrics print                                         | `destroy_process_group()` was removed from `launch_train.py`'s `finally` block. Restore it.        |

#### Tier 3b — N2 single-run smoke

The N2 path runs `model.predict(...)` rather than `model.get_action(...)`,
so it exercises a different code path through `Gr00tN2Policy._get_action`.
The smoke verifies the auto-dispatch in `build_live_policy` actually picks
the N2 factory and that `_make_n2_concrete` populates every attribute
`Gr00tN2Policy._get_action` reads.

```bash
cd /mnt/amlfs-07/shared/rpunamiya/gr00t
NUM_GPUS=2 \
OUT=/tmp/n2_openloop_smoke_$(date +%s) \
MAX_STEPS=1 \
SAMPLE_MODE=fixed \
NUM_RANDOM_SAMPLES=2 \
STEPS_PER_TRAJ=25 \
SMOKE_TIMEOUT=900 \
  bash groot/core/training/scripts/n2/debug/debug_n2_backpack_openloop_smoke.sh \
  2>&1 | tee /tmp/n2_openloop_smoke.log
```

The script's defaults point at:
* `N2_YAML=groot/core/training/configs/experiments/n2/backpack/backpack_200traj_hist.yaml`
* `N2_CKPT=/mnt/amlfs-07/shared/checkpoints/n2_sonic_latent_v2_1776454917/`
* `N2_TOKENIZER=/mnt/amlfs-02/shared/ckpts/cosmos25/cosmos_reason1/`
* `N2_DIFFUSION=/mnt/amlfs-02/shared/ckpts/cosmos25/Cosmos-Predict2.5-2B-diffusers/transformer`
* `N2_VAE=/mnt/amlfs-02/shared/ckpts/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth`

Override any of those env vars to point at your local N2 setup.

**Pass criterion (Tier 3b):** stdout shows a non-NaN
`open_loop/avg_mse` block for the configured embodiment/spec.
The N2-specific `[Gr00tN2Policy] preprocess=… predict=… postprocess=…
total=…` log line confirms the predict path was hit (rather than falling
through to the N1 `model.get_action` branch).

> **Known wart on H100.** The N2 smoke currently fails on `gr00t-dev` (H100)
> at the *training-step* forward — `flex_attention.create_block_mask` raises
> a CUDA device-side assert inside `n2/components/sequence_packing.py`. This
> is independent of the in-process eval (the crash happens before
> `on_step_end` fires), and the N2 dispatch logic itself is covered by the
> CPU unit tests in `test_live_policy_adapter.py` (`test_classify_n2_*`,
> `test_make_n2_concrete_*`). When the N2 training-step issue is resolved
> on H100 — or when running on GB200 where the N2 ckpts were trained — the
> smoke will produce the expected `open_loop/avg_mse` block. Cross-reference
> the N2 single-node runbook (`debug_n2_single_node.sh`) for parallel
> evidence: if that fails, the smoke will too.

### Sanity smoke test (no checkpoint, optional)

You can also exercise just the IID tests (tiers 1 + 2) in CI:

```bash
PYTHONPATH=. pytest tests/groot/core/training/open_loop -v -k iid
```

---

## 8. Common issues / FAQ

* **Q: My model crashes at the first eval call with
  `TypeError: get_action() got an unexpected keyword argument 'options'`.**
  A: Older N1.7 model `get_action(self, inputs, rtc_params=None, **kwargs)`
  doesn't accept `options=`. The shared `Gr00tPolicy._get_action` calls
  `self.model.get_action(**collated_inputs, options=options)`. In the
  in-process callback this is now handled automatically by
  `build_live_policy`'s auto-dispatch — the N2 path goes through
  `Gr00tN2Policy._get_action` which calls `model.predict(...)`, so the
  `options=` kwarg only matters for N1.x models. If you still hit it
  (older N1.7 weights), drop the `options=` kwarg in
  `Gr00tPolicy._get_action` or upgrade the model class.

* **Q: Adapter raises `TypeError: Cannot dispatch LivePolicyAdapter`.**
  A: The classifier didn't see N1 or N2 surface on your live model. Check
  that the unwrapped model has `predict + has_action_head` (N2) or
  `action_head` / `wrapped_model` (N1). If you've added a third model
  family, add a new entry to `_CONCRETE_POLICY_FACTORIES` in
  `live_policy.py` with the appropriate constructor closure.

* **Q: `RuntimeError: FlashAttention only support fp16 and bf16 data type`
  inside `evaluate_single_trajectory` / `siglip` attention.**
  A: The autocast wrap in `LivePolicyAdapter._get_action` was removed or
  bypassed. The live training model may be fp32 (e.g. when started from
  a `--model.no-load_bf16` run); `Gr00tPolicy._get_action` casts inputs to
  bf16 but FlashAttention also requires bf16 weights. The adapter wraps
  the forward in `torch.amp.autocast(dtype=torch.bfloat16)` to make the
  fp32 model handle the bf16 inputs on the fly. Restore the autocast
  context in `_get_action`.

* **Q: After eval prints metrics, `torch.distributed.run` hangs forever.**
  A: This is the `pt_elastic` cleanup hang — the workers print
  `WARNING: destroy_process_group() was not called before program exit`
  and the elastic launcher's rendezvous health-check never resolves.
  `launch_train.py` now wraps `run(config)` in a `try/finally` that calls
  `dist.destroy_process_group()`. If you regress that, the smoke
  scripts' `timeout ${SMOKE_TIMEOUT:-600}` wrapper will SIGTERM the run
  rather than hang CI for hours.

* **Q: The first eval call takes ~2 minutes.**
  A: Cold start cost — `LeRobotEpisodeLoader` is parsing metadata + the
  video backend (`torchcodec`) is warming up. Subsequent eval calls
  reuse the cached loader. To shorten the first call, set
  `open_loop_eval_in_process.traj_ids` to one trajectory and
  `open_loop_eval_in_process.steps_per_traj` to ~50.

* **Q: Plots aren't appearing in wandb.**
  A: Plots are written to `save_plot_dir`. Logging the *files* to wandb
  is not implemented (the legacy on-save callback does this for the
  `dream0` policy_type only). To wire it up, replace the
  `evaluate_single_trajectory` call site in `_run_eval` with a wrapper
  that also calls `wandb.log({...: wandb.Image(plot_path)})` on rank 0.

* **Q: Why does the in-process callback auto-detect, while the legacy
  callback uses `policy_type: 'gr00t' | 'dream0'`?**
  A: The two paths have different inputs. The legacy callback runs
  `groot.control` ZMQ servers and needs to know which client class to
  instantiate — that's a runtime decision the user makes. The in-process
  callback has the live `nn.Module` in hand, so duck-typing it is
  strictly more information than re-asserting in YAML. Auto-detect is
  intentionally inconsistent with the legacy path.

* **Q: Why is this not just using the existing `OpenLoopEvalCallback`
  in `experiment/utils.py`?**
  A: That one (a) lazily imports `groot.control` /
  `groot.vla.omni.policy` (not in standalone `groot.core` installs),
  (b) reloads the model from a checkpoint and starts a ZMQ policy
  server (heavy), and (c) only runs on rank 0. The in-process variant
  fixes all three.

---

## 9. Future improvements (good first PRs)

* DeepSpeed ZeRO-3 support (wrap inference in
  `deepspeed.zero.GatheredParameters([param_list])`).
* Per-action-key MSE / MAE breakdown (the legacy callback in commit
  `6b206188` did this — useful for catching gripper-channel regressions).
* Wandb plot upload (see FAQ).
* Per-key reduction of bf16 → fp32 *before* averaging across ranks to
  tighten IID equality.
