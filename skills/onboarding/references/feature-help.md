<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# Repo Tour and Feature Help

### Explain and orient

1. Read `README.md` and the [feature checklist](../../../docs/agentification.md).
   Explain that GR00T 2 is a robot policy: post-training adapts it to recorded
   demonstrations, open-loop evaluation compares predictions with recorded
   actions, closed-loop evaluation measures success in a simulator, and policy
   serving exposes inference to a client. TensorRT accelerates compatible
   inference workloads. AutoResearch searches recipes; AutoDebug investigates
   failures. Do not equate open-loop error with real robot task success.
2. Give a short map: `docs/` for setup and walkthroughs, `examples/` for dataset
   preparation, `groot/infra/oss.py` for the public CLI,
   `groot/infra/benchmark/specs/` for pinned benchmark recipes,
   `scripts/deployment/` for TensorRT and Thor, and `skills/` for agent workflows.
3. Tailor the next step to the user's goal and experience. If their goal is
   unknown, explain the choices and ask which they want to try. Hardware, tokens,
   and a dataset are not prerequisites for a tour. Do not print token values.
4. Finish with a relevant guide, a concrete example prompt or command, and what
   its successful result would look like. An explanation-only request is complete
   when the user has that orientation; do not launch the minimal example.

### Help use a feature

Read the guide for the selected feature from the checklist. Follow its actual
CLI and platform requirements rather than inventing a new execution wrapper.
Read the selected command's parser helpers, defaults, and validators in
`groot/infra/oss.py` and any imported parser modules to explain flags and draft
commands. This requires only reading source text. Do not run Python, uv, or
runtime `--help` just to browse or explain a feature. Check the runtime only
for requested setup, actual dry-run resolution, or execution.

- Installation: select training, inference, optional TensorRT, or Thor from
  `docs/installation.md`; check the user's OS and GPU before installing.
- Dataset preparation or a full first-run smoke: use `docs/minimal_example.md`
  and the [local workflow reference](local-workflows.md). Keep the user's own dataset unchanged.
- Training, open-loop evaluation, train-eval, or serving: use the [local workflow reference](local-workflows.md)
  and the input/output contracts in `README.md`.
- Closed-loop evaluation: use `README.md` and the selected backend's setup docs.
  Identify the simulator, suite, checkpoint, embodiment, episode count and limits,
  and output directory. Distinguish local Ray from a user-provided cluster.
- Reproducible benchmarks: use `train-eval --benchmark-recipe` and
  `docs/libero_family_benchmarks.md`. The recipe owns experiment settings; do not
  duplicate them with ordinary train/eval flags. Dry-run the selected recipe.
- TensorRT or Thor: read `scripts/deployment/README.md` and the platform's
  installation instructions. Match checkpoint, codec, embodiment, denoising
  steps, and target GPU. Keep engine caches outside the checkout. `build-trt`
  has no dry-run; inspecting help is not a successful engine build. CUDA graph
  serving requires the documented real-shaped client warmup.
- AutoResearch, baseline preparation, visualization, or AutoDebug: route to the
  corresponding skill in `skills/README.md`; do not substitute a GPU smoke run.

For execution, report the exact command, resulting artifacts or endpoint, and
which success checks ran. Separate command resolution from real execution.
If the host lacks required hardware or access, finish the available preparation
and report the specific remaining requirement; do not claim runtime success.

## Local execution workflow

1. Work from the OSS package root, the directory containing `pyproject.toml`.
2. Ordinary local training defaults to `n2_oss_pretrain`. Do not pass an
   experiment name positionally. For benchmark recipes, let the recipe select
   the experiment and data configuration.
3. Use a local LeRobot-compatible dataset path for v1. If the user gives a
   Hugging Face dataset ID, ask for the local materialized dataset path unless
   the CLI adds first-class dataset download flags.
4. Use `--dry-run` first for commands that support it unless the user asks to
   launch immediately. For `build-trt`, inspect help and the checkpoint contract
   before the real build; no dry-run is supported.
5. Report the exact command, artifacts or endpoint, and the first actionable
   failure if the command fails.

Read [local-workflows.md](local-workflows.md) when using
local train/eval/serve commands or the full first-run execution smoke. It retains
the detailed walkthrough and command examples; skip it for a repo tour.
