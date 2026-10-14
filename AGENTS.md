# Agent Instructions

This repository supports OSS GR00T 2 post-training, open-loop and simulator
evaluation, reproducible benchmarks, policy serving, and TensorRT deployment.
Start with [`README.md`](README.md). The
[documented feature checklist](docs/agentification.md) maps user goals to guides,
entrypoints, required inputs, success checks, and validation limits.

## Default Agents and Skills

Use these two OSS agent definitions as the defaults for this package:

- [`autoresearch`](.agents/autoresearch.toml) for model-driven research and OSS
  post-training/evaluation. Its configured skills are
  [`onboarding`](skills/onboarding/SKILL.md) and
  [`autoresearch`](skills/autoresearch/SKILL.md).
- [`autodebug`](.agents/autodebug.toml) for debugging repository issues. Its
  configured skill is [`autodebug`](skills/autodebug/SKILL.md).

Treat each agent TOML file as the source of truth for its enabled skills, and
follow the task-specific local post-training and evaluation rules below.

## Default Action

When the user asks for a repo tour, getting-started help, setup, training,
evaluation, benchmarks, serving, or deployment, use the `onboarding` skill.
For general onboarding or a menu request, use
[`onboarding`](skills/onboarding/SKILL.md) to generate a CLI-style chat
menu by reading the current `groot/infra/oss.py` parser source as text.
Menu discovery and explanations do not run Python, uv, or runtime `--help`. Keep the
actual subcommand names; direct task requests bypass the menu. Show `debug`
and `research` separately as agent-workflow choices that route to AutoDebug and
AutoResearch with the existing context and execution scope.
For a tour, explain features and help select a next step without requiring a
dataset, credentials, or GPU. For execution, read the selected feature guide;
onboarding does not require a full training smoke before using another feature.

Use the `autoresearch` agent to improve an
existing post-training recipe and `autodebug` to diagnose a repository failure.

Do not use OSMO, GitLab staging pipelines, or the `golden-run` skill for local
OSS post-train work.

## Shared CLI Command

Run from the OSS package root:

```bash
uv run python -m groot.infra.oss train \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag>
```

Run local post-train followed by evaluation:

```bash
uv run python -m groot.infra.oss train-eval \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag>
```

Run evaluation for an existing checkpoint:

```bash
uv run python -m groot.infra.oss eval \
  --checkpoint <hf-uri-or-local-path> \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag>
```

Ordinary local training defaults to `n2_oss_pretrain`; benchmark recipes select
their own experiment and data settings. Do not pass an
experiment name positionally.

## Required Inputs

Local training requires a LeRobot dataset supplied with `--dataset-path` (or an
explicit `data.datasets.*.dataset_paths` override via `--set`). Model
dependencies use the default `hf://model/...` URIs; set `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` if those artifacts require
authentication.

## Expected Agent Flow

1. Identify the selected workflow and its required inputs from the feature
   guide; only training and open-loop evaluation require a dataset.
2. Use `--dry-run` first when the selected command supports it and setup is
   unclear; `build-trt` requires real target hardware and has no dry-run.
3. Use the `onboarding` skill to run the shared `python -m groot.infra.oss` CLI
   from the OSS package root.
4. Report the local command, output checkpoint or eval JSON path, and first actionable failure if one occurs.

## Source of Truth

This repository is a materialized copy of upstream `gr00t/oss`.

- Make source changes in the upstream GR00T OSS tree first.
- Sync the materialized copy after upstream changes are committed.
- Do not create Isaac-only behavior unless the user explicitly asks for it.

## Smoke Test

Local smoke checks should stay short and prove that the post-train and
evaluation entrypoints build the expected local commands before running a full
training or evaluation job.
