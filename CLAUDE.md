# Agent Instructions

This repository is optimized for local OSS post-training and open-loop
evaluation.

## Default Agents and Skills

Use these three OSS agent definitions as the defaults for this package:

- [`autoresearch`](.agents/autoresearch.toml) for model-driven research and OSS
  post-training/evaluation. Its configured skills are
  [`insights`](skills/insights/SKILL.md),
  [`onboarding`](skills/onboarding/SKILL.md), and
  [`autoresearch`](skills/autoresearch/SKILL.md).
- [`autodebug`](.agents/autodebug.toml) for debugging repository issues. Its
  configured skills are [`insights`](skills/insights/SKILL.md) and
  [`autodebug`](skills/autodebug/SKILL.md).
- [`faq`](.agents/faq.toml) for concise, source-grounded documentation and
  FAQ answers. Its configured skill is
  [`insights`](skills/insights/SKILL.md).

Treat each agent TOML file as the source of truth for its enabled skills, and
follow the task-specific local post-training and evaluation rules below.

## Default Action

When the user asks to set up OSS GR00T or locally train, evaluate, or serve a
policy, use the `onboarding` skill. Use the `autoresearch` agent to improve an
existing post-training recipe, `autodebug` to diagnose a repository failure,
and `faq` for product, support, concept, or documentation questions.

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
uv run python -m groot.infra.oss eval --checkpoint <hf-uri-or-local-path>
```

The public CLI uses its fixed `n2_oss_pretrain` experiment. Do not pass an
experiment name positionally.

## Required Inputs

Local training requires a LeRobot dataset supplied with `--dataset-path` (or an
explicit `data.datasets.*.dataset_paths` override via `--set`). Model
dependencies use the default `hf://model/...` URIs; set `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` if those artifacts require
authentication.

## Expected Agent Flow

1. Confirm the user wants a local run and has a dataset path and required HF access.
2. Use `--dry-run` first when setup is unclear.
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
