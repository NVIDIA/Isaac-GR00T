# Agent Instructions

This repository is optimized for local OSS post-training and open-loop
evaluation.

## Default Agents and Skills

Use these three OSS agent definitions as the defaults for this package:

- [`autoresearch`](.agents/autoresearch/autoresearch.toml) for model-driven
  research and OSS post-training/evaluation. Its configured skills are
  [`insights`](skills/insights/SKILL.md),
  [`evaluation`](skills/evaluation/SKILL.md), and
  [`post-train`](skills/post-train/SKILL.md).
- [`autodebug`](.agents/autodebug/autodebug.toml) for debugging repository
  issues. Its configured skill is [`insights`](skills/insights/SKILL.md).
- [`faq`](.agents/faq/faq.toml) for concise, source-grounded documentation and
  FAQ answers. Its configured skill is
  [`insights`](skills/insights/SKILL.md).

Treat each agent TOML file as the source of truth for its enabled skills, and
follow the task-specific local post-training and evaluation rules below.

## Default Action

When the user asks to launch, inspect, retry, or debug OSS post-training, use the
`post-train` Codex skill.

When the user asks to evaluate an existing checkpoint, use the `evaluation`
Codex skill. When the user asks for post-train and evaluation
together, use `uv run python -m groot.oss train-eval` so both jobs run in one
local process chain.

Do not use OSMO, GitLab staging pipelines, or the `golden-run` skill for local
OSS post-train work.

## Shared CLI Command

Run from the OSS package root:

```bash
uv run python -m groot.oss train n2-minwam-sim
```

Run local post-train followed by evaluation:

```bash
uv run python -m groot.oss train-eval n2-minwam-sim
```

Run evaluation for an existing checkpoint:

```bash
uv run python -m groot.oss eval n2-minwam-sim --checkpoint <hf-uri-or-local-path>
```

If the user does not specify a variant, use the default configured by the skill.

## Required Environment

Local training requires model dependency paths:

```bash
export COSMOS_MODEL_PATH=<local-cosmos-tokenizer-path>
export DIFFUSION_TRANSFORMER_PATH=<local-transformer-path>
export VAE_PATH=<local-vae-path>
```

Set `HF_TOKEN` or a Hugging Face token alias if the configured artifacts require
authentication.

## Expected Agent Flow

1. Confirm the user wants a local run and has the required local paths or HF access.
2. Use `--dry-run` first when setup is unclear.
3. Run the shared `python -m groot.oss` CLI from the OSS package root.
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
