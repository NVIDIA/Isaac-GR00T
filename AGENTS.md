# Agent Instructions

This repository is optimized for Codex-driven OSS post-training and open-loop
evaluation.

## Default Action

When the user asks to launch, inspect, retry, or debug OSS post-training, use the
`post-train` Codex skill.

When the user asks to evaluate an existing checkpoint, use the `evaluation`
Codex skill. When the user asks for post-train and evaluation
together, use the `post-train` skill with `--eval` so both jobs run in one
pipeline.

Do not start from the training Python entry point. Do not use the `golden-run`
skill for OSS post-train work.

## Skill Command

Run from the repository root:

```bash
uv run python .codex/skills/post-train/scripts/post_train.py n2-minwam-sim-h100 high
```

Run post-train followed by evaluation:

```bash
uv run python .codex/skills/post-train/scripts/post_train.py n2-minwam-sim-h100 high --eval
```

Run evaluation for an existing checkpoint:

```bash
uv run python .codex/skills/evaluation/scripts/evaluation.py n2-minwam-sim-h100 high --checkpoint <hf-uri-or-shared-path>
```

Use `low`, `normal`, or `high` as the optional priority argument. If the user does
not specify a variant, use the default configured by the skill.

## Required Environment

For local Codex runs, set GitLab connection variables before triggering:

```bash
export GITLAB_HOST=gitlab-master.nvidia.com
export GITLAB_PROJECT_PATH=gr00t-release/isaac-gr00t-n2
export GITLAB_TOKEN=<token>
```

Look in `~/.credentials` before asking the user for a token.

## Expected Agent Flow

1. Confirm the current branch is the intended branch.
2. Trigger the post-train or evaluation skill helper.
3. Report the pipeline URL and whether an MR comment was posted.
4. Monitor the staging job when the user asks for status or failure triage.
5. If the job fails, summarize the first actionable failure from the job trace.

## Source of Truth

This repository is a materialized copy of upstream `gr00t/oss`.

- Make source changes in the upstream GR00T OSS tree first.
- Sync the materialized copy after upstream changes are committed.
- Do not create Isaac-only behavior unless the user explicitly asks for it.

## Smoke Test

`test.oss` is the CI smoke test for the skill path. It should stay short and only
prove that the post-train and evaluation workflows can launch and reach a few
steps.
