# Isaac-GR00T-N2

This repository is set up for the Codex `post-train` and `evaluation`
skills. The primary workflow is not to run training or evaluation code by hand;
it is to ask Codex to trigger and monitor OSS CI pipelines.

## Use the Skill

From the repository root, run the skill helper:

```bash
uv run python .codex/skills/post-train/scripts/post_train.py n2-minwam-sim-h100 high
```

Run post-train followed by evaluation in the same pipeline:

```bash
uv run python .codex/skills/post-train/scripts/post_train.py n2-minwam-sim-h100 high --eval
```

Run evaluation for an existing checkpoint:

```bash
uv run python .codex/skills/evaluation/scripts/evaluation.py n2-minwam-sim-h100 high --checkpoint <hf-uri-or-shared-path>
```

The first argument selects the post-train variant and compute type. The default
supported run is:

- `n2-minwam-sim-h100`

The second argument is the OSMO priority:

- `high`
- `normal`
- `low`

If you omit both arguments, the skill triggers the first configured OSS
post-train variant at high priority.

## What the Skill Does

The skill triggers a GitLab pipeline for the current branch with
`POST_TRAIN_E2E_LEAF=true`. That selects the OSS post-train staging job, which
builds or reuses the train image and launches the configured H100 post-train run.
With `--eval`, the pipeline also sets `OPEN_LOOP_EVAL_E2E_LEAF=true` and runs
evaluation against the produced `FINAL_CHECKPOINT_PATH`.

The `evaluation` skill triggers only the evaluation staging job and requires
an existing checkpoint URI or shared filesystem path.

When an open merge request exists for the current branch, the skill also posts a
comment with the triggered pipeline link.

## Required Environment

If you are outside GitLab CI, set:

```bash
export GITLAB_HOST=gitlab-master.nvidia.com
export GITLAB_PROJECT_PATH=gr00t-release/isaac-gr00t-n2
export GITLAB_TOKEN=<token>
```

If a token is needed in Codex, check `~/.credentials` first.

## Codex Prompt Examples

```text
Use the post-train skill to launch n2-minwam-sim-h100 at low priority.
```

```text
Check the latest OSS post-train pipeline for this branch and summarize failures.
```

```text
Trigger the default OSS post-train run and comment on the MR.
```

```text
Run OSS evaluation for hf://model/nvidia/GR00T-N2-MinWAM-Stack-Bowls-PostTrain@main/checkpoint-100.
```

```text
Run post-train and evaluation together for n2-minwam-sim-h100.
```

## Validation Surface

The smoke-test CI path for the post-train skill is `test.oss`. It submits
`tests/oss_workflows/n2_minwam_sim_posttrain.yaml`, which is intentionally small:
one H100 node and only a few training steps. The evaluation skill uses
`staging.evaluation_openloop.h100`, which submits
`tests/oss_workflows/n2_minwam_sim_policy_server_eval.yaml` directly and streams
the OSMO workflow logs into GitLab.

Use the smoke test to prove the skill path works. Do not treat it as a full
quality or convergence run.
