---
name: prepare-autoresearch-baseline
description: Prepare a working GR00T N2 training and evaluation workflow as an AutoResearch baseline by creating task-owned wrappers and configs, optionally running the baseline, verifying model and metric artifacts, generating autoresearch.yaml, and running preflight. Use when a user needs to set up or convert a local or remote-infrastructure baseline before launching AutoResearch.
---

# Prepare AutoResearch Baseline

Prepare and validate the inputs consumed by `skills/autoresearch`. Do not launch
or resume AutoResearch with this skill.

## Safety and scope

- Read `skills/autoresearch/README.md`, its generic YAML template, and the
  closest example before creating task files.
- Start from a train/eval workflow that already works or that the user can
  explain. Do not invent scheduler, simulator, robot, dataset, or scoring logic.
- Treat inspection and preflight as read-only. Run train or eval only when the
  user explicitly asks to execute or verify the baseline.
- Never silently change datasets, evaluation behavior, metric paths, branches,
  compute budgets, or existing baseline artifacts.
- Never overwrite an existing result or output directory without permission.
- Keep infrastructure submission, reattachment, polling, and artifact checks
  inside the user-owned wrappers.

## Required inputs

Collect only missing information:

- Git workspace and baseline branch or commit;
- task directory for committed wrappers and configs;
- working train and eval commands or scripts;
- training and evaluation datasets and configs;
- baseline artifact directory outside the workspace;
- result metric path and `minimize` or `maximize` direction;
- requested research ideas and must/must-not-change constraints, if any;
- whether to create files only or also run the baseline.

If no working train or eval contract exists, report what is missing and stop.

## Workflow

### 1. Inspect the existing workflow

Run from the repository root. Read:

- `skills/autoresearch/README.md`;
- `skills/autoresearch/scripts/autoresearch.yaml`;
- `skills/autoresearch/examples/yam_stack_bowl` for local/open-loop work, or
  `skills/autoresearch/examples/genie_sim_osmo` for remote/closed-loop work.

Inspect the user's commands, configs, expected artifacts, Git revision, and
smoke-test cost. Preserve working behavior and paths unless the user requests a
change.

### 2. Create the smallest task-owned interface

Create only missing files. Prefer this layout when the user has no established
layout:

```text
task_dir/
├── train_baseline.sh
├── eval_baseline.sh
├── train_config.yaml
├── eval_config.yaml
└── autoresearch.yaml
```

Reuse existing scripts and configs instead of copying them when possible.
Place any config that AutoResearch may edit inside the Git workspace and commit
it. Keep wrapper scripts, datasets, evaluation code, and evaluation config
frozen. Put only intended training-side code and config in `editable_paths`.
When the training config is editable, declare it under `locked_train_config`
with the eval dataset paths so candidates cannot change the training data,
budget, or base checkpoint, or train on evaluation data.

Adapt wrappers to the environment contract supplied by AutoResearch:

- Train with `REPO_ROOT` at the candidate source revision and write the final
  model to `TRAINING_OUTPUT_DIR` (also exposed as `OUTPUT_DIR`). Wait until local
  or remote training finishes before returning.
- Produce `config.json` and at least one `*.safetensors` file in the training
  output directory, unless `train.checkpoint` explicitly configures another
  path.
- Evaluate `CHECKPOINT_PATH`, wait until evaluation finishes, and write a JSON
  result to `RESULT_PATH` (normally `<trial>/eval/results.json`).
- Use `AUTORESEARCH_TRIAL_ID`, `AUTORESEARCH_STAGE`, and
  `AUTORESEARCH_ATTEMPT` as stable remote job identity inputs when applicable.
- Return nonzero when submission, execution, artifact validation, or metric
  production fails.

Do not add a `baseline_root` key to `autoresearch.yaml`; it is only a convenient
directory used while producing the baseline artifacts.

### 3. Validate without execution

Before running expensive work:

1. Check shell syntax with `bash -n` for new shell wrappers.
2. Check that referenced scripts, datasets, configs, workspace, branch, and
   local dependencies exist.
3. Use a wrapper's own dry-run or validation option when it has one. Do not
   submit a job merely to validate setup.
4. Ensure `output_dir` is outside the Git workspace and all YAML script/result
   paths are absolute.

If a baseline result already exists, verify that its configured dot-separated
metric is present, finite, and exactly matches `baseline.baseline_result`.

### 4. Run the baseline only when requested

Before execution, report the train script, eval script, datasets, Git revision,
baseline artifact directory, and expected cost. Do not delete existing
artifacts. For a conventional baseline directory, expose the same paths used by
AutoResearch:

If baseline training fails with a confirmed GPU OOM and cannot fit through
resource-compatible memory settings that preserve the requested experiment
contract, propose LoRA as a fallback. Do not enable LoRA silently: get the
user's approval, record it as the baseline training method, and keep that method
fixed for subsequent AutoResearch candidates so their results remain comparable.

```bash
RUN_ROOT=/absolute/path/to/baseline \
TRAINING_OUTPUT_DIR=/absolute/path/to/baseline/training \
OUTPUT_DIR=/absolute/path/to/baseline/training \
bash /absolute/path/to/train_baseline.sh

RUN_ROOT=/absolute/path/to/baseline \
CHECKPOINT_PATH=/absolute/path/to/baseline/training \
EVAL_OUTPUT_DIR=/absolute/path/to/baseline/eval \
RESULT_PATH=/absolute/path/to/baseline/eval/results.json \
bash /absolute/path/to/eval_baseline.sh
```

After training, verify the final model artifacts. After evaluation, verify the
result JSON and extract the exact metric value; never infer or round it. Write
that value to `baseline.baseline_result`. A baseline checkpoint is not required
after the baseline result is established.

### 5. Generate and preflight autoresearch.yaml

Start from `skills/autoresearch/scripts/autoresearch.yaml`. Configure absolute
`baseline.train_script`, `baseline.eval_script`, and `baseline.eval_result`
paths, the exact metric and direction, a locally visible baseline Git revision,
and a study `output_dir` outside the workspace.

Distill the user's requested research ideas into one concise, self-contained
`user_guidance` string. Preserve explicit parameter names or values, allowed
change scope, and must-not-change constraints. Omit conversational context and
do not invent requirements. If the user provided no research guidance, leave
the field empty or omit it.

Run the existing read-only validator:

```bash
.venv/bin/python skills/autoresearch/scripts/check.py \
  /absolute/path/to/autoresearch.yaml
```

Fix only the first actionable error and rerun preflight. If no baseline result
exists and execution was not requested, leave the setup incomplete and report
the exact baseline command the user must run; never invent a score.

## Handoff

Report:

- files created or reused;
- baseline commands run, if any;
- final model and result JSON paths;
- metric path, exact value, and direction;
- preflight result;
- remaining blocker or the command to invoke `$autoresearch` next.
