# Genie Sim on OSMO: Closed-Loop AutoResearch

This example post-trains GR00T N2 on OSMO and evaluates each candidate in Genie
Sim. The evaluation runs a policy-server task and a simulator task together,
then writes a closed-loop **success_rate**. Higher is better.

The checked-in settings are a lightweight execution smoke test: 10 training
steps, one Genie Sim episode, and at most one subtask. They are not a meaningful
training benchmark.

For the framework-level contract, see
[AutoResearch](../../README.md).

## Execution flow

~~~text
AutoResearch controller
  └─ train_baseline.sh
       └─ one-task OSMO workflow
            └─ RUN_ROOT/training
  └─ eval_baseline.sh
       └─ two-task OSMO group
            ├─ GR00T policy server
            └─ Genie Sim simulator
                 └─ RUN_ROOT/eval/results.json
~~~

The controller does not contain OSMO-specific logic. The user-owned wrappers
submit, reattach to, poll, and verify OSMO workflows. AutoResearch only observes
their exit codes and expected artifacts.

## Files

| Path | Purpose | AutoResearch may edit it |
| --- | --- | --- |
| **autoresearch.yaml** | Study, Git, objective, wrapper, retry, and parallelism settings | No |
| **train_baseline.sh** | Submit/reattach/poll the training workflow | No |
| **eval_baseline.sh** | Submit/reattach/poll the closed-loop eval workflow | No |
| **train_config.yaml** | N2 data, model, and training recipe | Yes |
| **eval_config.yaml** | OSMO pool and fixed Genie Sim evaluation settings | No |
| **workflows/train_genie_sim_osmo_n2.yaml** | One-GPU OSMO training workflow | No |
| **workflows/eval_genie_sim_osmo_n2.yaml** | Policy-server plus simulator workflow | No |
| **osmo_scripts/** | Code executed inside the OSMO tasks | No |

Trial workflow names include the stable trial ID, stage, and attempt. After a
controller restart, each wrapper queries that name and reattaches instead of
blindly submitting a duplicate job.

## 1. Prerequisites

The controller host and OSMO tasks must be able to access:

- the configured training dataset and starting model assets;
- the configured Genie Sim checkout;
- the study **output_dir**, including trial checkpoints and results;
- the Git repository and the exact candidate commits.

The checked-in workflows expect an OSMO credential named **github-token** and
clone the [Isaac-GR00T-CRA `main` repository](https://github.com/NVIDIA/Isaac-GR00T-CRA/tree/main).
Change the workflow and wrapper settings if your repository or credential name
differs.

Verify OSMO access before launching:

~~~bash
osmo version
osmo login
osmo profile list
osmo pool list
~~~

Replace **your-osmo-pool** with a pool available to your account, and confirm it
can access **AUTORESEARCH_ROOT** and run the public NVIDIA NGC PyTorch and Isaac
Sim images in the workflow.
Each task bootstraps `uv` and validates `uv.lock` before running. The training
and policy tasks then install the locked project environment, so they do not
depend on Python packages preinstalled in the base image.

The controller also needs the repository development environment and an
authenticated Codex CLI:

~~~bash
uv sync --locked --extra dev
codex --version
~~~

The policy-server task installs the full training environment inside its OSMO
container. The controller itself does not need to build FlashAttention.

## 2. Configure the example

Choose one writable shared base directory that is visible to the controller and
OSMO workers:

~~~bash
export AUTORESEARCH_ROOT=/path/to/autoresearch
export OSMO_POOL=your-available-pool
~~~

The checked-in configurations resolve their repository, dataset, model cache,
Genie Sim, baseline, and study paths below this directory. Update these settings
before running:

### autoresearch.yaml

- **workspace** and **output_dir**.
- all three absolute paths under **baseline**.
- **base_branch** and, optionally, **base_commit**.
- **baseline_result** after the baseline evaluation.
- **OSMO_POOL** in the controller environment, plus **PRIORITY** under
  **train.env** and **eval.env** if the default is unsuitable.
- **max_trials**, **max_parallel_trials**, and **max_retry** for the available
  compute budget.

If **base_commit** is omitted, AutoResearch resolves the current head of
**base_branch**. If it is present, that exact commit is used. Because OSMO pods
clone the repository, the base commit and every candidate commit must be pushed
and reachable from the configured remote.

### train_config.yaml

- training dataset path and embodiment;
- starting checkpoint;
- diffusion model and VAE paths;
- training recipe.

The wrapper copies this file into each run directory and supplies the run's
training output path. Do not use its checked-in **output_dir** to identify a
trial.

### eval_config.yaml

- OSMO pool and priority;
- **geniesim_root**;
- embodiment and Genie Sim task;
- episode/subtask counts;
- policy action horizon, denoising steps, and port.

### OSMO workflows

Review container images, CPU/GPU/memory/storage requests, timeouts, Git
credential mapping, repository URL, and shared mount assumptions. Training uses
one GPU. Each evaluation uses two one-GPU tasks concurrently.

The checked-in **max_parallel_trials: 4** can therefore consume up to four GPUs
during a training wave and up to eight GPUs while four closed-loop evaluations
run together. For the first smoke test, use **max_trials: 1** and
**max_parallel_trials: 1**.

## 3. Validate without submitting a job

First validate the training workflow template:

~~~bash
osmo workflow validate \
  skills/autoresearch/examples/genie_sim_osmo/workflows/train_genie_sim_osmo_n2.yaml \
  --pool your-osmo-pool
~~~

Then validate the rendered evaluation workflow through its wrapper. Use a fresh
temporary path so an existing result cannot trigger the wrapper's resume shortcut:

~~~bash
RUN_ROOT=/tmp/genie-sim-osmo-validation \
AUTORESEARCH_ROOT=/tmp/autoresearch-validation \
VALIDATE_ONLY=1 \
bash skills/autoresearch/examples/genie_sim_osmo/eval_baseline.sh
~~~

Finally run the AutoResearch preflight:

~~~bash
.venv/bin/python skills/autoresearch/scripts/check.py \
  skills/autoresearch/examples/genie_sim_osmo/autoresearch.yaml
~~~

These commands validate configuration and workflow rendering. They do not
submit training or evaluation jobs.

## 4. Run and record the baseline

The following commands submit real OSMO workflows:

~~~bash
export RUN_ROOT="${AUTORESEARCH_ROOT}/genie_sim_osmo/baseline"
WORKFLOW_NAME=genie-sim-osmo-baseline-train \
  bash skills/autoresearch/examples/genie_sim_osmo/train_baseline.sh
WORKFLOW_NAME=genie-sim-osmo-baseline-eval \
  bash skills/autoresearch/examples/genie_sim_osmo/eval_baseline.sh
~~~

The expected durable artifacts are:

~~~text
$RUN_ROOT/
├── config/
│   └── train_config.yaml
├── training/
│   ├── config.json
│   └── *.safetensors
└── eval/
    ├── results.json
    └── genie_sim_benchmark.log
~~~

The result converter writes a payload containing **success_rate**,
**num_results**, checkpoint path, task name, and the raw Genie Sim result files.
Copy the exact **success_rate** value into **baseline.baseline_result** in
**autoresearch.yaml**.

Both wrappers reuse completed durable artifacts. The explicit workflow names
above also make the baseline resumable: if interrupted, rerun the same command
with the same **RUN_ROOT** and **WORKFLOW_NAME**. The wrapper first reuses a
valid artifact or reattaches to that OSMO workflow.

## 5. Launch or resume AutoResearch

~~~bash
bash skills/autoresearch/scripts/launch_autoresearch.sh \
  skills/autoresearch/examples/genie_sim_osmo/autoresearch.yaml
~~~

AutoResearch writes **state.json**, trial logs, checkpoints, and results under
**output_dir**. Running the same command again resumes that state. Each trial
wrapper receives the candidate branch and exact revision, submits its remote
work, waits for completion, and returns only after verifying the expected
artifact.

To start an independent study, choose a new **output_dir**. Keep an interrupted
study's state and trial directories so wrapper reattachment and controller
recovery can work.

## Inspecting failures

The wrapper output records the OSMO workflow name. Inspect a failed or stalled
workflow with:

~~~bash
osmo workflow query WORKFLOW_ID --format-type json
osmo workflow logs WORKFLOW_ID -n 10000
~~~

Common causes are:

- the candidate commit was not pushed or cannot be fetched by the workflow;
- the **github-token** credential is absent or invalid;
- controller and OSMO tasks do not share the configured checkpoint/result paths;
- the selected pool cannot run both evaluation tasks or mount Genie Sim;
- the policy server never becomes reachable by the simulator;
- Genie Sim produces no recognized success value, so result conversion fails;
- a workflow completes without the required final model or **results.json**.

Fix the wrapper, workflow, path, or infrastructure issue and rerun the same
study. AutoResearch will resume from **state.json** and apply the configured
stage retry limit.
