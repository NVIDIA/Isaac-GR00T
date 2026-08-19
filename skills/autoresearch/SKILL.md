---
name: autoresearch
description: Set up, validate, smoke-test, launch, resume, inspect, and debug GR00T N2 AutoResearch studies driven by autoresearch.yaml and user-owned train/eval scripts. Use when Codex needs to prepare a new AutoResearch task, verify its baseline and execution contract, operate a local or remote-infrastructure study, summarize state or stage failures, or render study results.
---

# AutoResearch Operator

Operate the repository implementation; do not create a second orchestration
layer. Keep autoresearch.yaml as the only user configuration interface.

## Safety and scope

- Treat validation as read-only. Never invoke train_script or eval_script during
  preflight.
- Launch a study only when the user explicitly asks to run, start, resume, or
  smoke-test it.
- Before launch, state the config path, output_dir, trial budget, parallelism,
  baseline metric, and scripts that will execute.
- Treat user scripts as opaque infrastructure adapters. Do not replace their
  scheduler submission or polling behavior inside the skill.
- Never silently change datasets, metric paths, baseline results, frozen paths,
  compute budgets, or an existing state.json.
- Do not delete or overwrite an existing output_dir to start fresh unless the
  user explicitly requests it.
- Do not run git fetch when base_commit is omitted. The implementation resolves
  the locally visible head of base_branch.

## Core workflow

Run commands from the repository root.

### 1. Locate the config and understand the request

Require one autoresearch.yaml path. Determine whether the user wants:

- preflight validation only;
- preparation of a new task;
- a lightweight smoke test;
- launch or resume;
- failure diagnosis or status;
- result visualization.

For a new task or wrapper-contract question, read README.md before editing
files. Use its current configuration and artifact contracts as the source of
truth.

### 2. Run deterministic preflight

Run:

    .venv/bin/python skills/autoresearch/scripts/check.py /absolute/path/to/autoresearch.yaml

This validates the YAML, baseline result and metric, absolute wrapper paths, Git
remote and revision, Codex executable, output boundary, and any existing saved
state. It does not run user train/eval scripts or create output state.

If preflight fails, report the first actionable error and fix only what the user
asked to fix. Re-run preflight after a fix.

### 3. Prepare a new task

When asked to create a task:

1. Start from skills/autoresearch/scripts/autoresearch.yaml.
2. Prefer the closest existing example:
   - skills/autoresearch/examples/yam_stack_bowl for local/open-loop work.
   - skills/autoresearch/examples/genie_sim_osmo for remote/closed-loop work.
3. Keep user wrapper scripts and evaluation logic frozen.
4. Put only intended training-side code and config in editable_paths.
5. Require an already-produced baseline eval JSON. Do not assume a baseline
   checkpoint is required.
6. Verify each wrapper independently with its own lightweight mechanism before
   launching AutoResearch.
7. Run preflight.

Do not invent infrastructure commands when no working user wrapper exists.
Explain the missing contract and ask the user to provide or complete it.

### 4. Smoke-test safely

A controller smoke test should normally use:

- a new study_id and output_dir;
- max_trials: 1;
- max_parallel_trials: 1;
- a short train/eval configuration already implemented by the user wrappers.

Changing max_trials does not shorten training. Confirm the wrappers themselves
use smoke-sized data and compute. Do not mutate a production config in place
unless the user requests that exact edit.

After preflight, launch only when the user asked to execute the smoke test.

### 5. Launch or resume

Run:

    bash skills/autoresearch/scripts/launch_autoresearch.sh /absolute/path/to/autoresearch.yaml

If output_dir/state.json exists, the command resumes it. Otherwise it creates a
new study. Keep the process attached or monitor its terminal output for
long-running work. Report wave, trial, stage, attempt, and the first actionable
failure rather than dumping entire logs.

Do not separately submit an OSMO or other remote job. The configured wrapper
owns submission, reattachment, polling, and artifact verification.

### 6. Inspect or diagnose

For status or failures:

1. Read output_dir/state.json.
2. Identify active_wave_index, best_trial_id, best_score, and unfinished or
   failed trials.
3. Inspect only the relevant trial's latest train/eval attempt logs and Codex
   attempt metadata.
4. Distinguish candidate-edit, publish, train, eval, and metric-parsing failures.
5. Check whether retry attempts remain.
6. Recommend the smallest fix. Do not edit or relaunch unless requested.

Treat an unscored failed trial as a failure, not as a metric regression.

### 7. Visualize results

For an HTML report, run:

    python3 skills/visualize-autoresearch/scripts/render.py /path/to/output_dir/state.json

Return the generated absolute HTML path. Do not modify state.json.

## Handoff format

Keep user-facing results compact:

- operation performed;
- config and output_dir;
- new or resumed study;
- current wave/trial/stage;
- best score and trial when available;
- first actionable failure when applicable;
- exact next command or next safe action.
