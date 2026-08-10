# GR00T N2 AutoResearch

## How it works

### 1. Basic idea

AutoResearch helps users find a better GR00T N2 post-training recipe for their
own dataset and embodiment. It turns recipe tuning into a sequence of measured
experiments:

1. Codex proposes one focused hypothesis and implements the corresponding code
   or configuration change.
2. AutoResearch trains a candidate model with that change.
3. It evaluates the candidate with the user's evaluation procedure.
4. It compares the resulting score with the best recipe found so far.
5. The experiment result becomes evidence for the hypotheses proposed next.

Each experiment is called a **trial**. A trial keeps the hypothesis, code
revision, training result, evaluation result, score, and feedback together, so a
score can always be traced back to the exact recipe that produced it.

### 2. Bring a baseline; run on any infrastructure

The user starts with a baseline that already works: training and evaluation
scripts, their configurations, and a baseline evaluation result. AutoResearch
uses the same scripts for every candidate.

The scripts hide infrastructure-specific behavior from the controller. A script
may run locally, submit an OSMO workflow, use another cluster scheduler, or wait
for an external process. As long as the training script returns a valid model
and the evaluation script returns a numeric result, AutoResearch can compare
experiments in the same way.

This supports three evaluation modes:

- **Open-loop evaluation:** run inference on a held-out dataset and measure a
  metric such as action MSE.
- **Closed-loop simulation:** run the trained policy in a simulator and measure
  a metric such as task success rate.
- **Manual evaluation:** ask a person to test the model on a real robot. The
  evaluation script waits for the person to provide the result, writes the
  result JSON, and then allows AutoResearch to continue.

AutoResearch treats all three as a scalar optimization problem. The study
configuration says which value to read and whether a lower or higher value is
better.

### 3. How waves and trials work

A **wave** is a group of trials that share the same parent recipe. All trials in
the wave can run in parallel because none depends on another trial in that same
wave.

After the wave finishes, the best trial that improves the global score becomes
the parent of the next wave. Another trial may also be useful even if it was not
the winner. AutoResearch queues that change and can later test it in combination
with the global best. If the Git changes conflict, that composition is skipped.

~~~mermaid
flowchart LR
    B["Baseline"]

    subgraph W1["Wave 1 — all trials start from the baseline"]
        T1["Trial 1<br/>Improved, but not the winner"]
        T2["Trial 2<br/>New global best"]
        T3["Trial 3<br/>Worse than baseline"]
    end

    subgraph W2["Wave 2 — all trials start from Trial 2"]
        T4["Trial 4<br/>New hypothesis"]
        T5["Trial 5<br/>Trial 2 + queued change from Trial 1"]
    end

    B --> T1
    B --> T2
    B --> T3
    T2 --> T4
    T2 --> T5
    T1 -. "queued change" .-> T5

    classDef baseline fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef useful fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef best fill:#bbf7d0,stroke:#15803d,stroke-width:3px,color:#14532d
    classDef worse fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef composition fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    class B baseline
    class T1 useful
    class T2 best
    class T3 worse
    class T5 composition
~~~

This is a greedy search with one global-best lineage. It keeps the search easy
to understand while still allowing several ideas to be evaluated in parallel
and useful non-winning changes to be revisited.

### 4. All study progress is recorded in state.json

**state.json** is the durable record of the study. It stores the baseline,
current best, active wave, trial lineage, hypotheses, stage status, scores, and
guidance for future trials.

AutoResearch saves this state as work completes. If the controller stops, the
same study can read **state.json** and continue instead of starting over. Large
artifacts such as models and logs stay in the corresponding trial folders;
**state.json** stores the information needed to locate and interpret them.

## How to use

### 0. Set up the environment

If the repository environment has not been set up yet, run this command from
the repository root:

```bash
bash skills/autoresearch/scripts/setup_env.sh
```

This installs the locked training and development dependencies required by
AutoResearch and the local train/eval scripts.

### 1. Prepare the inputs

#### Training and evaluation datasets

Prepare the post-training dataset and a separate evaluation dataset. Keep their
contents fixed throughout the study. If the evaluation is closed-loop or
manual, keep the simulator tasks, real-robot tasks, episode definitions, and
scoring procedure fixed instead.

#### Training and evaluation scripts

Provide one training script and one evaluation script that already reproduce
the baseline.

The training script may run locally or submit remote work, but it must wait for
that work to finish and return only after producing a valid final model. By
default, AutoResearch expects:

~~~text
output_dir/<trial_id>/training/
├── config.json
└── *.safetensors
~~~

The evaluation script receives that model, performs the task-specific
evaluation, and writes:

~~~text
output_dir/<trial_id>/eval/results.json
~~~

The result file must contain the finite numeric metric configured for the study.
For remote infrastructure, the scripts are also responsible for job submission,
polling, recovery, and artifact verification.

#### Training and evaluation configs

Put the training recipe in a dedicated config file used by the training script.
This config is normally part of **editable_paths**, allowing Codex to tune
learning rates, sampling, architecture options, and other training choices.

Keep the evaluation config frozen. Changing the eval dataset, task, metric,
horizon, or scoring rule during a study would make scores from different trials
incomparable.

#### Baseline eval result

Run the baseline training and evaluation before starting AutoResearch. Keep the
result JSON and identify the exact numeric value to optimize.

For example:

- open-loop: **results.local.average.mse**, with **direction: minimize**;
- closed-loop: **success_rate**, with **direction: maximize**.

Copy the exact value into **baseline.baseline_result** in
**autoresearch.yaml**. AutoResearch does not run the baseline automatically, and
it does not need the baseline checkpoint after this result has been produced.

### 2. Configure the study

Copy [the generic template](scripts/autoresearch.yaml) into the task folder.
This YAML is the only configuration interface for AutoResearch.

The main arguments are:

| Arguments | Description |
| --- | --- |
| **study_id** | Stable study name used in trial IDs, branches, logs, and state |
| **workspace** | Git repository containing the code Codex may edit |
| **output_dir** | Directory for study state and trial artifacts; it must be outside **workspace** |
| **max_trials** | Number of candidate experiments, excluding the baseline |
| **max_parallel_trials** | Maximum number of trials that may run together in one wave |
| **max_retry** | Retries after a failed Codex, publish, training, or evaluation operation |
| **baseline.train_script** | Absolute path to the training script |
| **baseline.eval_script** | Absolute path to the evaluation script |
| **baseline.eval_result** | Absolute path to the existing baseline result JSON |
| **baseline.eval_metric** | Dot-separated path to the numeric metric inside that JSON |
| **baseline.direction** | Whether lower (**minimize**) or higher (**maximize**) is better |
| **baseline.baseline_result** | Exact metric value produced by the baseline |
| **baseline.min_delta** | Smallest score improvement accepted as better |
| **baseline.script_timeout_s** | Maximum duration of one train or eval script invocation |
| **git.git_remote** | Git remote used to publish candidate branches |
| **git.base_branch** | Branch containing the baseline code |
| **git.base_commit** | Optional exact baseline commit |
| **editable_paths** | Repository paths Codex may change |
| **frozen_paths** | Repository paths that must remain unchanged |
| **idea_types** | Research categories assigned to new trials |
| **agent** | Codex executable, model, timeout, and sandbox settings |

With **max_retry: 2**, an operation can run at most three times: one initial
attempt and two retries.

If **git.base_commit** is omitted, AutoResearch uses the locally visible head of
**git.base_branch** without fetching. Pinning an exact commit is recommended for
a long-running study. Commit all baseline code before launch because
uncommitted workspace changes are not included.

For the first smoke test, use a short training/evaluation setup,
**max_trials: 1**, and **max_parallel_trials: 1**. The trial count controls how
many experiments run; the training and evaluation configs control the cost of
each experiment.

### 3. Run or resume AutoResearch

You can launch or resume a study by:

~~~bash
bash skills/autoresearch/scripts/launch_autoresearch.sh \
  /absolute/path/to/autoresearch.yaml
~~~

Or ask Codex to use the repository skill:

~~~text
Use $autoresearch to validate and run or resume
/absolute/path/to/autoresearch.yaml. Report progress and failures.
~~~

If **output_dir/state.json** already exists, the launch command resumes that
study. Otherwise it initializes a new one. A failed operation is retried
according to **max_retry**, and one failed trial does not stop the other trials
in its wave. Use a different **output_dir** when starting an independent study.

Two complete smoke-test examples are included:

- [Yam Stack Bowl](examples/yam_stack_bowl/README.md): local post-training with
  open-loop MSE evaluation.
- [Genie Sim on OSMO](examples/genie_sim_osmo/README.md): OSMO post-training
  with closed-loop simulation success rate.

### 4. Inspect the results and visualize state.json

AutoResearch stores all durable outputs under **output_dir**:

~~~text
output_dir/
├── state.json
└── STUDY_wave1_trial1/
    ├── codex/
    ├── train.attempt-0001.stdout.log
    ├── train.attempt-0001.stderr.log
    ├── eval.attempt-0001.stdout.log
    ├── eval.attempt-0001.stderr.log
    ├── training/
    │   ├── config.json
    │   └── *.safetensors
    └── eval/
        └── results.json
~~~

Each trial folder contains the Codex proposal and diff, train/eval logs, final
model, and evaluation result. A retry writes a new attempt log instead of
overwriting the previous failure.

At the study level, **state.json** records the metric definition, baseline and
best scores, best trial, active wave, and changes waiting for composition. Each
trial record contains its parent, hypothesis, code revision, changed files,
train/eval status, attempts, errors, score, result summary, and guidance for
future trials.

You can visualize state.json by:

~~~bash
python3 skills/visualize-autoresearch/scripts/render.py \
  /path/to/output_dir/state.json
~~~

It generates a self-contained HTML report **state.visualization.html** beside **state.json**.

You can also ask:

~~~text
Use $visualize-autoresearch to render
/path/to/output_dir/state.json and return the HTML path.
~~~

The report shows summary metrics and an interactive trial graph. Solid arrows
show the parent lineage, dashed arrows show composition, and green/red intensity
shows how much a trial improved or regressed relative to its parent. Selecting a
trial opens its hypothesis, result, code, stages, errors, and checkpoint. The
HTML is portable and does not modify **state.json**.
