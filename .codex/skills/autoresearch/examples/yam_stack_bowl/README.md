# Yam Stack Bowl: Local Open-Loop AutoResearch

This example post-trains GR00T N2 on a small Yam Stack Bowl dataset and evaluates
the result locally with open-loop action MSE. It is intentionally a smoke test:
training runs for 10 steps and evaluation uses trajectory 0 for 10 steps.
The smoke recipe updates the full model with Adafactor so it fits on one L40 GPU.

The objective is **results.local.average.mse**, and lower is better. This example
proves the local AutoResearch flow; increase the data and training/evaluation
budgets before using its result as a production recipe.

For the framework-level contract, see
[AutoResearch](../../README.md).

## Files

| File | Purpose | AutoResearch may edit it |
| --- | --- | --- |
| **autoresearch.yaml** | Study, Git, objective, path, retry, and parallelism settings | No |
| **train_baseline.sh** | User-owned local training wrapper | No |
| **eval_baseline.sh** | User-owned local evaluation wrapper | No |
| **train_config.yaml** | N2 data, model, and training recipe | Yes |
| **eval_config.yaml** | Fixed open-loop evaluation settings | No |

The wrappers call the general scripts in **skills/autoresearch/scripts**. During a
trial, AutoResearch supplies the trial output directory and exact candidate Git
revision through environment variables. The wrapper interface stays fixed while
Codex changes only paths listed under **editable_paths**.

## 1. Configure the example

Edit the following values before running:

1. In **autoresearch.yaml**:
   - **workspace**: the local repository checkout.
   - **output_dir**: a new durable study directory.
   - all three absolute paths under **baseline**.
   - **base_branch** and, optionally, **base_commit**.
   - **baseline_result** after running the baseline.
2. In **train_config.yaml**:
   - the training dataset path.
   - embodiment and modality settings.
   - starting checkpoint and training recipe.
3. In **eval_config.yaml**:
   - the held-out evaluation dataset path.
   - embodiment-compatible horizon and sampling settings.
4. In **train_baseline.sh** and **eval_baseline.sh**:
   - change their default absolute repository/config/output paths if this
     checkout is not at the paths currently shown.

The configured training and evaluation datasets must be separate. The current
example expects:

~~~text
/mnt/amlfs-01/home/yuhlu/AutoResearch_yam_stack_bowl/datasets/train
/mnt/amlfs-01/home/yuhlu/AutoResearch_yam_stack_bowl/datasets/eval
~~~

If **base_commit** is omitted, AutoResearch resolves the current head of
**base_branch**. If it is present, that exact commit is used. The base branch and
commit must be available from the configured Git remote.

## 2. Set up the local environment

From the repository root, create the full training environment:

~~~bash
bash skills/autoresearch/scripts/setup_env.sh
~~~

This installs the locked training and development dependencies and builds
FlashAttention with conservative build parallelism.

Also make sure the Codex CLI is installed and authenticated:

~~~bash
codex --version
~~~

The first access to Hugging Face model assets may download them. Later runs use
the normal Hugging Face cache.

## 3. Run and record the baseline

Use the same output root referenced by **baseline.eval_result**:

~~~bash
export RUN_ROOT=/mnt/amlfs-01/home/yuhlu/AutoResearch_yam_stack_bowl_full/baseline
bash skills/autoresearch/examples/yam_stack_bowl/train_baseline.sh
bash skills/autoresearch/examples/yam_stack_bowl/eval_baseline.sh
~~~

The expected artifacts are:

~~~text
$RUN_ROOT/
├── training/
│   ├── config.json
│   └── *.safetensors
└── eval/
    ├── results.json
    └── plots/
~~~

Read **results.local.average.mse** from **eval/results.json** and copy that exact
number to **baseline.baseline_result** in **autoresearch.yaml**.

Both wrappers are resumable:

- training reuses a final model containing **config.json** and a safetensors
  file;
- evaluation reuses a non-empty **results.json**;
- numbered checkpoint directories are deleted only after the final model has
  been saved successfully.

To intentionally rerun the baseline, use a new **RUN_ROOT**, or remove the
specific completed artifact after confirming it is no longer needed.

## 4. Validate without training

Run the AutoResearch preflight from the repository root:

~~~bash
.venv/bin/python skills/autoresearch/scripts/check.py \
  skills/autoresearch/examples/yam_stack_bowl/autoresearch.yaml
~~~

This checks the configuration, Git base, paths, baseline metric, Codex
availability, and wrapper contract. It does not run training or evaluation.

## 5. Launch or resume AutoResearch

~~~bash
bash skills/autoresearch/scripts/launch_autoresearch.sh \
  skills/autoresearch/examples/yam_stack_bowl/autoresearch.yaml
~~~

The current example runs five trials sequentially. AutoResearch writes
**state.json**, per-trial logs, final models, and evaluation results under
**output_dir**. Running the same command again resumes the existing state instead
of creating a new study.

To start a separate study, choose a new **output_dir**. Do not delete an active
study's **state.json** merely to retry a failed stage; stage retries and study
resume are already part of the controller.

## Common problems

- **Preflight reports a baseline mismatch:** read the configured metric from the
  configured result file and update **baseline_result** exactly.
- **Training cannot find data:** update both dataset paths and verify the
  training/eval folders are readable from the local machine.
- **Model downloads stall or fail:** check Hugging Face authentication, cache
  permissions, and network access.
- **A checkpoint is rejected:** the trial training directory must contain
  **config.json** and at least one **.safetensors** file.
- **The smoke result is noisy:** this is expected for 10 training steps and 10
  evaluation points. Use the smoke only to verify execution before scaling up.
