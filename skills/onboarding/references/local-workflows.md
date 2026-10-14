<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

## User Onboarding Walkthrough

Use this walkthrough when a user wants to verify their OSS installation before
using their own dataset. Use `docs/installation.md` for platform and environment
setup and `docs/minimal_example.md` for the dataset, training, evaluation, and serving
walkthrough.

### 1. Confirm the required inputs

Before the full training smoke, identify:

- An H100-class CUDA GPU with at least 80 GB of memory for the N2 smoke.
- A writable dataset path, Hugging Face cache, GR00T artifact cache, and output
  directory on ordinary local storage. Do not require shared internal storage.
- A Hugging Face token with read access to every model repository referenced by
  the selected experiment.
- The dataset embodiment tag. The public onboarding dataset uses
  `so100_2rgb`.

Set the runtime paths explicitly:

```bash
export HF_TOKEN="your_hugging_face_token"
export HF_HOME=/workspace/cache/huggingface
export GROOT_HF_ARTIFACT_CACHE_ROOT=/workspace/cache/groot
export TRAINING_OUTPUT_ROOT=/workspace/outputs
```

If the environment is not installed yet, follow `docs/installation.md`:

- Full onboarding, post-training, and train-eval use the x86 training setup.
- Evaluation or serving with an existing checkpoint can use the x86 inference
  setup; an installed training environment also supports these workflows.
- TensorRT requires the optional TensorRT setup and a compatible checkpoint.
  Automatic bundles require four denoising steps. The minimal walkthrough uses
  PyTorch with two denoising steps; run its commands as written.
- Jetson Thor export, build, validation, and serving use the separate Thor setup.

Use the installation guide's commands and dependency checks rather than
maintaining a separate package-install recipe. For the full onboarding smoke
test, run the exact workflow commands from `docs/minimal_example.md`.

### 2. Prepare the public onboarding dataset

Download the pinned three-episode SO100 LeRobot dataset and generate its GR00T
metadata:

```bash
uv run python examples/prepare_minimal_dataset.py \
  --output-dir /workspace/data/gr00t_red_test
```

Confirm that the command reports `/workspace/data/gr00t_red_test`. The dataset
preparation step needs network access and writable local storage, but no GPU or
model checkpoint.

### 3. Resolve the commands before using the GPU

Dry-run `train`, `eval`, and `policy-server` using the commands in
`docs/minimal_example.md`. Verify that:

- Training resolves the local dataset and `so100_2rgb` embodiment override.
- Evaluation resolves both a dataset and a checkpoint.
- Policy serving resolves a checkpoint but does not require a dataset.

Stop and report the first resolution failure before downloading model weights
or starting a GPU run.

### 4. Run one real training step

Use the exact post-training command from `docs/minimal_example.md`, including
the SO100 temporal modality configuration and `training.torch_compile=false`.
The successful completion signal is:

```text
/workspace/outputs/so100-smoke/checkpoint-1
```

Do not treat model loading or dataset initialization alone as success. Confirm
that the optimizer completes step 1 and the checkpoint exists.

### 5. Evaluate the new checkpoint

Run the guide's two-point open-loop evaluation with:

- `--checkpoint /workspace/outputs/so100-smoke/checkpoint-1`
- `--embodiment-tag so100_2rgb`
- `--denoising-steps 2`
- An explicit `--output-json` path

Success means model inference runs and the JSON report is written. Record the
report path and aggregate metrics; the metric values are not expected to be
meaningful after one training step.

### 6. Verify policy serving

Start the guide's policy-server command from the new local checkpoint. Wait for
the server-ready log and verify the endpoint with the repository client:

```bash
python -c 'from groot.core.policy.server_client import PolicyClient; c = PolicyClient(address="tcp://127.0.0.1:5555", timeout_ms=5000); assert c.ping(); c.close()'
```

Stop the server cleanly after the ping succeeds.

### 7. Optionally verify the combined entrypoint

For release validation, also run the guide's `train-eval` command. It should
write a fresh `checkpoint-1`, pass that checkpoint directly to open-loop eval,
and produce a JSON report. Skip this duplicate training run for ordinary user
onboarding when the separate training and evaluation checks already passed.

### 8. Report onboarding status

Report each stage separately: environment, dataset preparation, dry runs,
training, evaluation, and policy server. Include exact local paths for the
checkpoint and eval report. The full execution smoke passes only when the optimizer step, eval inference,
and policy-server ping all succeed using local paths. A repo tour or an
inference-only onboarding request has its own completion criteria above.

## Post-Training

Dry-run training on a local dataset:

```bash
uv run python -m groot.infra.oss train \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag> \
  --dry-run
```

Short smoke training run:

```bash
uv run python -m groot.infra.oss train \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag> \
  --num-gpus 8 \
  --batch-size 4 \
  --max-steps 2
```

Override the starting checkpoint:

```bash
uv run python -m groot.infra.oss train \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag> \
  --pretrain-checkpoint <hf-uri-or-local-path>
```

## Open-Loop Evaluation

Dry-run evaluation:

```bash
uv run python -m groot.infra.oss eval \
  --checkpoint <hf-uri-or-local-path> \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag> \
  --dry-run
```

Quick local smoke eval:

```bash
uv run python -m groot.infra.oss eval \
  --checkpoint <hf-uri-or-local-path> \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag> \
  --traj-ids 0 \
  --steps 20 \
  --local-only
```

Evaluate multiple trajectories and write an explicit report:

```bash
uv run python -m groot.infra.oss eval \
  --checkpoint <hf-uri-or-local-path> \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag> \
  --traj-ids 0 1 2 \
  --output-json /tmp/groot_oss_open_loop_eval.json
```

## Policy Server

Dry-run the policy-server launch:

```bash
uv run python -m groot.infra.oss policy-server \
  --checkpoint <hf-uri-or-local-path> \
  --embodiment-tag <checkpoint-embodiment-tag> \
  --dry-run
```

Start a GR00T policy server:

```bash
uv run python -m groot.infra.oss policy-server \
  --checkpoint <hf-uri-or-local-path> \
  --embodiment-tag <checkpoint-embodiment-tag> \
  --host 0.0.0.0 \
  --port 5555
```

Use `--host 127.0.0.1` when the server should only accept local clients. Use
`--device cpu` only for lightweight debugging; real N2 policy serving should run
on CUDA.

## Train Then Evaluate

Run post-training followed by eval on the same dataset:

```bash
uv run python -m groot.infra.oss train-eval \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag>
```

## Dataset And Artifacts

- Pass `--embodiment-tag` alongside `--dataset-path`; the same tag controls
  training dataset interpretation and open-loop evaluation.
- Use `--checkpoint` for both local checkpoints and `hf://model/...` URIs.
- Use `--dataset-path` for local training and evaluation datasets.
- `policy-server` resolves `hf://model/...` checkpoints through the same
  Hugging Face cache path as evaluation before launching the core policy server.
- Set `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` when private
  Hugging Face artifacts are needed.
- Do not download, rewrite, or convert datasets unless the user explicitly asks
  for that work.
