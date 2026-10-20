# Minimal End-to-End Example

This guide prepares a small public SO100 LeRobot dataset and verifies
post-training, open-loop evaluation, and policy serving in the installed
Isaac GR00T 2 training environment. It uses only local data, cache, and output
paths. The model artifacts are large, so treat this as an integration smoke test rather than a
meaningful training run.

Run every command below from the package root, the directory containing
`pyproject.toml`.

## 1. Install the training environment

Follow the [x86 training installation](installation.md#x86-training), including
its prerequisites and dependency checks. This walkthrough requires one
H100-class GPU with at least 80 GB of memory, at least 200 GB of free local
disk, and a Hugging Face token with access to the selected experiment's model
repositories.

The walkthrough uses PyTorch inference with two denoising steps. TensorRT is
optional for other inference workflows; its setup and checkpoint requirements
are described in the [installation guide](installation.md#optional-tensorrt-on-x86).

The OSS launcher uses synchronous checkpoint saving by default. This avoids
requiring a very large `/dev/shm` allocation in ordinary containers. Runtimes
provisioned with sufficient shared memory may opt into asynchronous saving with
`--set training.use_async_checkpoint_save=true`.

## 2. Configure credentials and local paths

**Inputs:** a Hugging Face read token and a writable location with enough space
for the dataset, model cache, checkpoints, and reports.

Keep every artifact under one configurable local directory:

```bash
export GROOT2_HOME="${GROOT2_HOME:-$HOME/gr00t2}"
export HF_HOME="$GROOT2_HOME/cache/huggingface"
export GROOT_HF_ARTIFACT_CACHE_ROOT="$GROOT2_HOME/cache/groot"
export TRAINING_OUTPUT_ROOT="$GROOT2_HOME/outputs"

mkdir -p \
  "$GROOT2_HOME/data" \
  "$HF_HOME" \
  "$GROOT_HF_ARTIFACT_CACHE_ROOT" \
  "$TRAINING_OUTPUT_ROOT"
```

Confirm that the token is active:

```bash
uv run hf auth whoami
```

No shared filesystem mount is required.

## 3. Prepare the mini dataset

**Inputs:** network access to the public
`akira-sasaki/gr00t_red_test` Hugging Face dataset and the writable data
directory from step 2. This step does not require a model checkpoint or GPU.

Download the pinned dataset revision and generate its GR00T modality and
statistics metadata:

```bash
uv run python examples/prepare_minimal_dataset.py \
  --output-dir "$GROOT2_HOME/data/gr00t_red_test"
```

The prepared dataset has three episodes and uses the `so100_2rgb` embodiment.

## 4. Run one training step

**Inputs:** the prepared dataset, its `so100_2rgb` embodiment tag, one H100-class
GPU, writable cache and output paths, and access to the default pretraining,
Cosmos, diffusion transformer, and VAE artifacts used by the experiment.

```bash
uv run python -m groot.infra.oss train \
  --dataset-path "$GROOT2_HOME/data/gr00t_red_test" \
  --embodiment-tag so100_2rgb \
  --output-dir "$TRAINING_OUTPUT_ROOT/so100-smoke" \
  --num-gpus 1 \
  --batch-size 1 \
  --max-steps 1 \
  --set 'data.video_backend="torchcodec"' \
  --set 'data.modality_configs={"so100_2rgb":{"video":{"delta_indices":[0,5,10,15,20]}}}' \
  --set 'data.override_pretraining_statistics=false' \
  --set 'training.save_steps=1' \
  --set 'training.torch_compile=false'
```

Success means the optimizer completes step 1 and the checkpoint exists at:

```text
$TRAINING_OUTPUT_ROOT/so100-smoke/checkpoint-1
```

## 5. Run open-loop evaluation

**Inputs:** the prepared dataset, the matching `so100_2rgb` embodiment tag, the
local checkpoint from step 4, one H100-class GPU, and a writable JSON output
path.

```bash
uv run python -m groot.infra.oss eval \
  --checkpoint "$TRAINING_OUTPUT_ROOT/so100-smoke/checkpoint-1" \
  --dataset-path "$GROOT2_HOME/data/gr00t_red_test" \
  --embodiment-tag so100_2rgb \
  --video-backend torchcodec \
  --traj-ids 0 \
  --steps 2 \
  --action-horizon 16 \
  --denoising-steps 2 \
  --local-only \
  --output-json "$TRAINING_OUTPUT_ROOT/so100-smoke/eval.json"
```

Success means model inference runs and `eval.json` is written.

## 6. Start the policy server

**Inputs:** the matching `so100_2rgb` embodiment tag, the local checkpoint from
step 4, one H100-class GPU, and an available TCP port. A dataset is not required.

```bash
uv run python -m groot.infra.oss policy-server \
  --checkpoint "$TRAINING_OUTPUT_ROOT/so100-smoke/checkpoint-1" \
  --embodiment-tag so100_2rgb \
  --host 0.0.0.0 \
  --port 5555 \
  --denoising-steps 2
```

The server stays in the foreground and listens at `tcp://0.0.0.0:5555`. Stop
it with `Ctrl-C` after it reports that it is ready.

## 7. Run training and evaluation together

**Inputs:** all training inputs from step 4 plus the evaluation settings from
step 5. Do not provide a checkpoint; this command passes its newly written
`checkpoint-1` directly to evaluation.

```bash
uv run python -m groot.infra.oss train-eval \
  --dataset-path "$GROOT2_HOME/data/gr00t_red_test" \
  --embodiment-tag so100_2rgb \
  --output-dir "$TRAINING_OUTPUT_ROOT/so100-train-eval" \
  --num-gpus 1 --batch-size 1 --max-steps 1 \
  --set 'data.video_backend="torchcodec"' \
  --set 'data.modality_configs={"so100_2rgb":{"video":{"delta_indices":[0,5,10,15,20]}}}' \
  --set 'data.override_pretraining_statistics=false' \
  --set 'training.torch_compile=false' \
  --video-backend torchcodec \
  --traj-ids 0 --steps 2 --action-horizon 16 \
  --denoising-steps 2 --local-only \
  --output-json "$TRAINING_OUTPUT_ROOT/so100-train-eval/eval.json"
```
