# GR00T Deployment & Inference Guide

Run TensorRT-accelerated inference for the GR00T N2 policy.

---

## Prerequisites

- A compatible N2 model checkpoint
- A CUDA-capable target platform
- The environment for the target platform, using the setup below

TensorRT engines are specific to the GPU and software stack on which they are built. Follow
the setup section for the target platform, then use the common export, build, validation, and
serving workflow below.

| Platform | Setup | Supported workflow |
| --- | --- | --- |
| Jetson Thor | [Jetson Thor Setup](#jetson-thor-setup) (bare metal) | Export, build, validate, and serve |


## TensorRT Acceleration

TensorRT replaces the N2 Transformer's and VAE encoder's normal PyTorch execution with
GPU-specific engines. The surrounding checkpoint loading, preprocessing, action decoding,
and policy server remain the same. There is no silent fallback: an invalid, foreign, or
unvalidated engine bundle stops policy construction.

### Qualified engine configurations

This runtime is unquantized BF16, batch size 1, 224 x 320 images, five history
frames per camera, joint denoising, conditional KV cache, two video denoising steps, and four
action denoising steps. Engines are built from the selected checkpoint, but this release admits
only the checkpoint layouts that have been validated end to end:

| Cameras | Action horizon |
| ---: | ---: |
| 3 | 80 |
| 4 | 48 |

## Export, build, and validate engines

Choose any writable directory for the generated ONNX files, TensorRT engines, and manifest.
Use the same directory for export, build, validation, and serving. The commands build the
engines locally; they do not download prebuilt engines.

Start from a clean Git checkout. The checkpoint can be in any local directory; it must contain
`config.json`, `processor_config.json`, `statistics.json`, and the model safetensors. All model
dependencies referenced by `config.json` must also resolve on the Thor.

Set the checkpoint and output paths. The commands below automatically select the embodiment
from a single-embodiment checkpoint:

```bash
export CHECKPOINT=/path/to/checkpoint
export TRT_ARTIFACT_DIR=/path/to/trt-output

export PROCESSOR_EMBODIMENT="$(python -c \
  'import json, sys; keys=list(json.load(open(sys.argv[1]))); assert len(keys) == 1, f"expected one embodiment, found {keys}"; print(keys[0])' \
  "$CHECKPOINT/statistics.json")"
export SERVER_EMBODIMENT="$(printf '%s' "$PROCESSOR_EMBODIMENT" | tr '[:lower:]' '[:upper:]')"

# This must print nothing: the engine manifest records clean Git revisions.
git status --short

# Export the checkpoint-shaped ONNX graphs.
python -P -m groot.deployment.n2_trt.tools.export_transformer \
  --checkpoint "$CHECKPOINT" \
  --embodiment "$PROCESSOR_EMBODIMENT" \
  --codec mp4v \
  --output-dir "$TRT_ARTIFACT_DIR"

python -P -m groot.deployment.n2_trt.tools.export_vae \
  --checkpoint "$CHECKPOINT" \
  --embodiment "$PROCESSOR_EMBODIMENT" \
  --output-dir "$TRT_ARTIFACT_DIR"

# Build GPU-specific TensorRT engines from those ONNX graphs.
python -P -m groot.deployment.n2_trt.tools.build_engines \
  --artifact-root "$TRT_ARTIFACT_DIR"

# Verify TensorRT outputs against the captured PyTorch references.
python -P -m groot.deployment.n2_trt.validation \
  --artifact-root "$TRT_ARTIFACT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --embodiment "$PROCESSOR_EMBODIMENT"
```

Validation writes `n2_trt_manifest.json`. Serving authenticates that manifest, the checkpoint,
all three engine files, their fixed I/O schemas, TensorRT version, and target GPU.

## Build or reuse engines automatically

For the standard OSS workflow, pass `--enable-trt`. The first invocation
exports, builds, and validates a target-local bundle; later invocations reuse
the content-addressed bundle after authenticating it against the checkpoint and
runtime:

```bash
python -P -m groot.infra.oss eval \
  --checkpoint "$CHECKPOINT" \
  --dataset-path /path/to/lerobot-dataset \
  --embodiment-tag "$PROCESSOR_EMBODIMENT" \
  --enable-trt
```

The default cache is `~/.cache/groot/tensorrt`. Override it with
`--trt-cache-dir` or `GROOT_TRT_CACHE_ROOT`; use `--rebuild-trt` with
`--enable-trt` to replace the matching cache entry. Automatic builds use the
checkout revision when Git metadata is available, or the installed wheel's
immutable `RECORD` contents. Unsupported checkpoint contracts and failed
validation stop the command without eager fallback. Automatic bundles
currently require `--denoising-steps 4`.

`train-eval --enable-trt` builds only after training has produced its final
checkpoint. `closed-loop-eval --enable-trt` resolves or builds a compatible
bundle on each Ray worker, so each worker uses its local GPU and cache. A
forced rebuild runs once for each distinct runtime/cache identity and is reused
by workers sharing that identity.

## Use an explicitly managed engine bundle

The OSS open-loop evaluator, closed-loop evaluator, and policy server all use
`--trt-engine-dir` when a bundle is built and retained outside the automatic
cache. The checkpoint must match the checkpoint used to export and validate the
bundle.

Run open-loop evaluation:

```bash
python -P -m groot.infra.oss eval \
  --checkpoint "$CHECKPOINT" \
  --dataset-path /path/to/lerobot-dataset \
  --embodiment-tag "$PROCESSOR_EMBODIMENT" \
  --trt-engine-dir "$TRT_ARTIFACT_DIR"
```

Run closed-loop evaluation with any installed OSS simulator backend:

```bash
python -P -m groot.infra.oss closed-loop-eval \
  --checkpoint "$CHECKPOINT" \
  --trt-engine-dir "$TRT_ARTIFACT_DIR" \
  --backend <python-module:BackendClass> \
  --suite <suite-name> \
  --embodiment-tag "$PROCESSOR_EMBODIMENT"
```

Start the policy server:

```bash
python -P -m groot.infra.oss policy-server \
  --checkpoint "$CHECKPOINT" \
  --embodiment-tag "$SERVER_EMBODIMENT" \
  --trt-engine-dir "$TRT_ARTIFACT_DIR" \
  --host 0.0.0.0 \
  --port 5555
```

To enable graph replay, add `--use-cuda-graph` to any of these commands. On the
policy server it fixes the prompt for the life of the process, disables video
prediction, and requires one real-shaped client warmup with
`PolicyClient.warmup_and_require_ready(observation)`. Benchmark both modes on
the target hardware.

TensorRT is an inference runtime and does not replace the trainable PyTorch
model used by `groot.infra.oss train`.

---

## Platform-Specific Setup

### Jetson Thor Setup

#### Bare Metal

```bash
# One-time install (validates aarch64/Python 3.12 and creates the locked
# CUDA 13 environment)
bash scripts/deployment/thor/install_deps.sh

# In each new shell
source .venv/bin/activate
source scripts/deployment/thor/activate_thor.sh
```

After activation, use plain `python` rather than `uv run` for this guide. Verify the environment
before building:

```bash
python -c \
  'import torch, tensorrt; print(torch.__version__, tensorrt.__version__, torch.cuda.get_device_name())'
```
