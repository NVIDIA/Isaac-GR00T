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

## Serve a policy

CUDA graph replay is disabled by default. Start the server:

```bash
python -P -m groot.core.policy.run_gr00t_server \
  --model-path "$CHECKPOINT" \
  --embodiment-tag "$SERVER_EMBODIMENT" \
  --trt-engine-dir "$TRT_ARTIFACT_DIR" \
  --host 0.0.0.0 \
  --port 5555
```

To enable graph replay, add `--use-cuda-graph` to the command. It fixes the prompt for the life of
the server process, disables video prediction, and requires one real-shaped client warmup with
`PolicyClient.warmup_and_require_ready(observation)`. Benchmark both modes on the target hardware.

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
