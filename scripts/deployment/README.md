# GR00T Deployment & Inference Guide

Run TensorRT-accelerated inference for the GR00T N2 policy.

---

## Prerequisites

- A compatible N2 model checkpoint
- A CUDA-capable target platform
- The environment for the target platform from the [installation guide](../../docs/installation.md)

TensorRT engines are specific to the GPU and software stack on which they are built. Follow
the installation path for the target platform, then use the common export, build, validation, and
serving workflow below.

| Platform | Setup | Supported workflow |
| --- | --- | --- |
| x86_64 dGPU | [Optional TensorRT setup](../../docs/installation.md#optional-tensorrt-on-x86) | Export, build, validate, and serve for compatible checkpoints |
| Jetson Thor | [Thor setup](../../docs/installation.md#jetson-thor) (bare metal) | Export, build, validate, and serve |


## TensorRT Acceleration

TensorRT replaces the N2 Transformer's and VAE encoder's normal PyTorch execution with
GPU-specific engines. The surrounding checkpoint loading, preprocessing, action decoding,
and policy server remain the same. There is no silent fallback: an invalid, foreign, or
unvalidated engine bundle stops policy construction.

### Qualified engine configurations

This runtime is unquantized BF16, batch size 1, 224 x 320 images, five history
frames per camera, joint denoising, conditional KV cache, two video denoising steps, and four
action denoising steps. Engines are built from the selected checkpoint. The following
layouts have been validated end to end; other layouts must satisfy the exporters'
constraints and pass validation before serving:

| Cameras | Action horizon |
| ---: | ---: |
| 3 | 80 |
| 4 | 48 |

## Build and validate engines

Choose a writable cache directory outside the Git checkout for the generated ONNX files,
TensorRT engines, and manifest, or omit `--trt-cache-dir` to use the default external cache.
Generated files inside the checkout can invalidate clean-source provenance. The command
builds engines locally; it does not download prebuilt engines. Serve using the validated
bundle path printed by the command.

Start from a clean Git checkout. The checkpoint can be in any local directory; it must contain
`config.json`, `processor_config.json`, `statistics.json`, and the model safetensors. All model
dependencies referenced by `config.json` must also resolve on the target host.

Set the checkpoint and cache paths. `build-trt` resolves the checkpoint contract, exports the
ONNX graphs, builds the GPU-specific engines, validates them against PyTorch, and prints the
validated bundle path. It automatically selects the workload from a single-embodiment
checkpoint; pass `--embodiment-tag` for a supported multi-embodiment checkpoint.

```bash
export CHECKPOINT=/path/to/checkpoint
export TRT_CACHE_DIR=/absolute/path/outside/checkout/trt-cache

# This must print nothing: the engine manifest records clean Git revisions.
git status --short

uv run python -P -m groot.infra.oss build-trt \
  --checkpoint "$CHECKPOINT" \
  --trt-cache-dir "$TRT_CACHE_DIR" \
  --codec mp4v
```

The build and explicit serving examples use `--codec mp4v`. Choose the codec appropriate
to your checkpoint and workflow, and use the same `--codec` for building and consuming
the bundle; a mismatch is rejected. The CLI default is `none`.

The command prints `TRT_ENGINE_DIR=<path>`. Save that path as `TRT_ARTIFACT_DIR` when using
an explicitly managed bundle. The later inference commands also require the selected workload:

To find the checkpoint's embodiment tags:

```bash
uv run python -c \
  'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["processor_kwargs"]["modality_configs"]))' \
  "$CHECKPOINT/processor_config.json"
```

```bash
export TRT_ARTIFACT_DIR=/path/printed/by/build-trt
export PROCESSOR_EMBODIMENT="<checkpoint-embodiment-tag>"
export SERVER_EMBODIMENT="$(printf '%s' "$PROCESSOR_EMBODIMENT" | tr '[:lower:]' '[:upper:]')"
```

Validation writes `n2_trt_manifest.json`; serving authenticates that manifest, the checkpoint,
all three engine files, their fixed I/O schemas, TensorRT version, and target GPU.

## Build or reuse engines automatically

For the standard OSS workflow, pass `--enable-trt`. The first invocation
exports, builds, and validates a target-local bundle; later invocations reuse
the content-addressed bundle after authenticating it against the checkpoint and
runtime:

```bash
uv run python -P -m groot.infra.oss eval \
  --checkpoint "$CHECKPOINT" \
  --dataset-path /path/to/lerobot-dataset \
  --embodiment-tag "$PROCESSOR_EMBODIMENT" \
  --codec none \
  --enable-trt
```

The default cache is `~/.cache/groot/tensorrt`. Override it with
`--trt-cache-dir` or `GROOT_TRT_CACHE_ROOT`; use `--rebuild-trt` with
`--enable-trt` to replace the matching cache entry. Automatic builds use the
checkout revision when Git metadata is available, or the installed wheel's
immutable `RECORD` contents. Unsupported checkpoint contracts and failed
validation stop the command without eager fallback. Automatic bundles
currently require `--denoising-steps 4`.

Keep cache overrides outside the checkout. The automatic example above selects
`--codec none` and builds or reuses a bundle for that codec; it does not reuse the
`mp4v` bundle from the explicit build example.

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
uv run python -P -m groot.infra.oss eval \
  --checkpoint "$CHECKPOINT" \
  --dataset-path /path/to/lerobot-dataset \
  --embodiment-tag "$PROCESSOR_EMBODIMENT" \
  --codec mp4v \
  --trt-engine-dir "$TRT_ARTIFACT_DIR"
```

Run closed-loop evaluation with any installed OSS simulator backend:

```bash
uv run python -P -m groot.infra.oss closed-loop-eval \
  --checkpoint "$CHECKPOINT" \
  --trt-engine-dir "$TRT_ARTIFACT_DIR" \
  --codec mp4v \
  --backend <python-module:BackendClass> \
  --suite <suite-name> \
  --embodiment-tag "$PROCESSOR_EMBODIMENT"
```

Start the policy server:

```bash
uv run python -P -m groot.infra.oss policy-server \
  --checkpoint "$CHECKPOINT" \
  --embodiment-tag "$SERVER_EMBODIMENT" \
  --codec mp4v \
  --trt-engine-dir "$TRT_ARTIFACT_DIR" \
  --host 0.0.0.0 \
  --port 5555
```

To enable graph replay, add `--use-cuda-graph` to any of these commands. On the
policy server it supports prompt changes with fixed input shapes, disables video
prediction, and requires one real-shaped client warmup with
`PolicyClient.warmup_and_require_ready(observation)`. Benchmark both modes on
the target hardware.

TensorRT is an inference runtime and does not replace the trainable PyTorch
model used by `groot.infra.oss train`.

---

## Platform-Specific Setup

### Jetson Thor Setup

Follow the [Thor installation and activation instructions](../../docs/installation.md#jetson-thor),
then return to this guide to build, validate, and serve your checkpoint.

### x86 Setup

Follow the [x86 TensorRT installation instructions](../../docs/installation.md#optional-tensorrt-on-x86),
then return to this guide to build, validate, and use your checkpoint's engines.
