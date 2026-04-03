# GR00T Deployment & Inference Guide

Run inference with PyTorch or TensorRT acceleration for the GR00T N1.7 policy.

---

## Prerequisites

- Model checkpoint: `nvidia/GR00T-N1.7-3B`
- Dataset in LeRobot format (e.g., `demo_data/libero_demo`)
- CUDA-enabled GPU
- Setup uv environment following README.md

| Platform | Installation |
|----------|-------------|
| **dGPU** (H100, A100, RTX 4090/5090, L20, RTX Pro 5000/6000, etc.) | `uv sync` — GPU deps (`flash-attn`, `onnx`, `tensorrt`) included |
| **Jetson Thor** | [Jetson Thor Setup](#jetson-thor-setup) (Docker or bare metal) |
| **DGX Spark** | [DGX Spark Setup](#dgx-spark-setup) (Docker or bare metal) |
| **Jetson Orin** | [Jetson Orin Setup](#jetson-orin-setup) (Docker or bare metal) |

- dGPU local environment: use the installation commands below, then use the PyTorch or TensorRT commands in this guide
- Thor Docker or bare metal: skip to [Jetson Thor Setup](#jetson-thor-setup)
- Spark Docker or bare metal: skip to [DGX Spark Setup](#dgx-spark-setup)
- Orin Docker or bare metal: skip to [Jetson Orin Setup](#jetson-orin-setup)

### dGPU Installation

```bash
uv sync
```

GPU dependencies (`flash-attn`, `onnx`, `tensorrt`) are included in the default install.

## Download Model and Dataset

Download the finetuned model to a local directory (HuggingFace does not support nested repo paths directly):

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO \
  --include "libero_10/config.json" "libero_10/embodiment_id.json" \
  "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
  "libero_10/processor_config.json" "libero_10/statistics.json" \
  --local-dir checkpoints/GR00T-N1.7-LIBERO
```

For demo dataset setup, see the [Getting Started section in the main README](../../README.md#getting-started).

---

## Quick Start: PyTorch Inference

Run inference on demo trajectories using PyTorch (no TRT setup needed):

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 1 2 3 4 \
  --inference-mode pytorch \
  --action-horizon 8
```

---

## TensorRT Acceleration

The `n17_full_pipeline` mode accelerates all model components with TRT engines.
Speedup varies by platform — see benchmark tables below for measured results on each device.

| Component | Engine | Notes |
|-----------|--------|-------|
| ViT | **TRT** | Qwen3-VL Vision (24 blocks, FP32 for accuracy) |
| LLM | **TRT** | Qwen3-VL Text Model (16 layers, with deepstack injection) |
| VL Self-Attention | **TRT** | SelfAttentionTransformer (4 layers, if present) |
| State Encoder | **TRT** | CategorySpecificMLP |
| Action Encoder | **TRT** | MultiEmbodimentActionEncoder |
| DiT | **TRT** | AlternateVLDiT (32 layers) |
| Action Decoder | **TRT** | CategorySpecificMLP |

Lightweight ops remain in PyTorch: `embed_tokens`, `masked_scatter`, `get_rope_index`, VLLN.

<details>
<summary>DiT-only mode (legacy from N1.6)</summary>

The `dit_only` export mode (`--export-mode dit_only`) optimizes only the action head DiT, leaving the backbone in PyTorch. This was the default in N1.6. For N1.7, **full_pipeline is recommended** as it accelerates the backbone (ViT + LLM) which dominates inference time.
</details>

### Build TRT Engines

The unified `build_trt_pipeline.py` script runs all steps (export ONNX → build engines → verify accuracy → benchmark) in a single command:

```bash
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag libero_sim
```

> **Finetuned models:** Replace `--model-path` with your checkpoint path. The pipeline is identical for base and finetuned models.

> **Note:** Engine build takes ~2-5 minutes depending on GPU. Engines are GPU-architecture-specific and must be rebuilt for different GPUs.

You can also run a subset of steps:

```bash
# Export + build only (skip verify and benchmark)
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag libero_sim \
  --steps export,build
```

<details>
<summary>What each step does</summary>

The pipeline runs 4 steps in sequence:

1. **Export to ONNX** (`export`) — Exports all model components (LLM, VL Self-Attention, State Encoder, Action Encoder, DiT, Action Decoder) to ONNX format under `<output-dir>/onnx/`.
2. **Build TensorRT Engines** (`build`) — Compiles each ONNX model into a GPU-specific TensorRT engine under `<output-dir>/engines/`.
3. **Verify Accuracy** (`verify`) — Runs PyTorch vs TRT output comparison. Expected: `Cosine Similarity: 0.999+` (PASS).
4. **Benchmark** (`benchmark`) — Measures E2E latency for PyTorch Eager, torch.compile, and TRT modes.

Each step can be run individually via `--steps <step>`. Verbose logs are written to `<output-dir>/pipeline.log`.
</details>

---

## Performance

### Benchmark Results

GR00T N1.7 Inference Timing (4 denoising steps):

| Device | Mode | Data Processing | Backbone | Action Head | E2E | Frequency | E2E Speedup |
|--------|------|-----------------|----------|-------------|-----|-----------|-------------|
| H100 | PyTorch Eager | 7 ms | 50 ms | 99 ms | 156 ms | 6.4 Hz | 1.00x |
| H100 | torch.compile | 7 ms | 51 ms | 14 ms | 72 ms | 13.9 Hz | 2.16x |
| H100 | **TensorRT (n17_full_pipeline)** | **7 ms** | **10 ms** | **13 ms** | **31 ms** | **32.7 Hz** | **5.11x** |
| Thor | PyTorch Eager | 8 ms | 55 ms | 75 ms | 139 ms | 7.2 Hz | 1.00x |
| Thor | torch.compile | 8 ms | 57 ms | 68 ms | 133 ms | 7.5 Hz | 1.10x |
| Thor | **TRT (n17_full_pipeline)** | **8 ms** | **30 ms** | **57 ms** | **94 ms** | **10.6 Hz** | **1.32x** |
| Spark | PyTorch Eager | 13 ms | 38 ms | 75 ms | 126 ms | 7.9 Hz | 1.00x |
| Spark | torch.compile | 13 ms | 39 ms | 56 ms | 109 ms | 9.2 Hz | 1.16x |
| Spark | **TensorRT (n17_full_pipeline)** | **13 ms** | **33 ms** | **52 ms** | **99 ms** | **10.1 Hz** | **1.28x** |
| Orin | PyTorch Eager | 10 ms | 128 ms | 204 ms | 341 ms | 2.9 Hz | 1.00x |
| Orin | torch.compile | 10 ms | 127 ms | 79 ms | 217 ms | 4.6 Hz | 1.57x |
| Orin | **TensorRT (dit_only)** | **10 ms** | **127 ms** | **78 ms** | **215 ms** | **4.7 Hz** | **1.59x** |

<details>
<summary>Raw benchmark output (H100)</summary>

```
Hardware: NVIDIA H100 80GB HBM3
Model: checkpoints/GR00T-N1.7-LIBERO/libero_10
Action Horizon: 40, Denoising Steps: 4

PyTorch Eager:
  E2E:             median=164.7 ms, mean=164.5 ± 2.6 ms (6.1 Hz)
  Backbone:        52.53 ms | Action Head: 103.30 ms

torch.compile:
  E2E:             median=72.7 ms, mean=73.0 ± 1.1 ms (13.8 Hz)
  Backbone:        51.99 ms | Action Head: 13.58 ms

TensorRT (n17_full_pipeline):
  E2E:             median=29.6 ms, mean=29.9 ± 0.8 ms (33.8 Hz)
  Backbone:        9.30 ms  | Action Head: 13.21 ms
```
</details>

### Standalone Inference with TRT

The standalone inference script serves as both an accuracy validation and a reference for deploying TRT inference in your own code. It runs per-step inference on real trajectories and compares action predictions:

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 1 2 3 4 \
  --inference-mode trt_full_pipeline \
  --trt-engine-path ./gr00t_trt_deployment/engines \
  --save-plot-path ./output/trt_inference.png
```

Expected accuracy: MSE/MAE match PyTorch within noise. TRT produces identical action quality. Speedup varies by platform — run `build_trt_pipeline.py --steps benchmark` on your hardware for exact numbers.

### Optional: LIBERO Closed-Loop Sim Evaluation

To validate TRT accuracy in end-to-end robotic tasks, run the LIBERO closed-loop evaluation. This requires a separate environment setup (~10-30 min, MuJoCo simulator + dependencies).

<details>
<summary>Setup, commands, and results (H100, 20 episodes)</summary>

Task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, 20 episodes:

| Mode | Success Rate |
|------|-------------|
| PyTorch | 100% (20/20) |
| TRT (n17_full_pipeline) | 95% (19/20) |

Difference is within simulation noise (p >> 0.05).

> **Note:** Use `--n-envs 1` for TRT evaluation (ViT engine has static shapes for single-observation inference).

```bash
# One-time LIBERO setup (~10 min)
bash gr00t/eval/sim/LIBERO/setup_libero.sh

# Activate LIBERO venv and install additional deps
source gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
uv pip install diffusers transformers accelerate safetensors torchcodec

# TRT full pipeline evaluation
python gr00t/eval/rollout_policy.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --env-name "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
  --n-episodes 20 --n-envs 1 --max-episode-steps 504 \
  --trt-engine-path ./gr00t_trt_deployment/engines \
  --trt-mode n17_full_pipeline
```
</details>

> Run `python scripts/deployment/build_trt_pipeline.py --steps benchmark` to generate benchmarks for your hardware.

---

## Platform-Specific Setup

> Jetson and Spark platforms use different dependency stacks than dGPU. Thor and Spark use CUDA 13 with PyTorch 2.10.0 from the [Jetson AI Lab cu130 index](https://pypi.jetson-ai-lab.io/sbsa/cu130). Orin uses CUDA 12.6 with PyTorch 2.10.0 from the [Jetson AI Lab cu126 index](https://pypi.jetson-ai-lab.io/jp6/cu126).

### Jetson Thor Setup

Thor uses CUDA 13 and Python 3.12, which require a different dependency stack than x86 or Orin.
Tested with JetPack 7.1.
There are two ways to run on Thor: Docker (recommended) or bare metal.

<details>
<summary><strong>Docker (Recommended)</strong></summary>

Build the Thor container from the repo root:

```bash
cd docker && bash build.sh --profile=thor && cd ..
```

Download the finetuned model (run once, on the host):

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
```

Start an interactive Docker session (recommended for multi-step TRT work):

```bash
docker run -it --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HF_HOME:-${HOME}/.cache/huggingface}":/root/.cache/huggingface \
  -w /workspace/repo \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  gr00t-thor \
  bash
```

Then inside the container, run the full TRT pipeline (export, build, verify, benchmark):

```bash
python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag libero_sim
```
</details>

<details>
<summary><strong>Bare Metal</strong></summary>

```bash
# One-time install (temporarily copies the Thor pyproject.toml and uv.lock to repo root,
# installs NVPL libs, uv, Python deps, and builds torchcodec from source against the
# system FFmpeg runtime)
bash scripts/deployment/thor/install_deps.sh

# In each new shell
source .venv/bin/activate
source scripts/activate_thor.sh
```

Then run the TRT pipeline or PyTorch inference as shown in the [TensorRT Acceleration](#tensorrt-acceleration) and [Quick Start](#quick-start-pytorch-inference) sections above.
The activation script exports the PyTorch and CUDA library/include paths that `torchcodec`
and `torch.compile` need on Thor.
</details>

---

### DGX Spark Setup

Spark uses CUDA 13 and Python 3.12 like Thor, but requires a dedicated dependency stack and
source-built `flash-attn` for `sm121`. There are two ways to run on Spark: Docker (recommended)
or bare metal.

<details>
<summary><strong>Docker (Recommended)</strong></summary>

Build the Spark container from the repo root:

```bash
cd docker && bash build.sh --profile=spark && cd ..
```

Download the finetuned model (run once, on the host):

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
```

Start an interactive Docker session (recommended for multi-step TRT work):

```bash
docker run -it --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HF_HOME:-${HOME}/.cache/huggingface}":/root/.cache/huggingface \
  -w /workspace/repo \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  gr00t-spark \
  bash
```

Then inside the container, run the full TRT pipeline (export, build, verify, benchmark):

```bash
python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag libero_sim
```
</details>

<details>
<summary><strong>Bare Metal</strong></summary>

```bash
# One-time install (temporarily copies the Spark pyproject.toml and uv.lock to repo root,
# installs NVPL libs, uv, Python deps, source-builds flash-attn for sm121, and builds
# torchcodec from source against the system FFmpeg runtime)
bash scripts/deployment/spark/install_deps.sh

# In each new shell
source .venv/bin/activate
source scripts/activate_spark.sh
```

Then run the TRT pipeline or PyTorch inference as shown in the [TensorRT Acceleration](#tensorrt-acceleration) and [Quick Start](#quick-start-pytorch-inference) sections above.
If you later rerun `uv sync`, rerun `bash scripts/deployment/spark/install_deps.sh` so the
Spark-specific `flash-attn` build is restored and revalidated.
</details>

---

### Jetson Orin Setup

> **Note:** On Orin, only the DiT (action head) TRT export is currently supported. Use `--export-mode dit_only` instead of `full_pipeline`. Full pipeline support is in progress.

Orin uses CUDA 12.6 and Python 3.10 (JetPack 6.2), which require a different dependency stack than x86 or Thor.
Tested with JetPack 6.2.
There are two ways to run on Orin: Docker (recommended) or bare metal.

<details>
<summary><strong>Docker (Recommended)</strong></summary>

Build the Orin container from the repo root:

```bash
cd docker && bash build.sh --profile=orin && cd ..
```

Download the finetuned model (run once, on the host):

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
```

Start an interactive Docker session (recommended for multi-step TRT work):

```bash
docker run -it --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HF_HOME:-${HOME}/.cache/huggingface}":/root/.cache/huggingface \
  -w /workspace/repo \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  gr00t-orin \
  bash
```

Then inside the container, run the TRT pipeline (DiT-only on Orin):

```bash
python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag libero_sim \
  --export-mode dit_only
```
</details>

<details>
<summary><strong>Bare Metal</strong></summary>

```bash
# One-time install (temporarily copies the Orin pyproject.toml and uv.lock to repo root,
# installs uv, Python deps, and builds torchcodec from source against JetPack's FFmpeg
# runtime)
bash scripts/deployment/orin/install_deps.sh

# In each new shell
source .venv/bin/activate
source scripts/activate_orin.sh
```

Then run the TRT pipeline (with `--export-mode dit_only`) or PyTorch inference as shown in the [TensorRT Acceleration](#tensorrt-acceleration) and [Quick Start](#quick-start-pytorch-inference) sections above.
The activation script exports the PyTorch and CUDA library/include paths that `torchcodec`
and `torch.compile` need on Orin.
</details>

> **Orin storage tip:** If your eMMC root is low on space, redirect the HuggingFace cache to an NVMe SSD with `export HF_HOME=/path/to/ssd/.cache/huggingface` before downloading models.

> **Orin TRT limitations:** TRT 10.3 on Orin does not support the backbone (LLM) engine — the build step will report a failure for `llm_bf16.engine` and that is expected. The remaining 6 engines build successfully. Use `--mode action_head` for verification and `--trt-mode dit_only` for inference:
> ```bash
> python scripts/deployment/build_trt_pipeline.py \
>   --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
>   --dataset-path demo_data/libero_demo \
>   --export-mode action_head \
>   --steps verify
>
> python scripts/deployment/standalone_inference_script.py \
>   --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
>   --dataset-path demo_data/libero_demo \
>   --embodiment-tag LIBERO_PANDA \
>   --traj-ids 0 \
>   --inference-mode tensorrt \
>   --trt-engine-path ./gr00t_n1d7_engines
> ```

---

## Command-Line Arguments

### `build_trt_pipeline.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint |
| `--dataset-path` | `demo_data/libero_demo` | Path to dataset (LeRobot format) |
| `--embodiment-tag` | Auto-detected | Embodiment tag (auto-detected from processor_config.json if single embodiment) |
| `--output-dir` | `./gr00t_trt_deployment` | Root output directory. ONNX → `<output-dir>/onnx/`, engines → `<output-dir>/engines/` |
| `--precision` | `bf16` | Precision for ONNX export and TRT engine build (`bf16`, `fp16`, `fp32`) |
| `--batch-size` | `1` | Batch size baked into exported ONNX/TRT models |
| `--export-mode` | `full_pipeline` | Export mode: `dit_only`, `action_head`, or `full_pipeline` |
| `--video-backend` | `torchcodec` | Video backend for dataset loading |
| `--workspace` | `8192` | TRT builder workspace size in MB |
| `--num-iterations` | `20` | Number of benchmark iterations |
| `--warmup` | `5` | Number of warmup iterations |
| `--skip-compile` | `false` | Skip torch.compile benchmark |
| `--steps` | `all` | Steps to run: `all` or comma-separated subset of `export,build,verify,benchmark` |
| `--log-file` | `<output-dir>/pipeline.log` | Log file path |

## Files

| File | Description |
|------|-------------|
| `build_trt_pipeline.py` | Unified pipeline: export ONNX, build engines, verify, benchmark |
| `standalone_inference_script.py` | Main inference script (PyTorch + DiT-only TensorRT) |
| `trt_torch.py` | TRT Engine wrapper class (load, bind, execute) |
| `trt_model_forward.py` | TRT forward functions and setup (backbone + action head) |

---

## Troubleshooting

### Engine Build Fails

- Ensure you have enough GPU memory (16GB+ recommended for full pipeline)
- Try reducing workspace size: `--workspace 4096`
- Ensure TensorRT version matches your CUDA version
- LLM engine requires `batch_size` dimension handling when using custom shape profiles

### ONNX Export Issues

- If export fails with COMPLEX128 error: ensure `_simple_causal_mask` is used (not HuggingFace's `create_causal_mask`)
- If `masked_scatter` size assertion fails: ensure `visual_pos_masks` has the correct number of True values matching deepstack tensor size
- Check that the dataset path is valid and contains at least one trajectory

### Accuracy Issues

- If cosine < 0.99: check that LLM export does NOT include the final RMSNorm (backbone returns pre-norm `hidden_states[-1]`)
- If output magnitude is ~12x too small: this is the norm bug — see above
- Run `build_trt_pipeline.py --steps verify --export-mode action_head` first to isolate backbone vs action head drift
