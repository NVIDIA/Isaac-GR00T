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
| **[Jetson Thor](https://developer.nvidia.com/embedded/jetson)** | [Jetson Thor Setup](#jetson-thor-setup) (Docker or bare metal) |
| **[DGX Spark](https://developer.nvidia.com/dgx-spark)** | [DGX Spark Setup](#dgx-spark-setup) (Docker or bare metal) |
| **[Jetson Orin](https://developer.nvidia.com/embedded/jetson)** | [Jetson Orin Setup](#jetson-orin-setup) (Docker or bare metal) |

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

For demo dataset setup, see the [Data Format section in the main README](../../README.md#data-format).

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

The `trt_full_pipeline` mode (passed via `--inference-mode trt_full_pipeline`
in `standalone_inference_script.py`) accelerates all model components with
TRT engines. Speedup varies by platform — see benchmark tables below for
measured results on each device. The same pipeline is referred to as
`n17_full_pipeline` inside the engine-loading and build scripts
(`trt_model_forward.py`, `build_trt_pipeline.py`); the two names describe
the same set of engines.

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
  --embodiment-tag LIBERO_PANDA
```

> **Finetuned models:** Replace `--model-path` with your checkpoint path. The pipeline is identical for base and finetuned models.

> **Note:** Engine build takes ~2-5 minutes depending on GPU. Engines are GPU-architecture-specific and must be rebuilt for different GPUs.

> **Batch size:** The `--batch-size` value is baked as a **static** dimension into the ONNX and TRT models. Engines built with one batch size cannot be used with a different batch size at runtime. If you need a different batch size, re-run the full pipeline (`--steps export,build,verify`) with the new `--batch-size` value.

You can also run a subset of steps:

```bash
# Export + build only (skip verify and benchmark)
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
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

GR00T N1.7 Inference Timing (4 denoising steps, 1 camera):

| Device | Mode | Data Processing | Backbone | Action Head | E2E | Frequency | E2E Speedup |
|--------|------|-----------------|----------|-------------|-----|-----------|-------------|
| **dGPU** | | | | | | | |
| H100 80GB HBM3 | PyTorch Eager | 6.2 ms | 31.3 ms | 48.2 ms | 85.8 ms | 11.7 Hz | 1.00x |
| | torch.compile | 6.2 ms | 30.4 ms | 12.0 ms | 48.6 ms | 20.6 Hz | 1.77x |
| | **TensorRT (Full Pipeline)** | **6.2 ms** | **8.8 ms** | **12.3 ms** | **27.9 ms** | **35.9 Hz** | **3.08x** |
| H20 96GB HBM3 | PyTorch Eager | 5.33 ms | 30.8 ms | 47.3 ms | 83.4 ms | 12.0 Hz | 1.00x |
| | torch.compile | 5.33 ms | 31.1 ms | 13.3 ms | 49.7 ms | 20.1 Hz | 1.68x |
| | **TensorRT (Full Pipeline)** | **5.33 ms** | **14.2 ms** | **14.5 ms** | **34.0 ms** | **29.4 Hz** | **2.45x** |
| RTX Pro 6000 Blackwell | PyTorch Eager | 4.8 ms | 29.3 ms | 44.0 ms | 78.4 ms | 12.8 Hz | 1.00x |
| | torch.compile | 4.8 ms | 29.4 ms | 16.5 ms | 50.7 ms | 19.7 Hz | 1.55x |
| | **TensorRT (Full Pipeline)** | **4.8 ms** | **9.9 ms** | **13.2 ms** | **27.9 ms** | **35.9 Hz** | **2.81x** |
| RTX Pro 5000 72GB | PyTorch Eager | 8.85 ms | 54.01 ms | 63.19 ms | 126.4 ms | 7.9 Hz | 1.00x |
| | torch.compile | 8.85 ms | 55.74 ms | 20.38 ms | 84.9 ms | 11.8 Hz | 1.49x |
| | **TensorRT (Full Pipeline)** | **8.85 ms** | **14.37 ms** | **17.33 ms** | **40.5 ms** | **24.7 Hz** | **3.13x** |
| L40 | PyTorch Eager | 6.6 ms | 42.8 ms | 78.9 ms | 128.3 ms | 7.8 Hz | 1.00x |
| | torch.compile | 6.6 ms | 42.7 ms | 19.8 ms | 69.0 ms | 14.5 Hz | 1.86x |
| | **TensorRT (Full Pipeline)** | **6.6 ms** | **13.1 ms** | **18.8 ms** | **38.4 ms** | **26.0 Hz** | **3.34x** |
| L20 | PyTorch Eager | 5.7 ms | 47.58 ms | 86.92 ms | 140.3 ms | 7.1 Hz | 1.00x |
| | torch.compile | 5.7 ms | 47.2 ms | 20.18 ms | 73.1 ms | 13.7 Hz | 1.92x |
| | **TensorRT (Full Pipeline)** | **5.7 ms** | **17.27 ms** | **19.79 ms** | **42.8 ms** | **23.3 Hz** | **3.28x** |
| **Jetson / Spark** | | | | | | | |
| DGX Spark | PyTorch Eager | 13.14 ms | 38.22 ms | 74.94 ms | 126.4 ms | 7.9 Hz | 1.00x |
| | torch.compile | 13.14 ms | 39.23 ms | 56.49 ms | 108.8 ms | 9.2 Hz | 1.16x |
| | **TensorRT (Full Pipeline)** | **13.14 ms** | **33.43 ms** | **52.37 ms** | **98.6 ms** | **10.1 Hz** | **1.28x** |
| AGX Thor | PyTorch Eager | 8.21 ms | 55.26 ms | 81.65 ms | 144.9 ms | 6.9 Hz | 1.00x |
| | torch.compile | 8.21 ms | 55.59 ms | 64.66 ms | 128.4 ms | 7.8 Hz | 1.13x |
| | **TensorRT (Full Pipeline)** | **8.21 ms** | **28.89 ms** | **56.64 ms** | **93.8 ms** | **10.7 Hz** | **1.54x** |
| Orin | PyTorch Eager | 9.45 ms | 127.6 ms | 205.39 ms | 342.8 ms | 2.9 Hz | 1.00x |
| | torch.compile | 9.45 ms | 128.59 ms | 78.94 ms | 217.0 ms | 4.6 Hz | 1.58x |
| | **TensorRT (DiT-only)** | **9.45 ms** | **128.38 ms** | **78.6 ms** | **216.5 ms** | **4.6 Hz** | **1.58x** |

> **Note:** Orin uses DiT-only TensorRT (`--inference-mode tensorrt`) because TRT 10.3 does not support the backbone engine. All other platforms use the full pipeline (`--inference-mode trt_full_pipeline`).

<details>
<summary>Raw benchmark output (H100 80GB HBM3)</summary>

```
Hardware: NVIDIA H100 80GB HBM3
Model: checkpoints/GR00T-N1.7-LIBERO/libero_10
1 camera, Denoising Steps: 4

PyTorch Eager:
  E2E:             85.8 ms (11.7 Hz)
  Data Processing: 6.2 ms | Backbone: 31.3 ms | Action Head: 48.2 ms

torch.compile:
  E2E:             48.6 ms (20.6 Hz), 1.77x speedup
  Data Processing: 6.2 ms | Backbone: 30.4 ms | Action Head: 12.0 ms

TensorRT (Full Pipeline):
  E2E:             27.9 ms (35.9 Hz), 3.08x speedup
  Data Processing: 6.2 ms | Backbone: 8.8 ms  | Action Head: 12.3 ms
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
  --embodiment-tag LIBERO_PANDA
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
  --embodiment-tag LIBERO_PANDA
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
  --embodiment-tag LIBERO_PANDA \
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

> **Orin TRT limitations:** TRT 10.3 on Orin does not support the backbone (LLM) engine — the build step will report a failure for `llm_bf16.engine` and that is expected. The remaining 6 engines build successfully. Use `--export-mode action_head` for verification and `--inference-mode tensorrt` (DiT-only TRT, backbone runs in PyTorch) for inference:
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
| `--batch-size` | `1` | Batch size baked into exported ONNX/TRT models (static — see note below) |
| `--export-mode` | `full_pipeline` | Export mode: `dit_only`, `action_head`, or `full_pipeline` |
| `--video-backend` | `torchcodec` | Video backend for dataset loading |
| `--workspace` | `8192` | TRT builder workspace size in MB |
| `--num-iterations` | `20` | Number of benchmark iterations |
| `--warmup` | `5` | Number of warmup iterations |
| `--skip-compile` | `false` | Skip torch.compile benchmark |
| `--steps` | `all` | Steps to run: `all` or comma-separated subset of `export,build,verify,benchmark` |
| `--log-file` | `<output-dir>/pipeline.log` | Log file path |

### `standalone_inference_script.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint |
| `--dataset-path` | `demo_data/droid_sample` | Path to dataset (LeRobot format) |
| `--embodiment-tag` | Auto-detected | Robot embodiment tag |
| `--traj-ids` | `[0]` | Episode indices to evaluate (space-separated) |
| `--steps` | `200` | Max steps per trajectory (capped by actual length) |
| `--action-horizon` | `16` | Action prediction horizon |
| `--inference-mode` | `pytorch` | `pytorch`, `tensorrt` (DiT-only TRT), or `trt_full_pipeline` (all engines) |
| `--trt-engine-path` | `./gr00t_n1d7_engines` | Directory containing pre-built TRT engines |
| `--denoising-steps` | `4` | Diffusion denoising iterations |
| `--save-plot-path` | `None` | Save per-trajectory GT-vs-predicted comparison plots |
| `--video-backend` | `torchcodec` | Video decoder: `torchcodec`, `decord`, or `torchvision_av` |
| `--skip-timing-steps` | `1` | Initial steps excluded from timing stats (warmup) |
| `--host` / `--port` | `127.0.0.1` / `5555` | Server address (when using client mode without `--model-path`) |
| `--seed` | `42` | Random seed for reproducibility |

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
