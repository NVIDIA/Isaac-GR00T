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

## Quick Start: PyTorch Mode

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

## TensorRT Acceleration (3x Faster)

The `n17_full_pipeline` mode accelerates all model components with TRT engines:

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

### Step 0: Download Model and Dataset

Download the finetuned model to a local directory (HuggingFace does not support nested repo paths directly):

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO \
  --include "libero_10/config.json" "libero_10/embodiment_id.json" \
  "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
  "libero_10/processor_config.json" "libero_10/statistics.json" \
  --local-dir checkpoints/GR00T-N1.7-LIBERO
```

For demo dataset setup, see the [Getting Started section in the main README](../../README.md#getting-started).

### Step 1: Export to ONNX

```bash
uv run python scripts/deployment/export_onnx_n1d7.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --output-dir ./gr00t_n1d7_onnx \
  --export-mode full_pipeline \
  --embodiment-tag LIBERO_PANDA
```

**Output:** ONNX files in `./gr00t_n1d7_onnx/` (LLM, VL Self-Attention, State Encoder, Action Encoder, DiT, Action Decoder)

> **Finetuned models:** Replace `--model-path` with your checkpoint path. The export pipeline is identical for base and finetuned models.

### Step 2: Build TensorRT Engines

```bash
uv run python scripts/deployment/build_tensorrt_engine.py \
  --mode full_pipeline \
  --onnx-dir ./gr00t_n1d7_onnx \
  --engine-dir ./gr00t_n1d7_engines \
  --precision bf16
```

> **Note:** Engine build takes ~2-5 minutes depending on GPU. Engines are GPU-architecture-specific and must be rebuilt for different GPUs.

### Step 3: Verify Accuracy

```bash
uv run python scripts/deployment/verify_n1d7_trt.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --engine-dir ./gr00t_n1d7_engines \
  --mode n17_full_pipeline
```

Expected output: `Cosine Similarity: 0.999+` (PASS).

---

### Step 4: Run Benchmark

```bash
uv run python scripts/deployment/benchmark_inference.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --trt-engine-path ./gr00t_n1d7_engines \
    --trt-mode n17_full_pipeline
```

## Performance

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint (local path) |
| `--dataset-path` | `demo_data/libero_demo` | Path to dataset |
| `--embodiment-tag` | `LIBERO_PANDA` | Embodiment tag |
| `--trt-engine-path` | (optional) | Path to TensorRT engine |
| `--num-iterations` | `20` | Number of benchmark iterations |
| `--warmup` | `5` | Number of warmup iterations |
| `--skip_compile` | `false` | Skip torch.compile benchmark |
| `--seed` | `42` | Random seed for reproducibility |

### N1.7 TRT (BF16, 4 denoising steps, ViT in PyTorch)

| Device | Mode | Data Processing | Backbone | Action Head | E2E | Frequency |
|--------|------|-----------------|----------|-------------|-----|-----------|
| H100 | PyTorch Eager | 7 ms | 50 ms | 99 ms | 156 ms | 6.4 Hz |
| H100 | torch.compile | 7 ms | 51 ms | 14 ms | 72 ms | 13.9 Hz |
| H100 | **TRT (n17_full_pipeline)** | **7 ms** | **27 ms** | **13 ms** | **47 ms** | **21.1 Hz** |
| Orin | PyTorch Eager | 10 ms | 128 ms | 204 ms | 341 ms | 2.9 Hz |
| Orin | torch.compile | 10 ms | 127 ms | 79 ms | 217 ms | 4.6 Hz |
| Orin | **TRT (dit_only)** | **10 ms** | **127 ms** | **78 ms** | **215 ms** | **4.7 Hz** |

| Device | Mode | E2E Speedup | Action Head Speedup |
|--------|------|-------------|---------------------|
| H100 | PyTorch Eager | 1.00x | 1.00x |
| H100 | torch.compile | 2.16x | 7.31x |
| H100 | TRT (n17_full_pipeline) | 3.29x | 7.62x |
| Orin | PyTorch Eager | 1.00x | 1.00x |
| Orin | torch.compile | 1.57x | 2.58x |
| Orin | TRT (dit_only) | 1.59x | 2.61x |

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

We are verified the accuracy of TensorRT-exported engines, and in Step 3 we did that. But if you want to test it in end2end robotic tasks, it takes a bit more setup effort to run the following validations with LIBERO.

### Optional: Standalone Inference Accuracy Validation

Compare PyTorch vs TRT action predictions on real trajectories (no extra setup needed):

<details>
<summary>Commands and results (H100, LIBERO, 5 trajectories)</summary>

```bash
# PyTorch baseline
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 1 2 3 4 \
  --inference-mode pytorch \
  --save-plot-path ./output/pytorch_inference.png

# TRT full pipeline
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 1 2 3 4 \
  --inference-mode trt_full_pipeline \
  --trt-engine-path ./gr00t_n1d7_engines \
  --save-plot-path ./output/trt_inference.png
```

| Mode | Avg MSE | Avg MAE | Avg Inference/Step |
|------|---------|---------|-------------------|
| PyTorch Eager | 0.002787 | 0.017541 | 243 ms |
| TRT (n17_full_pipeline) | 0.002811 | 0.017527 | 35 ms |

MSE/MAE match within noise — TRT produces identical action quality.

> **Note on timing gap:** Standalone per-step times (243 ms PyTorch, 35 ms TRT) differ from benchmark E2E times (165 ms, 30 ms) above. We believe this is because the standalone script uses `time.time()` without `torch.cuda.synchronize()`, which can overcount PyTorch async GPU ops. TRT engine calls are synchronous by default, so TRT timing is consistent across both scripts. The benchmark numbers (with GPU sync) are more accurate for pure inference throughput.
</details>

### Optional: LIBERO Closed-Loop Sim Evaluation

Robot task success rate comparison. Requires separate LIBERO environment setup (~10-30 min, MuJoCo simulator + dependencies).

<details>
<summary>Setup, commands, and results (H100, 20 episodes)</summary>

Task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, 20 episodes:

| Mode | Success Rate |
|------|-------------|
| PyTorch | 100% (20/20) |
| TRT (n17_full_pipeline) | 95% (19/20) |

Difference is within simulation noise (p >> 0.05).

> **Note:** LIBERO evaluation requires a separate environment setup which can take some effort.
> Use `--n-envs 1` for TRT evaluation (ViT engine has static shapes for single-observation inference).

```bash
# One-time LIBERO setup (~10 min)
bash gr00t/eval/sim/LIBERO/setup_libero.sh

# Activate LIBERO venv and install additional deps
source gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
uv pip install diffusers transformers accelerate safetensors torchcodec

# PyTorch baseline
python gr00t/eval/rollout_policy.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --env-name "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
  --n-episodes 20 --n-envs 1 --max-episode-steps 504

# TRT full pipeline
python gr00t/eval/rollout_policy.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --env-name "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
  --n-episodes 20 --n-envs 1 --max-episode-steps 504 \
  --trt-engine-path ./gr00t_n1d7_engines \
  --trt-mode n17_full_pipeline
```
</details>

> Run `python scripts/deployment/benchmark_inference.py` to generate benchmarks for your hardware.

> Jetson and Spark platforms use different dependency stacks than dGPU. Thor and Spark use CUDA 13 with PyTorch 2.10.0 from the [Jetson AI Lab cu130 index](https://pypi.jetson-ai-lab.io/sbsa/cu130). Orin uses CUDA 12.6 with PyTorch 2.10.0 from the [Jetson AI Lab cu126 index](https://pypi.jetson-ai-lab.io/jp6/cu126). See the platform-specific setup sections below.
---

## Platform-Specific Setup (TODO: verify install and update benchmark numbers)

Thor uses CUDA 13 and Python 3.12, which require a different dependency stack than x86 or Orin.
Tested with JetPack 7.1.
There are two ways to run on Thor: Docker (recommended) or bare metal.

### Docker (Recommended)

Build the Thor container from the repo root:

```bash
cd docker && bash build.sh --profile=thor && cd ..
```

Download the finetuned model (run once, on the host):

```bash
huggingface-cli download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
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

Then inside the container, run the full pipeline:

```bash
# Step 1: PyTorch inference (quick sanity check)
python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 \
  --inference-mode pytorch \
  --denoising-steps 4

# Step 2: Export to ONNX
python scripts/deployment/export_onnx_n1d7.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --output-dir ./gr00t_n1d7_onnx \
  --export-mode full_pipeline \
  --embodiment-tag LIBERO_PANDA

# Step 3: Build TensorRT engines
python scripts/deployment/build_tensorrt_engine.py \
  --mode full_pipeline \
  --onnx-dir ./gr00t_n1d7_onnx \
  --engine-dir ./gr00t_n1d7_engines \
  --precision bf16

# Step 4: Verify TRT accuracy
python scripts/deployment/verify_n1d7_trt.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --engine-dir ./gr00t_n1d7_engines \
  --mode n17_full_pipeline

# Step 5: Benchmark (PyTorch + torch.compile + TRT)
python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_n1d7_engines \
  --trt-mode n17_full_pipeline
```

### Bare Metal

```bash
# One-time install (temporarily copies the Thor pyproject.toml and uv.lock to repo root,
# installs NVPL libs, uv, Python deps, and builds torchcodec from source against the
# system FFmpeg runtime)
bash scripts/deployment/thor/install_deps.sh

# In each new shell
source .venv/bin/activate
source scripts/activate_thor.sh
```

Then run inference or benchmarks as shown in the Quick Start section above.
The activation script exports the PyTorch and CUDA library/include paths that `torchcodec`
and `torch.compile` need on Thor.

---

## DGX Spark Setup

Spark uses CUDA 13 and Python 3.12 like Thor, but requires a dedicated dependency stack and
source-built `flash-attn` for `sm121`. There are two ways to run on Spark: Docker (recommended)
or bare metal.

### Docker (Recommended)

Build the Spark container from the repo root:

```bash
cd docker && bash build.sh --profile=spark && cd ..
```

Download the finetuned model (run once, on the host):

```bash
huggingface-cli download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
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

Then inside the container, run the full pipeline:

```bash
# Step 1: PyTorch inference (quick sanity check)
python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 \
  --inference-mode pytorch \
  --denoising-steps 4

# Step 2: Export to ONNX
python scripts/deployment/export_onnx_n1d7.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --output-dir ./gr00t_n1d7_onnx \
  --export-mode full_pipeline \
  --embodiment-tag LIBERO_PANDA

# Step 3: Build TensorRT engines
python scripts/deployment/build_tensorrt_engine.py \
  --mode full_pipeline \
  --onnx-dir ./gr00t_n1d7_onnx \
  --engine-dir ./gr00t_n1d7_engines \
  --precision bf16

# Step 4: Verify TRT accuracy
python scripts/deployment/verify_n1d7_trt.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --engine-dir ./gr00t_n1d7_engines \
  --mode n17_full_pipeline

# Step 5: Benchmark (PyTorch + torch.compile + TRT)
python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_n1d7_engines \
  --trt-mode n17_full_pipeline
```

### Bare Metal

```bash
# One-time install (temporarily copies the Spark pyproject.toml and uv.lock to repo root,
# installs NVPL libs, uv, Python deps, source-builds flash-attn for sm121, and builds
# torchcodec from source against the system FFmpeg runtime)
bash scripts/deployment/spark/install_deps.sh

# In each new shell
source .venv/bin/activate
source scripts/activate_spark.sh
```

Then run inference or benchmarks as shown in the Quick Start section above.
Use `export_onnx_n1d7.py` and `build_tensorrt_engine.py` to prepare a Spark-specific TensorRT
engine when you want the fastest action-head path. If you later rerun `uv sync`, rerun
`bash scripts/deployment/spark/install_deps.sh` so the Spark-specific `flash-attn` build is
restored and revalidated.

---

## Jetson Orin Setup

Orin uses CUDA 12.6 and Python 3.10 (JetPack 6.2), which require a different dependency stack than x86 or Thor.
Tested with JetPack 6.2.
There are two ways to run on Orin: Docker (recommended) or bare metal.

### Docker (Recommended)

Build the Orin container from the repo root:

```bash
cd docker && bash build.sh --profile=orin && cd ..
```

Download the finetuned model (run once, on the host):

```bash
huggingface-cli download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
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

Then inside the container, run the full pipeline:

```bash
# Step 1: PyTorch inference (quick sanity check)
python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 \
  --inference-mode pytorch \
  --denoising-steps 4

# Step 2: Export to ONNX
python scripts/deployment/export_onnx_n1d7.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --output-dir ./gr00t_n1d7_onnx \
  --export-mode full_pipeline \
  --embodiment-tag LIBERO_PANDA

# Step 3: Build TensorRT engines
python scripts/deployment/build_tensorrt_engine.py \
  --mode full_pipeline \
  --onnx-dir ./gr00t_n1d7_onnx \
  --engine-dir ./gr00t_n1d7_engines \
  --precision bf16

# Step 4: Verify TRT accuracy (backbone TRT not supported on Orin; use action_head mode)
python scripts/deployment/verify_n1d7_trt.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --engine-dir ./gr00t_n1d7_engines \
  --mode action_head

# Step 5: Benchmark PyTorch + torch.compile + TRT DiT
python scripts/deployment/benchmark_inference.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --trt-engine-path ./gr00t_n1d7_engines \
  --trt-mode dit_only
```

### Bare Metal

```bash
# One-time install (temporarily copies the Orin pyproject.toml and uv.lock to repo root,
# installs uv, Python deps, and builds torchcodec from source against JetPack's FFmpeg
# runtime)
bash scripts/deployment/orin/install_deps.sh

# In each new shell
source .venv/bin/activate
source scripts/activate_orin.sh
```

Then run inference or benchmarks as shown in the Quick Start section above.
The activation script exports the PyTorch and CUDA library/include paths that `torchcodec`
and `torch.compile` need on Orin.

> **Orin storage tip:** If your eMMC root is low on space, redirect the HuggingFace cache to an NVMe SSD with `export HF_HOME=/path/to/ssd/.cache/huggingface` before downloading models.

> **Orin TRT limitations:** TRT 10.3 on Orin does not support the backbone (LLM) engine — the build step will report a failure for `llm_bf16.engine` and that is expected. The remaining 6 engines build successfully. Use `--mode action_head` for verification and `--trt-mode dit_only` for inference:
> ```bash
> python scripts/deployment/verify_n1d7_trt.py \
>   --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
>   --dataset-path demo_data/libero_demo \
>   --engine-dir ./gr00t_n1d7_engines \
>   --mode action_head
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

### `export_onnx_n1d7.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint |
| `--dataset-path` | (required) | Path to dataset (for input shape capture) |
| `--embodiment-tag` | Auto-detected | Embodiment tag (auto-detected from model config if single embodiment) |
| `--output-dir` | `./gr00t_n1d7_onnx` | Output directory for ONNX models |
| `--export-mode` | `dit_only` | `dit_only`, `action_head`, or `full_pipeline` |
| `--video-backend` | `torchcodec` | Video backend |
| `--precision` | `bf16` | Export precision (`bf16`) |

### `build_tensorrt_engine.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `single` | `single` (one engine) or `full_pipeline` (all 6) |
| `--onnx-dir` | `./gr00t_n1d7_onnx` | Directory with ONNX models (full_pipeline mode) |
| `--engine-dir` | `./gr00t_n1d7_engines` | Directory to save engines (full_pipeline mode) |
| `--onnx` | — | Path to single ONNX model (single mode) |
| `--engine` | — | Path to save single engine (single mode) |
| `--precision` | `bf16` | Precision (`fp32`, `fp16`, `bf16`) |
| `--workspace` | `8192` | Workspace size in MB |

### `verify_n1d7_trt.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint (local path) |
| `--dataset-path` | `demo_data/libero_demo` | Path to dataset |
| `--engine-dir` | `./gr00t_n1d7_engines` | Directory with TRT engines |
| `--mode` | `action_head` | `action_head` or `n17_full_pipeline` |
| `--embodiment-tag` | `LIBERO_PANDA` | Embodiment tag |

---

## Architecture

```
Full Pipeline TRT (6 engines):

┌──────────────────────────────────────────────────────────────────────┐
│                         GR00T N1.7 Policy                            │
│                                                                      │
│  ┌─────────────────── Backbone ───────────────────┐  ┌────────────┐  │
│  │                                                │  │ Action Head │  │
│  │  [ViT TRT] → embed_tokens → masked_scatter    │  │            │  │
│  │              → get_rope_index                  │  │ [State Enc]│  │
│  │              → [LLM TRT]                       │  │ [Act Enc]  │  │
│  │                  (with deepstack injection)    │  │ [DiT]      │  │
│  │                                                │  │ [Act Dec]  │  │
│  └────────────────────────────────────────────────┘  └────────────┘  │
│                                                                      │
│  ████ = TRT Engine    plain text = PyTorch (<1ms)                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `standalone_inference_script.py` | Main inference script (PyTorch + DiT-only TensorRT) |
| `export_onnx_n1d7.py` | Export N1.7 model components to ONNX (ViT, LLM, action head) |
| `build_tensorrt_engine.py` | Build TensorRT engines from ONNX models |
| `trt_torch.py` | TRT Engine wrapper class (load, bind, execute) |
| `trt_model_forward.py` | TRT forward functions and setup (backbone + action head) |
| `verify_n1d7_trt.py` | Accuracy verification (PyTorch vs TRT output comparison) |
| `benchmark_inference.py` | Benchmark timing for data processing, backbone, action head |

---

## Troubleshooting

### Engine Build Fails

- Ensure you have enough GPU memory (16GB+ recommended for full pipeline)
- Try reducing workspace size: `--workspace 4096`
- Ensure TensorRT version matches your CUDA version
- LLM engine requires `batch_size` dimension handling — update `build_tensorrt_engine.py` if using custom shape profiles

### ONNX Export Issues

- If export fails with COMPLEX128 error: ensure `_simple_causal_mask` is used (not HuggingFace's `create_causal_mask`)
- If `masked_scatter` size assertion fails: ensure `visual_pos_masks` has the correct number of True values matching deepstack tensor size
- Check that the dataset path is valid and contains at least one trajectory

### Accuracy Issues

- If cosine < 0.99: check that LLM export does NOT include the final RMSNorm (backbone returns pre-norm `hidden_states[-1]`)
- If output magnitude is ~12x too small: this is the norm bug — see above
- Run `verify_n1d7_trt.py --mode action_head` first to isolate backbone vs action head drift
