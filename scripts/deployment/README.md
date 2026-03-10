# GR00T Deployment & Inference Guide

Run inference with PyTorch or TensorRT acceleration for the GR00T N1.7 policy.

---

## Prerequisites

- Model checkpoint: `nvidia/GR00T-N1.7-3B`
- Dataset in LeRobot format (e.g., `demo_data/gr1.PickNPlace`)
- CUDA-enabled GPU
- FFmpeg (required by `torchcodec` video backend): `sudo apt-get install -y ffmpeg`

## Choose Your Setup

- dGPU local environment: use the installation commands below, then use the PyTorch or TensorRT commands in this guide
- Thor Docker or bare metal: skip to [Jetson Thor Setup](#jetson-thor-setup)
- Orin Docker or bare metal: skip to [Jetson Orin Setup](#jetson-orin-setup)

### dGPU Installation

**PyTorch mode** (default installation):
```bash
uv sync
```

**TensorRT mode** (includes ONNX and TensorRT dependencies):
```bash
uv sync --extra gpu
```

---

## Quick Start: PyTorch Mode

```bash
python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/gr1.PickNPlace \
  --embodiment-tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

---

## TensorRT Full Pipeline (5x Faster)

The full-pipeline TRT mode exports **all 6 model components** to TensorRT engines:

| Engine | Description |
|--------|-------------|
| ViT | Qwen3-VL Vision Encoder (24 blocks + PatchMerger + DeepStack) |
| LLM | Qwen3-VL Text Model (16 layers, with deepstack injection) |
| State Encoder | CategorySpecificMLP |
| Action Encoder | MultiEmbodimentActionEncoder |
| DiT | AlternateVLDiT (32 layers) |
| Action Decoder | CategorySpecificMLP |

Lightweight ops remain in PyTorch (<1ms): `embed_tokens`, `masked_scatter`, `get_rope_index`, VLLN.

### Step 1: Export to ONNX

```bash
uv run python scripts/deployment/export_onnx_n1d7.py \
  --model_path nvidia/GR00T-N1.7-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --output_dir ./gr00t_n1d7_onnx \
  --export_mode full_pipeline
```

**Output:** 6 ONNX files in `./gr00t_n1d7_onnx/` (~5 GB total)

> **Finetuned models:** Replace `--model_path` with your checkpoint path. The export pipeline is identical for base and finetuned models.

### Step 2: Build TensorRT Engines

```bash
uv run python scripts/deployment/build_tensorrt_engine.py \
  --mode full_pipeline \
  --onnx_dir ./gr00t_n1d7_onnx \
  --engine_dir ./gr00t_n1d7_engines \
  --precision bf16
```

**Output:** 6 `.engine` files in `./gr00t_n1d7_engines/` (~4.3 GB total)

> **Note:** Engine build takes ~2-5 minutes depending on GPU. Engines are GPU-architecture-specific and must be rebuilt for different GPUs.

### Step 3: Verify Accuracy

```bash
uv run python scripts/deployment/verify_n1d7_trt.py \
  --model_path nvidia/GR00T-N1.7-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --engine_dir ./gr00t_n1d7_engines \
  --mode n17_full_pipeline
```

Expected output: `Cosine Similarity: 0.999987` (PASS).

---

### Step 4: Run Benchmark

```bash
uv run python scripts/deployment/benchmark_inference.py \
    --model_path nvidia/GR00T-N1.7-3B \
    --trt_engine_path ./gr00t_n1d7_engines \
    --trt_mode n17_full_pipeline
```

## Performance

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `nvidia/GR00T-N1.7-3B` | Path to model checkpoint |
| `--dataset-path` | `demo_data/gr1.PickNPlace` | Path to dataset |
| `--embodiment-tag` | `GR1` | Embodiment tag |
| `--trt-engine-path` | (optional) | Path to TensorRT engine |
| `--num-iterations` | `20` | Number of benchmark iterations |
| `--warmup` | `5` | Number of warmup iterations |
| `--skip_compile` | `false` | Skip torch.compile benchmark |
| `--seed` | `42` | Random seed for reproducibility |
### N1.7 Full Pipeline TRT (BF16, 4 denoising steps, single view)

| Device | Mode | Data Processing | Backbone | Action Head | E2E | Frequency |
|--------|------|-----------------|----------|-------------|-----|-----------|
| H100 | PyTorch Eager | 4 ms | 49 ms | 95 ms | 148 ms | 6.8 Hz |
| H100 | torch.compile | 4 ms | 48 ms | 11 ms | 63 ms | 15.8 Hz |
| H100 | **TRT Full Pipeline** | **4 ms** | **7 ms** | **13 ms** | **23 ms** | **42.6 Hz** |

Speedup vs Eager: torch.compile **2.33x**, TRT Full Pipeline **6.29x**

TODO: test compatibility and get results on other platforms
### N1.6 DiT-Only TRT (BF16, 4 denoising steps, single view)

> The DiT-only mode (`--export_mode dit_only`) optimizes only the action head DiT,
> leaving the backbone in PyTorch.

GR00T-N1.6-3B inference timing:

| Device | Mode | Data Processing | Backbone | Action Head | E2E | Frequency |
|--------|------|-----------------|----------|-------------|-----|-----------|
| RTX 5090 | PyTorch Eager | 2 ms | 18 ms | 38 ms | 58 ms | 17.3 Hz |
| RTX 5090 | torch.compile | 2 ms | 18 ms | 16 ms | 37 ms | 27.3 Hz |
| RTX 5090 | TensorRT | 2 ms | 18 ms | 11 ms | 31 ms | 32.1 Hz |
| H100 | PyTorch Eager | 4 ms | 23 ms | 49 ms | 77 ms | 13.0 Hz |
| H100 | torch.compile | 4 ms | 23 ms | 11 ms | 38 ms | 26.3 Hz |
| H100 | TensorRT | 4 ms | 22 ms | 10 ms | 36 ms | 27.9 Hz |
| RTX 4090 | PyTorch Eager | 2 ms | 25 ms | 55 ms | 82 ms | 12.2 Hz |
| RTX 4090 | torch.compile | 2 ms | 25 ms | 17 ms | 44 ms | 22.8 Hz |
| RTX 4090 | TensorRT | 2 ms | 24 ms | 16 ms | 43 ms | 23.3 Hz |
| Thor | PyTorch Eager | 5 ms | 38 ms | 74 ms | 117 ms | 8.6 Hz |
| Thor | torch.compile | 5 ms | 39 ms | 61 ms | 105 ms | 9.5 Hz |
| Thor | TensorRT | 5 ms | 38 ms | 49 ms | 92 ms | 10.9 Hz |
| Orin | PyTorch Eager | 6 ms | 93 ms | 202 ms | 300 ms | 3.3 Hz |
| Orin | torch.compile | 6 ms | 93 ms | 101 ms | 199 ms | 5.0 Hz |
| Orin | TensorRT | 6 ms | 95 ms | 72 ms | 173 ms | 5.8 Hz |

### Speedup vs PyTorch Eager

| Device | Mode | E2E Speedup | Action Head Speedup |
|--------|------|-------------|---------------------|
| RTX 5090 | PyTorch Eager | 1.00x | 1.00x |
| RTX 5090 | torch.compile | 1.58x | 2.32x |
| RTX 5090 | TensorRT | 1.86x | 3.59x |
| H100 | PyTorch Eager | 1.00x | 1.00x |
| H100 | torch.compile | 2.02x | 4.60x |
| H100 | TensorRT | 2.14x | 4.80x |
| RTX 4090 | PyTorch Eager | 1.00x | 1.00x |
| RTX 4090 | torch.compile | 1.87x | 3.26x |
| RTX 4090 | TensorRT | 1.92x | 3.48x |
| Thor | PyTorch Eager | 1.00x | 1.00x |
| Thor | torch.compile | 1.11x | 1.20x |
| Thor | TensorRT | 1.27x | 1.49x |
| Orin | PyTorch Eager | 1.00x | 1.00x |
| Orin | torch.compile | 1.50x | 2.00x |
| Orin | TensorRT | 1.73x | 2.80x |

> Run `python scripts/deployment/benchmark_inference.py` to generate benchmarks for your hardware.
> See `GR00T_inference_timing.ipynb` for detailed analysis and visualizations.

> Jetson platforms use different dependency stacks than dGPU. Thor uses CUDA 13 with PyTorch 2.10.0 from the [Jetson AI Lab cu130 index](https://pypi.jetson-ai-lab.io/sbsa/cu130). Orin uses CUDA 12.6 with PyTorch 2.10.0 from the [Jetson AI Lab cu126 index](https://pypi.jetson-ai-lab.io/jp6/cu126). See the platform-specific setup sections below.
---

## Jetson Thor Setup

Thor uses CUDA 13 and Python 3.12, which require a different dependency stack than x86 or Orin.
Tested with JetPack 7.1.
There are two ways to run on Thor: Docker (recommended) or bare metal.

### Docker (Recommended)

Build the Thor container from the repo root:

```bash
cd docker && bash build.sh --profile=thor && cd ..
```

Run inference:

```bash
docker run --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HOME}/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  gr00t-thor \
  python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --dataset-path demo_data/gr1.PickNPlace \
    --embodiment-tag GR1 \
    --traj-ids 0 \
    --inference-mode pytorch \
    --denoising-steps 4
```

Run benchmarks:

```bash
docker run --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HOME}/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  gr00t-thor \
  python scripts/deployment/benchmark_inference.py
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

## Jetson Orin Setup

Orin uses CUDA 12.6 and Python 3.10 (JetPack 6.2), which require a different dependency stack than x86 or Thor.
Tested with JetPack 6.2.
There are two ways to run on Orin: Docker (recommended) or bare metal.

### Docker (Recommended)

Build the Orin container from the repo root:

```bash
cd docker && bash build.sh --profile=orin && cd ..
```

Run inference:

```bash
docker run --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HOME}/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  gr00t-orin \
  python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --dataset-path demo_data/gr1.PickNPlace \
    --embodiment-tag GR1 \
    --traj-ids 0 \
    --inference-mode pytorch \
    --denoising-steps 4
```

Run benchmarks:

```bash
docker run --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "$(pwd)":/workspace/repo \
  -v "${HOME}/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  gr00t-orin \
  python scripts/deployment/benchmark_inference.py
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
> Experiments on Thor and Orin used different dependency stacks. Thor with CUDA 13, PyTorch 2.9, using supporting packages sourced from the [Jetson AI Lab cu130 index](https://pypi.jetson-ai-lab.io/sbsa/cu130); and Orin with CUDA 12.6, PyTorch 2.8, using supporting packages sourced from the [Jetson AI Lab cu126 index](https://pypi.jetson-ai-lab.io/jp6/cu126).

---

## Command-Line Arguments

### `export_onnx_n1d7.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to model checkpoint |
| `--dataset_path` | (required) | Path to dataset (for input shape capture) |
| `--embodiment_tag` | `GR1` | Embodiment tag |
| `--output_dir` | `./gr00t_n1d7_onnx` | Output directory for ONNX models |
| `--export_mode` | `dit_only` | `dit_only`, `action_head`, or `full_pipeline` |
| `--video_backend` | `torchcodec` | Video backend |
| `--precision` | `bf16` | Export precision (`bf16`) |

### `build_tensorrt_engine.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `single` | `single` (one engine) or `full_pipeline` (all 6) |
| `--onnx_dir` | `./gr00t_n1d7_onnx` | Directory with ONNX models (full_pipeline mode) |
| `--engine_dir` | `./gr00t_n1d7_engines` | Directory to save engines (full_pipeline mode) |
| `--onnx` | — | Path to single ONNX model (single mode) |
| `--engine` | — | Path to save single engine (single mode) |
| `--precision` | `bf16` | Precision (`fp32`, `fp16`, `bf16`) |
| `--workspace` | `8192` | Workspace size in MB |

### `verify_n1d7_trt.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `nvidia/GR00T-N1.7-3B` | Path to model checkpoint |
| `--dataset_path` | `demo_data/gr1.PickNPlace` | Path to dataset |
| `--engine_dir` | `./gr00t_n1d7_engines` | Directory with TRT engines |
| `--mode` | `action_head` | `action_head` or `n17_full_pipeline` |
| `--embodiment_tag` | `GR1` | Embodiment tag |

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
