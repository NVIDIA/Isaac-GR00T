# GR00T Deployment & Inference Guide

Run inference with PyTorch or TensorRT acceleration for the GR00T N1.7 policy.

---

## Prerequisites

- Model checkpoint: `nvidia/GR00T-N1.7-3B`
- Dataset in LeRobot format (e.g., `demo_data/libero_demo`)
- CUDA-enabled GPU
- FFmpeg (required by `torchcodec` video backend): `sudo apt-get install -y ffmpeg`

## Choose Your Setup

- dGPU local environment: use the installation commands below, then use the PyTorch or TensorRT commands in this guide
- Thor Docker or bare metal: skip to [Jetson Thor Setup](#jetson-thor-setup)
- Spark Docker or bare metal: skip to [DGX Spark Setup](#dgx-spark-setup)
- Orin Docker or bare metal: skip to [Jetson Orin Setup](#jetson-orin-setup)

### dGPU Installation

```bash
uv sync
```

GPU dependencies (`flash-attn`, `onnx`, `tensorrt`) are included in the default install.

---

## Quick Start: PyTorch Mode

First, download the finetuned model to a local directory (HuggingFace does not support nested repo paths directly):
```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
```

```bash
python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

---

## TensorRT Acceleration (3x Faster)

The `n17_full_pipeline` mode accelerates the LLM and action head with TRT engines while keeping the ViT in PyTorch for accuracy:

| Component | Engine | Notes |
|-----------|--------|-------|
| ViT | **PyTorch** | Kept in PyTorch for accuracy (see note below) |
| LLM | **TRT** | Qwen3-VL Text Model (16 layers, with deepstack injection) |
| VL Self-Attention | **TRT** | SelfAttentionTransformer (4 layers, if present) |
| State Encoder | **TRT** | CategorySpecificMLP |
| Action Encoder | **TRT** | MultiEmbodimentActionEncoder |
| DiT | **TRT** | AlternateVLDiT (32 layers) |
| Action Decoder | **TRT** | CategorySpecificMLP |

Lightweight ops remain in PyTorch: `embed_tokens`, `masked_scatter`, `get_rope_index`, VLLN.

> **TensorRT version:** We tested on both CUDA 12.8 and CUDA 13.0. TensorRT 12 libraries work on both, so they are used by default on standard dGPU setups. On CUDA 13.0 platforms (DGX Spark, Jetson Thor), TensorRT 13 is also supported and is set as the default in those platform-specific dependency stacks.

<!-- TODO: Investigate and fix ViT TRT accuracy issue.
     The ONNX export wrapper produces 0.998 cosine vs PyTorch ViT, but the TRT-built engine
     diverges to ~0.57 cosine. This is likely a TRT builder kernel fusion bug with the patched
     SDPA attention across 24 ViT blocks. Once fixed, the ViT engine can be re-enabled for
     full backbone TRT acceleration (~7ms backbone instead of ~27ms). -->

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

> **Note:** Engine build takes ~2-5 minutes depending on GPU. Engines are GPU-architecture-specific and must be rebuilt for different GPUs. The ViT engine (if present) is automatically skipped at runtime — ViT stays in PyTorch for accuracy.

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

| Device | Mode | E2E Speedup | Action Head Speedup |
|--------|------|-------------|---------------------|
| H100 | PyTorch Eager | 1.00x | 1.00x |
| H100 | torch.compile | 2.16x | 7.31x |
| H100 | TRT (n17_full_pipeline) | 3.29x | 7.62x |

TODO: test compatibility and get results on other platforms
### DiT-Only TRT (BF16, 4 denoising steps, single view)

> The DiT-only mode (`--export-mode dit_only`) optimizes only the action head DiT,
> leaving the backbone in PyTorch.

GR00T N1.7 DiT-only inference timing:

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
| Spark | PyTorch Eager | 2 ms | 33 ms | 76 ms | 112 ms | 8.9 Hz |
| Spark | torch.compile | 2 ms | 33 ms | 54 ms | 89 ms | 11.2 Hz |
| Spark | TensorRT | 2 ms | 32 ms | 48 ms | 84 ms | 11.9 Hz |
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
| Spark | PyTorch Eager | 1.00x | 1.00x |
| Spark | torch.compile | 1.25x | 1.41x |
| Spark | TensorRT | 1.33x | 1.58x |
| Orin | PyTorch Eager | 1.00x | 1.00x |
| Orin | torch.compile | 1.50x | 2.00x |
| Orin | TensorRT | 1.73x | 2.80x |

> Run `python scripts/deployment/benchmark_inference.py` to generate benchmarks for your hardware.
> See `GR00T_inference_timing.ipynb` for detailed analysis and visualizations.

> Jetson and Spark platforms use different dependency stacks than dGPU. Thor and Spark use CUDA 13 with PyTorch 2.10.0 from the [Jetson AI Lab cu130 index](https://pypi.jetson-ai-lab.io/sbsa/cu130). Orin uses CUDA 12.6 with PyTorch 2.10.0 from the [Jetson AI Lab cu126 index](https://pypi.jetson-ai-lab.io/jp6/cu126). See the platform-specific setup sections below.
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
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo \
    --embodiment-tag LIBERO_PANDA \
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

## DGX Spark Setup

Spark uses CUDA 13 and Python 3.12 like Thor, but requires a dedicated dependency stack and
source-built `flash-attn` for `sm121`. There are two ways to run on Spark: Docker (recommended)
or bare metal.

### Docker (Recommended)

Build the Spark container from the repo root:

```bash
cd docker && bash build.sh --profile=spark && cd ..
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
  gr00t-spark \
  python scripts/deployment/standalone_inference_script.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo \
    --embodiment-tag LIBERO_PANDA \
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
  gr00t-spark \
  python scripts/deployment/benchmark_inference.py
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
Use `export_onnx_n1d6.py` and `build_tensorrt_engine.py` to prepare a Spark-specific TensorRT
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
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo \
    --embodiment-tag LIBERO_PANDA \
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
