# GR00T N1.6 TensorRT Deployment Guide

TensorRT acceleration for GR00T N1.6 inference. Converts 6 model components to TRT engines for faster inference on NVIDIA GPUs.

## What Is Optimized

The following 6 trainable components are exported to ONNX and compiled into TRT engines:

| Engine | Description | Notes |
|--------|-------------|-------|
| **ViT** (Siglip2) | Vision encoder, 27 transformer layers | Largest backbone component |
| **LLM** (Qwen3) | Language model, 16 transformer layers | Exported with eager attention (no flash) |
| **State Encoder** | Embodiment-specific state MLP | Small, BF16 even in FP8 mode |
| **Action Encoder** | Multi-embodiment action encoder | Small, BF16 even in FP8 mode |
| **DiT** | Flow-matching diffusion transformer, 32 layers | Runs 4× per inference (denoising loop) |
| **Action Decoder** | Embodiment-specific action MLP | Small, BF16 even in FP8 mode |

### What Is NOT Optimized (remains in PyTorch)

- **pixel_shuffle_back + MLP1**: Data-dependent reshape after ViT (~1ms)
- **VLLN**: Single LayerNorm(2048), negligible latency
- **Position embedding**: Lightweight lookup in denoising loop
- **Embedding layer + vision token scatter**: Python-level token placement logic
- **Data processing**: Tokenizer, image preprocessing (CPU-bound, ~5ms)

These components contribute ~6ms combined overhead that is included in all reported timings below.

## Prerequisites

- CUDA-enabled GPU with sufficient VRAM (tested on H100 80GB)
- Model checkpoint (base or finetuned)
- Dataset in LeRobot format (needed during export to capture input shapes)
- FP8 mode requires Hopper architecture (H100) or newer; RTX 4090 and Jetson Thor support BF16 only

### Installation

```bash
uv sync --extra=gpu
```

This installs `flash-attn`, `onnx`, and `tensorrt`.

---

## Quick Start

The pipeline script handles export, build, validation, and benchmarking:

```bash
# BF16
bash scripts/deployment/run_trt_pipeline.sh export_bf16       # ~5-10 min on H100
bash scripts/deployment/run_trt_pipeline.sh build_bf16        # ~10-20 min on H100
bash scripts/deployment/run_trt_pipeline.sh compare_detailed  # ~2 min
bash scripts/deployment/run_trt_pipeline.sh benchmark         # ~5 min

# FP8 (H100 / Hopper only)
bash scripts/deployment/run_trt_pipeline.sh export_fp8        # ~15-20 min (includes calibration)
bash scripts/deployment/run_trt_pipeline.sh build_fp8         # ~10-20 min
bash scripts/deployment/run_trt_pipeline.sh compare_detailed_fp8
bash scripts/deployment/run_trt_pipeline.sh benchmark_fp8
```

> **Build times**: ONNX export takes ~5-10 minutes, TRT engine build takes ~10-20 minutes per precision mode. Engines are GPU-architecture specific — they must be rebuilt when switching between different GPUs (e.g., H100 vs RTX 4090).

### Exporting a Finetuned Model

The export workflow is identical for base and finetuned models. Override `MODEL` with your checkpoint path:

```bash
MODEL=/path/to/finetuned/checkpoint bash scripts/deployment/run_trt_pipeline.sh export_bf16
MODEL=/path/to/finetuned/checkpoint bash scripts/deployment/run_trt_pipeline.sh build_bf16
```

The checkpoint directory must contain:
- `config.json` — model architecture config
- `model-*.safetensors` — model weights (one or more shards)
- `model.safetensors.index.json` — weight shard index
- `processor_config.json` — processor configuration

Example with a finetuned LIBERO checkpoint:

```bash
MODEL=/tmp/libero_10/checkpoint-20000 \
DATASET=path/to/libero_dataset \
bash scripts/deployment/run_trt_pipeline.sh export_bf16
```

### Changing Embodiment or Dataset

```bash
MODEL=/path/to/checkpoint \
EMBODIMENT=NEW_EMBODIMENT \
DATASET=path/to/your/dataset \
bash scripts/deployment/run_trt_pipeline.sh export_bf16
```

---

## Embodiment Compatibility

All testing was performed with **GR1** embodiment using `demo_data/gr1.PickNPlace`. The TRT engines use dynamic inputs (batch size, sequence length) and the `embodiment_id` input selects the correct MLP branch at runtime, so **a single set of engines supports multiple embodiments without re-exporting**.

However, there are constraints that may require re-exporting:

| Constraint | Default Value | Impact |
|------------|--------------|--------|
| `max_state_dim` | 128 | Embodiments with state dim > 128 need re-export |
| `max_action_dim` | 128 | Embodiments with action dim > 128 need re-export |
| `action_horizon` | 50 | Fixed at export time |
| `image_size` | 252 (auto-detected) | Different image resolutions need re-export (ViT engine is resolution-specific) |
| `llm_seq_len` | Dynamic (min=1, max=512) | Multi-view setups produce longer sequences; covered by dynamic shape profiles |

---

## Running Inference with TRT Engines

### Policy Server (for sim evaluation or real robot)

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model_path nvidia/GR00T-N1.6-3B \
    --embodiment_tag GR1 \
    --port 5556 \
    --trt_engine_path ./groot_n1d6_engines \
    --trt_mode full_pipeline
```

### Standalone Benchmark

```bash
uv run python scripts/deployment/benchmark_inference.py \
    --model_path nvidia/GR00T-N1.6-3B \
    --dataset_path demo_data/gr1.PickNPlace \
    --trt_engine_path ./groot_n1d6_engines \
    --trt_mode full_pipeline
```

---

## Performance (H100 80GB)

Model: GR00T-N1.6-3B, 4 denoising steps, GR1 embodiment, single-view.

### Inference Timing

| Mode | Data Proc | Backbone | Action Head | E2E | Frequency |
|------|-----------|----------|-------------|-----|-----------|
| PyTorch Eager | 5 ms | 39 ms | 93 ms | 137 ms | 7.3 Hz |
| torch.compile | 5 ms | 39 ms | 11 ms | 55 ms | 18.3 Hz |
| **TRT BF16** | 5 ms | **6 ms** | **12 ms** | **24 ms** | **42.3 Hz** |
| **TRT FP8** | 4 ms | **5 ms** | **10 ms** | **19 ms** | **52.5 Hz** |

> **Note**: These frequencies measure model inference time only (data processing + backbone + action head). In a real deployment loop, additional overhead from sensor I/O, image preprocessing, network latency (if using server/client), and control loop timing will reduce the effective frequency. Expect actual deployment frequency to be lower than these numbers.

> **Other GPUs**: These numbers are from H100. Performance on RTX 4090, A100, or Jetson Thor will differ. FP8 is only available on Hopper (H100) and newer architectures. Run `bash scripts/deployment/run_trt_pipeline.sh benchmark` on your target hardware to measure actual performance.

---

## Accuracy Validation

### Method

Accuracy is verified at three levels:

1. **Per-Component Isolation** (`--compare-detailed`): Each TRT engine is tested independently — the same input is fed to both the PyTorch module and the TRT engine, and their outputs are compared via cosine similarity. This isolates each engine's precision from upstream drift.

2. **End-to-End Action Output** (`--compare`): Both PyTorch and TRT pipelines process the same observation with the same random noise. The final action predictions are compared.

3. **Task Success Rate** (RoboCasa simulation): Cosine similarity alone does not guarantee correct robot behavior over long episodes. The full RoboCasa GR1 tabletop evaluation suite (24 tasks, 20 episodes each) measures task success rate for each inference backend.

### Per-Component Cosine Similarity (BF16)

| Component | Cosine Similarity | Status |
|-----------|-------------------|--------|
| ViT (Siglip2) | 0.998375 | WARN (flash→TRT attention drift, inherent to export) |
| LLM (Qwen3) | 0.999777 | PASS |
| State Encoder | 1.000000 | PASS |
| Action Encoder | 0.999996 | PASS |
| DiT | 0.999993 | PASS |
| Action Decoder | 1.000000 | PASS |
| **E2E Action** | **0.999998** | **PASS** |

### RoboCasa Task Success Rate (24 tasks, 20 episodes each)

| Mode | Avg Success Rate | Statistically Significant vs Eager? |
|------|-----------------|-------------------------------------|
| PyTorch Eager | 0.5048 | — |
| TRT BF16 | 0.4618 | NO (t=-1.78, p>0.05) |
| TRT FP8 | 0.5060 | NO (t=+0.04, p>0.05) |

Neither TRT mode shows statistically significant degradation. Differences are within simulation variance.

> **Note**: The RoboCasa evaluation takes **3+ hours per backend** (24 tasks × 20 episodes × 720 max steps). Running all three backends sequentially takes ~10 hours. Use `bash scripts/deployment/eval_trt_robocasa.sh quick` for a faster smoke test (2 episodes).

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     GR00T Policy                                 │
│                                                                  │
│  Backbone:                                                       │
│  ┌──────────┐   ┌──────────────┐   ┌─────────┐   ┌──────────┐  │
│  │ ViT (TRT)│ → │pixel_shuffle │ → │MLP1 (PT)│ → │ LLM (TRT)│  │
│  │ Siglip2  │   │  (PyTorch)   │   │         │   │  Qwen3   │  │
│  └──────────┘   └──────────────┘   └─────────┘   └──────────┘  │
│                                                                  │
│  Action Head:                                                    │
│  ┌────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────────┐   │
│  │VLLN(PT)│→ │State Enc(TRT)│→ │DiT (TRT)│→ │Act Dec (TRT) │   │
│  └────────┘  └──────────────┘  │×4 steps  │  └──────────────┘   │
│              ┌──────────────┐  │          │                      │
│              │Act Enc (TRT) │→ │          │                      │
│              └──────────────┘  └─────────┘                       │
└──────────────────────────────────────────────────────────────────┘

TRT = TensorRT engine    PT = PyTorch (small ops, not worth TRT overhead)
```

---

## Files

| File | Description |
|------|-------------|
| `run_trt_pipeline.sh` | Pipeline runner: export → build → validate → benchmark |
| `export_onnx_n1d6.py` | Export 6 components to ONNX (BF16 or FP8 with calibration) |
| `build_tensorrt_engine.py` | Build TRT engines from ONNX (auto shape profiles from metadata) |
| `benchmark_inference.py` | Benchmark and accuracy comparison (`--compare`, `--compare-detailed`) |
| `trt_model_forward.py` | TRT forward functions and engine setup (monkey-patches policy) |
| `trt_torch.py` | TRT engine wrapper class |
| `eval_trt_robocasa.sh` | RoboCasa sim evaluation across backends (eager vs BF16 vs FP8) |
| `vit_trt_compile.py` | Experimental: torch_tensorrt ViT compilation (preserves SDPA) |
| `accuracy_thresholds.py` | Centralized accuracy thresholds for validation |

---

## Troubleshooting

### Engine Build Fails
- Ensure 8GB+ GPU memory available
- Try `--workspace 4096` to reduce memory usage
- Engines are GPU-architecture specific — rebuild when switching GPUs

### TRT Version Mismatch
- Engines must be built with the same TRT version used at runtime
- Check: `python -c "import tensorrt; print(tensorrt.__version__)"`
- Rebuild engines after TRT version upgrades

### flash-attn Not Found
- Run `uv sync --extra=gpu` (bare `uv sync` removes flash-attn)

### ONNX Export Fails
- Ensure model loads in PyTorch first
- Ensure dataset has at least 1 trajectory
- Check `export_metadata.json` is generated in output dir

### FP8 Not Available
- FP8 requires Hopper architecture (H100) or newer
- RTX 4090 (Ada) and Jetson Thor support BF16 only
