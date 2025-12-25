# GR00T Deployment & Inference Guide

Run inference with PyTorch or TensorRT acceleration for the GR00T policy.

---

## Prerequisites

- Model checkpoint (e.g., `nvidia/GR00T-N1.6-3B`)
- Dataset in LeRobot format
- CUDA-enabled GPU

### Installation

**PyTorch mode** (default installation):
```bash
uv sync
```

**TensorRT mode** (includes ONNX and TensorRT dependencies):
```bash
uv sync --extra tensorrt
```

---

## Quick Start: PyTorch Mode

```bash
python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

---

## TensorRT Mode (2x Faster)

### Step 1: Export to ONNX

```bash
python scripts/deployment/export_onnx_n1d6.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag GR1 \
  --output-dir ./groot_n1d6_onnx
```

**Output:** `./groot_n1d6_onnx/dit_model.onnx`

### Step 2: Build TensorRT Engine

```bash
python scripts/deployment/build_tensorrt_engine.py \
  --onnx ./groot_n1d6_onnx/dit_model.onnx \
  --engine ./groot_n1d6_onnx/dit_model_bf16.trt \
  --precision bf16
```

**Output:** `./groot_n1d6_onnx/dit_model_bf16.trt`

> **Note:** Engine build takes ~5-10 minutes depending on GPU. The engine is GPU-specific and needs to be rebuilt for different GPU architectures.

### Step 3: Run with TensorRT

```bash
python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode tensorrt \
  --trt-engine-path ./groot_n1d6_onnx/dit_model_bf16.trt \
  --action-horizon 8
```

---

## Command-Line Arguments

### `standalone_inference_script.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint |
| `--dataset-path` | (required) | Path to LeRobot dataset |
| `--embodiment-tag` | `GR1` | Embodiment tag |
| `--traj-ids` | `[0]` | List of trajectory IDs to evaluate |
| `--steps` | `200` | Max steps per trajectory |
| `--action-horizon` | `16` | Action horizon for inference |
| `--inference-mode` | `pytorch` | `pytorch` or `tensorrt` |
| `--trt-engine-path` | `./groot_n1d6_onnx/dit_model_bf16.trt` | TensorRT engine path |
| `--denoising-steps` | `4` | Number of denoising steps |
| `--skip-timing-steps` | `1` | Steps to skip for timing (warmup) |
| `--seed` | `42` | Random seed for reproducibility |
| `--video-backend` | `torchcodec` | Video backend (`decord`, `torchvision_av`, `torchcodec`) |

### `export_onnx_n1d6.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint |
| `--dataset-path` | (required) | Path to dataset (for input shape capture) |
| `--embodiment-tag` | `GR1` | Embodiment tag |
| `--output-dir` | `./groot_n1d6_onnx` | Output directory for ONNX model |
| `--video-backend` | `torchcodec` | Video backend |

### `build_tensorrt_engine.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--onnx` | (required) | Path to ONNX model |
| `--engine` | (required) | Path to save TensorRT engine |
| `--precision` | `bf16` | Precision (`fp32`, `fp16`, `bf16`, `fp8`) |
| `--workspace` | `8192` | Workspace size in MB |

### `benchmark_inference.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `nvidia/GR00T-N1.6-3B` | Path to model checkpoint |
| `--dataset-path` | `demo_data/gr1.PickNPlace` | Path to dataset |
| `--embodiment-tag` | `GR1` | Embodiment tag |
| `--trt-engine-path` | (optional) | Path to TensorRT engine |
| `--num-iterations` | `20` | Number of benchmark iterations |
| `--warmup` | `5` | Number of warmup iterations |
| `--skip-compile` | `false` | Skip torch.compile benchmark |
| `--seed` | `42` | Random seed for reproducibility |

---

## Benchmarks

### Component-wise Breakdown

> **Note:** The backbone (Vision Encoder + Language Model) timing is the same across all modes (Eager, torch.compile, TensorRT). Only the **Action Head (DiT)** is optimized with torch.compile or TensorRT, which is why you see significant speedups in the Action Head column while the Backbone column remains constant.

GR00T-N1.6-3B inference timing (4 denoising steps):

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

> Experiments on Thor and Orin used different dependency stacks. Thor with CUDA 13, PyTorch 2.9, using supporting packages sourced from the [Jetson AI Lab cu130 index](https://pypi.jetson-ai-lab.io/sbsa/cu130); and Orin with CUDA 12.6, PyTorch 2.8, using supporting packages sourced from the [Jetson AI Lab cu126 index](https://pypi.jetson-ai-lab.io/jp6/cu126).
---

## Troubleshooting

### Engine Build Fails

- Ensure you have enough GPU memory (8GB+ recommended)
- Try reducing workspace size: `--workspace 4096`
- Ensure TensorRT version matches your CUDA version

### ONNX Export Issues

- If export fails, ensure the model loads correctly in PyTorch first
- Check that the dataset path is valid and contains at least one trajectory

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GR00T Policy                             │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ Vision Encoder│  │Language Model │  │  Action Head    │  │
│  │(Cosmos-Reason)│──│(Cosmos-Reason)│──│    (DiT)        │  │
│  └───────────────┘  └───────────────┘  └─────────────────┘  │
│                                              ▲              │
│                                              │              │
│                                    ┌─────────┴─────────┐    │
│                                    │ TensorRT Engine   │    │
│                                    │ (dit_model.trt)   │    │
│                                    └───────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

The TensorRT optimization targets the **DiT (Diffusion Transformer)** component of the action head, which is the main computational bottleneck during inference.

---

## Files

| File | Description |
|------|-------------|
| `standalone_inference_script.py` | Main inference script (PyTorch + TensorRT) |
| `export_onnx_n1d6.py` | Export DiT model to ONNX format |
| `build_tensorrt_engine.py` | Build TensorRT engine from ONNX |
| `benchmark_inference.py` | Benchmark data processing, backbone, action head, and E2E timing |
| `GR00T_inference_timing.ipynb` | Inference timing analysis notebook with visualizations |
