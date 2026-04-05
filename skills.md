# skills.md — Copy-Pasteable Commands for Isaac GR00T

Quick-reference command snippets organized by task. Each block is self-contained and ready to paste into a terminal.

---

## Setup

```bash
# Clone and install
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
uv sync --all-extras
```

```bash
# Platform-specific install (pick one)
bash scripts/deployment/dgpu/install_deps.sh     # dGPU (H100, A100, RTX)
bash scripts/deployment/orin/install_deps.sh      # Jetson Orin
bash scripts/deployment/thor/install_deps.sh      # Jetson Thor
bash scripts/deployment/spark/install_deps.sh     # DGX Spark
```

```bash
# Activate platform environment (pick one, run in each new shell)
source scripts/activate_orin.sh
source scripts/activate_thor.sh
source scripts/activate_spark.sh
```

---

## Lint and Format

```bash
# Full lint + format check (same as CI)
pre-commit run --all-files
```

```bash
# Auto-fix lint issues
ruff check --fix gr00t/ tests/
```

```bash
# Format only
ruff format gr00t/ tests/
```

---

## Testing

```bash
# All CPU tests
python -m pytest tests/ groot_infra/tests/ -m "not gpu" -v --timeout=300
```

```bash
# All GPU tests
python -m pytest tests/ groot_infra/tests/ -m gpu -v --timeout=300
```

```bash
# Single test file
python -m pytest tests/examples/test_so100.py -v --timeout=300
```

```bash
# Tests matching a keyword
python -m pytest tests/ -v -k "finetune" --timeout=300
```

---

## Fine-Tuning

```bash
# Fine-tune on SO100 demo data (single GPU)
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/cube_to_bowl_5 \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/SO100/modality_config.py \
  --output-dir output/so100_finetune
```

```bash
# Fine-tune with multiple GPUs
NUM_GPUS=4 bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <your_dataset> \
  --embodiment-tag <your_tag> \
  --modality-config-path <your_config.py> \
  --output-dir output/<run_name>
```

```bash
# Fine-tune with custom hyperparameters
MAX_STEPS=5000 GLOBAL_BATCH_SIZE=16 SAVE_STEPS=500 \
  bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <your_dataset> \
  --embodiment-tag <your_tag> \
  --output-dir output/<run_name>
```

---

## Inference

```bash
# Start inference server with a trained model
python gr00t/eval/run_gr00t_server.py \
  --model-path <checkpoint_path> \
  --embodiment-tag <tag> \
  --port 5555
```

```bash
# Start server with replay policy (for testing without a model)
python gr00t/eval/run_gr00t_server.py \
  --dataset-path demo_data/cube_to_bowl_5 \
  --modality-config-path examples/SO100/modality_config.py \
  --embodiment-tag new_embodiment
```

---

## Deployment (ONNX + TensorRT)

```bash
# Step 1: Export to ONNX
python scripts/deployment/export_onnx_n1d7.py \
  --model-path <checkpoint_path> \
  --output-dir output/onnx
```

```bash
# Step 2: Build TensorRT engines
python scripts/deployment/build_trt_pipeline.py \
  --onnx-dir output/onnx \
  --output-dir output/trt
```

```bash
# Step 3: Benchmark
python scripts/deployment/benchmark_inference.py \
  --model-path <checkpoint_or_trt_path>
```

```bash
# Verify TensorRT engines
python scripts/deployment/verify_n1d7_trt.py \
  --trt-dir output/trt
```

---

## Build and Validation

```bash
# Validate lockfile is up to date
uv lock --locked
```

```bash
# Build distributable wheel
uv build
```

```bash
# Check ATTRIBUTIONS.md is current
python scripts/generate_attributions.py --check
```

```bash
# Regenerate ATTRIBUTIONS.md
python scripts/generate_attributions.py
```

---

## Data Preparation

```bash
# Download DROID sample dataset
python scripts/download_droid_sample.py
```

```bash
# Convert LeRobot v3 dataset to v2 format
python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --input-path <v3_dataset> \
  --output-path <v2_dataset>
```

```bash
# Patch an existing checkpoint
python scripts/patch_checkpoint.py \
  --checkpoint-path <path> \
  --output-path <patched_path>
```
