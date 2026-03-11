#!/usr/bin/env bash
# =============================================================================
# GR00T N1.6 TensorRT Full Pipeline: Export, Build, Validate, Benchmark
#
# Usage:
#   bash scripts/deployment/run_trt_pipeline.sh [STEP] [OPTIONS]
#
# Steps (run one at a time, or "all" for the full sequence):
#   export_bf16         - Export all 6 components to ONNX (BF16)
#   build_bf16          - Build TRT engines from ONNX (BF16)
#   compare             - Accuracy validation: TRT vs PyTorch (quick 3-level)
#   compare_detailed    - Per-component accuracy isolation (8 tests)
#   benchmark           - Performance benchmark (Eager / TRT)
#   export_fp8          - Export with FP8 quantization (requires nvidia-modelopt)
#   build_fp8           - Build TRT engines from ONNX (FP8)
#   compare_fp8         - Accuracy validation for FP8 engines
#   compare_detailed_fp8 - Per-component accuracy isolation for FP8 engines
#   benchmark_fp8       - Performance benchmark for FP8 engines
#   profile_vit         - ViT FP8 per-layer cosine profiling (identify high-drift layers)
#   all                 - Run export_bf16 → build_bf16 → compare → benchmark
#
# Examples:
#   bash scripts/deployment/run_trt_pipeline.sh export_bf16
#   bash scripts/deployment/run_trt_pipeline.sh compare
#   bash scripts/deployment/run_trt_pipeline.sh profile_vit
#   bash scripts/deployment/run_trt_pipeline.sh all
#   MODEL=./my_model DATASET=./my_data bash scripts/deployment/run_trt_pipeline.sh all
#
# Acceptance Criteria (source: scripts/deployment/accuracy_thresholds.py):
#
#   BF16:
#     Backbone   cosine >= 0.999, L1 <= 0.01
#     Action     cosine >= 0.99,  L1 <= 0.05
#     E2E action cosine >= 0.99
#     Component  PASS >= 0.999, WARN >= 0.99, else FAIL
#
#   FP8 (relaxed):
#     Backbone   cosine >= 0.99,  L1 <= 0.05
#     Action     cosine >= 0.98,  L1 <= 0.1
#     E2E action cosine >= 0.98
#     Component  PASS >= 0.99,  WARN >= 0.98, else FAIL
#
# Quick validation:
#   bash scripts/deployment/run_trt_pipeline.sh compare           # BF16
#   bash scripts/deployment/run_trt_pipeline.sh compare_fp8       # FP8
#   bash scripts/deployment/run_trt_pipeline.sh compare_detailed  # per-component
# =============================================================================
set -euo pipefail

# ── Configurable paths (override via environment variables) ──────────────────
MODEL="${MODEL:-nvidia/GR00T-N1.6-3B}"
DATASET="${DATASET:-demo_data/gr1.PickNPlace}"
EMBODIMENT="${EMBODIMENT:-gr1}"
ONNX_DIR="${ONNX_DIR:-./groot_n1d6_onnx}"
ENGINE_DIR="${ENGINE_DIR:-./groot_n1d6_engines}"
ONNX_DIR_FP8="${ONNX_DIR_FP8:-./groot_n1d6_onnx_fp8}"
ENGINE_DIR_FP8="${ENGINE_DIR_FP8:-./groot_n1d6_engines_fp8}"
CALIB_DATASET="${CALIB_DATASET:-${DATASET}}"
CALIB_SIZE="${CALIB_SIZE:-100}"
NUM_ITER="${NUM_ITER:-20}"
PRECISION="${PRECISION:-bf16}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Helper ───────────────────────────────────────────────────────────────────
banner() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
    echo ""
}

# ── Step functions ───────────────────────────────────────────────────────────

step_export_bf16() {
    banner "Step 1/4: Export ONNX (BF16) — ViT + LLM + StateEnc + ActionEnc + DiT + ActionDec"
    uv run python "${SCRIPT_DIR}/export_onnx_n1d6.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --output_dir "${ONNX_DIR}" \
        --export_mode full_pipeline \
        --precision bf16
}

step_build_bf16() {
    banner "Step 2/4: Build TRT Engines (BF16)"
    uv run python "${SCRIPT_DIR}/build_tensorrt_engine.py" \
        --mode full_pipeline \
        --onnx_dir "${ONNX_DIR}" \
        --engine_dir "${ENGINE_DIR}" \
        --precision bf16
}

step_compare() {
    banner "Step 3/4: Accuracy Validation — PyTorch vs TRT"
    uv run python "${SCRIPT_DIR}/benchmark_inference.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --trt_engine_path "${ENGINE_DIR}" \
        --trt_mode full_pipeline \
        --compare
}

step_benchmark() {
    banner "Step 4/4: Performance Benchmark"
    uv run python "${SCRIPT_DIR}/benchmark_inference.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --trt_engine_path "${ENGINE_DIR}" \
        --trt_mode full_pipeline \
        --skip_compile \
        --num_iterations "${NUM_ITER}"
}

step_compare_detailed() {
    banner "Detailed Accuracy: Per-component PyTorch vs TRT"
    uv run python "${SCRIPT_DIR}/benchmark_inference.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --trt_engine_path "${ENGINE_DIR}" \
        --trt_mode full_pipeline \
        --compare-detailed
}

step_export_fp8() {
    banner "FP8: Export ONNX (FP8 quantized)"
    echo "Requires: uv pip install 'nvidia-modelopt[torch]'"
    uv run python "${SCRIPT_DIR}/export_onnx_n1d6.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --output_dir "${ONNX_DIR_FP8}" \
        --export_mode full_pipeline \
        --precision fp8 \
        --calib_dataset_path "${CALIB_DATASET}" \
        --calib_size "${CALIB_SIZE}"
}

step_build_fp8() {
    banner "FP8: Build TRT Engines"
    uv run python "${SCRIPT_DIR}/build_tensorrt_engine.py" \
        --mode full_pipeline \
        --onnx_dir "${ONNX_DIR_FP8}" \
        --engine_dir "${ENGINE_DIR_FP8}" \
        --precision fp8
}

step_compare_fp8() {
    banner "FP8: Accuracy Validation — PyTorch vs TRT (FP8)"
    uv run python "${SCRIPT_DIR}/benchmark_inference.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --trt_engine_path "${ENGINE_DIR_FP8}" \
        --trt_mode full_pipeline \
        --compare
}

step_compare_detailed_fp8() {
    banner "FP8: Detailed Accuracy — Per-component PyTorch vs TRT (FP8)"
    uv run python "${SCRIPT_DIR}/benchmark_inference.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --trt_engine_path "${ENGINE_DIR_FP8}" \
        --trt_mode full_pipeline \
        --compare-detailed
}

step_benchmark_fp8() {
    banner "FP8: Performance Benchmark"
    uv run python "${SCRIPT_DIR}/benchmark_inference.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --trt_engine_path "${ENGINE_DIR_FP8}" \
        --trt_mode full_pipeline \
        --skip_compile \
        --num_iterations "${NUM_ITER}"
}

step_profile_vit() {
    banner "FP8: ViT Per-Layer Cosine Profiling"
    echo "Requires: uv pip install 'nvidia-modelopt[torch]'"
    uv run python "${SCRIPT_DIR}/profile_vit_fp8_layers.py" \
        --model_path "${MODEL}" \
        --dataset_path "${DATASET}" \
        --embodiment_tag "${EMBODIMENT}" \
        --num_samples "${PROFILE_SAMPLES}" \
        --calib_dataset_path "${CALIB_DATASET}" \
        --calib_size "${CALIB_SIZE}" \
        --test_partial_quant
}

step_all() {
    step_export_bf16
    step_build_bf16
    step_compare
    step_benchmark
}

# ── Main ─────────────────────────────────────────────────────────────────────

STEP="${1:-}"

if [ -z "${STEP}" ]; then
    echo "Usage: bash $0 <step>"
    echo ""
    echo "BF16 steps: export_bf16 | build_bf16 | compare | compare_detailed | benchmark"
    echo "FP8 steps:  export_fp8 | build_fp8 | compare_fp8 | compare_detailed_fp8 | benchmark_fp8"
    echo "Profiling:  profile_vit"
    echo "Full:       all"
    echo ""
    echo "Environment variables (override defaults):"
    echo "  MODEL          = ${MODEL}"
    echo "  DATASET        = ${DATASET}"
    echo "  EMBODIMENT     = ${EMBODIMENT}"
    echo "  ONNX_DIR       = ${ONNX_DIR}       (BF16 ONNX)"
    echo "  ENGINE_DIR     = ${ENGINE_DIR}     (BF16 engines)"
    echo "  ONNX_DIR_FP8   = ${ONNX_DIR_FP8}   (FP8 ONNX)"
    echo "  ENGINE_DIR_FP8 = ${ENGINE_DIR_FP8} (FP8 engines)"
    echo "  NUM_ITER         = ${NUM_ITER}"
    echo "  PROFILE_SAMPLES  = ${PROFILE_SAMPLES}"
    exit 1
fi

case "${STEP}" in
    export_bf16)  step_export_bf16 ;;
    build_bf16)   step_build_bf16 ;;
    compare)           step_compare ;;
    compare_detailed)  step_compare_detailed ;;
    benchmark)         step_benchmark ;;
    export_fp8)             step_export_fp8 ;;
    build_fp8)              step_build_fp8 ;;
    compare_fp8)            step_compare_fp8 ;;
    compare_detailed_fp8)   step_compare_detailed_fp8 ;;
    benchmark_fp8)          step_benchmark_fp8 ;;
    profile_vit)            step_profile_vit ;;
    all)                    step_all ;;
    *)
        echo "ERROR: Unknown step '${STEP}'"
        echo "Valid BF16 steps: export_bf16 | build_bf16 | compare | compare_detailed | benchmark"
        echo "Valid FP8 steps:  export_fp8 | build_fp8 | compare_fp8 | compare_detailed_fp8 | benchmark_fp8"
        echo "Profiling: profile_vit"
        echo "Full: all"
        exit 1
        ;;
esac

banner "Done: ${STEP}"
