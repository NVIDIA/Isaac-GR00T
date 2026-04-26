#!/bin/bash

#SBATCH --job-name=gr00t_h100_4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=h100
#SBATCH --gres gpu:4
#SBATCH --comment="gr00t_h100_4 training"

# slurm-gres-viz
# —highprio 
# srun --gpus=1 --nodes=1 --pty /bin/bash

# 환경 설정
cd /rlwrld2/home/david/gr00tN17/Isaac-GR00T
source /workspace/home/rlwrld/david/miniconda3/etc/profile.d/conda.sh
conda activate gr00tN17
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH=/rlwrld2/home/david/gr00tN17/Isaac-GR00T
python -c "import gr00t; print(gr00t.__file__)"
echo "Conda environment 'gr00tN17' activated."
echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
echo "PYTHONPATH=$PYTHONPATH"

# Set wandb API key for authentication
export WANDB_API_KEY=981eedf5f279635b72ecbd74304b034fd59288e4

# Force wandb to stay offline to avoid network/filesystem issues that cause Bus errors
# Check GPUS
nvidia-smi
nproc

export WANDB_MODE=disabled
export WANDB_DISABLED=true
# wandb.init() needs time to start its internal service; 1s is too short
# and causes `CommError: Run initialization has timed out`.
export WANDB_INIT_TIMEOUT=120

echo "========== 학습 시작 =========="

# Get CONFIG_NAME from command line argument
if [ -z "$1" ]; then
    echo "Error: CONFIG_NAME argument is required"
    echo "Usage: sbatch gr00t_training.sbatch <CONFIG_NAME>"
    echo "Example: sbatch gr00t_training.sbatch aws_mouse"
    exit 1
fi

CONFIG_NAME=$1
PROJECT_NAME=gr00t
MAX_STEPS=30000
SAVE_STEPS=10000

echo "Using CONFIG_NAME: $CONFIG_NAME"

# ---- State dropout (SD) -----------------------------------------------------
# STATE_DROPOUT_PROB is resolved in this order:
#   1) env var STATE_DROPOUT_PROB if exported
#   2) trailing _SD<NN> in CONFIG_NAME  (e.g. mousebox_packing_0421_SD30 -> 0.30)
#   3) default 0.0
# When (2) is used, SD_SUFFIX stays empty so OUTPUT_DIR isn't doubled-suffixed.
_SD_IN_NAME=0
if [ -z "${STATE_DROPOUT_PROB:-}" ]; then
    if [[ "$CONFIG_NAME" =~ _SD([0-9]+)(_|$) ]]; then
        _SD_PCT_NAME="${BASH_REMATCH[1]}"
        STATE_DROPOUT_PROB=$(awk "BEGIN {printf \"%g\", $_SD_PCT_NAME/100}")
        _SD_IN_NAME=1
        echo "Derived STATE_DROPOUT_PROB=$STATE_DROPOUT_PROB from CONFIG_NAME suffix _SD${_SD_PCT_NAME}"
    else
        STATE_DROPOUT_PROB=0.0
    fi
fi

SD_SUFFIX=""
if [ "$_SD_IN_NAME" = "0" ] && awk "BEGIN {exit !($STATE_DROPOUT_PROB > 0)}"; then
    SD_PCT=$(awk "BEGIN {printf \"%02d\", $STATE_DROPOUT_PROB * 100}")
    SD_SUFFIX="_SD${SD_PCT}"
fi
echo "STATE_DROPOUT_PROB=$STATE_DROPOUT_PROB (suffix: '${SD_SUFFIX}')"



OUTPUT_DIR=output/train/${CONFIG_NAME}${SD_SUFFIX}_h100_4
# OUTPUT_DIR=output/train/${CONFIG_NAME}
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR already exists. Erasing..."
    # erase the directory
    rm -rf $OUTPUT_DIR
fi
mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Reduce allocator fragmentation; cheap, no downside.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training..."

# ---- N1.7 finetune arguments (edit or export these before sbatch) ----
# Base checkpoint: any GR00T N1.7 HF repo or local path.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
# If DATA_YAML is set, use the multi-dataset launcher; otherwise DATASET_PATH is required.
DATA_YAML="${DATA_YAML:-}"
# Single-dataset path (only used when DATA_YAML is empty).
DATASET_PATH="${DATASET_PATH:-}"
# REQUIRED: embodiment tag (see gr00t/data/embodiment_tags.py).
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
# Optional: python file defining a custom modality config (leave empty to use default).
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-}"
# Derive a per-job port to avoid EADDRINUSE when two jobs land on the same node
# or when relaunching before TIME_WAIT clears. Falls back to 29500 outside SLURM.
MASTER_PORT="${MASTER_PORT:-$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))}"
# ----------------------------------------------------------------------

# ---- Data input auto-resolution --------------------------------------------
# If neither DATA_YAML nor DATASET_PATH is set, look for configs/<CONFIG_NAME>.yaml
# so that `sbatch this.sbatch <name>` just works when you drop YAMLs into configs/.
if [ -z "$DATA_YAML" ] && [ -z "$DATASET_PATH" ]; then
    _CANDIDATE="configs/${CONFIG_NAME}.yaml"
    if [ -f "$_CANDIDATE" ]; then
        DATA_YAML="$_CANDIDATE"
        echo "Auto-resolved DATA_YAML=$DATA_YAML from CONFIG_NAME"
    fi
fi

COMMON_ARGS=(
  --num_gpus 4
  # 4 x H100-80GB, ZeRO-2, NO --tune_visual:
  #   per-device microbatch = global_batch_size / (num_gpus * grad_accum) = 128 / (4*2) = 16
  # Effective total batch per optimizer step = 128 (across all 4 GPUs).
  # H100 80GB has identical memory capacity to A100 80GB but ~2-3x faster in bf16.
  # For more throughput on H100: drop grad_accum to 1 (microbatch 32).
  # If you OOM: raise grad_accum to 4 (microbatch 8) or add --gradient_checkpointing.
  --global_batch_size 128
  --gradient_accumulation_steps 2
  --output_dir "$OUTPUT_DIR"
  --max_steps "$MAX_STEPS"
  --save_steps "$SAVE_STEPS"
  --dataloader_num_workers 16
  --learning_rate 1e-4
  --state_dropout_prob "$STATE_DROPOUT_PROB"
)

if [ -n "$MODALITY_CONFIG_PATH" ]; then
  COMMON_ARGS+=(--modality_config_path "$MODALITY_CONFIG_PATH")
fi

if [ -n "$DATA_YAML" ]; then
  echo "Using multi-dataset launcher with DATA_YAML=$DATA_YAML"
  LAUNCH_ARGS=(
    --base_model_path "$BASE_MODEL_PATH"
    --data_yaml "$DATA_YAML"
    "${COMMON_ARGS[@]}"
  )
  torchrun --nproc_per_node=4 --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune_multi.py "${LAUNCH_ARGS[@]}"
else
  if [ -z "$DATASET_PATH" ]; then
    echo "Error: either DATA_YAML or DATASET_PATH must be set." >&2
    exit 2
  fi
  echo "Using single-dataset launcher with DATASET_PATH=$DATASET_PATH"
  LAUNCH_ARGS=(
    --base_model_path "$BASE_MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --embodiment_tag "$EMBODIMENT_TAG"
    "${COMMON_ARGS[@]}"
  )
  torchrun --nproc_per_node=4 --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py "${LAUNCH_ARGS[@]}"
fi

echo "========== 학습 종료 (업로드 시도) =========="

# # ------------------------------------------------------------------
# # [업로드 섹션]
# # 폴더(-d)가 실제로 존재하는지 확인 후 업로드를 진행합니다.
# # ------------------------------------------------------------------

# for STEP in $(seq $SAVE_STEPS $SAVE_STEPS $MAX_STEPS); do
#     CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint-$STEP"
#     REPO_NAME="Joonwoo-David/${PROJECT_NAME}_${CONFIG_NAME}_cp${STEP}"
    
#     if [ -d "$CHECKPOINT_DIR" ]; then
#         echo "Found checkpoint-$STEP. Uploading to $REPO_NAME..."
#         hf upload "$REPO_NAME" "$CHECKPOINT_DIR" .
#     else
#         echo "Skip: checkpoint-$STEP 폴더가 생성되지 않았습니다."
#     fi
# done


# echo "========== 모든 작업 완료 =========="