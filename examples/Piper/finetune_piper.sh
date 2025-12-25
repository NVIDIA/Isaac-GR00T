#!/bin/bash
set -x -e


# usage:
# ACTION_SIZE=32 bash finetune_piper.sh
# GLOBAL_BATCH_SIZE=128 bash finetune_piper.sh


# ============================================
# 可配置参数（可通过环境变量覆盖）
# ============================================
ACTION_SIZE=${ACTION_SIZE:-16}           # Action horizon: 16 或 32
NUM_GPUS=${NUM_GPUS:-8}                  # GPU 数量
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-192}
MAX_STEPS=${MAX_STEPS:-10000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MASTER_PORT=${MASTER_PORT:-29500}

# ============================================
# 路径配置
# ============================================
export PROJECT_ROOT=$(pwd)/../../
export DATASET_ROOT=/home/lancel/Datasets/piper
cd $PROJECT_ROOT

source .venv/bin/activate

# 根据 ACTION_SIZE 选择配置文件
CONFIG_FILE=$PROJECT_ROOT/examples/Piper/piper_config_action${ACTION_SIZE}.py
OUTPUT_DIR=$PROJECT_ROOT/exps/piper_joint_relative_action${ACTION_SIZE}_1202_n16

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Supported ACTION_SIZE: 16, 32"
    exit 1
fi

echo "=========================================="
echo "Training with ACTION_SIZE=${ACTION_SIZE}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Distributed 训练
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path $DATASET_ROOT/data_1202_lerobot_joint \
    --modality_config_path $CONFIG_FILE \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps $MAX_STEPS \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate $LEARNING_RATE \
    --use_wandb \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --video_backend decord
