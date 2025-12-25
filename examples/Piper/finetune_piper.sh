set -x -e

# 设置 GPU 数量（distributed 模式）
export NUM_GPUS=8  # 根据实际可用GPU数量调整
export PROJECT_ROOT=$(pwd)/../../
export DATASET_ROOT=/home/lancel/Projects/2024-05-21-Robotics/piper/data
cd $PROJECT_ROOT


# Configuration type: joint_space, task_space_rot6d, task_space_rpy, task_space_quat
CONFIG_TYPE=${CONFIG_TYPE:-joint_space}
source .venv/bin/activate

# Distributed 训练模式：使用 torchrun 启动多GPU训练
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path $DATASET_ROOT/data_1202_lerobot_joint \
    --modality_config_path $PROJECT_ROOT/examples/Piper/piper_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir $PROJECT_ROOT/exps/piper_joint_relative_1202_n16 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 32 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --video_backend decord

