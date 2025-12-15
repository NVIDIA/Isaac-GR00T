set -x -e

export NUM_GPUS=2

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/ \
    --embodiment_tag LIBERO_SIM \
    --num_gpus $NUM_GPUS \
    --output_dir /tmp/libero_goal \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 128 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 12 \
    --num_shards_per_epoch 10000 \
    --shard_size 512 \
    --episode_sampling_rate 0.05
