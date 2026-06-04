set -x -e

export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /home/fortress/.cache/huggingface/lerobot/astribot/fold_cloth_better \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/astribot/astribot_with_torso_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir outputs/astribot_fold_cloth_better \
    --save-total-limit 5 \
    --save-steps 5000 \
    --max-steps 30000 \
    --global-batch-size 128 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 8