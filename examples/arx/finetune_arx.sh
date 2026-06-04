export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /home/ailab/lerobot/datasets/arx/fold_cloth \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/arx/arx_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir outputs/arx_fold_cloth \
    --save-total-limit 5 \
    --save-steps 200 \
    --max-steps 2000 \
    --global-batch-size 1 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4