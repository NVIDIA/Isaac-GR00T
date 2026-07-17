#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper to submit the G1-Dex3 smoke test to the AWS Batch FSS queue.
# Mirrors the local target command:
#   MAX_STEPS=5 SAVE_STEPS=100 GLOBAL_BATCH_SIZE=2 EPISODE_SAMPLING_RATE=1.0 \
#   USE_WANDB=0 DATALOADER_NUM_WORKERS=0 bash examples/finetune.sh ...
#
# Prerequisites (see scripts/sagemaker/README.md):
#   * export HF_TOKEN=...            (gated nvidia/Cosmos-Reason2-2B accepted on HF)
#   * image built + pushed to ECR    (or pass --build-image)
#   * dataset uploaded to S3         (set S3_DATASET below)
set -x -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUSTOMIZE these:
USER_NAME="${USER_NAME:-claire-yang}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-default}"
S3_DATASET="${S3_DATASET:-s3://robotics-cam-data/scratch/claireyang/gr00t_lerobot_datasets/place_cube_in_hand/curated_50both_25left_25right/}"
S3_REMOTE_SYNC="${S3_REMOTE_SYNC:-s3://CHANGE-ME-bucket}"

python3 "$DIR/launch_sagemaker.py" \
    --user "$USER_NAME" \
    --region us-east-1 \
    --profile "$AWS_PROFILE_NAME" \
    --instance-type p4de \
    --s3-dataset "$S3_DATASET" \
    --s3-remote-sync "$S3_REMOTE_SYNC" \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1Dex3/g1_dex3_modality_config.py \
    --letter-box-transform true \
    --max-steps 5 \
    --save-steps 100 \
    --global-batch-size 2 \
    --episode-sampling-rate 1.0 \
    --use-wandb 0 \
    --dataloader-num-workers 0
