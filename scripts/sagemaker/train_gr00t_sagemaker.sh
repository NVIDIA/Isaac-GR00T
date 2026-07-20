#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Submit a full G1-Dex3 finetuning run to SageMaker (estimator.fit, single node).
# Every knob is env-overridable, e.g. for a quick smoke test:
#   MAX_STEPS=5 GLOBAL_BATCH_SIZE=8 INSTANCE_TYPE=p4 bash scripts/sagemaker/train_gr00t_sagemaker.sh
#
# Prerequisites (see scripts/sagemaker/README.md):
#   * export HF_TOKEN=...            (gated nvidia/Cosmos-Reason2-2B accepted on HF)
#   * export WANDB_API_KEY=...       (only if USE_WANDB=1)
#   * image built + pushed to ECR    (or pass --build-image)
#   * dataset in S3 (S3_DATASET), outputs bucket (S3_REMOTE_SYNC)
set -x -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- account / data ---
USER_NAME="${USER_NAME:-claire-yang}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-default}"
# S3_DATASET must point at the LeRobot dataset ROOT (the dir containing meta/ data/ videos/).
S3_DATASET="${S3_DATASET:-s3://claireyang/gr00t_lerobot_datasets/curated_50both_25left_25right/}"
S3_REMOTE_SYNC="${S3_REMOTE_SYNC:-s3://claireyang/gr00t_finetuning/}"

# --- instance + hyperparameters (real-run defaults) ---
# p4/p4de/p5 all have 8 GPUs, so GLOBAL_BATCH_SIZE MUST be a multiple of 8
# (GR00T asserts global_batch_size % num_gpus == 0; per-device batch = global / 8).
INSTANCE_TYPE="${INSTANCE_TYPE:-p4de}"        # 8x A100 80GB (account has 64 on-demand quota; p4d=0)
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"  # per-device 4 on 8 GPUs; 80GB has room to raise to 64
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
USE_WANDB="${USE_WANDB:-1}"

# --wait streams logs and blocks until the run finishes (hours). Off by default for
# a full run -- submit and monitor async. Set WAIT=1 to stream instead.
WAIT="${WAIT:-0}"

set +x  # don't echo the secret under `set -x`
if [ "$USE_WANDB" = "1" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: USE_WANDB=1 but WANDB_API_KEY is not exported; the job may stall at wandb login." >&2
fi
set -x

ARGS=(
    --user "$USER_NAME"
    --region us-east-1
    --profile "$AWS_PROFILE_NAME"
    --submit-mode fit
    --instance-type "$INSTANCE_TYPE"
    --s3-dataset "$S3_DATASET"
    --s3-remote-sync "$S3_REMOTE_SYNC"
    --base-model-path nvidia/GR00T-N1.7-3B
    --embodiment-tag NEW_EMBODIMENT
    --modality-config-path examples/G1Dex3/g1_dex3_modality_config.py
    --letter-box-transform true
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --episode-sampling-rate "$EPISODE_SAMPLING_RATE"
    --use-wandb "$USE_WANDB"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
)
if [ "$WAIT" = "1" ]; then
    ARGS+=(--wait)
fi

python3 "$DIR/launch_sagemaker.py" "${ARGS[@]}"
