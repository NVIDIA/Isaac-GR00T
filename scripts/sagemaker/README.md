# GR00T finetuning on SageMaker

Run the existing `examples/finetune.sh` flow as a SageMaker training job. The launcher structure is
adapted from [`TRI-ML/batch_test`](https://github.com/TRI-ML/batch_test).

**Submission mode (`--submit-mode`):**
- **`fit`** (default) — submit directly to SageMaker via `estimator.fit()`. No AWS Batch queue needed.
  This is the correct mode for account **124224456861**, which has **no** `SAGEMAKER_TRAINING` Batch
  queue (all its Batch queues are plain `ECS`/`ECS_FARGATE` container queues).
- **`queue`** — submit to an AWS Batch `SAGEMAKER_TRAINING` Fair-Share queue (the batch_test model).
  Only works in an account/region where such a queue exists; pass `--queue-name` to target it.

Single-node, multi-GPU only. `examples/finetune.sh` owns the `torchrun --nproc_per_node` fan-out, so
the estimator does **not** enable `distribution={"torch_distributed": ...}`.

## Files

| File | Runs where | Role |
| ---- | ---------- | ---- |
| `launch_sagemaker.py` | your laptop | Builds the PyTorch estimator + submits it (`fit` or `queue`) |
| `sagemaker_entry.py` | in the container | Maps SageMaker env/paths → `examples/finetune.sh` |
| `train_gr00t_sagemaker.sh` | your laptop | Convenience wrapper with the G1-Dex3 smoke-test args pre-filled |
| `../../docker/Dockerfile.sagemaker` | build time | `FROM gr00t:latest` + `sagemaker-training` + baked repo at `/opt/gr00t` |

## Customization knobs (what YOU fill in for your account)

| Knob | Where | Change from the checked-in default |
| ---- | ----- | ---------------------------------- |
| **IAM execution role ARN** | `launch_sagemaker.py` → `roles["us-east-1"]` | pre-set to `arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess` (the batch_test role in TRI account 124224456861) |
| **ECR image URI** | `launch_sagemaker.py` → `IMAGE` | pre-set to `124224456861.dkr.ecr.us-east-1.amazonaws.com/claireyang/gr00t:latest` (or pass `--build-image`) |
| **AWS profile** | `--profile` / `AWS_PROFILE_NAME` in the wrapper | pre-set to `robotics-shared-sagemaker-ml-dev` (SSO profile into account 124224456861) |
| **S3 dataset path** | `--s3-dataset` (mounted at `/opt/ml/input/data/training`) | `s3://claireyang/gr00t_lerobot_datasets/place_cube_in_hand/...` — must point at the LeRobot **root** (the dir holding `meta/ data/ videos/`) |
| **S3 output / remote-sync** | `--s3-remote-sync` (or `S3_REMOTE_SYNC` env) | your bucket; drives `output_path` + `checkpoint_s3_uri` |
| **HF token** | `HF_TOKEN` env var (injected into the container) | **required** — pre-accept the gated `nvidia/Cosmos-Reason2-2B` license on HuggingFace first |
| **Instance type** | `--instance-type` / `INSTANCE_MAPPER` | `p4de` (A100 80GB × 8) is a good default; full finetune needs >22 GB/GPU |
| **Submit mode** | `--submit-mode` | `fit` (default; direct to SageMaker). Use `queue` + `--queue-name` only in an FSS-enabled account |
| **`tri.project` / owner tag** | estimator `tags` | `MM:PJ-0077` / `{user}@tri.global` → your project code + email |
| **Hyperparameters** | `--max-steps`, `--save-steps`, `--global-batch-size`, `--episode-sampling-rate`, `--use-wandb`, `--dataloader-num-workers`, `--embodiment-tag`, `--modality-config-path`, `--letter-box-transform` | the finetune settings |

## One-time setup

```bash
# 0a. Client-side launcher deps (boto3 + SageMaker SDK) into the env you launch from
#     -- NOT part of GR00T's training deps:
uv pip install boto3
uv pip install 'sagemaker<3'

# 0b. Accept the gated model license once (as your HF user):
#     https://huggingface.co/nvidia/Cosmos-Reason2-2B  -> "Agree and access"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

aws sso login --profile robotics-shared-sagemaker-ml-dev
aws sts get-caller-identity --profile robotics-shared-sagemaker-ml-dev --query Account --output text   # must print 124224456861

# 1. Build the base GR00T image, then the SageMaker image (from the repo root):
bash docker/build.sh
docker build -f docker/Dockerfile.sagemaker -t gr00t-finetune-sagemaker .
#    (Option B) skip the local base build by pointing at a registry base:
#    docker build -f docker/Dockerfile.sagemaker \
#      --build-arg BASE_IMAGE=<acct>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag> \
#      -t gr00t-finetune-sagemaker .

# 2. Tag + push to your ECR repo (or let launch_sagemaker.py --build-image do it):
AWS_REGION=us-east-1
IMAGE=124224456861.dkr.ecr.us-east-1.amazonaws.com/claireyang/gr00t:latest
aws ecr describe-repositories --repository-names claireyang/gr00t --region $AWS_REGION \
  || aws ecr create-repository --repository-name claireyang/gr00t --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS \
  --password-stdin 124224456861.dkr.ecr.us-east-1.amazonaws.com
docker tag gr00t-finetune-sagemaker $IMAGE
docker push $IMAGE
# -> already set as launch_sagemaker.py's IMAGE constant.

## Submit

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export S3_DATASET=s3://claireyang/gr00t_lerobot_datasets/curated_50both_25left_25right/
export S3_REMOTE_SYNC=s3://claireyang/gr00t_finetuning/
export WANDB_API_KEY=WANDB_API_KEY
bash scripts/sagemaker/train_gr00t_sagemaker.sh
```

## Monitor

With `--submit-mode fit` (default): the wrapper passes `--wait`, so logs stream to your terminal until
the job ends. Otherwise watch the **SageMaker console → Training jobs**, or:
```bash
aws sagemaker describe-training-job --training-job-name <job> --region us-east-1 \
  --query TrainingJobStatus --output text
```
A successful smoke test writes checkpoints to
`s3://<remote-sync>/sagemaker/<user>/gr00t-finetune/<job>/checkpoint/`.

(In `queue` mode you'd instead monitor via `batchy` / the AWS Batch console, as in `batch_test`.)

## Local dry-run (no AWS)

Verify the entry point reproduces the target command on 1 GPU before touching SageMaker:

```bash
export SM_CHANNEL_TRAINING=$PWD/demo_data SM_NUM_GPUS=1 HF_TOKEN=hf_xxx
export GR00T_REPO_DIR=$PWD HF_HOME=/tmp/hf_home
# /opt/ml/checkpoints isn't writable off-SageMaker, so point the output elsewhere:
export SM_CHECKPOINT_DIR=/tmp/g1dex3_smoketest
python scripts/sagemaker/sagemaker_entry.py \
  --base-model-path nvidia/GR00T-N1.7-3B --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/G1Dex3/g1_dex3_modality_config.py --letter-box-transform true \
  --dataset-subdir place_cube_in_hand --max-steps 5 --save-steps 100 \
  --global-batch-size 2 --episode-sampling-rate 1.0 --use-wandb 0 --dataloader-num-workers 0
```
