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
```

## Submit (cloud)

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export S3_DATASET=s3://claireyang/gr00t_lerobot_datasets/curated_50both_25left_25right/
export S3_REMOTE_SYNC=s3://claireyang/gr00t_finetuning/
export WANDB_API_KEY=WANDB_API_KEY
bash scripts/sagemaker/train_gr00t_sagemaker.sh
```

**Cloud gotcha — input channel must be writable (`File`, not `FastFile`).** GR00T's dataset init
writes statistics back into `<dataset>/meta/` (`stats.json` via `generate_stats`, and
`relative_stats.json` via `generate_rel_stats` — the latter dumps *unconditionally*, even when the
cache is fresh). A `FastFile` channel is a read-only S3 FUSE mount, so those writes fail early on
rank 0 with `OSError: [Errno 30] Read-only file system: .../meta/.stats.json.*.tmp` (surfaces as a
torchrun `ChildFailedError`, exit code 1). The launcher therefore pins `input_mode="File"`, which
stages the dataset (~1.2 GB) onto the writable instance volume.

**Cloud gotcha — pre-bake stats into S3 for multi-GPU jobs (else a NCCL watchdog timeout).** GR00T
generates stats inside `run_or_wait_on_rank0()` (`gr00t/utils/dist_utils.py`): rank 0 runs the ~6 min
`generate_rel_stats` pass (loads every trajectory for each relative-action key — `left_arm_eef` +
`right_arm_eef`) while ranks 1–7 block at the `all_reduce` collective in its `finally`. If rank 0's work
exceeds the NCCL watchdog (default **600 s**), the waiting ranks abort with `c10::DistBackendError` /
`Signal 6 (SIGABRT)` — surfaces as a torchrun `ChildFailedError`, exit code 1, ~10 min in. Fix: upload a
`stats.json`/`relative_stats.json` produced by an earlier run into the S3 dataset's `meta/` so rank 0
skips the slow pass (`generate_stats` early-returns; `generate_rel_stats` skips `calculate_stats_for_key`
and only does a sub-second dump — harmless on the writable File mount). This is effectively **required**
at 8-GPU scale, not just a speedup.
```bash
# stats files are written by the (root) container as mode 600 -- take ownership first
sudo chown "$USER:$USER" <dataset>/meta/stats.json <dataset>/meta/relative_stats.json
aws s3 cp <dataset>/meta/stats.json          s3://<bucket>/<dataset>/meta/stats.json          --region us-east-1
aws s3 cp <dataset>/meta/relative_stats.json s3://<bucket>/<dataset>/meta/relative_stats.json --region us-east-1
```
Alternative (no pre-bake): raise the `init_process_group` timeout so waiting ranks tolerate rank 0's
long pass — a code change that leaves the underlying "one rank busy for minutes under a collective"
fragility in place.

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

## Local mode debug (SageMaker `local_gpu`) — recommended before any cloud run

Exercises the **full container path** (image → entry point → channel mounting → `finetune.sh` → real
GR00T training) on the puget's own GPUs. SageMaker runs the ECR image via local docker, so iterations
are far faster and cheaper than a cloud job. Debug your settings here first.

**Prerequisites (one time):**
```bash
# 1. SDK local extras (local mode drives `docker compose`):
uv pip install 'sagemaker[local]'

# 2. docker + compose on the puget:
sudo apt install docker.io
docker compose version                                   # any 2.x/3.x+ (see Compose v3+ gotcha below)
sudo usermod -aG docker $USER                            # then re-login (see docker-group gotcha below)

# 2b. NVIDIA Container Toolkit so the container can see the GPUs. Docker 29 discovers
#     GPUs via CDI, so a CDI spec must exist -- without it you get
#     "failed to discover GPU vendor from CDI" (see NVIDIA/CDI gotcha below):
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi   # must print GPUs

# 3. Build the image and tag it as the ECR URI, so local mode uses it WITHOUT pulling:
bash docker/build.sh
docker build -f docker/Dockerfile.sagemaker -t gr00t-finetune-sagemaker .
docker tag gr00t-finetune-sagemaker 124224456861.dkr.ecr.us-east-1.amazonaws.com/claireyang/gr00t:latest
```

**Run it:**
```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # real token: the container downloads the gated backbone
LOCAL=1 bash scripts/sagemaker/train_gr00t_sagemaker.sh
```

`LOCAL=1` (handled in the wrapper) swaps in a **local `file://` dataset** (`LOCAL_DATASET_DIR`, mounted
in File mode — no S3 download), writes outputs to `file:///tmp/gr00t_local_out`, and applies small
debug defaults (`MAX_STEPS=5`, `GLOBAL_BATCH_SIZE=8`, `USE_WANDB=0`). Override any inline:
```bash
LOCAL=1 MAX_STEPS=2 GLOBAL_BATCH_SIZE=1 \
  LOCAL_DATASET_DIR=/abs/path/to/some_lerobot_root \
  bash scripts/sagemaker/train_gr00t_sagemaker.sh
```

**Gotchas:**
- `local_gpu` exposes **all** the puget's GPUs → `finetune.sh` runs `torchrun` across all of them and
  requires `GLOBAL_BATCH_SIZE % <#GPUs> == 0` (default `8` covers 1/2/4/8-GPU boxes; lower it for a
  smaller box).
- `LOCAL_DATASET_DIR` must be the LeRobot **root** (the dir holding `meta/ data/ videos/`).
- If the SDK rejects the `file://` output path, set `S3_REMOTE_SYNC=s3://<bucket>/...` — the container
  still runs locally; only the final model upload touches S3.
- The `smddp`/`nccl` backend note from the SageMaker onboarding guide does **not** apply here: we omit
  `distribution={"torch_distributed": ...}`, so `finetune.sh` just uses plain `torchrun` with `nccl`.
- **Compose v3+ / `ImportError: Docker Compose is not installed`** (even though `docker compose version`
  works): SageMaker SDK v2's local-mode probe (`sagemaker/local/image.py`) hardcodes a `"v2" in
  \`docker compose version\`` string check that fails on Compose v3/v4/v5. It then falls back to looking
  for a standalone `docker-compose` binary. Fix without touching the venv or migrating to SDK v3 — add a
  shim that forwards the old name to the plugin:
  ```bash
  mkdir -p ~/.local/bin
  printf '#!/usr/bin/env bash\nexec docker compose "$@"\n' > ~/.local/bin/docker-compose
  chmod +x ~/.local/bin/docker-compose   # ensure ~/.local/bin is on PATH
  ```
  Remove the shim once off SageMaker SDK v2 (v3's `ModelTrainer` dropped the check).
- **`permission denied ... unix:///var/run/docker.sock`**: your user isn't in (or hasn't picked up) the
  `docker` group. `sudo usermod -aG docker $USER` adds you, but an **already-open** shell keeps its old
  groups — `id` won't list `docker` until you start a fresh login session. Either re-login (fixes all
  future shells) or, to test in the current shell without re-login, wrap the run:
  ```bash
  sg docker -c "LOCAL=1 bash scripts/sagemaker/train_gr00t_sagemaker.sh"   # inherits exported env (HF_TOKEN, conda PATH)
  ```
- **`Error response from daemon: failed to discover GPU vendor from CDI: no known GPU vendor found`**:
  Docker 29 exposes the compose `deploy.resources.reservations.devices` GPU request through CDI, which
  needs an NVIDIA CDI spec that only the NVIDIA Container Toolkit generates. Run step 2b above. Confirm
  with plain docker first (`docker run --rm --gpus all ... nvidia-smi`) — this error is **not**
  SageMaker-specific, so if bare docker fails, so will local mode. **Regenerate the spec after any GPU
  driver update** (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`); a stale spec pointing
  at old device nodes resurfaces the same error.

## Entry-point smoke test (no docker, no AWS)

Even lighter than local mode — runs `sagemaker_entry.py` directly in your GR00T env to verify the
env/path mapping reproduces the target `finetune.sh` command on 1 GPU (no container involved):

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

## Command reference

Recommended order: **entry-point smoke → local mode → cloud**.

| Goal | Command |
| ---- | ------- |
| Install client-side launcher deps | `uv pip install boto3 'sagemaker[local]'` |
| Build the training image | `bash docker/build.sh && docker build -f docker/Dockerfile.sagemaker -t gr00t-finetune-sagemaker .` |
| Build from a registry base (skip local base build) | `docker build -f docker/Dockerfile.sagemaker --build-arg BASE_IMAGE=<uri> -t gr00t-finetune-sagemaker .` |
| Tag + push to ECR | `aws ecr get-login-password --region us-east-1 \| docker login --username AWS --password-stdin 124224456861.dkr.ecr.us-east-1.amazonaws.com && docker tag gr00t-finetune-sagemaker $IMAGE && docker push $IMAGE` |
| Entry-point smoke (no docker/AWS) | see *Entry-point smoke test* above |
| **Local mode debug** | `LOCAL=1 bash scripts/sagemaker/train_gr00t_sagemaker.sh` |
| Local mode, override knobs | `LOCAL=1 MAX_STEPS=2 GLOBAL_BATCH_SIZE=1 bash scripts/sagemaker/train_gr00t_sagemaker.sh` |
| Cloud run (fit) | `bash scripts/sagemaker/train_gr00t_sagemaker.sh` |
| Cloud run, build+push image first | `python scripts/sagemaker/launch_sagemaker.py --build-image --user claire-yang --s3-dataset s3://… --s3-remote-sync s3://…` |
| Cloud run, stream logs | `WAIT=1 bash scripts/sagemaker/train_gr00t_sagemaker.sh` |
| Job status | `aws sagemaker describe-training-job --training-job-name <job> --region us-east-1 --query TrainingJobStatus --output text` |

Env vars the wrapper reads: `LOCAL`, `LOCAL_DATASET_DIR`, `USER_NAME`, `AWS_PROFILE_NAME`, `S3_DATASET`,
`S3_REMOTE_SYNC`, `INSTANCE_TYPE`, `MAX_STEPS`, `SAVE_STEPS`, `GLOBAL_BATCH_SIZE`,
`EPISODE_SAMPLING_RATE`, `DATALOADER_NUM_WORKERS`, `USE_WANDB`, `WAIT`, plus the secrets `HF_TOKEN`
(required) and `WANDB_API_KEY` (only if `USE_WANDB=1`).
