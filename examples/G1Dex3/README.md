# G1-Dex3 embodiment

Modality config for the Unitree G1 bimanual robot with Dex3 hands, produced by
`convert_mcap_to_lerobot.py`. Registers under `EmbodimentTag.NEW_EMBODIMENT`.

- **State / action keys** (index splits must match `meta/modality.json` exactly):
  `left_arm_eef (9) | left_hand (7) | right_arm_eef (9) | right_hand (7)`
- **Action space:** arms are **relative end-effector** (9D `xyz + rot6d`); hands are **absolute** Dex3
  joint targets. The dataset stores absolute poses on disk — GR00T's processor converts arm actions to
  relative because they're marked `rep=RELATIVE, type=EEF, format=XYZ_ROT6D` in
  [`g1_dex3_modality_config.py`](g1_dex3_modality_config.py). `meta/modality.json` only carries index
  splits, so this file is what makes the action space relative at training time.
- **Video keys:** `ego_view`, `left_wrist`, `right_wrist`. `ACTION_HORIZON = 16`.

## Local finetuning

```bash
MAX_STEPS=5 SAVE_STEPS=100 GLOBAL_BATCH_SIZE=2 EPISODE_SAMPLING_RATE=1.0 USE_WANDB=0 DATALOADER_NUM_WORKERS=0 \
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/place_cube_in_hand \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/G1Dex3/g1_dex3_modality_config.py \
  --letter-box-transform \
  --output-dir /tmp/g1dex3_smoketest
```

`--letter-box-transform` pads each camera view to a square before resizing, so the ego view and the two
wrist views (different native aspect ratios) stay stackable — required for this multi-camera dataset.

## Finetuning on SageMaker

The same flow runs as a SageMaker training job via `scripts/sagemaker/` (single-node, multi-GPU;
submitted directly with `estimator.fit()` by default). See
[`scripts/sagemaker/README.md`](../../scripts/sagemaker/README.md) for the full setup, then submit with
`scripts/sagemaker/train_gr00t_sagemaker.sh`.

### Customization knobs (what YOU fill in for your account)

These are the values that differ from the checked-in defaults; all live in
`scripts/sagemaker/launch_sagemaker.py` except `HF_TOKEN` (runtime env).

| Knob | Where | Change from the checked-in default |
| ---- | ----- | ---------------------------------- |
| **IAM execution role ARN** | `launch_sagemaker.py` → `roles["us-east-1"]` | pre-set to `arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess` (the batch_test role in TRI account 124224456861) |
| **ECR image URI** | `launch_sagemaker.py` → `IMAGE` | pre-set to `124224456861.dkr.ecr.us-east-1.amazonaws.com/claireyang/gr00t:latest` (or pass `--build-image`) |
| **S3 dataset path** | `--s3-dataset` (mounted at `/opt/ml/input/data/training`) | `s3://claireyang/gr00t_lerobot_datasets/place_cube_in_hand/...` — must point at the LeRobot **root** (the dir holding `meta/ data/ videos/`) |
| **S3 output / remote-sync** | `--s3-remote-sync` (or `S3_REMOTE_SYNC` env) | your bucket; drives `output_path` + `checkpoint_s3_uri` |
| **HF token** | `HF_TOKEN` env var (injected into the container) | **required** — pre-accept the gated `nvidia/Cosmos-Reason2-2B` license on HuggingFace first |
| **Instance type** | `--instance-type` / `INSTANCE_MAPPER` | `p4de` (A100 80GB × 8) is a good default; full finetune needs >22 GB/GPU |
| **Submit mode** | `--submit-mode` | `fit` (default; direct to SageMaker). Account 124224456861 has no `SAGEMAKER_TRAINING` Batch queue, so `queue` mode is not usable there |
| **`tri.project` / owner tag** | estimator `tags` | `MM:PJ-0077` / `{user}@tri.global` → your project code + email |
| **Hyperparameters** | `--max-steps`, `--save-steps`, `--global-batch-size`, `--episode-sampling-rate`, `--use-wandb`, `--dataloader-num-workers`, `--embodiment-tag`, `--modality-config-path`, `--letter-box-transform` | the finetune settings |
