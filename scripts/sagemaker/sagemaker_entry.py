#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""In-container entry point for running GR00T finetuning as a SageMaker training job.

SageMaker's ``sagemaker-training`` toolkit invokes this script (via the
``SAGEMAKER_PROGRAM`` env var baked into the image) as::

    python sagemaker_entry.py --key value --key value ...

where every hyperparameter passed to the estimator arrives as a ``--key value``
CLI pair (SageMaker serializes all hyperparameter values to strings).

This script does not train directly -- it translates SageMaker's conventions
(S3 input channels mounted under ``/opt/ml/input/data/<channel>``, artifacts
written to ``/opt/ml/checkpoints``, GPU count in ``SM_NUM_GPUS``) into the exact
env vars + CLI flags that the existing, tested ``examples/finetune.sh`` expects,
then execs it. ``finetune.sh`` owns the single-node ``torchrun`` fan-out, so the
estimator must NOT enable ``distribution={"torch_distributed": ...}``.
"""

import argparse
import os
import subprocess
import sys


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # --- passthrough to finetune.sh CLI flags ---
    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--modality-config-path", default=None)
    parser.add_argument(
        "--letter-box-transform",
        default="false",
        help="Truthy string (true/1/yes) adds --letter-box-transform to finetune.sh",
    )
    parser.add_argument(
        "--dataset-subdir",
        default="",
        help="Optional path under the mounted training channel that holds the "
        "LeRobot dataset root (leave empty if the channel root IS the dataset).",
    )
    # --- mapped to finetune.sh env vars ---
    parser.add_argument("--max-steps", default="10000")
    parser.add_argument("--save-steps", default="1000")
    parser.add_argument("--global-batch-size", default="32")
    parser.add_argument("--episode-sampling-rate", default="0.1")
    parser.add_argument("--use-wandb", default="0")
    parser.add_argument("--dataloader-num-workers", default="4")
    # argparse would otherwise choke on any hyperparameter we didn't declare;
    # keep unknown ones so new knobs don't require editing this file.
    args, unknown = parser.parse_known_args()
    args.extra = unknown
    return args


def resolve_dataset_path(dataset_subdir: str) -> str:
    """The dataset arrives via the SageMaker ``training`` input channel."""
    channel_root = os.environ.get("SM_CHANNEL_TRAINING")
    if channel_root is None:
        raise RuntimeError(
            "SM_CHANNEL_TRAINING is not set -- pass the dataset as the 'training' "
            "input channel (inputs={'training': 's3://...'}) or set the env var for "
            "a local dry-run."
        )
    return os.path.join(channel_root, dataset_subdir) if dataset_subdir else channel_root


def num_gpus() -> str:
    if "SM_NUM_GPUS" in os.environ:
        return os.environ["SM_NUM_GPUS"]
    try:
        import torch

        return str(torch.cuda.device_count() or 1)
    except Exception:
        return "1"


def main() -> None:
    args = parse_args()

    repo_dir = os.environ.get("GR00T_REPO_DIR", "/opt/ml/code")
    # SageMaker streams /opt/ml/checkpoints to checkpoint_s3_uri during the run;
    # GR00T writes checkpoint-*/, processor/, experiment_cfg/ under output_dir.
    output_dir = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = resolve_dataset_path(args.dataset_subdir)

    # Env vars that finetune.sh reads (defaults live in the shell script).
    env = os.environ.copy()
    env["NUM_GPUS"] = num_gpus()
    env["MAX_STEPS"] = args.max_steps
    env["SAVE_STEPS"] = args.save_steps
    env["GLOBAL_BATCH_SIZE"] = args.global_batch_size
    env["EPISODE_SAMPLING_RATE"] = args.episode_sampling_rate
    env["USE_WANDB"] = args.use_wandb
    env["DATALOADER_NUM_WORKERS"] = args.dataloader_num_workers
    # HF_TOKEN / HF_HOME are injected by the launcher / Dockerfile; the GR00T
    # loader threads HF_TOKEN through transformers_loading_kwargs automatically.

    cmd = [
        "bash",
        "examples/finetune.sh",
        "--base-model-path",
        args.base_model_path,
        "--dataset-path",
        dataset_path,
        "--embodiment-tag",
        args.embodiment_tag,
        "--output-dir",
        output_dir,
    ]
    if args.modality_config_path:
        cmd += ["--modality-config-path", args.modality_config_path]
    if _str2bool(args.letter_box_transform):
        cmd += ["--letter-box-transform"]
    if args.extra:
        cmd += ["--", *args.extra]

    print(f"[sagemaker_entry] repo_dir={repo_dir}", flush=True)
    print(f"[sagemaker_entry] dataset_path={dataset_path}", flush=True)
    print(f"[sagemaker_entry] output_dir={output_dir} NUM_GPUS={env['NUM_GPUS']}", flush=True)
    print(f"[sagemaker_entry] exec: {' '.join(cmd)}", flush=True)

    completed = subprocess.run(cmd, cwd=repo_dir, env=env)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
