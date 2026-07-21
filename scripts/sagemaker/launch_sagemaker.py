#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Launch GR00T finetuning as a SageMaker training job.

Default submission is estimator.fit() (direct to SageMaker, no Batch queue).
--submit-mode queue targets an AWS Batch SAGEMAKER_TRAINING FSS queue instead,
but account 124224456861 has no such queue (all its Batch queues are ECS), so
fit is the working path there.

Adapted from TRI-ML/batch_test/launch_sagemaker.py. Differences from that
open_lm launcher:
  * NAME + entry point target the GR00T finetuning wrapper (scripts/sagemaker/sagemaker_entry.py).
  * The dataset is passed as a real SageMaker input channel ('training'),
    because GR00T reads a local LeRobot dataset root -- it does not stream S3
    manifests the way open_lm did (batch_test used inputs=[None]).
  * HF_TOKEN is injected into the container so the gated VLM backbone
    (nvidia/Cosmos-Reason2-2B) can be downloaded at runtime.
  * distribution={"torch_distributed": ...} is intentionally OMITTED -- this is
    single-node multi-GPU and examples/finetune.sh owns the torchrun fan-out.

======================================================================
CUSTOMIZE THESE FOR YOUR ACCOUNT (see the knobs table in the README):
  - roles[...]            : your SageMaker execution role ARN(s)
  - IMAGE                 : your ECR image URI (built from docker/Dockerfile.sagemaker)
  - --s3-dataset          : S3 URI of your LeRobot dataset root
  - --s3-remote-sync      : your S3 bucket for outputs/checkpoints
  - HF_TOKEN env var      : set before running; pre-accept the gated HF license
  - tags (tri.project)    : your project code / owner email
======================================================================
"""

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
import time

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


# NOTE: sagemaker.aws_batch.training_queue is imported lazily inside the "queue"
# submit-mode branch -- it only exists in newer SDKs and is only needed when
# targeting an AWS Batch SAGEMAKER_TRAINING queue. Default submission is
# estimator.fit(), which has no such dependency.


logging.getLogger().setLevel(logging.ERROR)

NAME = "gr00t-finetune"

INSTANCE_MAPPER = {
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
}

# only role is in us-east-1
roles = {
    "us-east-1": "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess",
}

# ECR image URI (or pass --build-image to build+push from the Dockerfile).
IMAGE = "124224456861.dkr.ecr.us-east-1.amazonaws.com/claireyang/gr00t:latest"

# Path to the SageMaker Dockerfile in this repo (used only with --build-image).
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.sagemaker"


def get_env_variable(var_name, default=None):
    return os.getenv(var_name, default)


def create_session(profile, region):
    try:
        if region is None:
            region = get_env_variable("AWS_DEFAULT_REGION", "us-west-2")
        if profile is not None:
            session = boto3.session.Session(profile_name=profile, region_name=region)
            boto3.setup_default_session(profile_name=profile, region_name=region)
        elif get_env_variable("AWS_PROFILE") is not None:
            session = boto3.session.Session(
                profile_name=get_env_variable("AWS_PROFILE"), region_name=region
            )
            boto3.setup_default_session(
                profile_name=get_env_variable("AWS_PROFILE"), region_name=region
            )
        else:
            boto3.setup_default_session(region_name=region)
            session = boto3.session.Session(region_name=region)
        return sagemaker.Session(session)
    except Exception as e:
        print(f"Failed to create session: {e}")
        return None


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_sagemaker_role_arn(region):
    try:
        return roles[region]
    except KeyError:
        raise ValueError(f"Region {region} not supported")


def _ecr_registry(uri):
    """Return the '<acct>.dkr.ecr.<region>.amazonaws.com' host of an ECR image URI, else None."""
    host = uri.split("/", 1)[0]
    return host if ".dkr.ecr." in host and ".amazonaws.com" in host else None


def get_image(base_image, profile="default", region="us-east-1"):
    """Build docker/Dockerfile.sagemaker and push to the IMAGE repo; return its URI.

    ``base_image`` is passed as the BASE_IMAGE build-arg (Option B) so the base can
    be a local tag (default ``gr00t:latest``) or a registry URI.
    """
    os.environ["AWS_PROFILE"] = f"{profile}"
    repo_uri = IMAGE.rsplit(":", 1)[0]  # strip the tag -> ".../claireyang/gr00t"
    repo_name = repo_uri.split(".amazonaws.com/", 1)[1]  # -> "claireyang/gr00t"
    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} "
        f"| docker login --username AWS --password-stdin"
    )

    print("Building GR00T SageMaker container")
    commands = []
    # If the base lives in ECR, authenticate to that registry so `docker build` can pull it.
    base_registry = _ecr_registry(base_image)
    if base_registry:
        commands.append(f"{login_cmd} {base_registry}")
    commands += [
        f"docker build --progress=plain -f {DOCKERFILE} "
        f"--build-arg BASE_IMAGE={base_image} -t {IMAGE} {REPO_ROOT}",
        f"{login_cmd} {_ecr_registry(IMAGE)}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {repo_name} || "
            f"aws --region {region} ecr create-repository --repository-name {repo_name}"
        ),
    ]
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {IMAGE}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return IMAGE


def get_job_name(base):
    now = datetime.now()
    now_ms_str = f"{now.microsecond // 1000:03d}"
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
    return "-".join([base, date_str])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run in SageMaker local_gpu mode")
    parser.add_argument("--user", required=True, help="User name (used in job/base names + tags)")

    # AWS / profile args
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument("--build-image", action="store_true", help="Build + push the ECR image")
    parser.add_argument(
        "--base-image",
        default="gr00t:latest",
        help="Base image for the build (BASE_IMAGE build-arg); local tag or ECR URI",
    )
    parser.add_argument(
        "--s3-remote-sync", default=None, help="S3 output root (else reads S3_REMOTE_SYNC env var)"
    )
    parser.add_argument(
        "--submit-mode",
        default="fit",
        choices=["fit", "queue"],
        help="'fit' submits directly to SageMaker (default; no Batch queue needed). "
        "'queue' targets an AWS Batch SAGEMAKER_TRAINING FSS queue (batch_test style; "
        "requires such a queue to exist -- account 124224456861 currently has none).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="fit mode: stream logs and block until the job finishes (good for smoke tests)",
    )
    parser.add_argument("--queue-name", default=None, help="queue mode: override the queue name")
    parser.add_argument("--priority", default=10, type=int, help="FSS priority (1-9999)")
    parser.add_argument("--fss-identifier", default="default", help="FSS share id (do not change)")

    # Instance args
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument(
        "--max-run", type=int, default=5 * 24 * 60 * 60, help="Max job runtime (default 5 days)"
    )

    # GR00T dataset + hyperparameters
    parser.add_argument("--s3-dataset", required=True, help="S3 URI of the LeRobot dataset root")
    parser.add_argument("--dataset-subdir", default="", help="Path under the channel to the dataset root")
    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--modality-config-path", default="examples/G1Dex3/g1_dex3_modality_config.py"
    )
    parser.add_argument("--letter-box-transform", default="true")
    parser.add_argument("--max-steps", default="5")
    parser.add_argument("--save-steps", default="100")
    parser.add_argument("--global-batch-size", default="2")
    parser.add_argument("--episode-sampling-rate", default="1.0")
    parser.add_argument("--use-wandb", default="0")
    parser.add_argument("--dataloader-num-workers", default="0")

    args = parser.parse_args()

    if args.s3_remote_sync is None:
        assert "S3_REMOTE_SYNC" in os.environ, (
            "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        )
        args.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]

    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, (
        "HF_TOKEN env var is required so the container can download the gated "
        "nvidia/Cosmos-Reason2-2B backbone. Pre-accept the license on HuggingFace first."
    )

    region = args.region
    image = (
        get_image(args.base_image, region=region, profile=args.profile)
        if args.build_image
        else IMAGE
    )
    if "CHANGE-ME" in image:
        raise SystemExit(
            "Set the IMAGE constant to your ECR image URI, or pass --build-image to build it."
        )
    if "CHANGE-ME" in get_sagemaker_role_arn(region):
        raise SystemExit(f"Set roles['{region}'] to your SageMaker execution role ARN before submitting.")

    if args.local:
        # Local mode runs the container via local docker. LocalSession (not the
        # cloud Session) is what triggers that path; local_code=True mounts the
        # source_dir directly instead of uploading it to S3.
        boto_sess = boto3.session.Session(profile_name=args.profile, region_name=region)
        sagemaker_session = sagemaker.LocalSession(boto_session=boto_sess)
        sagemaker_session.config = {"local": {"local_code": True}}
    else:
        sagemaker_session = create_session(args.profile, region)
    role = get_sagemaker_role_arn(region)
    print(f"SageMaker Execution Role: {role}")

    base_job_name = f"{args.user.replace('.', '-')}-{NAME}"
    job_name = get_job_name(base_job_name)
    # rstrip the trailing slash: S3_REMOTE_SYNC often ends in '/', and without this
    # the f-string yields a doubled slash (s3://bucket//sagemaker/...), which S3
    # preserves as a literal empty key segment -- artifacts then land under a path
    # that doesn't match the clean one you'd eyeball in the console.
    output_root = f"{args.s3_remote_sync.rstrip('/')}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    hyperparameters = {
        "base-model-path": args.base_model_path,
        "embodiment-tag": args.embodiment_tag,
        "modality-config-path": args.modality_config_path,
        "letter-box-transform": args.letter_box_transform,
        "dataset-subdir": args.dataset_subdir,
        "max-steps": args.max_steps,
        "save-steps": args.save_steps,
        "global-batch-size": args.global_batch_size,
        "episode-sampling-rate": args.episode_sampling_rate,
        "use-wandb": args.use_wandb,
        "dataloader-num-workers": args.dataloader_num_workers,
    }

    # Secrets/config forwarded into the container. HF_TOKEN is required (gated
    # backbone); WANDB_API_KEY is forwarded only if set locally -- without it,
    # --use-wandb 1 would stall/fail at wandb.init on the training instance.
    container_env = {
        "HF_TOKEN": hf_token,
        "SM_USE_RESERVED_CAPACITY": "1",
        "NCCL_DEBUG": "INFO",
    }
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        container_env["WANDB_API_KEY"] = wandb_key
    elif args.use_wandb not in ("0", "false", "False", ""):
        print(
            "WARNING: --use-wandb is on but WANDB_API_KEY is not set in your shell; "
            "the job may stall at wandb login. Export WANDB_API_KEY or set --use-wandb 0."
        )

    estimator = PyTorch(
        entry_point="sagemaker_entry.py",
        source_dir=str(Path(__file__).parent),
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image,
        instance_count=1,  # single-node; finetune.sh owns the torchrun fan-out
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        output_path=output_s3,
        checkpoint_s3_uri=None if args.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if args.local else "/opt/ml/checkpoints",
        code_location=output_s3,
        max_run=args.max_run,
        # File (not FastFile) on cloud too: GR00T's dataset init writes stats back
        # into <dataset>/meta/ (generate_stats + the unconditional generate_rel_stats
        # dump). FastFile mounts the channel read-only, so those writes fail with
        # "OSError: [Errno 30] Read-only file system". File mode stages the dataset
        # (only ~1.2 GB here) onto the writable instance volume. See README gotchas.
        input_mode="File",
        environment=container_env,
        # Warm pools don't exist in local mode.
        keep_alive_period_in_seconds=None if args.local else 60,
        tags=[
            {"Key": "tri.project", "Value": "LBM:PJ-0109"},
            {"Key": "tri.owner.email", "Value": "claire.yang@tri.global"},
        ],
    )

    inputs = {"training": args.s3_dataset}

    # Direct submission to SageMaker (default). No Batch queue involved -- this is
    # the right path for account 124224456861, which has no SAGEMAKER_TRAINING queue.
    # --local also uses fit() (SageMaker local_gpu mode).
    if args.local or args.submit_mode == "fit":
        print(f"Submitting SageMaker training job directly (estimator.fit): {job_name}")
        estimator.fit(inputs=inputs, job_name=job_name, wait=args.wait)
        if not args.wait:
            print(
                f"Submitted. Monitor in the SageMaker console (Training jobs) or:\n"
                f"  aws sagemaker describe-training-job --training-job-name {job_name} "
                f"--region {region} --query TrainingJobStatus --output text"
            )
        return

    # Queue mode: requires an AWS Batch SAGEMAKER_TRAINING service-environment queue.
    from sagemaker.aws_batch.training_queue import TrainingQueue as Queue

    queue_name = args.queue_name or f"fss-{INSTANCE_MAPPER[args.instance_type]}-{region}".replace(
        ".", "-"
    )
    queue = Queue(queue_name)
    print(f"Starting training job on queue {queue.queue_name}")
    queued_jobs = queue.map(
        estimator,
        inputs=[inputs],
        job_names=[job_name],
        priority=args.priority,
        share_identifier=args.fss_identifier,
        timeout={"attemptDurationSeconds": args.max_run},
    )
    print(f"Queued jobs: {queued_jobs}")


if __name__ == "__main__":
    main()
