# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: T201
"""Shared golden-run training submit helper."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from typing import NamedTuple

import yaml

from groot.infra.ci.golden_run_config import CONFIG_BACKED_LAUNCHER, get_variant

CONFIG_PATH_ENV_VAR = "GOLDEN_RUN_CONFIG_PATH"
CHECKPOINT_PATH_TEMPLATE_ENV_VAR = "GOLDEN_RUN_CHECKPOINT_PATH_TEMPLATE"
_DEFAULT_CHECKPOINT_PATH_TEMPLATE = "s3://{user_slug}_{job_name}/checkpoint-10000"
_SUBMIT_RESULT_FILENAME = "dashboard_submit_result.json"
_DELTACAT_ENV_NAMES = (
    "DELTACAT_BETA_SERVER_URL",
    "DELTACAT_SERVER_URL",
    "DELTACAT_USER_ID",
    "DELTACAT_TOKEN_ID",
    "DELTACAT_BEARER_TOKEN",
    "DELTACAT_WORKER_TOKEN",
    "DELTACAT_PACKDS_DECODE_WORKERS",
    "DELTACAT_PACKDS_EPISODE_ROWS_PREFETCH_BATCH_SIZE",
    "DELTACAT_PACKDS_MAX_BLOB_CACHE_BYTES",
    "DELTACAT_PACKDS_MAX_DECODED_CACHE_BYTES",
    "DELTACAT_PACKDS_S3_MAX_POOL_CONNECTIONS",
    "DATALOADER_NUM_WORKERS",
    "GRADIENT_ACCUMULATION_STEPS",
    "NCCL_TIMEOUT",
)


class _ExperimentSubmitProxy:
    def __getattr__(self, name: str):
        from groot.core.training.experiment import submit as module

        return getattr(module, name)


experiment_submit = _ExperimentSubmitProxy()


class _ExperimentConfigFiles(NamedTuple):
    data_config: Path
    training_config: Path
    env_vars: dict[str, str]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    print_env = subparsers.add_parser("print-env")
    _add_variant_args(print_env)

    submit = subparsers.add_parser("submit")
    _add_variant_args(submit)
    submit.add_argument("--cluster-id", required=True)
    submit.add_argument("--ckpts-path-output", type=Path, required=True)
    submit.add_argument("--log-file", type=Path, required=True)
    submit.add_argument("--submission-id-output", type=Path)
    submit.add_argument("--status-poll-interval", type=float, default=60.0)
    submit.add_argument("--status-error-limit", type=int, default=10)
    submit.add_argument("--submission-id-wait-attempts", type=int, default=10)

    args = parser.parse_args()
    if args.command == "print-env":
        _print_env(args.run_type, args.compute_type)
    elif args.command == "submit":
        _submit(args)


def _add_variant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-type", required=True)
    parser.add_argument("--compute-type", required=True)


def _print_env(run_type: str, compute_type: str) -> None:
    variant = _get_variant(run_type)
    platform = variant.platform(compute_type)
    values = {
        "GOLDEN_RUN_VARIANT_LABEL": variant.label,
        "GOLDEN_RUN_LAUNCHER": variant.launcher,
        "NUM_GPU": str(platform.num_gpus),
        "NUM_GPUS": str(platform.num_gpus),
        "NUM_NODES": str(platform.num_nodes),
        "BATCH_SIZE": str(platform.batch_size),
        "GRADIENT_ACCUMULATION_STEPS": str(platform.gradient_accumulation_steps),
    }
    if platform.pool:
        values["OSMO_POOL"] = platform.pool
    if variant.experiment:
        values["GOLDEN_RUN_EXPERIMENT"] = variant.experiment
    if variant.artifacts is not None:
        values.update(
            {
                "GOLDEN_RUN_BASE_MODEL_REPO_ID": variant.artifacts.base_model.repo_id,
                "GOLDEN_RUN_BASE_MODEL_REVISION": variant.artifacts.base_model.revision,
                "GOLDEN_RUN_DATASET_REPO_ID": variant.artifacts.dataset.repo_id,
                "GOLDEN_RUN_DATASET_REVISION": variant.artifacts.dataset.revision,
                "GOLDEN_RUN_DATASET_LOCAL_NAME": variant.artifacts.dataset.local_name,
                "GOLDEN_RUN_FINAL_MODEL_REPO_ID": variant.artifacts.final_model.repo_id,
                "GOLDEN_RUN_FINAL_MODEL_REVISION": variant.artifacts.final_model.revision,
            }
        )
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")


def _submit(args: argparse.Namespace) -> None:
    variant = _get_variant(args.run_type)
    platform = variant.platform(args.compute_type)
    job_name = os.environ.get("JOB_NAME") or _default_job_name(args.run_type, args.compute_type)

    env = os.environ.copy()
    env.update(
        {
            "JOB_NAME": job_name,
            "NUM_GPUS": str(platform.num_gpus),
            "BATCH_SIZE": str(platform.batch_size),
            "GRADIENT_ACCUMULATION_STEPS": str(platform.gradient_accumulation_steps),
            "CLUSTER_TYPE": args.compute_type,
        }
    )
    submit_result_path = args.ckpts_path_output.with_name(_SUBMIT_RESULT_FILENAME)
    submit_result_path.unlink(missing_ok=True)
    env["GEAR_SUBMIT_RESULT_PATH"] = str(submit_result_path)

    print(f"Submitting golden run {args.run_type!r} on {args.compute_type}: {variant.label}")
    print(f"Launcher: {variant.launcher}")
    print(f"Ray job filter: {job_name}")
    if variant.launcher == CONFIG_BACKED_LAUNCHER:
        if variant.experiment:
            _submit_experiment_backed_variant(
                args=args,
                variant=variant,
                env=env,
                job_name=job_name,
            )
        else:
            _submit_config_backed_variant(
                args=args,
                variant=variant,
                env=env,
                job_name=job_name,
            )
    else:
        launcher = Path(variant.launcher)
        cmd = [
            sys.executable,
            str(launcher),
            "submit",
            "--cluster-id",
            args.cluster_id,
        ]
        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            _print_subprocess_output(exc.stdout or exc.output)
            raise
        _print_subprocess_output(result.stdout)
    checkpoint_step = variant.eval.checkpoint_step if variant.eval is not None else 10000
    checkpoint_root = _read_dashboard_checkpoint_root(submit_result_path)
    if checkpoint_root:
        _write_checkpoint_path(
            output=args.ckpts_path_output,
            checkpoint_root=checkpoint_root,
            checkpoint_step=checkpoint_step,
        )
    else:
        print(
            "Dashboard submit result did not include checkpoint root; falling back to legacy CI path."
        )
        _write_checkpoint_path(
            output=args.ckpts_path_output,
            job_name=job_name,
            checkpoint_step=checkpoint_step,
        )

    submission_id = _recover_submission_id(
        cluster_id=args.cluster_id,
        job_name=job_name,
        attempts=args.submission_id_wait_attempts,
    )
    if args.submission_id_output is not None:
        args.submission_id_output.write_text(f"{submission_id}\n")
    print(f"Ray submission ID: {submission_id}")

    _tail_logs_and_poll_terminal_status(
        args.cluster_id,
        submission_id,
        args.log_file,
        args.status_poll_interval,
        getattr(args, "status_error_limit", 10),
    )


def _submit_config_backed_variant(
    *, args: argparse.Namespace, variant, env: dict, job_name: str
) -> None:
    if not variant.data_config or not variant.training_config:
        raise ValueError("config-backed golden run requires data_config and training_config")
    _submit_launch_trainer_variant(
        args=args,
        variant=variant,
        env=env,
        job_name=job_name,
        data_config=variant.data_config,
        training_config=variant.training_config,
        launch_args=[
            "run",
            "--data-config",
            variant.data_config,
            "--training-config",
            variant.training_config,
        ],
    )


def _submit_experiment_backed_variant(
    *,
    args: argparse.Namespace,
    variant,
    env: dict,
    job_name: str,
) -> None:
    if not variant.experiment:
        raise ValueError("experiment-backed golden run requires experiment")

    with tempfile.TemporaryDirectory(prefix="golden-run-experiment-") as output_root:
        files = _resolve_experiment_configs(
            experiment=variant.experiment,
            compute_type=args.compute_type,
            output_root=Path(output_root),
        )
        for key, value in files.env_vars.items():
            env.setdefault(key, value)
        launch_args = [
            "run",
            "--experiment",
            variant.experiment,
            "--compute-type",
            args.compute_type,
        ]
        _submit_launch_trainer_variant(
            args=args,
            variant=variant,
            env=env,
            job_name=job_name,
            data_config=str(files.data_config),
            training_config=str(files.training_config),
            launch_args=launch_args,
        )


def _resolve_experiment_configs(
    *, experiment: str, compute_type: str, output_root: Path
) -> _ExperimentConfigFiles:
    _ = output_root
    raise RuntimeError(
        "experiment-backed golden runs require the internal golden-run submitter; "
        f"cannot resolve experiment {experiment!r} for {compute_type!r} from the public OSS package"
    )


def _submit_launch_trainer_variant(
    *,
    args: argparse.Namespace,
    variant,
    env: dict,
    job_name: str,
    data_config: str,
    training_config: str,
    launch_args: list[str],
) -> None:
    entrypoint = shlex.join(
        ["python", "-m", "groot.internal.infra.job.main.launch_trainer", *launch_args]
    )
    start_from_checkpoint = _training_config_value(
        Path(training_config),
        "training",
        "start_from_checkpoint",
        env,
    )
    if start_from_checkpoint:
        start_from_checkpoint = _resolve_checkpoint_path(
            start_from_checkpoint,
            args.compute_type,
        )
    config_overrides = ["--load_config_path", data_config]
    if start_from_checkpoint:
        config_overrides.extend(["--training.start_from_checkpoint", start_from_checkpoint])
    dataset_root = _config_backed_data_root(args.compute_type)
    if variant.uses_deltacat:
        env.update(_deltacat_env_from_os_environ())
        env.setdefault("MALLOC_ARENA_MAX", os.environ.get("MALLOC_ARENA_MAX", "2"))
        env.setdefault("MALLOC_TRIM_THRESHOLD_", os.environ.get("MALLOC_TRIM_THRESHOLD_", "0"))
    pip_packages = experiment_submit.get_pip_packages(
        "n2",
        args.compute_type,
        extra_packages=_deltacat_runtime_pip_packages() if variant.uses_deltacat else (),
        **({"transformers": "4.57.6"} if variant.run_type.startswith("n2-minwam") else {}),
    )
    with _temporary_environ(env):
        experiment_submit.submit_to_cluster(
            cluster_id=args.cluster_id,
            job_name=job_name,
            entrypoint=entrypoint,
            entrypoint_prefix="pip uninstall -y apex 2>/dev/null; ",
            env_vars=env,
            extra_env={
                "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
                "TORCHINDUCTOR_CUDAGRAPH_TREES": "0",
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor_clean",  # noqa: S108
            },
            pip_packages=pip_packages,
            start_from_checkpoint=start_from_checkpoint,
            datasets=experiment_submit._datasets_from_config(  # noqa: SLF001
                data_config, dataset_root
            ),
            lfs_prefix=experiment_submit.get_cluster_paths(args.compute_type)["checkpoints"],
            data_config_path=data_config,
            config_overrides=config_overrides,
            wandb_project=env.get("WANDB_PROJECT", "groot_n2_posttrain"),
        )


def _deltacat_env_from_os_environ() -> dict[str, str]:
    return {name: os.environ[name] for name in _DELTACAT_ENV_NAMES if os.environ.get(name)}


def _deltacat_runtime_pip_packages() -> tuple[str, ...]:
    return (
        "msgpack==1.0.7",
        *experiment_submit.get_project_extra_packages("deltacat"),
    )


def _config_backed_data_root(compute_type: str) -> str:
    dataset_root = Path(experiment_submit.get_cluster_paths(compute_type)["datasets"])
    if dataset_root.name == "datasets":
        return str(dataset_root.parent)
    return str(dataset_root)


def _training_config_value(
    training_config: Path,
    section: str,
    key: str,
    env: dict[str, str],
) -> str | None:
    raw = yaml.safe_load(training_config.read_text()) or {}
    if not isinstance(raw, dict):
        return None
    section_body = raw.get(section)
    if not isinstance(section_body, dict):
        return None
    value = section_body.get(key)
    if value is None:
        return None
    return _expand_placeholders(str(value), env)


def _expand_placeholders(value: str, env: dict[str, str]) -> str:
    cluster_type = env.get("CLUSTER_TYPE", "h100")
    paths = experiment_submit.get_cluster_paths(cluster_type)
    values = {
        "JOB_NAME": env.get("JOB_NAME", ""),
        "USER_SLUG": env.get("USER_SLUG", ""),
        "BATCH_SIZE": env.get("BATCH_SIZE", ""),
        "NUM_GPUS": env.get("NUM_GPUS", ""),
        **{f"PATHS.{name}": path for name, path in paths.items()},
    }
    expanded = value
    for key, replacement in values.items():
        expanded = expanded.replace("{" + key + "}", replacement)
    return expanded


def _resolve_checkpoint_path(value: str, compute_type: str) -> str:
    if "://" in value or Path(value).is_absolute():
        return value
    return str(Path(experiment_submit.get_cluster_paths(compute_type)["checkpoints"]) / value)


@contextmanager
def _temporary_environ(env: dict[str, str]):
    old_values = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _recover_submission_id(cluster_id: str, job_name: str, attempts: int) -> str:
    pattern = re.compile(rf"{re.escape(job_name)}_[a-f0-9]{{6}}")
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--active",
                    "gear",
                    "ray",
                    "status",
                    cluster_id,
                    "--filter",
                    job_name,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result = None
            print("Ray status query timed out while waiting for submission registration.")
        match = None if result is None else pattern.search(result.stdout)
        if match:
            return match.group(0)
        print(f"[submission-id-wait {attempt}/{attempts}] not registered yet, sleeping 10s...")
        time.sleep(10)
    raise RuntimeError(f"could not recover Ray submission_id for {job_name}")


def _read_dashboard_checkpoint_root(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    checkpoint_root = payload.get("checkpoint_path")
    return checkpoint_root.strip() if isinstance(checkpoint_root, str) else ""


def _checkpoint_root_for_eval(checkpoint_root: str) -> str:
    """Convert a dashboard checkpoint root to the eval checkpoint path.

    Examples:
      checkpoints/project/00075_job
        -> s3://checkpoints/project/00075_job/checkpoint-10000
      msc://checkpoints/project/00075_job/
        -> s3://checkpoints/project/00075_job/checkpoint-10000
    """
    checkpoint_root = checkpoint_root.strip().rstrip("/")
    if checkpoint_root.startswith("msc://"):
        return f"s3://{checkpoint_root[len('msc://') :]}"
    if checkpoint_root.startswith("s3://"):
        return checkpoint_root
    if "://" in checkpoint_root:
        return checkpoint_root
    return f"s3://{checkpoint_root}"


def _write_checkpoint_path(
    output: Path,
    job_name: str | None = None,
    checkpoint_root: str | None = None,
    checkpoint_step: int = 10000,
) -> None:
    if checkpoint_root:
        output.write_text(
            f"{_checkpoint_root_for_eval(checkpoint_root)}/checkpoint-{checkpoint_step}\n"
        )
        return

    if job_name is None:
        raise ValueError("job_name must be set when dashboard checkpoint root is unavailable")
    user_slug = os.environ.get("USER_SLUG", "")
    if not user_slug:
        raise ValueError("USER_SLUG must be set to derive the golden-run checkpoint path")
    checkpoint_path = _checkpoint_path_template().format(user_slug=user_slug, job_name=job_name)
    # The legacy template baked /checkpoint-10000 into the path. Swap in the
    # variant's actual step so non-10k runs land on the right checkpoint.
    checkpoint_path = checkpoint_path.replace("/checkpoint-10000", f"/checkpoint-{checkpoint_step}")
    output.write_text(f"{checkpoint_path}\n")


def _create_ray_client(cluster_id: str):
    from gear.gear_ray import GearRayJobSubmissionClient

    return GearRayJobSubmissionClient(cluster_id)


def _tail_logs_and_poll_terminal_status(
    cluster_id: str,
    submission_id: str,
    log_file: Path,
    interval: float,
    status_error_limit: int = 10,
) -> None:
    if status_error_limit < 1:
        raise ValueError("status_error_limit must be at least 1")
    client = _create_ray_client(cluster_id)

    tail_thread = threading.Thread(
        target=_tail_logs_worker,
        args=(client.client, submission_id, log_file),
        daemon=True,
        name=f"ray-log-tail-{submission_id}",
    )
    tail_thread.start()
    try:
        _poll_terminal_status_with_client(
            client.client,
            submission_id,
            interval,
            status_error_limit=status_error_limit,
        )
    finally:
        tail_thread.join(timeout=2.0)


def _tail_logs_worker(client, submission_id: str, log_file: Path) -> None:
    try:
        _tail_logs_with_client(client, submission_id, log_file)
    except Exception as exc:  # noqa: BLE001
        print(f"Ray log tail failed: {exc!r}; polling Ray status until terminal.")


def _tail_logs_with_client(client, submission_id: str, log_file: Path) -> None:
    async def consume_logs() -> None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as out:
            async for log in client.tail_job_logs(submission_id):
                print(log, end="" if log.endswith("\n") else "\n")
                out.write(log)

    asyncio.run(consume_logs())
    print("Ray log tail exited; polling Ray status until terminal.")


def _poll_terminal_status_with_client(
    client,
    submission_id: str,
    interval: float,
    status_error_limit: int = 10,
) -> None:
    attempt = 1
    consecutive_errors = 0
    while True:
        try:
            status = client.get_job_status(submission_id)
        except Exception as exc:
            consecutive_errors += 1
            print(
                f"Ray status query failed ({consecutive_errors}/{status_error_limit}): "
                f"{exc!r}; treating as transient."
            )
            if consecutive_errors >= status_error_limit:
                raise RuntimeError(
                    f"Ray status query failed {consecutive_errors} consecutive times for "
                    f"{submission_id}; cluster may be gone or unreachable"
                ) from exc
        else:
            consecutive_errors = 0
            status_name = _job_status_name(status)
            print(f"{status_name} {submission_id}")
            if status_name == "SUCCEEDED":
                print(f"Ray job {submission_id} SUCCEEDED.")
                return
            if status_name in {"FAILED", "STOPPED"}:
                raise RuntimeError(f"Ray job {submission_id} reached a failed terminal state")

        print(f"[ray-status {attempt}] {submission_id} not terminal yet, sleeping {interval:g}s...")
        attempt += 1
        time.sleep(interval)


def _job_status_name(status) -> str:
    name = getattr(status, "name", None)
    return str(name or status)


def _tail_logs(cluster_id: str, submission_id: str, log_file: Path) -> None:
    cmd = ["uv", "run", "--active", "gear", "ray", "tail", cluster_id, submission_id]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as out:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            out.write(line)
        rc = proc.wait()
    if rc == 0:
        print("Ray log tail exited; polling Ray status until terminal.")
    else:
        print("Ray log tail exited non-zero; polling Ray status until terminal.")


def _print_subprocess_output(output: str | bytes | None) -> None:
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    print(output, end="" if output.endswith("\n") else "\n")


def _poll_terminal_status(cluster_id: str, submission_id: str, interval: float) -> None:
    attempt = 1
    while True:
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--active",
                    "gear",
                    "ray",
                    "status",
                    cluster_id,
                    "--filter",
                    submission_id,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result = None
            print("Ray status query timed out; treating as transient.")
        if result is not None:
            print(result.stdout)
        if result is not None and result.returncode == 0:
            if re.search(r"^SUCCEEDED ", result.stdout, re.MULTILINE):
                print(f"Ray job {submission_id} SUCCEEDED.")
                return
            if re.search(r"^(FAILED|STOPPED) ", result.stdout, re.MULTILINE):
                raise RuntimeError(f"Ray job {submission_id} reached a failed terminal state")
        else:
            print(
                "Ray status query failed; treating as transient while waiting for terminal status."
            )

        print(f"[ray-status {attempt}] {submission_id} not terminal yet, sleeping {interval:g}s...")
        attempt += 1
        time.sleep(interval)


def _default_job_name(run_type: str, compute_type: str) -> str:
    short_sha = os.environ.get("CI_COMMIT_SHORT_SHA", "local")
    safe_run_type = re.sub(r"[^a-zA-Z0-9]+", "_", run_type).strip("_")
    return f"golden_run_{safe_run_type}_{compute_type}_{short_sha}"


def _get_variant(run_type: str):
    config_path = os.getenv(CONFIG_PATH_ENV_VAR)
    return get_variant(run_type, Path(config_path) if config_path else None)


def _checkpoint_path_template() -> str:
    return os.getenv(CHECKPOINT_PATH_TEMPLATE_ENV_VAR, _DEFAULT_CHECKPOINT_PATH_TEMPLATE)


if __name__ == "__main__":
    main()
