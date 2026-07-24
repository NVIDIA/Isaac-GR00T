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
import getpass
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import traceback

import ray
from ray.train._internal import utils as _ray_train_utils
from ray.train.torch import train_loop_utils
from ray.util import scheduling_strategies
import torch
import tyro

from groot.core.data.dataset import deltacat_sampled_stats
from groot.core.data.dataset.build_artifacts import DatasetBuildArtifacts
from groot.core.training.configs import base_config
from groot.core.training.experiment import experiment, memory_monitor
from groot.core.training.experiment.gear_registration import (
    preserve_checkpoint_root_enabled,
    resolve_training_output_dir,
)
from groot.core.utils.storage import cache as cache_entrypoint

_DRY_RUN_PROPAGATED_VARS = ("_GROOT_DRY_RUN_STUB_MODEL", "IDENTITY_FORWARD_SLEEP_MS")
_TORCH_COMPILE_PROPAGATED_VARS = (
    "GLOBAL_DISABLE_TTT_COMPILE",
    "GROOT_N2_COMPILE_DIAGNOSTICS_INTERVAL",
    "GROOT_N2_COMPILE_DIAGNOSTICS_STEPS",
    "GROOT_TTT_COMPILE_TRANSFORMER_BLOCKS",
    "GROOT_TTT_INNER_COMPILE_MODE",
    "GROOT_TTT_TRANSFORMER_COMPILE_MODE",
    "TORCHDYNAMO_VERBOSE",
    "TORCHINDUCTOR_CUDAGRAPH_TREES",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "TORCH_LOGS",
)
_DELTACAT_PROPAGATED_VARS = (
    "DELTACAT_BETA_SERVER_URL",
    "DELTACAT_SERVER_URL",
    "DELTACAT_USER_ID",
    "DELTACAT_TOKEN_ID",
    "DELTACAT_BEARER_TOKEN",
    "DELTACAT_WORKER_TOKEN",
    "DELTACAT_CLIENT_TIMEOUT",
    "DELTACAT_PACKDS_DECODE_WORKERS",
    "DELTACAT_PACKDS_MAX_BLOB_CACHE_BYTES",
    "DELTACAT_PACKDS_MAX_DECODED_CACHE_BYTES",
)
_PROPAGATED_WORKER_ENV_VARS = (
    *_DRY_RUN_PROPAGATED_VARS,
    *_TORCH_COMPILE_PROPAGATED_VARS,
    *_DELTACAT_PROPAGATED_VARS,
    "GROOT_DISTRIBUTED_TIMEOUT_MINUTES",
    "TRAINING_OUTPUT_DIR",
    "TRAINING_MSC_CHECKPOINT_URL",
    "GEAR_PRESERVE_CHECKPOINT_ROOT",
)


def _collect_worker_env_vars() -> dict[str, str]:
    return {key: os.environ[key] for key in _PROPAGATED_WORKER_ENV_VARS if os.environ.get(key)}


def _build_ray_init_runtime_env() -> dict | None:
    env_vars = _collect_worker_env_vars()
    runtime_env = {"env_vars": env_vars} if env_vars else {}
    runtime_pip = os.environ.get("GROOT_RAY_RUNTIME_PIP", "")
    if runtime_pip:
        runtime_env["pip"] = shlex.split(runtime_pip)
    return runtime_env or None


def train_func(cfg, dataset_build_artifacts=None):
    # Disable local autotune cache to avoid corrupted .best_config JSON from
    # multi-GPU write races. Kernels will re-autotune each run (one-time cost).
    import torch._inductor.config as _ic

    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    _ic.autotune_local_cache = False
    _ic.autotune_remote_cache = False
    logging.info(
        "TorchInductor env: TORCHINDUCTOR_FORCE_DISABLE_CACHES=%s TORCHINDUCTOR_CUDAGRAPH_TREES=%s",
        os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "<unset>"),
        os.environ.get("TORCHINDUCTOR_CUDAGRAPH_TREES", "<unset>"),
    )
    if not _env_flag("TORCHINDUCTOR_CUDAGRAPH_TREES", True):
        # PyTorch does not apply TORCHINDUCTOR_CUDAGRAPH_TREES as an env-driven
        # config override in this environment, so wire the probe knob directly.
        _ic.triton.cudagraph_trees = False
        logging.info("Disabled TorchInductor cudagraph trees via TORCHINDUCTOR_CUDAGRAPH_TREES=0")

    experiment.run(
        cfg,
        dataset_build_artifacts=dataset_build_artifacts,
    )


class TorchTrainer:
    def __init__(self, placement_group) -> None:
        # create placement group
        self.placement_group = placement_group
        self.workers = []
        self.training_fn = None
        self.training_fn_cfg = None
        self.dataset_build_artifacts = None
        self.num_workers = None
        # Store the initial number of GPUs in the cluster
        self.initial_cluster_gpus = ray.cluster_resources().get("GPU", 0)
        self.gpus_lost = 0
        logging.info(f"Initial cluster GPUs: {self.initial_cluster_gpus}")

    def setup_workers(
        self,
        training_fn,
        training_fn_cfg,
        dataset_build_artifacts=None,
    ):
        self.training_fn = training_fn
        self.training_fn_cfg = training_fn_cfg
        self.dataset_build_artifacts = dataset_build_artifacts
        self.workers = []

        # Check if number of GPUs in the cluster has decreased
        current_cluster_gpus = ray.cluster_resources().get("GPU", 0)
        if current_cluster_gpus < self.initial_cluster_gpus:
            self.gpus_lost = self.initial_cluster_gpus - current_cluster_gpus
            logging.warning(
                f"Cluster GPUs decreased from {self.initial_cluster_gpus} to "
                f"{current_cluster_gpus}. GPUs lost: {self.gpus_lost}"
            )

        # Only create as many workers as (bundle_specs - gpus_lost)
        num_workers_to_create = max(1, len(self.placement_group.bundle_specs) - self.gpus_lost)
        logging.info(
            f"Creating {num_workers_to_create} workers out of {len(self.placement_group.bundle_specs)} bundles"
        )

        # Only use the first num_workers_to_create bundles
        for i, bundle in enumerate(self.placement_group.bundle_specs):
            if i >= num_workers_to_create:
                break

            self.workers.append(
                _TrainingWorker.options(
                    num_cpus=bundle.get("CPU", 0),
                    num_gpus=bundle.get("GPU", 0),  # CPU-only bundles omit GPU
                    scheduling_strategy=scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group
                    ),
                ).remote(
                    training_fn,
                    training_fn_cfg,
                    dataset_build_artifacts,
                )
            )

        if self.workers:
            head_node_addr, port = ray.get(self.workers[0].get_address.remote())
            ray.get(
                [
                    worker.setup_env_vars.remote(head_node_addr, port, i, len(self.workers))
                    for i, worker in enumerate(self.workers)
                ]
            )

    def _cleanup_workers(self):
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []

    def train(
        self,
        training_fn,
        training_fn_cfg,
        *,
        dataset_build_artifacts=None,
    ):
        self.setup_workers(
            training_fn,
            training_fn_cfg,
            dataset_build_artifacts,
        )
        retries = 0
        max_retries = max(self.training_fn_cfg.training.max_retries, 1)
        while retries < max_retries:
            try:
                ray.get([worker.train.remote() for worker in self.workers])
                break  # If successful, exit the loop
            except ray.exceptions.ActorDiedError as e:
                assert "Worker exits with an exit code 0" in str(
                    e
                ), f"Actor died with non-zero exit code: {e}"
                logging.info("Training already finished")
                break
            except Exception as e:
                retries += 1
                # 1. Dump error to file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                error_path = os.path.join(
                    self.training_fn_cfg.training.output_dir.format(user=getpass.getuser()),
                    f"error_{timestamp}.txt",
                )
                os.makedirs(os.path.dirname(error_path), exist_ok=True)
                with open(error_path, "w") as f:
                    f.write(f"Error occurred during training:\n{e!s}\n")
                    f.write("\nTraceback:\n")
                    f.write(traceback.format_exc())
                logging.exception("Error occurred during training")

                # Append a worker_retry status event so the dashboard
                # timeline distinguishes same-run retries from a fresh
                # launcher submission. set_status=False because we do NOT
                # want the scalar status to flip during transient retries
                # -- it stays "running". Last retry that exhausts
                # max_retries falls through to the raise below, where
                # rank-0's training_failed event (emitted by
                # experiment.py) records the terminal transition.
                self._append_retry_event(
                    retries=retries,
                    max_retries=max_retries,
                    error_path=error_path,
                    exc=e,
                )

                # 2. Tear down workers
                self._cleanup_workers()

                # 3. Setup new workers with GPU loss detection built in
                self.setup_workers(
                    self.training_fn,
                    self.training_fn_cfg,
                    self.dataset_build_artifacts,
                )

                # 4. Continue loop to retry training
                logging.warning(
                    f"Training failed, error logged to {error_path}. Retrying with new workers..."
                )
        if retries >= max_retries:
            raise RuntimeError("Training failed")

    @staticmethod
    def _append_retry_event(
        *, retries: int, max_retries: int, error_path: str, exc: BaseException
    ) -> None:
        """Best-effort dashboard event for a same-run worker retry.

        Imported lazily so a missing/unreachable dashboard module never
        breaks the retry loop -- the file-on-Lustre error log is still the
        durable record of the failure.
        """
        try:
            from groot.core.training.experiment.gear_registration import (
                append_training_run_status_event,
            )

            experiment_id = os.environ.get("GEAR_EXPERIMENT_ID", "")
            training_run_id = os.environ.get("GEAR_TRAINING_RUN_ID", "")
            if not experiment_id or not training_run_id:
                return
            append_training_run_status_event(
                experiment_id,
                training_run_id,
                status="running",
                message=f"worker_retry {retries}/{max_retries}: {type(exc).__name__}",
                metadata={
                    "event": "worker_retry",
                    "retry": retries,
                    "max_retries": max_retries,
                    "error_path": error_path,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)[:512],
                },
                set_status=False,
            )
        except Exception as patch_exc:  # noqa: BLE001
            logging.warning("Dashboard worker_retry event append failed (non-fatal): %s", patch_exc)

    def shutdown(self):
        ray.util.remove_placement_group(self.placement_group)


@ray.remote
class _TrainingWorker:
    def __init__(
        self,
        training_fn,
        training_fn_cfg,
        dataset_build_artifacts=None,
    ) -> None:
        self.training_fn = training_fn
        self.training_fn_cfg = training_fn_cfg
        self.dataset_build_artifacts = dataset_build_artifacts

    def get_address(self):
        return _ray_train_utils.get_address_and_port()

    def setup_env_vars(self, addr, port, world_rank, world_size):
        # HERE: Set all important environment variables in this function
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(world_rank)
        os.environ["LOCAL_RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(world_size)

        device = train_loop_utils.get_device()
        os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)

    def train(self):
        try:
            # Clear stale TorchInductor cache to avoid corrupted autotune entries
            import shutil

            cache_dir = "/tmp/torchinductor_root/"  # noqa: S108
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                logging.info(f"Cleared TorchInductor cache at {cache_dir}")

            device = train_loop_utils.get_device()
            # Guard for CPU-only runs (launch_identity_cpu_dry_run /
            # ``--training.num_gpus 0``). torch.cuda.set_device crashes
            # when CUDA isn't available.
            if torch.cuda.is_available():
                torch.cuda.set_device(device)
            self.training_fn(
                self.training_fn_cfg,
                self.dataset_build_artifacts,
            )
        except Exception as e:
            if "ECC" in str(e) or "CUDA driver initialization failed" in str(e):
                # Force stop Ray if we encounter GPU errors
                logging.exception("GPU error during training")
                subprocess.run(["ray", "stop", "--force"], check=False)
            elif "trainer_state.json" in str(e):
                # Parse the path from the exception message
                error_msg = str(e)
                path_match = re.search(r"'([^']*trainer_state\.json)'", error_msg)
                if path_match:
                    path = path_match.group(1)
                else:
                    raise

                ckpt_dir = os.path.dirname(path)
                # delete the parent folder of trainer_state.json
                subprocess.run(["rm", "-rf", ckpt_dir], check=False)
                raise RuntimeError(
                    f"Cannot find trainer_state.json in {ckpt_dir}, deleting it and re-running"
                ) from e
            else:
                raise


def main():
    # Use tyro for clean CLI
    config = tyro.cli(
        base_config.Config, default=base_config.get_default_config(), description=__doc__
    )

    # Load config from path if provided
    if config.load_config_path:
        assert Path(
            config.load_config_path
        ).exists(), f"Config path does not exist: {config.load_config_path}"
        config = config.load(Path(config.load_config_path))
        config.load_config_path = None
        logging.info(f"Loaded config from {config.load_config_path}")

        # Override with any command-line options.
        config = tyro.cli(base_config.Config, default=config, description=__doc__)

    preserve_checkpoint_root = preserve_checkpoint_root_enabled()
    api_checkpoint_path = os.environ.get("GEAR_CHECKPOINT_PATH", "")
    api_msc_checkpoint_url = (
        ""
        if preserve_checkpoint_root
        else experiment.dashboard_checkpoint_path_to_msc_url(api_checkpoint_path)
    )
    if config.training.msc_checkpoint_url and api_msc_checkpoint_url:
        logging.info(
            "Overriding training.msc_checkpoint_url on driver from %s to dashboard-allocated %s",
            config.training.msc_checkpoint_url,
            api_msc_checkpoint_url,
        )
        config.training.msc_checkpoint_url = api_msc_checkpoint_url

    resolved_output_dir = resolve_training_output_dir(config.training.output_dir)
    if resolved_output_dir != config.training.output_dir:
        logging.info(
            "Resolved canonical training.output_dir on driver from %s to %s",
            config.training.output_dir,
            resolved_output_dir,
        )
        config.training.output_dir = resolved_output_dir
    if preserve_checkpoint_root:
        requested_msc_checkpoint_url = os.environ.get("TRAINING_MSC_CHECKPOINT_URL", "")
        if (
            requested_msc_checkpoint_url
            and config.training.msc_checkpoint_url != requested_msc_checkpoint_url
        ):
            logging.info(
                "Preserving launcher training.msc_checkpoint_url on driver: %s",
                requested_msc_checkpoint_url,
            )
            config.training.msc_checkpoint_url = requested_msc_checkpoint_url

    # Attach to an existing Ray cluster when RAY_ADDRESS is set (the
    # multi-node CPU path; SkyPilot's start_cluster exports this env
    # var on every pod after the head finishes bootstrap). Otherwise
    # ray.init() with no address starts a fresh local single-pod
    # cluster — the single-node CPU and the GPU paths.
    #
    # Propagate launch-time knobs into Ray workers via runtime_env.env_vars.
    # When RAY_ADDRESS is set we attach to an existing user-Ray whose worker
    # processes were spawned by the cluster's Ray daemon with the daemon's env;
    # the launcher's os.environ from sky exec does not reach them otherwise.
    #
    # _GROOT_DRY_RUN_STUB_MODEL: boolean flag (any truthy ⇒ "1") read in
    #   experiment.run to shadow pipeline._create_model with IdentityModel
    #   before pipeline.setup. (Not wired as a worker_process_setup_hook —
    #   that hook fires before Ray binds CUDA_VISIBLE_DEVICES for the
    #   actor, so importing the model package early can pin CUDA to all
    #   visible GPUs, breaking per-rank device assignment.)
    # IDENTITY_FORWARD_SLEEP_MS: read by IdentityModel.__init__ to inject
    #   forward-pass latency in the dry-run; without this propagation the
    #   sleep silently no-ops on the GPU (RAY_ADDRESS-attach) path.
    # These variables must be present before the actor imports TTT modules,
    # because compile settings are read at import time. True Ray-job
    # submissions may already contain the same keys in their job runtime_env;
    # retry without an explicit runtime_env when Ray reports that duplicate.
    runtime_env = _build_ray_init_runtime_env()
    try:
        ray.init(address=os.environ.get("RAY_ADDRESS"), runtime_env=runtime_env)
    except ValueError as exc:
        message = str(exc)
        if runtime_env and "runtime_env" in message and "RAY_OVERRIDE_JOB_RUNTIME_ENV" in message:
            logging.warning(
                "ray.init rejected duplicate runtime_env keys; assuming the Ray job "
                "runtime_env already contains launch env and retrying without explicit runtime_env."
            )
            ray.init(address=os.environ.get("RAY_ADDRESS"))
        else:
            raise
    memory_monitor.start()
    if config.data.download_cache:
        cache_entrypoint.maybe_download_cache_contents(config)
    config.data.materialize_deltacat_as_of_timestamps()
    dataset_build_artifacts = None
    if config.data.has_deltacat_sources():
        memory_monitor.phase("deltacat-stats-preflight")
        dataset_build_artifacts = DatasetBuildArtifacts(
            deltacat_stats_by_embodiment=deltacat_sampled_stats.compute_from_config(
                config.data,
                model_config=config.model,
                allow_ray=True,
            )
        )
        memory_monitor.phase("post-stats-pre-train")
    num_gpus = config.training.num_gpus

    # Get the maximum available GPUs
    available_gpus = ray.cluster_resources().get("GPU", 0)

    if num_gpus < 0:
        raise ValueError(f"num_gpus must be >= 0, got {num_gpus}")
    cpu_only = num_gpus == 0

    if not cpu_only and available_gpus > 0 and num_gpus > available_gpus:
        # Cap num_gpus at the maximum available GPUs
        logging.warning(
            f"Requested {num_gpus} GPUs but only {available_gpus} available. Capping at {available_gpus}."
        )
        num_gpus = int(available_gpus)

    # Create placement group and wait indefinitely for resources (allows job queuing).
    if cpu_only:
        # CPU-only run (launch_identity_cpu_dry_run / --training.num_gpus 0).
        # One bundle per CPU worker; bundles omit "GPU" so Ray doesn't try
        # to claim GPU resources from the pool.
        #
        # Placement strategy selection (CPU_PLACEMENT_STRATEGY env, default
        # SPREAD):
        #   SPREAD       — best-effort even spread across distinct nodes;
        #       Ray fills nodes round-robin when bundles > nodes. Default,
        #       robust regardless of per-node CPU budget.
        #   STRICT_SPREAD — exactly one bundle per node; requires
        #       num_cpu_workers ≤ num_nodes. Use when you need a hard
        #       guarantee of 1-rank-per-pod.
        #   PACK         — minimize node count. AVOID for multinode CPU
        #       dry-runs: when per-bundle CPU fits one node (e.g. 24 × 5 =
        #       120 < 128 on 32xlarge), PACK crams all bundles onto the head
        #       and OOMs it.
        #   "auto"       — STRICT_SPREAD when num_cpu_workers ≤
        #       SKYPILOT_NUM_NODES, else SPREAD.
        num_cpu_workers = config.training.num_cpu_workers
        if num_cpu_workers < 1:
            raise ValueError(f"num_cpu_workers must be >= 1, got {num_cpu_workers}")
        strategy = os.environ.get("CPU_PLACEMENT_STRATEGY", "SPREAD").upper()
        if strategy == "AUTO":
            try:
                num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))
            except ValueError:
                num_nodes = 1
            strategy = "STRICT_SPREAD" if num_cpu_workers <= max(num_nodes, 1) else "SPREAD"
        if num_cpu_workers == 1:
            strategy = "PACK"  # 1-bundle case: SPREAD/STRICT_SPREAD are nonsense
        placement_group = ray.util.placement_group(
            [{"CPU": config.training.dataloader_num_workers + 1} for _ in range(num_cpu_workers)],
            strategy=strategy,
        )
        logging.info(
            f"Waiting for CPU-only placement group ({num_cpu_workers} worker(s), strategy={strategy}, no GPUs)..."
        )
    else:
        placement_group = ray.util.placement_group(
            [{"CPU": config.training.dataloader_num_workers, "GPU": 1} for _ in range(num_gpus)]
        )
        logging.info(f"Waiting for placement group with {num_gpus} GPUs...")
    trainer = TorchTrainer(placement_group)
    memory_monitor.phase("training")
    trainer.train(
        train_func,
        config,
        dataset_build_artifacts=dataset_build_artifacts,
    )
    trainer.shutdown()


if __name__ == "__main__":
    main()
