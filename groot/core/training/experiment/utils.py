# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from concurrent.futures import ThreadPoolExecutor
import glob
import json
import logging
from pathlib import Path
import shutil
import time

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from groot.core.utils.storage.msc import (
    list_directories_from_path,
    resolve_checkpoint_path,
)


class UnfreezeBackboneCallback(TrainerCallback):
    """Calls model.maybe_unfreeze_backbone(global_step) each step so the VLM backbone can be unfrozen after K steps."""

    def __init__(self, model):
        self.model = model

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, _control: TrainerControl, **kwargs
    ):
        if self.model is not None and hasattr(self.model, "maybe_unfreeze_backbone"):
            self.model.maybe_unfreeze_backbone(state.global_step)


class StragglerDetectionCallback(TrainerCallback):
    """Callback for nvidia_resiliency_ext straggler detection with HuggingFace Trainer."""

    def __init__(
        self,
        report_time_interval: int = 60,
        scores_to_compute: list[str] | None = None,
        gather_on_rank0: bool = True,
        profiling_interval: int = 1,
        perf_threshold: float = 0.7,
    ):
        self.report_time_interval = report_time_interval
        self.scores_to_compute = scores_to_compute or [
            "relative_perf_scores",
            "individual_perf_scores",
        ]
        self.gather_on_rank0 = gather_on_rank0
        self.profiling_interval = profiling_interval
        self.perf_threshold = perf_threshold
        self._initialized = False

    def _get_straggler_module(self):
        """Return the straggler module."""
        try:
            from nvidia_resiliency_ext.attribution import straggler
        except ModuleNotFoundError as error:
            if error.name != "nvidia_resiliency_ext.attribution":
                raise
            # Older training images still install NVRx 0.3.0.
            from nvidia_resiliency_ext import straggler

        return straggler

    def on_train_begin(self, args, state, _control, **kwargs):
        """Verify straggler detector is initialized."""
        try:
            straggler = self._get_straggler_module()
            if hasattr(straggler.Detector, "initialized") and straggler.Detector.initialized:
                self._initialized = True
                logging.info("Straggler detection callback ready")
            else:
                self._initialized = False
                logging.warning("Straggler detector not initialized. Reporting disabled.")
        except ImportError as e:
            logging.warning(
                f"Straggler detection disabled: {e}. "
                "Training will continue without straggler detection."
            )
            self._initialized = False

    def on_step_end(self, args, state, _control, **kwargs):
        """Generate straggler report if time interval has elapsed."""
        if not self._initialized:
            return

        try:
            straggler = self._get_straggler_module()
            if not (hasattr(straggler.Detector, "initialized") and straggler.Detector.initialized):
                return

            report = straggler.Detector.generate_report_if_interval_elapsed()
            if report is not None:
                self._handle_report(report, state)
        except Exception as e:
            logging.warning(f"Error generating straggler report: {e}")

    def _handle_report(self, report, state):
        """Process and log straggler report."""
        import torch.distributed as dist

        if dist.is_initialized():
            global_rank = dist.get_rank()
        else:
            global_rank = 0

        should_log = (self.gather_on_rank0 and global_rank == 0) or not self.gather_on_rank0

        # Prepare wandb metrics (only on rank 0)
        wandb_metrics = {}

        if should_log:
            logging.info(f"\n{'='*60}")
            logging.info(f"Straggler Report (step {state.global_step})")
            logging.info(f"{'='*60}")

            section_scores = getattr(report, "section_relative_perf_scores", None)
            local_summaries = getattr(report, "local_section_summaries", None)
            total_straggler_ranks = set()

            if section_scores:
                for section_name, rank_scores in section_scores.items():
                    if rank_scores:
                        scores_list = [(rank, score) for rank, score in rank_scores.items()]
                        scores_list.sort(key=lambda x: x[1])

                        slowest_rank, slowest_score = scores_list[0]
                        fastest_rank, fastest_score = scores_list[-1]

                        stragglers = [(r, s) for r, s in scores_list if s < self.perf_threshold]
                        for r, _ in stragglers:
                            total_straggler_ranks.add(r)

                        wandb_metrics[f"slowest_rank/{section_name}_rank"] = slowest_rank
                        wandb_metrics[f"slowest_rank/{section_name}_score"] = slowest_score
                        wandb_metrics[f"straggler/{section_name}_fastest_score"] = fastest_score
                        wandb_metrics[f"straggler/{section_name}_num_below_threshold"] = len(
                            stragglers
                        )

                        timing_str = ""
                        if local_summaries and section_name in local_summaries:
                            from enum import Enum

                            summary = local_summaries[section_name]
                            for key, val in summary.items():
                                key_name = key.name if isinstance(key, Enum) else str(key)
                                if "AVG" in key_name:
                                    timing_str = f", avg={val:.1f}ms"
                                    wandb_metrics[f"timing/{section_name}_avg_ms"] = val
                                elif "MAX" in key_name:
                                    wandb_metrics[f"timing/{section_name}_max_ms"] = val
                                elif "MIN" in key_name:
                                    wandb_metrics[f"timing/{section_name}_min_ms"] = val

                        if stragglers:
                            strag_str = ", ".join([f"r{r}={s:.3f}" for r, s in stragglers[:5]])
                            if len(stragglers) > 5:
                                strag_str += f" (+{len(stragglers)-5})"
                            logging.warning(
                                f"  {section_name}: {len(stragglers)} SLOW (below {self.perf_threshold}): {strag_str}{timing_str}"
                            )
                        else:
                            logging.info(
                                f"  {section_name}: slowest=rank {slowest_rank} ({slowest_score:.4f}), all OK{timing_str}"
                            )

            wandb_metrics["straggler/total_slow_ranks"] = len(total_straggler_ranks)
            if total_straggler_ranks:
                logging.warning(
                    f"  TOTAL: {len(total_straggler_ranks)} ranks below threshold {self.perf_threshold}: {sorted(total_straggler_ranks)[:10]}"
                )
            else:
                logging.info(f"  All ranks performing well (>= {self.perf_threshold})")

            # Node-level straggler report (for forwarding to infra/Osmo team)
            rank_to_node = getattr(report, "rank_to_node", None)
            if rank_to_node and total_straggler_ranks and section_scores:
                self._log_node_report(section_scores, rank_to_node, total_straggler_ranks)

            logging.info(f"{'='*60}\n")

            # Log to wandb
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(wandb_metrics)
            except Exception:
                pass

    def _log_node_report(self, section_scores, rank_to_node, straggler_ranks):
        """Generate a node-level straggler report for infra team."""
        # Group straggler ranks by node
        node_stragglers = {}
        for rank in straggler_ranks:
            node = rank_to_node.get(rank, "unknown")
            if node not in node_stragglers:
                node_stragglers[node] = []
            # Get worst score across sections for this rank
            worst_score = 1.0
            worst_section = ""
            for section, scores in section_scores.items():
                if scores and rank in scores:
                    if scores[rank] < worst_score:
                        worst_score = scores[rank]
                        worst_section = section
            node_stragglers[node].append((rank, worst_score, worst_section))

        # Sort nodes by number of straggler ranks (worst nodes first)
        sorted_nodes = sorted(node_stragglers.items(), key=lambda x: len(x[1]), reverse=True)

        logging.warning("  --- Node Straggler Report ---")
        for node, ranks_info in sorted_nodes:
            avg_score = sum(s for _, s, _ in ranks_info) / len(ranks_info)
            worst = min(ranks_info, key=lambda x: x[1])
            rank_ids = [r for r, _, _ in ranks_info]
            pct_slower = int((1 - avg_score) * 100)
            logging.warning(
                f"  Node {node}: {len(ranks_info)} slow rank(s) {rank_ids}, "
                f"avg {pct_slower}% slower, worst=rank {worst[0]} ({worst[2]} score={worst[1]:.3f})"
            )

    def on_train_end(self, args, state, _control, **kwargs):
        """Shutdown straggler detector at training end."""
        if self._initialized:
            try:
                straggler = self._get_straggler_module()
                if hasattr(straggler.Detector, "initialized") and straggler.Detector.initialized:
                    straggler.Detector.shutdown()
                    logging.info("✓ Straggler detection shutdown complete")
            except Exception as e:
                logging.warning(f"Error shutting down straggler detector: {e}")
            finally:
                self._initialized = False


class SaveVlModelCallback(TrainerCallback):
    def __init__(self, processor=None):
        """
        Args:
            processor: The processor to save alongside the model
        """
        self.processor = processor.vlm_processor

    def on_save(self, args, state, _control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            # Save the model to vl_model directory
            vl_model_dir = checkpoint_dir / "vl_model"
            assert "model" in kwargs, "Model not found in kwargs"
            model = kwargs["model"]
            print(f"Saving model to {vl_model_dir}")
            model.save_pretrained(vl_model_dir)

            # Save the processor to the same vl_model directory
            assert self.processor is not None, "Processor not found"
            print(f"Saving processor to {vl_model_dir}")
            self.processor.save_pretrained(vl_model_dir)


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, _control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            t_start = time.time()
            print(
                f"[CP-SAVE][callback=CheckpointFormatCallback] on_save BEGIN at step {state.global_step}"
            )
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)

            duration = time.time() - t_start
            print(
                f"[CP-SAVE][callback=CheckpointFormatCallback] on_save END at step {state.global_step}, "
                f"time={duration:.2f}s"
            )


class BestMetricCheckpointCallback(TrainerCallback):
    """This callback saves the best checkpoint based on the metric."""

    def __init__(
        self, metric_name: str, greater_is_better: bool = True, exp_cfg_dir: Path | None = None
    ):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self.exp_cfg_dir = exp_cfg_dir
        self._best_checkpoint_dir = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        if state.is_world_process_zero and metrics is not None:
            current_metric = metrics.get(self.metric_name, None)
            if current_metric is not None:
                is_better = (
                    self.greater_is_better
                    if current_metric > self.best_metric
                    else not self.greater_is_better
                )
                if is_better:
                    self.best_metric = current_metric
                    best_checkpoint_dir = (
                        Path(args.output_dir)
                        / f"checkpoint-{state.global_step}-best-{self.metric_name}_{current_metric}"
                    )
                    best_checkpoint_dir.mkdir(exist_ok=True)
                    model.save_pretrained(best_checkpoint_dir)
                    # Copy experiment config directory if provided
                    if self.exp_cfg_dir is not None:
                        exp_cfg_dst = best_checkpoint_dir / self.exp_cfg_dir.name
                        if self.exp_cfg_dir.exists():
                            print(
                                f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                            )
                            shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

                    print(
                        f"Best checkpoint saved to {best_checkpoint_dir} with metric {self.metric_name} = {current_metric}"
                    )

                    if (
                        self._best_checkpoint_dir is not None
                        and Path(self._best_checkpoint_dir).exists()
                    ):
                        shutil.rmtree(self._best_checkpoint_dir)

                    self._best_checkpoint_dir = str(best_checkpoint_dir)


def _normalize_pdx_checkpoint_root(checkpoint_root: str | Path) -> Path:
    root = str(checkpoint_root).strip()
    if root.startswith("s3://"):
        root = root.removeprefix("s3://")
    if "://" in root:
        raise ValueError(f"PDX checkpoint root must be a GearCheckpoints path: {checkpoint_root}")
    root = root.rstrip("/")
    path = Path(root)
    if path.parts[:2] != ("GearCheckpoints", "trinity-ckpt-repository") or len(path.parts) != 3:
        raise ValueError(
            "PDX checkpoint root must be in the format "
            "GearCheckpoints/trinity-ckpt-repository/<run_name>"
        )
    return path


def get_pdx_checkpoints_dir(
    run_name: str,
    user_prefix: str | None = None,
    pdx_checkpoint_root: str | Path | None = None,
) -> Path:
    """Get the PDX checkpoints directory for a given run name."""
    if pdx_checkpoint_root:
        return _normalize_pdx_checkpoint_root(pdx_checkpoint_root)

    if user_prefix is None:
        from gear.data import get_s3_credentials_from_file

        creds = get_s3_credentials_from_file()
        user = creds["aws_access_key_id"].split(":")[0]
    else:
        user = user_prefix
    return Path("GearCheckpoints/trinity-ckpt-repository") / f"{user}_{run_name}"


def remove_pdx_directory(run_name: str) -> list[str]:
    """Remove all files and directories for a given run name in the PDX bucket."""
    from gear.data import get_s3_client

    s3 = get_s3_client()
    bucket = "GearCheckpoints"
    assert run_name.startswith(
        "GearCheckpoints/"
    ), f"Run name {run_name} must start with GearCheckpoints/"
    prefix = run_name[len("GearCheckpoints/") :]
    if not prefix.endswith("/"):
        prefix += "/"

    deleted_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in page_iterator:
        if "Contents" in page:
            # Prepare a batch of up to 1000 keys per delete call
            objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects_to_delete})
            deleted_keys.extend([f"{bucket}/{obj['Key']}" for obj in page["Contents"]])
    print(f"Deleted keys: {deleted_keys}")
    return deleted_keys


def list_pdx_files_and_dirs(run_name: str, bucket: str = "GearCheckpoints") -> list[str]:
    """List directories for a given run name in the PDX bucket."""
    from gear.data import get_s3_client

    s3 = get_s3_client()
    directories = []
    paginator = s3.get_paginator("list_objects_v2")
    assert run_name.startswith(bucket + "/"), f"Run name {run_name} must start with {bucket}/"
    prefix = run_name[len(bucket + "/") :]
    if not prefix.endswith("/"):
        prefix += "/"
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

    for page in page_iterator:
        if "CommonPrefixes" in page:
            for prefix_obj in page["CommonPrefixes"]:
                suffix = Path(f"{bucket}/{prefix_obj['Prefix']}")
                directories.append(str(suffix))

        # Add files
        if "Contents" in page:
            for file_obj in page["Contents"]:
                directories.append(f"{bucket}/{file_obj['Key']}")

    return directories


def check_checkpoints_dir_exists(checkpoints_dir: str):
    """Check if the checkpoints directory exists.

    For MSC URLs, lists the directory contents to verify it exists and contains experiment_cfg.
    For GearCheckpoints paths, checks PDX storage.
    For local paths, checks filesystem existence.
    """
    # Validate MSC URLs by listing directory contents
    if checkpoints_dir.startswith("msc://"):
        try:
            # List ALL directories in the remote directory (not just checkpoints)
            directory_names = list_directories_from_path(checkpoints_dir)

            # Verify directory exists and contains experiment_cfg
            if len(directory_names) > 0:
                if "experiment_cfg" not in directory_names:
                    return False
                return True
            else:
                return False
        except Exception as e:
            # If listing fails, assume directory doesn't exist
            print(f"Error listing MSC directory {checkpoints_dir}: {e}")
            return False
    elif checkpoints_dir.startswith("GearCheckpoints/"):
        assert checkpoints_dir.split("/")[0] == "GearCheckpoints"
        assert checkpoints_dir.split("/")[1] == "trinity-ckpt-repository"
        assert (
            len(checkpoints_dir.split("/")) == 3
        ), "PDX checkpoints must be in the format of GearCheckpoints/trinity-ckpt-repository/user_run_name"
        directories = list_pdx_files_and_dirs(checkpoints_dir)
        if len(directories) > 0:
            assert "experiment_cfg" in [
                directory.split("/")[-1] for directory in directories
            ], f"Checkpoints directory {checkpoints_dir} does not contain experiment_cfg directory"
            return True
        else:
            return False
    else:
        return Path(checkpoints_dir).exists()


def download_pdx_directory_with_retries(src: str, dst: str, max_retries: int = 3):
    """Download a PDX directory with retries."""
    from gear.data import gear_download

    for attempt in range(max_retries):
        try:
            gear_download(
                src=src,
                dst=dst,
            )
            return
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = min(2**attempt, 60)  # Exponential backoff, max 60 seconds
                print(f"Download failed for {src} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Download failed for {src} after {max_retries} attempts: {str(e)}")


def ensure_local_checkpoint(checkpoint_dir):
    checkpoint_dir = str(checkpoint_dir)

    # Find LFS directory
    lfs_dirs = glob.glob("/mnt/*lfs*")
    assert lfs_dirs, "No LFS directory found in /mnt/"
    lfs_dir = sorted(lfs_dirs)[0]

    cache_dir = Path(f"{lfs_dir}/shared/ckpt_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir.startswith("msc://"):
        # Download MSC checkpoint to cache while resolving the path
        # checkpoint_dir: msc://GearCheckpoints/$USER/$JOB_NAME/checkpoint-xxxx
        # lfs_checkpoint_dir: $cache_dir/url_hash/checkpoint-xxxx
        lfs_checkpoint_dir = resolve_checkpoint_path(
            checkpoint_path=checkpoint_dir,
            output_dir=str(cache_dir),
            global_rank=0,
        )
        # Extract checkpoint name from URL for wandb naming
        checkpoint_dir_split = checkpoint_dir.rstrip("/").split("/")
        checkpoint_name = f"{checkpoint_dir_split[-2]}_{checkpoint_dir_split[-1]}"
    elif checkpoint_dir.startswith("GearCheckpoints/"):
        # Use parent dir name, otherwise the name is usually just "checkpoint-xxxx"
        checkpoint_name = f"{Path(checkpoint_dir).parent.name}_{Path(checkpoint_dir).name}"
        lfs_checkpoint_dir = str(cache_dir / checkpoint_name)
        download_pdx_directory_with_retries(
            src=str(Path(checkpoint_dir) / "*"),
            dst=lfs_checkpoint_dir,
        )
    elif Path(checkpoint_dir).exists():
        lfs_checkpoint_dir = checkpoint_dir
        # Use parent dir name, otherwise the name is usually just "checkpoint-xxxx"
        checkpoint_name = f"{Path(checkpoint_dir).parent.name}_{Path(checkpoint_dir).name}"
    else:
        raise FileNotFoundError(f"{checkpoint_dir} not found")

    return cache_dir, checkpoint_name, lfs_checkpoint_dir


def upload_checkpoint_with_retries(
    checkpoint_dir: Path, step: int, pdx_path: Path, max_retries: int = 3
):
    """Upload a checkpoint with retries."""
    from gear.data import gear_upload

    for attempt in range(max_retries):
        try:
            gear_upload(
                src=str(checkpoint_dir),
                dst=str(pdx_path / f"checkpoint-{step}"),
                filters=[
                    ("--exclude", "global_step*"),
                    ("--exclude", "rng_state*"),
                    ("--exclude", "optimizer.pt"),
                    ("--exclude", "optimizer_rank*.pt"),
                    ("--exclude", "dcp_state"),
                ],
            )
            # create and upload sentinel file to tell the evaluator daemon that the checkpoint is fully uploaded
            sentinel_file = checkpoint_dir / "SUCCESS"
            sentinel_file.touch()
            gear_upload(
                src=str(sentinel_file),
                dst=str(pdx_path / f"checkpoint-{step}" / "SUCCESS"),
            )
            return
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = min(2**attempt, 60)  # Exponential backoff, max 60 seconds
                print(
                    f"Upload failed for checkpoint {step} (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Upload failed for checkpoint {step} after {max_retries} attempts: {str(e)}")


def check_pdx_checkpoint_upload_completed(pdx_checkpoint: Path) -> bool:
    """Check if the checkpoint is completed."""
    files_and_dirs = list_pdx_files_and_dirs(
        str(pdx_checkpoint),
    )
    print(f"files_and_dirs found: {files_and_dirs}")
    files_and_dirs = [item.split("/")[-1] for item in files_and_dirs]
    if "SUCCESS" in files_and_dirs:
        print("SUCCESS exists, so the checkpoint is completed.")
        return True
    else:
        # check the existence of all files
        safetensors = []
        for item in files_and_dirs:
            if item.endswith(".safetensors"):
                assert item.startswith("model-"), f"{item=}"
                assert "-of-" in item, f"{item=}"
                safetensors.append(item)
        if len(safetensors) > 0:
            # extract the total number of safetensors based on of-<number>.safetensors
            total_num_safetensors = int(safetensors[0].split("-of-")[1].split(".safetensors")[0])
            expected_files_and_dirs = [
                "experiment_cfg",
                "config.json",
                "latest",
                "model.safetensors.index.json",
                "scheduler.pt",
                "trainer_state.json",
                "wandb_config.json",
                "zero_to_fp32.py",
            ]
            for i in range(1, total_num_safetensors + 1):
                expected_files_and_dirs.append(
                    f"model-{i:05d}-of-{total_num_safetensors:05d}.safetensors"
                )
            missing_files = set(expected_files_and_dirs) - set(files_and_dirs)
            if len(missing_files) > 0:
                print(f"Missing files: {missing_files}")
                return False
            else:
                print("SUCCESS does not exist, but all other files exist.")
                return True
        else:
            # No safetensors found, so it's not completed
            return False


class CheckpointUploadCallback(TrainerCallback):
    """This callback uploads checkpoints at regular intervals plus the last N checkpoints of the full training run"""

    def __init__(
        self,
        run_name: str,
        exp_cfg_dir: Path | None = None,
        pdx_checkpoint_root: str | Path | None = None,
        upload_every: int = 1000,
        last_n_checkpoints: int = 5,
        max_concurrent_uploads: int = 2,
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
            pdx_checkpoint_root: Optional dashboard-allocated PDX root to upload into.
            upload_every: Upload checkpoints every N steps
            last_n_checkpoints: Number of final checkpoints to upload
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.upload_every = upload_every
        self.last_n_checkpoints = last_n_checkpoints
        self.checkpoints_to_upload: set[int] = set()
        self.upload_executor = ThreadPoolExecutor(max_workers=max_concurrent_uploads)

        self.pdx_path = get_pdx_checkpoints_dir(
            run_name=run_name,
            pdx_checkpoint_root=pdx_checkpoint_root,
        )
        self.initial_upload_done = False
        self.trainer = None

    def on_train_begin(self, args, state, _control, **kwargs):
        """Upload initial metadata and calculate which checkpoints to upload."""
        from gear.data import gear_upload

        if state.is_world_process_zero:
            if not self.initial_upload_done:
                exp_cfg_pdx_path = self.pdx_path / "experiment_cfg"
                gear_upload(
                    src=str(self.exp_cfg_dir),
                    dst=str(exp_cfg_pdx_path),
                )
                print(f"Uploaded experiment_cfg to {exp_cfg_pdx_path}")
                wandb_config_pdx_path = self.pdx_path / "wandb_config.json"
                gear_upload(
                    src=str(self.exp_cfg_dir).replace("experiment_cfg", "wandb_config.json"),
                    dst=str(wandb_config_pdx_path),
                )
                print(f"Uploaded wandb_config.json to {wandb_config_pdx_path}")
                self.initial_upload_done = True

            # Calculate checkpoints to upload based on upload_every
            max_steps = args.max_steps
            save_steps = args.save_steps

            # Add checkpoints at upload_every interval
            current_step = self.upload_every
            while current_step <= max_steps:
                self.checkpoints_to_upload.add(current_step)
                current_step += self.upload_every

            # Add last N checkpoints
            for i in range(self.last_n_checkpoints):
                step = max_steps - i * save_steps
                if step > 0:
                    self.checkpoints_to_upload.add(int(step))

            print(f"Will upload checkpoints at steps: {sorted(self.checkpoints_to_upload)}")

    def _wait_for_async_checkpoint_writes(self):
        """Wait for safetensors background writes to complete before uploading.

        Only waits for safetensors (model shards) since the upload excludes
        optimizer state (optimizer.pt, optimizer_rank*.pt, dcp_state).
        Waiting for DCP async_save here would deadlock: its future.result()
        may require a collective barrier but only rank 0 reaches this callback.
        """
        if self.trainer is None:
            return
        self.trainer._wait_for_pending_safetensors_worker()

    @staticmethod
    def _verify_checkpoint_files(checkpoint_dir: Path, step: int):
        """Verify that expected safetensors files are present on disk before uploading."""
        found = list(checkpoint_dir.glob("*.safetensors"))
        assert len(found) > 0, f"Checkpoint {step}: no safetensors files found in {checkpoint_dir}"

        index_file = checkpoint_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            expected_shards = sorted(set(index.get("weight_map", {}).values()))
            missing = [s for s in expected_shards if not (checkpoint_dir / s).exists()]
            assert (
                len(missing) == 0
            ), f"Checkpoint {step}: missing safetensors shards {missing} in {checkpoint_dir}"

        print(
            f"[CP-SAVE][callback=CheckpointUploadCallback] verified "
            f"{len(found)} safetensors file(s) present for step {step}"
        )

    def _background_wait_and_upload(self, checkpoint_dir: Path, step: int):
        """Background thread: wait for async checkpoint writes to land on disk, then upload.

        Moving the wait into the background thread avoids blocking the training loop.
        This is safe because the trainer's own _save_checkpoint will wait for any
        pending async writes before dispatching new ones at the next save step.
        """
        t_wait_start = time.time()
        self._wait_for_async_checkpoint_writes()
        wait_duration = time.time() - t_wait_start
        print(
            f"[CP-SAVE][callback=CheckpointUploadCallback] background wait done for step {step}, "
            f"wait_time={wait_duration:.2f}s"
        )
        self._verify_checkpoint_files(checkpoint_dir, step)
        upload_checkpoint_with_retries(checkpoint_dir, step, self.pdx_path)
        print(f"Uploaded checkpoint {step} to {self.pdx_path}")

    def on_save(self, args, state, _control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero and state.global_step in self.checkpoints_to_upload:
            t_start = time.time()
            print(
                f"[CP-SAVE][callback=CheckpointUploadCallback] on_save BEGIN at step {state.global_step}"
            )
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            assert (
                checkpoint_dir.exists()
            ), f"Checkpoint {state.global_step} not found at {checkpoint_dir}"
            if args.max_steps == state.global_step:
                self._wait_for_async_checkpoint_writes()
                self._verify_checkpoint_files(checkpoint_dir, state.global_step)
                upload_checkpoint_with_retries(
                    checkpoint_dir,
                    state.global_step,
                    self.pdx_path,
                )
                print(f"Uploaded last checkpoint {state.global_step} to {self.pdx_path}")
            else:
                step = state.global_step
                future = self.upload_executor.submit(
                    self._background_wait_and_upload,
                    checkpoint_dir,
                    step,
                )
                future.add_done_callback(
                    lambda f, _step=step: (
                        None
                        if f.exception() is None
                        else print(
                            f"[CP-SAVE][callback=CheckpointUploadCallback] upload failed for step "
                            f"{_step}: {f.exception()}"
                        )
                    )
                )
                print(f"Submitted checkpoint {step} for background wait+upload")

            duration = time.time() - t_start
            print(
                f"[CP-SAVE][callback=CheckpointUploadCallback] on_save END at step {state.global_step}, "
                f"time={duration:.2f}s (upload dispatched async, not included)"
            )

    def __del__(self):
        print("Training completed. Waiting for remaining uploads to finish...")
        self.upload_executor.shutdown(wait=True)
        print("All uploads completed.")


class RLCallback(TrainerCallback):
    def on_optimizer_step(self, args, state, _control, **kwargs):
        model = kwargs.get("model")
        assert model is not None, "Model not found in kwargs"
        assert hasattr(model, "q_model"), "Q model not found in RL model"
        model.q_model.update_target_q_model()


class OpenLoopEvalCallback(TrainerCallback):
    """Callback to run open-loop evaluation on saved checkpoints.

    NOTE: This callback lazily imports from groot.control at runtime.
    It only works when the full groot package is available. If using
    groot.core standalone, do not enable open_loop_eval in the config.
    """

    def __init__(
        self,
        eval_configs: list[dict[str, str]],
        traj_ids: list[int],
        action_horizon: int,
        steps_per_traj: int,
        plot_indices: list[int] | None = None,
        video_backend: str = "torchcodec",
        policy_type: str = "gr00t",
        trainer=None,
    ):
        self.eval_configs = eval_configs
        self.traj_ids = traj_ids
        self.action_horizon = action_horizon
        self.steps_per_traj = steps_per_traj
        self.plot_indices = plot_indices
        self.video_backend = video_backend
        self.policy_type = policy_type
        self.trainer = trainer

    def on_save(
        self, args: TrainingArguments, state: TrainerState, _control: TrainerControl, **kwargs
    ):
        import torch.distributed as dist

        _use_barrier = dist.is_initialized() and dist.get_world_size() > 1
        _gloo_group = dist.new_group(backend="gloo") if _use_barrier else None

        should_eval = state.is_world_process_zero and bool(self.eval_configs)

        if should_eval:
            if self.trainer is not None:
                self.trainer._wait_for_pending_safetensors_worker()

        if not should_eval:
            if _use_barrier:
                dist.barrier(group=_gloo_group)
                dist.destroy_process_group(_gloo_group)
            return

        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        print(f"\n{'='*80}")
        print(f"Starting open-loop evaluation for checkpoint {state.global_step}")
        print(f"{'='*80}\n")

        import logging
        import os

        import numpy as np
        import torch
        import wandb

        from groot.core.data.schema import EmbodimentTag
        from groot.core.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        if self.policy_type == "dream0":
            from groot.vla.omni.oss_eval.open_loop_eval import evaluate_single_trajectory
            from groot.vla.omni.policy.dream0_policy import DreamZeroPolicy
        else:
            import random

            from groot.control.main.vla.open_loop_eval import (
                evaluate_single_trajectory,
                plot_trajectory_results_training_online,
            )
            from groot.control.policy.groot_n1_policy import GrootN1ServerPolicy

        for eval_config in self.eval_configs:
            dataset_path = eval_config["dataset_path"]
            embodiment_tag = eval_config["embodiment_tag"]
            eval_type = eval_config["eval_type"]

            try:
                dataset_name_base = os.path.splitext(os.path.basename(dataset_path.rstrip("/")))[0]
                dataset_name = f"{embodiment_tag}_{eval_type}"
                print(f"Evaluating on {embodiment_tag}/{eval_type}: {dataset_name_base}")

                if self.policy_type == "dream0":
                    policy = DreamZeroPolicy(
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        model_path=str(checkpoint_path),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                else:
                    eval_port = random.randint(7000, 8000)
                    print(f"  Using port {eval_port} for policy server")
                    policy = GrootN1ServerPolicy(
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        model_path=str(checkpoint_path),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        host="127.0.0.1",
                        port=eval_port,
                        policy_type="omni_dp",
                    )

                modality = policy.get_modality_config()

                dataset = LeRobotEpisodeLoader(
                    dataset_path=dataset_path,
                    modality_configs=modality,
                    video_backend=self.video_backend,
                    video_backend_kwargs=None,
                )

                all_mse = []
                all_mae = []

                for traj_id in self.traj_ids:
                    if traj_id >= len(dataset):
                        logging.warning(
                            f"Trajectory ID {traj_id} out of range (dataset has {len(dataset)} trajectories)"
                        )
                        continue

                    print(f"  [Open-loop evaluation] Evaluating trajectory {traj_id}...")

                    if self.policy_type == "dream0":
                        plot_path = f"/tmp/open_loop_eval/{dataset_name}_step{state.global_step}_traj{traj_id}.jpeg"
                        video_path = f"/tmp/open_loop_eval/{dataset_name}_step{state.global_step}_traj{traj_id}.mp4"
                        mse, mae = evaluate_single_trajectory(
                            policy=policy,
                            loader=dataset,
                            traj_id=traj_id,
                            embodiment_tag=EmbodimentTag(embodiment_tag),
                            steps=self.steps_per_traj,
                            action_horizon=self.action_horizon,
                            save_plot_path=plot_path,
                            save_video_path=video_path,
                        )
                        if wandb.run is not None:
                            if os.path.exists(plot_path):
                                wandb.log(
                                    {
                                        f"{dataset_name}/traj_{traj_id}/plot": wandb.Image(
                                            plot_path
                                        ),
                                        "eval_step": state.global_step,
                                    }
                                )
                            actual_video_path = video_path.replace(".mp4", f"_{traj_id}.mp4")
                            if os.path.exists(actual_video_path):
                                wandb.log(
                                    {
                                        f"{dataset_name}/traj_{traj_id}/video": wandb.Video(
                                            actual_video_path
                                        ),
                                        "eval_step": state.global_step,
                                    }
                                )
                    else:
                        mse, mae, _, _ = evaluate_single_trajectory(
                            policy=policy,
                            loader=dataset,
                            traj_id=traj_id,
                            embodiment_tag=EmbodimentTag(embodiment_tag),
                            steps=self.steps_per_traj,
                            action_horizon=self.action_horizon,
                            plot_save_dir="./offline_open_loop_eval_plots",
                            plot_indices=self.plot_indices,
                            log_to_wandb=(wandb.run is not None),
                            global_step=state.global_step,
                            dataset_name=dataset_name,
                            checkpoint_name=f"checkpoint-{state.global_step}",
                            plot_function=plot_trajectory_results_training_online,
                        )

                    all_mse.append(mse)
                    all_mae.append(mae)
                    print(f"    MSE: {mse:.6f}, MAE: {np.mean(mae):.6f}")

                if all_mse and wandb.run is not None:
                    avg_mse = np.mean(all_mse)
                    avg_mae = np.mean(all_mae)

                    wandb.log(
                        {
                            f"{embodiment_tag}/{eval_type}/avg_mse": avg_mse,
                            f"{embodiment_tag}/{eval_type}/avg_mae": avg_mae,
                            f"{embodiment_tag}/{eval_type}/num_trajectories": len(all_mse),
                            "eval_step": state.global_step,
                        }
                    )

                    print(f"\n  Summary for {embodiment_tag}/{eval_type}:")
                    print(f"    Average MSE: {avg_mse:.6f}")
                    print(f"    Average MAE: {avg_mae:.6f}")
                    print(f"    Evaluated {len(all_mse)} trajectories")

                if self.policy_type != "dream0":
                    print("  Cleaning up policy server...")
                    if hasattr(policy, "socket") and hasattr(policy.socket, "close"):
                        policy.socket.close()
                    if hasattr(policy, "context") and hasattr(policy.context, "term"):
                        policy.context.term()
                del policy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("  Policy cleanup complete")

            except Exception as e:
                print(f"Error evaluating dataset {dataset_path}: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"\n{'='*80}")
        print(f"Open-loop evaluation completed for checkpoint {state.global_step}")
        print(f"{'='*80}\n")

        if _use_barrier:
            dist.barrier(group=_gloo_group)
            dist.destroy_process_group(_gloo_group)
