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
"""MSC Checkpoint Callback for direct remote storage uploads."""

import logging
from pathlib import Path
import shutil
import threading
import time

from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)

try:
    import multistorageclient as msc
    from multistorageclient.types import ExecutionMode, PatternType

    MSC_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing msc: {e}")
    msc = None  # type: ignore
    ExecutionMode = None  # type: ignore
    PatternType = None  # type: ignore
    MSC_AVAILABLE = False
    logging.warning(
        "multi-storage-client not available. MSCCheckpointCallback will not function. "
        "Install with: pip install multi-storage-client[boto3]"
    )


class MSCCheckpointCallback(TrainerCallback):
    """Callback to upload checkpoints directly to remote storage using multi-storage-client.

    This callback uploads checkpoints from ephemeral storage to S3/object storage using MSC,
    eliminating the need for Lustre as an intermediate storage layer.

    Every checkpoint saved by Trainer (based on save_steps) will be uploaded to remote storage.

    Args:
        remote_checkpoint_dir: Remote directory path (s3://bucket/path or msc://profile/path)
        local_checkpoint_dir: Local output directory where checkpoints are saved
        exp_cfg_dir: Path to local experiment_cfg directory (uploaded once at train begin)
        wandb_config_file: Path to local wandb_config.json file (uploaded once at train begin)
        max_concurrent_uploads: Maximum number of concurrent background uploads
        sync_mode: If True, force synchronous uploads for all checkpoints (default: False for async)
        delete_after_upload: If True, delete local checkpoint after successful upload
        execution_mode: MSC execution mode ("local" or "ray" for Ray-based parallel, default: "local")
        max_retries: Maximum number of retry attempts for failed uploads (default: 3)

    Attributes:
        trainer: Trainer instance (used to wait for async safetensors/optimizer writes).
            Set after construction via ``callback.trainer = trainer``.
    """

    # Files to exclude from checkpoint uploads.
    # Only safetensors model shards, config, trainer_state, and metadata are
    # needed for remote resume.  Optimizer state (per-rank .pt and DCP) and
    # RNG state are rank-specific and not uploaded.
    CHECKPOINT_EXCLUDE_PATTERNS = [
        "rng_state*",
        "optimizer.pt",
        "*optim_states.pt",
        "optimizer_rank*.pt",
        "dcp_state",
    ]

    def __init__(
        self,
        remote_checkpoint_dir: str,
        local_checkpoint_dir: str | Path,
        exp_cfg_dir: Path | None = None,
        wandb_config_file: Path | None = None,
        max_concurrent_uploads: int = 2,
        sync_mode: bool = False,
        delete_after_upload: bool = True,
        execution_mode: str = "local",
        max_retries: int = 3,
    ):
        if not MSC_AVAILABLE:
            raise ImportError(
                "multi-storage-client is required for MSCCheckpointCallback. "
                "Install with: pip install multi-storage-client[boto3]"
            )

        self.remote_checkpoint_dir = remote_checkpoint_dir
        self.local_checkpoint_dir = Path(local_checkpoint_dir)
        self.exp_cfg_dir = exp_cfg_dir
        self.wandb_config_file = wandb_config_file
        self.sync_mode = sync_mode
        self.delete_after_upload = delete_after_upload
        self.max_retries = max_retries
        self.max_concurrent_uploads = max_concurrent_uploads
        self.trainer = None  # Set after construction via .trainer attribute
        self.upload_threads = []  # Track active background upload threads
        if execution_mode == "ray":
            self.execution_mode = ExecutionMode.RAY  # type: ignore
        else:
            self.execution_mode = ExecutionMode.LOCAL  # type: ignore

        self.initial_upload_done = False

        logger.info(
            f"MSCCheckpointCallback initialized: remote_dir={remote_checkpoint_dir}, "
            f"local_dir={self.local_checkpoint_dir}, sync_mode={sync_mode}, delete_after={delete_after_upload}, "
            f"execution_mode={self.execution_mode})"
        )

    def on_train_begin(self, args, state, _control, **_kwargs):
        """Upload initial metadata (experiment_cfg and wandb_config.json) once at training start."""
        if not state.is_world_process_zero:
            return

        if self.initial_upload_done:
            return

        if not MSC_AVAILABLE or msc is None or ExecutionMode is None:
            logger.error("MSC is not available")
            return

        # Upload experiment_cfg directory if provided
        if self.exp_cfg_dir is not None and self.exp_cfg_dir.exists():
            remote_exp_cfg_path = f"{self.remote_checkpoint_dir}/experiment_cfg"
            try:
                logger.debug(
                    f"Uploading experiment_cfg from {self.exp_cfg_dir} to {remote_exp_cfg_path}"
                )

                # upload the entire directory via sync
                msc.sync(
                    source_url=str(self.exp_cfg_dir),
                    target_url=remote_exp_cfg_path,
                    execution_mode=self.execution_mode,
                )
                logger.info(f"Successfully uploaded experiment_cfg to {remote_exp_cfg_path}")
            except Exception as e:
                logger.error(f"Failed to upload experiment_cfg: {e}")
                raise

        # Upload wandb_config.json if provided
        if self.wandb_config_file is not None and self.wandb_config_file.exists():
            remote_wandb_config_path = f"{self.remote_checkpoint_dir}/wandb_config.json"
            try:
                logger.debug(
                    f"Uploading wandb_config.json from {self.wandb_config_file} to {remote_wandb_config_path}"
                )

                # Use msc.write() for single file to avoid directory creation issue
                with open(self.wandb_config_file, "rb") as f:
                    msc.write(  # type: ignore
                        url=remote_wandb_config_path,
                        body=f.read(),
                    )
                logger.info(
                    f"Successfully uploaded wandb_config.json to {remote_wandb_config_path}"
                )
            except Exception as e:
                logger.error(f"Failed to upload wandb_config.json: {e}")
                raise

        self.initial_upload_done = True

    def _wait_for_async_checkpoint_writes(self):
        """Wait for safetensors background writes to complete before uploading.

        Only waits for safetensors (model shards) since the upload excludes
        optimizer state (optimizer.pt, optimizer_rank*.pt, dcp_state via
        CHECKPOINT_EXCLUDE_PATTERNS). Waiting for DCP async_save here would
        deadlock on the final step: its future.result() may require a collective
        barrier but only rank 0 reaches this callback.
        """
        if self.trainer is None:
            return
        self.trainer._wait_for_pending_safetensors_worker()

    def _background_wait_and_upload(self, local_path: Path, remote_path: str, step: int):
        """Background thread: wait for async checkpoint writes to land on disk, then upload.

        Moving the wait into the background thread avoids blocking the training loop.
        This is safe because the trainer's own _save_checkpoint will wait for any
        pending async writes before dispatching new ones at the next save step.
        """
        t_wait_start = time.time()
        self._wait_for_async_checkpoint_writes()
        wait_duration = time.time() - t_wait_start
        print(
            f"[CP-SAVE][callback=MSCCheckpointCallback] background wait done for step {step}, "
            f"wait_time={wait_duration:.2f}s"
        )
        self._upload_checkpoint(local_path, remote_path, step)

    def on_save(self, args, state, _control, **_kwargs):
        """Called after the HF's Trainer saves a checkpoint. Uploads every checkpoint repository to remote storage."""
        if not state.is_world_process_zero:
            return

        t_start = time.time()
        print(
            f"[CP-SAVE][callback=MSCCheckpointCallback] on_save BEGIN at step {state.global_step}"
        )

        checkpoint_dir = self.local_checkpoint_dir / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint {state.global_step} not found at {checkpoint_dir}")
            return

        remote_checkpoint_path = f"{self.remote_checkpoint_dir}/checkpoint-{state.global_step}"

        is_final_checkpoint = args.max_steps == state.global_step

        if is_final_checkpoint or self.sync_mode:
            # Synchronous: wait for async writes, then upload (must complete before exit)
            self._wait_for_async_checkpoint_writes()
            success = self._upload_checkpoint(
                checkpoint_dir, remote_checkpoint_path, state.global_step
            )
            if not success:
                logger.error(
                    f"Failed to upload checkpoint {state.global_step} to {remote_checkpoint_path} "
                    f"({'final' if is_final_checkpoint else 'sync'})"
                )
        else:
            # Asynchronous: dispatch wait + upload in background thread so
            # training resumes immediately without blocking on I/O.
            # Clean up completed threads first
            self.upload_threads = [t for t in self.upload_threads if t.is_alive()]

            # Start background thread that waits for async writes then uploads
            thread = threading.Thread(
                target=self._background_wait_and_upload,
                args=(checkpoint_dir, remote_checkpoint_path, state.global_step),
            )
            thread.start()
            self.upload_threads.append(thread)
            logger.info(
                f"MSCCheckpointCallback: step={state.global_step} submitted for background wait+upload "
                f"({len(self.upload_threads)} active threads)"
            )

        duration = time.time() - t_start
        print(
            f"[CP-SAVE][callback=MSCCheckpointCallback] on_save END at step {state.global_step}, "
            f"time={duration:.2f}s"
        )

    def _upload_checkpoint(self, local_path: Path, remote_path: str, step: int) -> bool:
        """Upload a checkpoint directory using MSC with retry logic.

        Returns:
            bool: True if upload succeeded, False if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Uploading checkpoint {step} from {local_path} to {remote_path} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )

                try:
                    # Create exclusion patterns for checkpoint upload
                    exclude_patterns = [
                        (PatternType.EXCLUDE, pattern)  # type: ignore
                        for pattern in self.CHECKPOINT_EXCLUDE_PATTERNS
                    ]

                    # Upload the entire directory via msc sync
                    upload_start_time = time.time()
                    msc.sync(  # type: ignore
                        source_url=str(local_path),
                        target_url=remote_path,
                        execution_mode=self.execution_mode,
                        patterns=exclude_patterns,
                    )

                    # Create SUCCESS sentinel file directly in remote storage to signal upload completion
                    # This matches the behavior of CheckpointUploadCallback

                    msc.write(  # type: ignore
                        url=f"{remote_path}/SUCCESS",
                        body=b"",  # Empty file serves as marker
                    )
                    logger.debug(f"Created sentinel file at {remote_path}/SUCCESS")
                    upload_duration = time.time() - upload_start_time
                    logger.info(
                        f"Successfully uploaded checkpoint {step} to {remote_path} "
                        f"(upload_time={upload_duration:.2f}s)"
                    )
                except Exception as e:
                    logger.error(f"Failed to upload checkpoint {step} to {remote_path}: {e}")
                    raise e

                # Delete local checkpoint if requested
                if self.delete_after_upload:
                    try:
                        shutil.rmtree(local_path)
                        logger.info(f"Deleted local checkpoint at {local_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete local checkpoint at {local_path}: {e}")

                return True  # Success

            except Exception as e:
                logger.error(
                    f"Failed to upload checkpoint {step} (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = min(2 ** (attempt + 1), 60)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed for checkpoint {step}. "
                        "Training will continue but checkpoint was not uploaded."
                    )

        return False  # All retries failed

    def on_train_end(self, args, state, _control, **_kwargs):
        """Wait for all background upload threads to complete before training ends."""
        if not state.is_world_process_zero:
            return

        if self.upload_threads:
            active_threads = [t for t in self.upload_threads if t.is_alive()]
            if active_threads:
                logger.info(
                    f"Waiting for {len(active_threads)} background upload threads to complete..."
                )
                for thread in active_threads:
                    thread.join()
                logger.info("All background uploads completed.")
