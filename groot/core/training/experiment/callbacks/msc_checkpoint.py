# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""MSC Checkpoint Callback for direct remote storage uploads."""

import logging
from pathlib import Path
import shutil
import threading
import time
from typing import ClassVar

from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)

try:
    import multistorageclient as msc
    from multistorageclient.types import ExecutionMode, PatternType

    MSC_AVAILABLE = True
except ImportError:
    logger.exception("Error importing msc")
    msc = None  # type: ignore[assignment]
    ExecutionMode = None  # type: ignore[assignment]
    PatternType = None  # type: ignore[assignment]
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
        upload_trainer_resume_state: If True, upload Trainer optimizer, scheduler,
            RNG, and distributed checkpoint state in addition to model files. Trainer
            subclasses can enable this for their MSC callback when exact resume is required.

    Attributes:
        trainer: Trainer instance (used to wait for async safetensors/optimizer writes).
            Set after construction via ``callback.trainer = trainer``.
    """

    # Files to exclude from checkpoint uploads.
    # Only safetensors model shards, config, trainer_state, and metadata are
    # needed for model-only remote loading. Optimizer state (per-rank .pt and
    # DCP) and RNG state are rank-specific and excluded by default. Callers can
    # opt into publishing a complete cross-run Trainer checkpoint.
    CHECKPOINT_EXCLUDE_PATTERNS: ClassVar[list[str]] = [
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
        upload_trainer_resume_state: bool = False,
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
        self.upload_trainer_resume_state = upload_trainer_resume_state
        self.trainer = None  # Set after construction via .trainer attribute
        self.upload_threads = []  # Track active background upload threads
        self._trainer_state_upload_errors: list[Exception] = []
        if execution_mode == "ray":
            self.execution_mode = ExecutionMode.RAY  # type: ignore[union-attr]
        else:
            self.execution_mode = ExecutionMode.LOCAL  # type: ignore[union-attr]

        self.initial_upload_done = False

        logger.info(
            f"MSCCheckpointCallback initialized: remote_dir={remote_checkpoint_dir}, "
            f"local_dir={self.local_checkpoint_dir}, sync_mode={sync_mode}, delete_after={delete_after_upload}, "
            f"execution_mode={self.execution_mode}, "
            f"upload_trainer_resume_state={self.upload_trainer_resume_state})"
        )

    def _checkpoint_exclude_patterns(self) -> list[str]:
        if self.upload_trainer_resume_state:
            return []
        return list(self.CHECKPOINT_EXCLUDE_PATTERNS)

    def on_train_begin(self, _args, state, _control, **_kwargs):
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
            except Exception:
                logger.exception("Failed to upload experiment_cfg")
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
                    msc.write(  # type: ignore[union-attr]
                        url=remote_wandb_config_path,
                        body=f.read(),
                    )
                logger.info(
                    f"Successfully uploaded wandb_config.json to {remote_wandb_config_path}"
                )
            except Exception:
                logger.exception("Failed to upload wandb_config.json")
                raise

        self.initial_upload_done = True

    def _wait_for_async_checkpoint_writes(self):
        """Wait for all local files selected for publication to become durable.

        The default model-only path waits only for safetensors. Complete
        Trainer-state publication uses the all-rank Trainer hook to also wait
        for optimizer shards and DCP state before rank 0 uploads the checkpoint.
        """
        if self.trainer is None:
            if self.upload_trainer_resume_state:
                raise RuntimeError(
                    "Complete Trainer-state publication requires an attached Trainer"
                )
            return
        if self.upload_trainer_resume_state:
            wait_for_resume_state = getattr(
                self.trainer,
                "_wait_for_pending_trainer_checkpoint_writes",
                None,
            )
            if not callable(wait_for_resume_state):
                raise RuntimeError(
                    "Complete Trainer-state publication requires the Trainer checkpoint "
                    "write synchronization hook"
                )
            wait_for_resume_state()
            return
        wait_for_model = getattr(self.trainer, "_wait_for_pending_safetensors_worker", None)
        if not callable(wait_for_model):
            raise TypeError("Checkpoint publication requires the Trainer model-write hook")
        wait_for_model()

    def _background_wait_and_upload(self, local_path: Path, remote_path: str, step: int):
        """Background thread: wait for async checkpoint writes to land on disk, then upload.

        Moving the wait into the background thread avoids blocking the training loop.
        This is safe because the trainer's own _save_checkpoint will wait for any
        pending async writes before dispatching new ones at the next save step.
        """
        t_wait_start = time.time()
        self._wait_for_async_checkpoint_writes()
        wait_duration = time.time() - t_wait_start
        logger.info(
            f"[CP-SAVE][callback=MSCCheckpointCallback] background wait done for step {step}, "
            f"wait_time={wait_duration:.2f}s"
        )
        self._upload_checkpoint(local_path, remote_path, step)

    def _raise_trainer_state_upload_error(self) -> None:
        if self._trainer_state_upload_errors:
            error = self._trainer_state_upload_errors[0]
            raise RuntimeError(f"Complete Trainer checkpoint upload failed: {error}") from error

    def _background_upload_trainer_state_checkpoint(
        self, local_path: Path, remote_path: str, step: int
    ) -> None:
        """Upload complete Trainer state and retain failures for the training thread."""
        try:
            success = self._upload_checkpoint(local_path, remote_path, step)
        except Exception as error:  # noqa: BLE001 - propagate background failures on train end
            self._trainer_state_upload_errors.append(error)
            return
        if not success:
            self._trainer_state_upload_errors.append(
                RuntimeError(f"checkpoint {step} upload to {remote_path} exhausted all retries")
            )

    def on_save(self, args, state, _control, **_kwargs):
        """Upload the checkpoint after Hugging Face Trainer finishes its save callback."""
        # Complete Trainer-state publication must enter the Trainer-owned wait
        # on every rank. In the ZeRO-2 async path, DCP completion may require
        # collectives and therefore cannot be awaited by rank 0 alone.
        if self.upload_trainer_resume_state:
            self._wait_for_async_checkpoint_writes()
        if not state.is_world_process_zero:
            return
        # Background upload failures cannot escape their worker thread. Surface
        # any previous failure on rank 0 before publishing the next checkpoint;
        # on_train_end performs the same check for the final outstanding upload.
        if self.upload_trainer_resume_state:
            self._raise_trainer_state_upload_error()

        t_start = time.time()
        logger.info(
            f"[CP-SAVE][callback=MSCCheckpointCallback] on_save BEGIN at step {state.global_step}"
        )

        checkpoint_dir = self.local_checkpoint_dir / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            message = f"Checkpoint {state.global_step} not found at {checkpoint_dir}"
            if self.upload_trainer_resume_state:
                raise RuntimeError(message)
            logger.error(message)
            return

        remote_checkpoint_path = f"{self.remote_checkpoint_dir}/checkpoint-{state.global_step}"

        is_final_checkpoint = args.max_steps == state.global_step

        if is_final_checkpoint or self.sync_mode:
            # Synchronous: wait for async writes, then upload (must complete before exit)
            if not self.upload_trainer_resume_state:
                self._wait_for_async_checkpoint_writes()
            success = self._upload_checkpoint(
                checkpoint_dir, remote_checkpoint_path, state.global_step
            )
            if not success:
                message = (
                    f"Failed to upload checkpoint {state.global_step} to {remote_checkpoint_path} "
                    f"({'final' if is_final_checkpoint else 'sync'})"
                )
                if self.upload_trainer_resume_state:
                    raise RuntimeError(message)
                logger.error(message)
        else:
            # Asynchronous: dispatch wait + upload in background thread so
            # training resumes immediately without blocking on I/O.
            # Clean up completed threads first
            self.upload_threads = [t for t in self.upload_threads if t.is_alive()]

            # Complete Trainer-state publication already performed its
            # collective wait above; repeating that wait from a rank-0-only
            # background thread would deadlock. Model-only publication retains
            # the existing background safetensors wait.
            upload_target = (
                self._background_upload_trainer_state_checkpoint
                if self.upload_trainer_resume_state
                else self._background_wait_and_upload
            )
            thread = threading.Thread(
                target=upload_target,
                args=(checkpoint_dir, remote_checkpoint_path, state.global_step),
            )
            thread.start()
            self.upload_threads.append(thread)
            logger.info(
                f"MSCCheckpointCallback: step={state.global_step} submitted for background wait+upload "
                f"({len(self.upload_threads)} active threads)"
            )

        duration = time.time() - t_start
        logger.info(
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
                        (PatternType.EXCLUDE, pattern)  # type: ignore[union-attr]
                        for pattern in self._checkpoint_exclude_patterns()
                    ]

                    # Upload the entire directory via msc sync
                    upload_start_time = time.time()
                    msc.sync(  # type: ignore[union-attr]
                        source_url=str(local_path),
                        target_url=remote_path,
                        execution_mode=self.execution_mode,
                        patterns=exclude_patterns,
                        ignore_hidden=not self.upload_trainer_resume_state,
                    )

                    # Create SUCCESS sentinel file directly in remote storage to signal upload completion
                    # This matches the behavior of CheckpointUploadCallback

                    msc.write(  # type: ignore[union-attr]
                        url=f"{remote_path}/SUCCESS",
                        body=b"",  # Empty file serves as marker
                    )
                    logger.debug(f"Created sentinel file at {remote_path}/SUCCESS")
                    upload_duration = time.time() - upload_start_time
                    logger.info(
                        f"Successfully uploaded checkpoint {step} to {remote_path} "
                        f"(upload_time={upload_duration:.2f}s)"
                    )
                except Exception:
                    logger.exception("Failed to upload checkpoint %d to %s", step, remote_path)
                    raise

                # Delete local checkpoint if requested
                if self.delete_after_upload:
                    try:
                        shutil.rmtree(local_path)
                        logger.info(f"Deleted local checkpoint at {local_path}")
                    except OSError as e:
                        logger.warning(f"Failed to delete local checkpoint at {local_path}: {e}")

                return True  # Success

            except Exception as e:  # noqa: BLE001 - storage clients expose backend exceptions
                logger.warning(
                    f"Failed to upload checkpoint {step} (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = min(2 ** (attempt + 1), 60)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.warning(
                        f"All {self.max_retries} attempts failed for checkpoint {step}. "
                        "Training will continue but checkpoint was not uploaded."
                    )

        return False  # All retries failed

    def on_train_end(self, _args, state, _control, **_kwargs):
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

        if self.upload_trainer_resume_state:
            self._raise_trainer_state_upload_error()
