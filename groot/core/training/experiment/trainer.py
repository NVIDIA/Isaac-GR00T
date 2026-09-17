# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import dataclasses
import json as _json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Any, ClassVar

from accelerate.utils import DistributedType
import structlog
import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils._pytree import tree_map
from transformers.trainer import (
    TRAINER_STATE_NAME,
    Trainer,
    TrainerState,
    get_last_checkpoint,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from groot.core.data.types import VLAStepDataId
from groot.core.training.experiment.dataloader import make_dataloader
from groot.core.utils.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from groot.core.utils.timer import ContextTimer

# Optional straggler detection support
try:
    import nvidia_resiliency_ext.straggler as straggler_module

    STRAGGLER_AVAILABLE = True
except ImportError:
    straggler_module = None
    STRAGGLER_AVAILABLE = False


def _parse_shard_size_to_bytes(max_shard_size: str) -> int:
    size_str = max_shard_size.strip().upper()
    units = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    for unit, mult in units.items():
        if size_str.endswith(unit):
            return int(float(size_str[: -len(unit)]) * mult)
    return int(size_str)


def _save_safetensors_subprocess(
    output_dir: str,
    state_dict: dict[str, torch.Tensor],
    rank: int,
    step: int,
    max_shard_size: str = "5GB",
    num_workers: int = 4,
) -> None:
    """Write model safetensors shards in a dedicated process."""
    from safetensors.torch import save_file

    # Keep checkpoint worker lower priority than training workers.
    try:
        os.nice(10)
    except PermissionError:
        print(f"[CP-SAVE][rank={rank}] os.nice(10) not permitted, continuing at default priority")

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError as e:
        print(f"[CP-SAVE][rank={rank}] Could not limit torch threads: {e}")

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    max_bytes = _parse_shard_size_to_bytes(max_shard_size)

    # Build shards without copying tensor storage.
    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0
    total_size = 0

    for key, tensor in state_dict.items():
        tensor_size = int(tensor.numel()) * int(tensor.element_size())
        if current_size + tensor_size > max_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size
        total_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    shard_filenames = [
        f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors" for i in range(num_shards)
    ]
    weight_map: dict[str, str] = {}
    for shard, filename in zip(shards, shard_filenames, strict=True):
        for key in shard.keys():
            weight_map[key] = filename

    print(
        f"[CP-SAVE][rank={rank}] subprocess writing {num_shards} safetensors shards "
        f"with {min(num_workers, num_shards)} workers at step {step}"
    )

    def _write_one_shard(shard_idx: int) -> None:
        shard_path = os.path.join(output_dir, shard_filenames[shard_idx])
        save_file(shards[shard_idx], shard_path)

    with ThreadPoolExecutor(max_workers=max(1, min(num_workers, num_shards))) as pool:
        futures = [pool.submit(_write_one_shard, i) for i in range(num_shards)]
        for future in as_completed(futures):
            future.result()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_file, "w") as f:
        _json.dump(index, f)

    print(
        f"[CP-SAVE][rank={rank}] subprocess safetensors write done at step {step}, "
        f"time={time.time() - t_start:.2f}s"
    )


class ProfCallback(TrainerCallback):
    """Callback to manage PyTorch profiler during training.

    Dynamically starts/stops the profiler within a specified session step window
    to minimize performance overhead. After profiling completes, the callback
    removes itself from the trainer to avoid any callback overhead on remaining steps.

    Note: PyTorch profiler may leave residual CUDA hooks that cause ~15-20% performance
    degradation even after the profiler is stopped. This is a known PyTorch limitation.
    For critical production runs, consider profiling in a separate short warmup job.

    Args:
        profile_dir: Directory to save profile traces
        upload_callback: Optional callback to trigger S3 upload after profiling
        profile_start_step: Session step to start profiling (default: 50)
        profile_end_step: Session step to stop profiling (default: 56)
        warmup_steps: Number of warmup steps for profiler schedule (default: 1)
        active_steps: Number of active profiling steps (default: 5)
        trainer: Trainer instance (required for self-removal after profiling)
        record_shapes: Record tensor shapes in profiler (default: False, lightweight)
        with_stack: Record Python stack traces (default: False, heavy overhead)
        profile_memory: Record memory allocation/deallocation (default: False)
    """

    def __init__(
        self,
        profile_dir,
        upload_callback=None,
        profile_start_step=None,
        profile_end_step=None,
        warmup_steps=1,
        active_steps=5,
        trainer=None,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ):
        self.profile_dir = profile_dir
        self.upload_callback = upload_callback
        self.profile_start_step = profile_start_step if profile_start_step is not None else 50
        self.profile_end_step = profile_end_step if profile_end_step is not None else 56
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.trainer = trainer
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.upload_triggered = False
        self.starting_global_step = None
        self.session_step = 0
        self.prof = None
        self.profiling_active = False
        self.profiling_complete = False
        self.removed_from_trainer = False

    def on_step_begin(self, args, state, _control, **kwargs):
        # Remove callback after upload triggered to eliminate all overhead
        if self.profiling_complete and self.upload_triggered and not self.removed_from_trainer:
            if self.trainer is not None and hasattr(self.trainer, "callback_handler"):
                try:
                    self.trainer.callback_handler.callbacks.remove(self)
                    self.removed_from_trainer = True
                    logging.info(
                        f"Removed ProfCallback from trainer at global step {state.global_step}"
                    )
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to remove ProfCallback: {e}")
            return

        # Early return if profiling already complete to avoid overhead
        if self.profiling_complete:
            return

        # Record starting global step on first call
        if self.starting_global_step is None:
            self.starting_global_step = state.global_step

        # Calculate session step
        self.session_step = state.global_step - self.starting_global_step

        # Start profiler when we reach the profiling window
        if self.session_step == self.profile_start_step and self.prof is None:
            logging.info(
                f"Starting profiler at global step {state.global_step} (session step {self.session_step})"
            )
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1,
                ),
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
            )
            self.prof.__enter__()
            self.profiling_active = True

    def on_step_end(self, args, state, _control, **kwargs):
        # Early return if profiling already complete to avoid overhead
        if self.profiling_complete:
            return

        # Recalculate session_step to ensure accuracy
        if self.starting_global_step is not None:
            self.session_step = state.global_step - self.starting_global_step

        # Step profiler if active
        if self.profiling_active and self.prof is not None:
            self.prof.step()

        # Stop profiler when we reach the end of profiling window
        if self.session_step == self.profile_end_step and self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.profiling_complete = True

            logging.info(
                f"Profiler stopped at global step {state.global_step} (session step {self.session_step})"
            )

            # Trigger upload if callback provided
            if self.upload_callback:
                logging.info(f"Triggering upload at global step {state.global_step}...")
                self.upload_callback()

            # Mark as ready for callback removal
            self.upload_triggered = True


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: int | None = None
) -> torch.Tensor:
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # See transformers/loss/loss_utils.py for reference
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


# Global variables for batched evaluation metrics
_eval_accuracy_accumulated_correct = 0
_eval_accuracy_accumulated_total = 0


def compute_eval_accuracy(
    eval_pred: EvalPrediction, compute_result: bool, action_offset: int | None = None
):
    logits = eval_pred.predictions[0]
    if action_offset is not None:
        logits = logits[..., action_offset:]
    preds = logits.argmax(axis=-1)
    labels = eval_pred.label_ids

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = ((preds == labels) & mask).sum()
    total = mask.sum()

    global _eval_accuracy_accumulated_correct, _eval_accuracy_accumulated_total
    _eval_accuracy_accumulated_correct += correct
    _eval_accuracy_accumulated_total += total

    if compute_result:
        accuracy = _eval_accuracy_accumulated_correct / max(_eval_accuracy_accumulated_total, 1)
        _eval_accuracy_accumulated_correct = 0
        _eval_accuracy_accumulated_total = 0
        return {"eval_accuracy": accuracy}
    else:
        return {}


def _get_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    stable_ratio: float = 0.8,
    decay_type: str = "cosine",
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a Warmup-Stable-Decay (WSD) learning rate schedule.

    Three phases:
        1. **Warmup**: linear ramp from 0 to 1 over ``num_warmup_steps``.
        2. **Stable**: constant at peak LR for ``stable_ratio`` fraction of
           *post-warmup* steps.
        3. **Decay**: decrease from peak to 0 using ``decay_type`` curve.
    """
    post_warmup_steps = num_training_steps - num_warmup_steps
    num_stable_steps = int(post_warmup_steps * stable_ratio)
    num_decay_steps = post_warmup_steps - num_stable_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        steps_after_warmup = current_step - num_warmup_steps
        if steps_after_warmup < num_stable_steps:
            return 1.0
        if num_decay_steps <= 0:
            return 1.0
        decay_progress = float(steps_after_warmup - num_stable_steps) / float(num_decay_steps)
        decay_progress = min(decay_progress, 1.0)
        if decay_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        elif decay_type == "linear":
            return 1.0 - decay_progress
        elif decay_type == "sqrt":
            return 1.0 - math.sqrt(decay_progress)
        else:
            raise ValueError(
                f"Unknown WSD decay type: {decay_type!r}. Use 'cosine', 'linear', or 'sqrt'."
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class DatasetResumeCallback(TrainerCallback):
    """Restore dataset-owned progress after Trainer state has been loaded.

    A complete parent resume keeps Hugging Face ``global_step`` absolute, while
    a child station owns a fresh dataset whose progress starts locally. Subtract
    ``global_step_offset`` for that dataset-only resume; the default of zero
    preserves the existing behavior for every other Trainer.
    """

    def __init__(self, dataset: Any, *, global_step_offset: int = 0) -> None:
        self.dataset = dataset
        self.global_step_offset = int(global_step_offset)
        if self.global_step_offset < 0:
            raise ValueError("Dataset resume global-step offset must be non-negative")

    def on_train_begin(self, args, state, control, **kwargs):
        trainer_global_step = int(state.global_step)
        global_step = trainer_global_step - self.global_step_offset
        if global_step < 0:
            raise ValueError(
                "Trainer global step precedes the dataset resume offset: "
                f"global_step={trainer_global_step}, offset={self.global_step_offset}"
            )
        if global_step <= 0:
            return

        resume_dataset = getattr(self.dataset, "resume_from_global_step", None)
        if callable(resume_dataset):
            resume_dataset(global_step)
            logging.info("Restored dataset at complete global step %d", global_step)
            return

        new_seed = self.dataset.seed + global_step
        self.dataset.reset_seed(new_seed)
        logging.warning(
            "Dataset has no resume_from_global_step(); reset seed to %d. "
            "The resumed experiment will not be reproducible.",
            new_seed,
        )


class GR00TTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        self._data_loading_enable_struct_logs = kwargs.pop("data_loading_enable_struct_logs", False)
        self._data_loading_logging_level = kwargs.pop("data_loading_logging_level", "INFO")
        self._data_loading_log_dir = kwargs.pop("data_loading_log_dir", None)
        self._dataloader_prefetch_factor = kwargs.pop("dataloader_prefetch_factor", None)
        self._training_enable_struct_logs = kwargs.pop("training_enable_struct_logs", False)
        self._training_log_dir = kwargs.pop("training_log_dir", None)
        self._training_logging_level = kwargs.pop("training_logging_level", "INFO")
        dataset_resume_global_step_offset = kwargs.pop("dataset_resume_global_step_offset", 0)
        straggler_config = kwargs.pop("straggler_config", None)
        super().__init__(
            *args,
            **kwargs,
            # compute_metrics=partial(compute_eval_accuracy, action_offset=self.action_offset),
        )
        self.add_callback(
            DatasetResumeCallback(
                self.train_dataset,
                global_step_offset=dataset_resume_global_step_offset,
            )
        )
        if self.args.sync_seq_len:
            assert self.args.seq_len_handle_strategy in (
                "pad",
                "truncate",
                "fixed",
            ), "seq_len_handle_strategy must be 'pad', 'truncate', or 'fixed'"
        if self.args.use_truncated_bptt:
            assert (
                self.args.truncated_bptt_chunk_size > 0
            ), "truncated_bptt_chunk_size must be greater than 0"

        # Performance metrics tracking using ContextTimer (same as BaseTrainer)
        self.timer = ContextTimer(self)

        # structlog is already configured (per-rank JSONL file) in experiment.py
        # before pipeline.setup(); just bind the rank contextvar and announce
        # this actor is ready.
        if self._training_enable_struct_logs and self._training_log_dir is not None:
            structlog.contextvars.bind_contextvars(rank=self.args.process_index)
            structlog.get_logger().info("training_actor_ready", rank=self.args.process_index)
            print(f"Struct log dir: {self._training_log_dir}")

        # Straggler detection config
        self._straggler_initialized = False
        if straggler_config is not None:
            self.straggler_enabled = straggler_config.enabled
            self.straggler_profile_cuda = straggler_config.profile_cuda
            self.straggler_report_time_interval_sec = straggler_config.report_time_interval_sec
            self.straggler_scores_to_compute = straggler_config.scores_to_compute
            self.straggler_gather_on_rank0 = straggler_config.gather_on_rank0
            self.straggler_profiling_interval = straggler_config.profiling_interval
        else:
            self.straggler_enabled = False
            self.straggler_profile_cuda = False

        if self.straggler_enabled and not STRAGGLER_AVAILABLE:
            logging.warning(
                "Straggler detection requested but nvidia_resiliency_ext not installed. "
                "Install with: pip install nvidia-resiliency-ext"
            )
            self.straggler_enabled = False

        self._aux_loss_metric_totals: dict[str, torch.Tensor] = {}
        self._aux_loss_metric_observations: dict[str, int] = {}
        self._last_aux_loss_log_step = 0
        self._aux_loss_accumulated_steps = 0
        self._nan_loss_diagnostics_dump_count = 0

    @staticmethod
    def get_datapoint_ids(batch_samples: list) -> list[VLAStepDataId]:
        """Extract VLAStepDataId entries from a list of collated batches.

        Handles both the BatchFeature(data={"inputs": ...}) shape used by the
        model collators and the flat-dict shape used by BasicDataCollator.
        """
        ids: list[VLAStepDataId] = []
        for batch in batch_samples:
            container = batch.get("inputs", batch)
            for v in container.get(VLAStepDataId.STEP_DATA_ID_KEY, []) or []:
                if isinstance(v, VLAStepDataId):
                    ids.append(v)
        return ids

    def _scalar_output_to_tensor(self, value: Any, device: torch.device) -> torch.Tensor | None:
        """Convert scalar model outputs to detached tensors for cadence-based logging."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return value.detach().to(device=device, dtype=torch.float32).reshape(())
        if isinstance(value, float | int):
            return torch.tensor(float(value), device=device, dtype=torch.float32)
        return None

    @staticmethod
    def _is_aux_loss_aggregation_key(key: str) -> bool:
        return key.startswith("_") or key.endswith("_loss_sum") or key.endswith("_loss_count")

    def _log_rank0_scalar_outputs(self, outputs: Any) -> None:
        """Log local rank-0 scalar outputs when distributed aux logging is disabled."""
        if getattr(self.args, "log_distributed_aux_losses", False):
            return
        if self.args.local_rank not in (-1, 0) or not hasattr(outputs, "items"):
            return

        log_dict = {}
        for key, value in outputs.items():
            if self._is_aux_loss_aggregation_key(key):
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                log_dict[f"train/{key}"] = value.item()
            elif isinstance(value, float | int):
                log_dict[f"train/{key}"] = float(value)

        if log_dict:
            self.log(log_dict)

    @staticmethod
    def _tensor_is_nonfinite(tensor: torch.Tensor) -> bool:
        if not (torch.is_floating_point(tensor) or tensor.is_complex()):
            return False
        detached = tensor.detach()
        if detached.numel() == 0:
            return False
        return not bool(torch.isfinite(detached).all().item())

    @staticmethod
    def _nonfinite_count(tensor: torch.Tensor, *, chunk_size: int = 1_000_000) -> int:
        if not (torch.is_floating_point(tensor) or tensor.is_complex()):
            return 0
        flat = tensor.detach().reshape(-1)
        total = 0
        for offset in range(0, flat.numel(), chunk_size):
            chunk = flat[offset : offset + chunk_size]
            total += int((~torch.isfinite(chunk)).sum().item())
        return total

    @staticmethod
    def _first_nonfinite_flat_indices(
        tensor: torch.Tensor,
        *,
        max_indices: int = 8,
        chunk_size: int = 1_000_000,
    ) -> list[int]:
        if max_indices <= 0 or not (torch.is_floating_point(tensor) or tensor.is_complex()):
            return []

        flat = tensor.detach().reshape(-1)
        found: list[int] = []
        for offset in range(0, flat.numel(), chunk_size):
            chunk = flat[offset : offset + chunk_size]
            bad_mask = ~torch.isfinite(chunk)
            if bool(bad_mask.any().item()):
                local_indices = bad_mask.nonzero(as_tuple=False).flatten()
                needed = max_indices - len(found)
                found.extend((local_indices[:needed] + offset).detach().cpu().tolist())
                if len(found) >= max_indices:
                    break
        return [int(idx) for idx in found]

    @staticmethod
    def _unravel_flat_index(flat_index: int, shape: list[int]) -> list[int]:
        if not shape:
            return []
        remaining = int(flat_index)
        coords: list[int] = []
        for size in reversed(shape):
            if size <= 0:
                coords.append(0)
                continue
            coords.append(remaining % size)
            remaining //= size
        return list(reversed(coords))

    @staticmethod
    def _safe_float(value: torch.Tensor) -> float | str | None:
        try:
            out = float(value.detach().float().cpu().item())
            if math.isnan(out):
                return "nan"
            if math.isinf(out):
                return "inf" if out > 0 else "-inf"
            return out
        except (RuntimeError, TypeError, ValueError):
            return None

    def _summarize_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        include_stats: bool = True,
        max_indices: int = 8,
    ) -> dict[str, Any]:
        detached = tensor.detach()
        shape = list(detached.shape)
        summary: dict[str, Any] = {
            "name": name,
            "type": "tensor",
            "shape": shape,
            "dtype": str(detached.dtype).removeprefix("torch."),
            "device": str(detached.device),
            "requires_grad": bool(getattr(tensor, "requires_grad", False)),
            "numel": int(detached.numel()),
        }

        if detached.numel() == 0:
            return summary

        if torch.is_floating_point(detached) or detached.is_complex():
            nonfinite_count = self._nonfinite_count(detached)
            summary["nonfinite_count"] = nonfinite_count
            if nonfinite_count:
                flat_indices = self._first_nonfinite_flat_indices(
                    detached,
                    max_indices=max_indices,
                )
                summary["first_nonfinite_flat_indices"] = flat_indices
                summary["first_nonfinite_indices"] = [
                    self._unravel_flat_index(idx, shape) for idx in flat_indices
                ]
            if include_stats:
                finite_count = int(detached.numel() - nonfinite_count)
                summary["finite_count"] = finite_count
                if finite_count:
                    if nonfinite_count == 0:
                        finite_values = detached
                    elif detached.numel() <= 10_000_000:
                        finite = torch.isfinite(detached)
                        finite_values = detached[finite]
                    else:
                        summary["stats_skipped"] = "tensor_too_large_with_nonfinite_values"
                        return summary
                    if finite_values.is_complex():
                        stat_values = finite_values.abs().float()
                        summary["stat"] = "abs"
                    else:
                        stat_values = finite_values.float()
                    summary["min"] = self._safe_float(stat_values.min())
                    summary["max"] = self._safe_float(stat_values.max())
                    summary["mean"] = self._safe_float(stat_values.mean())
                    summary["std"] = self._safe_float(stat_values.std(unbiased=False))
            return summary

        if detached.dtype == torch.bool:
            summary["true_count"] = int(detached.sum().item())
            summary["false_count"] = int(detached.numel() - detached.sum().item())
            summary["nonfinite_count"] = 0
            return summary

        if include_stats:
            summary["min"] = self._safe_float(detached.min())
            summary["max"] = self._safe_float(detached.max())
        summary["nonfinite_count"] = 0
        return summary

    def _iter_named_tensors(
        self,
        value: Any,
        *,
        prefix: str = "",
        max_depth: int = 6,
    ):
        if isinstance(value, torch.Tensor):
            yield prefix or "tensor", value
            return

        if max_depth <= 0:
            return

        if hasattr(value, "data") and isinstance(value.data, Mapping):
            value = value.data

        if isinstance(value, Mapping):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._iter_named_tensors(
                    child,
                    prefix=child_prefix,
                    max_depth=max_depth - 1,
                )
            return

        if isinstance(value, list | tuple):
            for idx, child in enumerate(value[:64]):
                child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                yield from self._iter_named_tensors(
                    child,
                    prefix=child_prefix,
                    max_depth=max_depth - 1,
                )

    def _summarize_named_tensors(
        self,
        value: Any,
        *,
        max_tensors: int = 128,
        include_stats: bool = True,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for name, tensor in self._iter_named_tensors(value):
            summaries.append(
                self._summarize_tensor(
                    name,
                    tensor,
                    include_stats=include_stats,
                )
            )
            if len(summaries) >= max_tensors:
                break
        return summaries

    def _summarize_parameter_finiteness(
        self,
        model: nn.Module,
        *,
        include_gradients: bool,
        max_entries: int = 32,
    ) -> dict[str, Any]:
        total_params = 0
        tensors_with_nonfinite_params = 0
        nonfinite_param_elements = 0
        first_nonfinite_params: list[dict[str, Any]] = []
        total_grad_tensors = 0
        tensors_with_nonfinite_grads = 0
        nonfinite_grad_elements = 0
        first_nonfinite_grads: list[dict[str, Any]] = []

        for name, param in model.named_parameters():
            total_params += int(param.numel())
            param_data = param.detach()
            if torch.is_floating_point(param_data) or param_data.is_complex():
                bad_param_count = self._nonfinite_count(param_data)
                if bad_param_count:
                    tensors_with_nonfinite_params += 1
                    nonfinite_param_elements += bad_param_count
                    if len(first_nonfinite_params) < max_entries:
                        first_nonfinite_params.append(
                            self._summarize_tensor(
                                name,
                                param_data,
                                include_stats=False,
                            )
                        )

            if include_gradients and param.grad is not None:
                total_grad_tensors += 1
                grad = param.grad.detach()
                if torch.is_floating_point(grad) or grad.is_complex():
                    bad_grad_count = self._nonfinite_count(grad)
                    if bad_grad_count:
                        tensors_with_nonfinite_grads += 1
                        nonfinite_grad_elements += bad_grad_count
                        if len(first_nonfinite_grads) < max_entries:
                            first_nonfinite_grads.append(
                                self._summarize_tensor(
                                    name,
                                    grad,
                                    include_stats=False,
                                )
                            )

        summary: dict[str, Any] = {
            "total_parameter_elements": total_params,
            "tensors_with_nonfinite_parameters": tensors_with_nonfinite_params,
            "nonfinite_parameter_elements": nonfinite_param_elements,
            "first_nonfinite_parameters": first_nonfinite_params,
        }
        if include_gradients:
            summary.update(
                {
                    "total_gradient_tensors": total_grad_tensors,
                    "tensors_with_nonfinite_gradients": tensors_with_nonfinite_grads,
                    "nonfinite_gradient_elements": nonfinite_grad_elements,
                    "first_nonfinite_gradients": first_nonfinite_grads,
                }
            )
        return summary

    def _nan_loss_diagnostics_path(self) -> Path:
        configured_dir = getattr(self.args, "nan_loss_diagnostics_dir", None)
        if configured_dir:
            diagnostics_dir = Path(configured_dir)
        else:
            diagnostics_dir = Path(self.args.output_dir) / "nan_loss_diagnostics"
        process_index = int(getattr(self.args, "process_index", 0))
        return diagnostics_dir / f"nonfinite_loss_rank{process_index}.jsonl"

    def _maybe_dump_nonfinite_loss_diagnostics(
        self,
        *,
        stage: str,
        model: nn.Module,
        inputs: Any,
        outputs: Any | None,
        loss: torch.Tensor,
        include_inputs: bool,
        include_outputs: bool,
        include_parameters: bool,
        include_gradients: bool,
    ) -> None:
        if not getattr(self.args, "enable_nan_loss_diagnostics", False):
            return
        if not isinstance(loss, torch.Tensor) or not self._tensor_is_nonfinite(loss):
            return

        dump_count = int(getattr(self, "_nan_loss_diagnostics_dump_count", 0))
        max_dumps = int(getattr(self.args, "nan_loss_diagnostics_max_dumps", 4) or 0)
        if max_dumps <= 0 or dump_count >= max_dumps:
            return
        self._nan_loss_diagnostics_dump_count = dump_count + 1

        try:
            path = self._nan_loss_diagnostics_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            process_index = int(getattr(self.args, "process_index", 0))
            local_rank = int(getattr(self.args, "local_rank", -1))
            payload: dict[str, Any] = {
                "schema_version": 1,
                "event": "nonfinite_loss",
                "stage": stage,
                "time_unix": time.time(),
                "global_step": int(getattr(self.state, "global_step", -1)),
                "process_index": process_index,
                "local_rank": local_rank,
                "world_size": int(getattr(self.args, "world_size", 1)),
                "loss": self._summarize_tensor("loss", loss),
            }
            if include_inputs:
                payload["inputs"] = self._summarize_named_tensors(inputs)
            if include_outputs and outputs is not None:
                payload["outputs"] = self._summarize_named_tensors(outputs)
            if include_parameters:
                payload["parameters"] = self._summarize_parameter_finiteness(
                    model,
                    include_gradients=include_gradients,
                )
            with open(path, "a") as f:
                f.write(_json.dumps(payload, sort_keys=True) + "\n")
            logging.error(
                "Non-finite loss diagnostics dumped at step %s rank %s stage %s to %s",
                payload["global_step"],
                process_index,
                stage,
                path,
            )
        except Exception:
            logging.exception("Failed to dump non-finite loss diagnostics")

    def _collect_nonfinite_scalar_outputs(self, outputs: Any) -> list[dict[str, Any]]:
        """Find scalar model outputs like action_loss that are NaN/Inf."""
        if not hasattr(outputs, "items"):
            return []

        nonfinite_outputs: list[dict[str, Any]] = []
        for key, value in outputs.items():
            if self._is_aux_loss_aggregation_key(key):
                continue

            if isinstance(value, torch.Tensor):
                if value.numel() != 1 or not self._tensor_is_nonfinite(value):
                    continue
                nonfinite_outputs.append(
                    self._summarize_tensor(
                        str(key),
                        value,
                        include_stats=True,
                    )
                )
                continue

            if isinstance(value, float | int):
                scalar = float(value)
                if math.isfinite(scalar):
                    continue
                nonfinite_outputs.append(
                    {
                        "name": str(key),
                        "type": type(value).__name__,
                        "value": "nan" if math.isnan(scalar) else ("inf" if scalar > 0 else "-inf"),
                    }
                )

        return nonfinite_outputs

    def _maybe_dump_nonfinite_model_output_diagnostics(
        self,
        *,
        stage: str,
        model: nn.Module,
        inputs: Any,
        outputs: Any,
    ) -> None:
        if not getattr(self.args, "enable_nan_loss_diagnostics", False):
            return

        nonfinite_outputs = self._collect_nonfinite_scalar_outputs(outputs)
        if not nonfinite_outputs:
            return

        dump_count = int(getattr(self, "_nan_loss_diagnostics_dump_count", 0))
        max_dumps = int(getattr(self.args, "nan_loss_diagnostics_max_dumps", 4) or 0)
        if max_dumps <= 0 or dump_count >= max_dumps:
            return
        self._nan_loss_diagnostics_dump_count = dump_count + 1

        try:
            path = self._nan_loss_diagnostics_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            process_index = int(getattr(self.args, "process_index", 0))
            local_rank = int(getattr(self.args, "local_rank", -1))
            payload: dict[str, Any] = {
                "schema_version": 1,
                "event": "nonfinite_model_output",
                "stage": stage,
                "time_unix": time.time(),
                "global_step": int(getattr(self.state, "global_step", -1)),
                "process_index": process_index,
                "local_rank": local_rank,
                "world_size": int(getattr(self.args, "world_size", 1)),
                "nonfinite_outputs": nonfinite_outputs,
                "inputs": self._summarize_named_tensors(inputs),
                "outputs": self._summarize_named_tensors(outputs),
                "parameters": self._summarize_parameter_finiteness(
                    model,
                    include_gradients=False,
                ),
            }
            with open(path, "a") as f:
                f.write(_json.dumps(payload, sort_keys=True) + "\n")
            logging.error(
                "Non-finite model output diagnostics dumped at step %s rank %s stage %s to %s",
                payload["global_step"],
                process_index,
                stage,
                path,
            )
        except Exception:
            logging.exception("Failed to dump non-finite model output diagnostics")

    def _add_aux_loss_metric(self, name: str, value: torch.Tensor) -> None:
        value = value.detach()
        with torch.no_grad():
            if name not in self._aux_loss_metric_totals:
                self._aux_loss_metric_totals[name] = torch.zeros_like(value)
                self._aux_loss_metric_observations[name] = 0
            self._aux_loss_metric_totals[name].add_(value)
            self._aux_loss_metric_observations[name] += 1

    def _accumulate_aux_loss_metrics(self, outputs: Any, device: torch.device) -> None:
        """Accumulate scalar outputs locally without per-step host syncs or collectives."""
        if not hasattr(outputs, "items"):
            return

        added_metric = False
        for key, value in outputs.items():
            if self._is_aux_loss_aggregation_key(key):
                continue

            scalar = self._scalar_output_to_tensor(value, device)
            if scalar is None:
                continue

            self._add_aux_loss_metric(key, scalar)
            added_metric = True

        if added_metric:
            self._aux_loss_accumulated_steps += 1

    @staticmethod
    def _resolve_aux_loss_metric_names(local_names: list[str]) -> list[str]:
        if dist.is_available() and dist.is_initialized():
            gathered_names: list[list[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_names, local_names)
            return sorted(
                {
                    metric_name
                    for rank_names in gathered_names
                    if rank_names is not None
                    for metric_name in rank_names
                }
            )
        return local_names

    @staticmethod
    def _distributed_mean_(tensor: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            # Gloo (CPU backend, used in CPU dry-runs) does not implement
            # ReduceOp.AVG. NCCL does. Use SUM + manual divide on Gloo.
            if dist.get_backend() == "gloo":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= dist.get_world_size()
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        return tensor

    def _maybe_build_global_training_metrics(
        self,
        device: torch.device,
        extra_metrics: dict[str, torch.Tensor | float] | None = None,
    ) -> dict[str, float]:
        """Build one packed distributed average for train_global/* metrics."""
        logging_steps = int(getattr(self.args, "logging_steps", 0) or 0)
        should_flush_aux_losses = False
        step = self._aux_loss_accumulated_steps

        if logging_steps > 0 and getattr(self.args, "log_distributed_aux_losses", False):
            gradient_accumulation_steps = int(
                getattr(self.args, "gradient_accumulation_steps", 1) or 1
            )
            log_every_accumulations = logging_steps * max(gradient_accumulation_steps, 1)
            should_flush_aux_losses = (
                step > self._last_aux_loss_log_step and step % log_every_accumulations == 0
            )

        local_means: dict[str, torch.Tensor] = {}
        if should_flush_aux_losses:
            for name, total in self._aux_loss_metric_totals.items():
                observations = self._aux_loss_metric_observations.get(name, 0)
                if observations <= 0:
                    continue
                local_means[name] = total.to(device=device, dtype=torch.float32) / observations

        for name, value in (extra_metrics or {}).items():
            scalar = self._scalar_output_to_tensor(value, device)
            if scalar is not None:
                local_means[name] = scalar

        metric_names = self._resolve_aux_loss_metric_names(sorted(local_means))
        if not metric_names:
            if should_flush_aux_losses:
                self._aux_loss_metric_totals.clear()
                self._aux_loss_metric_observations.clear()
                self._last_aux_loss_log_step = step
            return {}

        zero = torch.zeros((), device=device, dtype=torch.float32)
        packed = torch.stack(
            [
                local_means.get(name, zero).to(device=device, dtype=torch.float32)
                for name in metric_names
            ]
        )
        self._distributed_mean_(packed)

        metrics = {
            f"train_global/{name}": packed[idx].item() for idx, name in enumerate(metric_names)
        }

        if should_flush_aux_losses:
            self._aux_loss_metric_totals.clear()
            self._aux_loss_metric_observations.clear()
            self._last_aux_loss_log_step = step

        return metrics if self.is_world_process_zero() else {}

    def _maybe_log_aux_loss_metrics(self, device: torch.device) -> None:
        """Log local-window aux means with one packed distributed average."""
        metrics = self._maybe_build_global_training_metrics(device)
        if metrics:
            self.log(metrics)

    def _straggler_section(self, name: str, profile_cuda: bool = False):
        """Context manager for straggler detection sections.

        If straggler detection is disabled or not initialized, returns a no-op context manager.

        Args:
            name: Name of the detection section (e.g., "data_loading", "model_forward")
            profile_cuda: Whether to profile CUDA kernels in this section
        """
        if self._straggler_initialized and STRAGGLER_AVAILABLE and straggler_module is not None:
            return straggler_module.Detector.detection_section(name, profile_cuda=profile_cuda)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext

            return nullcontext()

    def _initialize_straggler_detector(self):
        """Initialize the straggler detector for distributed training monitoring."""
        if self._straggler_initialized:
            return

        try:
            if straggler_module is None:
                raise RuntimeError("straggler_module is None")
            straggler_module.Detector.initialize(
                scores_to_compute=self.straggler_scores_to_compute,
                gather_on_rank0=self.straggler_gather_on_rank0,
                report_time_interval=self.straggler_report_time_interval_sec,
                profiling_interval=self.straggler_profiling_interval,
            )
            self._straggler_initialized = True
            logging.info("Straggler detection initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize straggler detection: {e}")
            self._straggler_initialized = False

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create the LR scheduler.

        When ``_use_wsd_scheduler`` is set on the training arguments, build a
        custom Warmup-Stable-Decay schedule. Otherwise, fall through to
        HuggingFace's default ``create_scheduler``.
        """
        if getattr(self.args, "_use_wsd_scheduler", False):
            if self.lr_scheduler is None:
                if optimizer is None:
                    optimizer = self.optimizer
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                self.lr_scheduler = _get_wsd_schedule(
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    stable_ratio=self.args.wsd_stable_ratio,
                    decay_type=self.args.wsd_decay_type,
                )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    _NO_DECAY_KEYWORDS: ClassVar[set[str]] = {"gating_alpha", "register_tokens"}

    def get_decay_parameter_names(self, model) -> list[str]:
        decay_parameters = super().get_decay_parameter_names(model)
        return [n for n in decay_parameters if not any(kw in n for kw in self._NO_DECAY_KEYWORDS)]

    def create_optimizer(self):
        """Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.args.use_muon:
            return self.create_optimizer_muon()
        else:
            return super().create_optimizer()

    def create_optimizer_muon(self):
        """Setup the optimizer using the Muon example pattern.

        Creates specific parameter groups for different types of parameters:
        - hidden_matrix_params: matrix parameters from model blocks (ndim >= 2, not embedding)
        - embed_params: embedding parameters
        - scalar_params: parameters with ndim < 2
        - head_params: language model head parameters
        """
        opt_model = self.model
        muon_lr = self.args.muon_lr

        # Get optimizer kwargs for Adam groups
        _, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

        # Filter out keys that MuonWithAuxAdam doesn't expect (like 'fused')
        filtered_optimizer_kwargs = {
            k: v for k, v in optimizer_kwargs.items() if k in ["lr", "betas", "eps", "weight_decay"]
        }

        if self.optimizer is None:
            # Get decay parameter names to determine which parameters should use weight decay
            decay_parameters = self.get_decay_parameter_names(opt_model)

            # Group parameters: hidden matrix params for Muon, everything else for Adam
            hidden_matrix_params_with_decay = []
            hidden_matrix_params_no_decay = []
            adam_params_with_decay = []
            adam_params_no_decay = []

            # Split parameters by type and decay status
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue

                should_decay = n in decay_parameters

                if p.ndim >= 2 and "embed" not in n and "lm_head" not in n:
                    # Hidden matrix parameters - use Muon
                    print("using muon for hidden matrix params", n)
                    if should_decay:
                        hidden_matrix_params_with_decay.append(p)
                    else:
                        hidden_matrix_params_no_decay.append(p)
                else:
                    print("using adam for non-hidden matrix params", n)
                    # Everything else (embed, scalar, head) - use Adam
                    if should_decay:
                        adam_params_with_decay.append(p)
                    else:
                        adam_params_no_decay.append(p)

            # Create Adam groups for non-matrix parameters (embed, scalar, head combined)
            adam_groups = []
            if adam_params_with_decay:
                adam_groups.append(
                    {"params": adam_params_with_decay, "weight_decay": self.args.weight_decay}
                )
            if adam_params_no_decay:
                adam_groups.append({"params": adam_params_no_decay, "weight_decay": 0.0})

            # Add common Adam parameters from filtered optimizer_kwargs
            adam_groups = [
                dict(**g, **filtered_optimizer_kwargs, use_muon=False) for g in adam_groups
            ]

            # Create Muon groups for hidden matrix params (with and without decay)
            muon_groups = []
            if hidden_matrix_params_with_decay:
                muon_groups.append(
                    {
                        "params": hidden_matrix_params_with_decay,
                        "lr": muon_lr,
                        "weight_decay": self.args.weight_decay,
                        "use_muon": True,
                    }
                )
            if hidden_matrix_params_no_decay:
                muon_groups.append(
                    {
                        "params": hidden_matrix_params_no_decay,
                        "lr": muon_lr,
                        "weight_decay": 0.0,
                        "use_muon": True,
                    }
                )

            # Combine all parameter groups
            param_groups = [*adam_groups, *muon_groups]

            # Create the optimizer using MuonWithAuxAdam
            if self.args.muon_impl == "single_batched":
                self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups, use_batching=True)
            elif self.args.muon_impl == "dist":
                self.optimizer = MuonWithAuxAdam(param_groups)
            else:
                raise ValueError(
                    f"Unknown muon_impl: {self.args.muon_impl}. Expected 'single_batched' or 'dist'."
                )

        return self.optimizer

    # ------------------------------------------------------------------
    # Async checkpoint save + DCP resume
    # ------------------------------------------------------------------

    def _get_zero_stage(self) -> int:
        """Read ZeRO optimization stage from the DeepSpeed engine config at runtime.

        Returns the ZeRO stage (0, 1, 2, or 3), or -1 if not available.
        This avoids needing a separate config field that could drift from the actual
        DeepSpeed configuration.
        """
        try:
            ds_config = getattr(self.deepspeed, "config", None) or {}
            return int(ds_config.get("zero_optimization", {}).get("stage", -1))
        except Exception:
            return -1

    def _init_dcp_async(self):
        """Lazy-initialize DCP async checkpointing with a dedicated Gloo process group."""
        if hasattr(self, "_dcp_initialized") and self._dcp_initialized:
            return True
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict_saver import async_save

            self._dcp = dcp
            self._dcp_async_save = async_save
            self._pending_dcp_future = None

            # Use process-based async to avoid GIL contention with training thread
            try:
                from torch.distributed.checkpoint.state_dict_saver import (
                    AsyncCheckpointerType,
                )

                self._async_checkpointer_type = AsyncCheckpointerType.PROCESS
                print("[CP-SAVE] Using AsyncCheckpointerType.PROCESS (no GIL contention)")
            except (ImportError, AttributeError):
                self._async_checkpointer_type = None
                print(
                    "[CP-SAVE] AsyncCheckpointerType.PROCESS not available, using default (thread)"
                )

            # Create a dedicated Gloo process group for DCP async save.
            # This isolates DCP's CPU-side coordination from the default process group,
            # preventing conflicts with straggler detection or other Gloo users.
            if dist.is_initialized():
                self._dcp_process_group = dist.new_group(backend="gloo")
                print(
                    "[CP-SAVE] DCP async checkpoint initialized with dedicated Gloo process group"
                )
            else:
                self._dcp_process_group = None
                print("[CP-SAVE] DCP async checkpoint initialized (no dist, no dedicated PG)")

            self._dcp_initialized = True
            return True
        except ImportError as e:
            print(
                f"[CP-SAVE] WARNING: DCP not available ({e}), falling back to synchronous DeepSpeed save"
            )
            self._dcp_initialized = False
            return False

    def _wait_for_pending_dcp_save(self, *, raise_on_error: bool = False):
        """Wait for any in-flight DCP async save to complete."""
        if not hasattr(self, "_pending_dcp_future") or self._pending_dcp_future is None:
            return
        try:
            t_start = time.time()
            self._pending_dcp_future.result()
            duration = time.time() - t_start
            print(f"[CP-SAVE] Waited {duration:.2f}s for previous DCP async save to complete")
        except Exception as e:
            print(f"[CP-SAVE] WARNING: Previous DCP async save failed: {e}")
            if raise_on_error:
                raise
        finally:
            self._pending_dcp_future = None

    def _wait_for_pending_safetensors_worker(self, *, raise_on_timeout: bool = False):
        """Wait for any in-flight background safetensors worker to complete."""
        worker = getattr(self, "_safetensors_worker", None)
        if worker is None:
            return
        timed_out = False
        try:
            t_start = time.time()
            if hasattr(worker, "exitcode"):
                worker.join(timeout=1800)  # 30 min max wait
                if worker.is_alive():
                    timed_out = True
                    message = "Background safetensors process still alive after 1800s timeout"
                    print(f"[CP-SAVE] WARNING: {message}")
                    if raise_on_timeout:
                        raise TimeoutError(message)
                elif worker.exitcode != 0:
                    raise RuntimeError(
                        f"Background safetensors process failed with exit code {worker.exitcode}"
                    )
                else:
                    duration = time.time() - t_start
                    print(
                        f"[CP-SAVE] Waited {duration:.2f}s for previous safetensors background process"
                    )
            else:
                worker.join(timeout=1800)
                if worker.is_alive():
                    timed_out = True
                    message = "Background safetensors thread still alive after 1800s timeout"
                    print(f"[CP-SAVE] WARNING: {message}")
                    if raise_on_timeout:
                        raise TimeoutError(message)
                else:
                    duration = time.time() - t_start
                    print(
                        f"[CP-SAVE] Waited {duration:.2f}s for previous safetensors background thread"
                    )
        except Exception as e:
            print(f"[CP-SAVE] WARNING: Safetensors background worker error: {e}")
            raise
        finally:
            # Retain the worker only for strict complete-checkpoint publication.
            # The default model-only path preserves its previous cleanup behavior.
            if not timed_out or not raise_on_timeout:
                self._safetensors_worker = None

    def _wait_for_pending_trainer_checkpoint_writes(self):
        """Collectively finish all files required for exact Trainer resume.

        MSC upload runs on world rank 0, while ZeRO optimizer shards and RNG
        files are produced by every rank. All ranks must enter this method so a
        DCP future that requires collectives can finish before rank 0 publishes
        the checkpoint and its SUCCESS sentinel.
        """
        failures: list[str] = []
        try:
            self._wait_for_pending_safetensors_worker(raise_on_timeout=True)
        except Exception as exc:
            failures.append(f"safetensors: {exc}")

        optimizer_thread = getattr(self, "_pending_opt_pt_thread", None)
        if optimizer_thread is not None:
            try:
                optimizer_thread.join()
            except Exception as exc:
                # Keep every rank on the shared failure-gather path even if a
                # local writer handle itself fails while joining.
                failures.append(f"optimizer shard: {exc}")
            finally:
                self._pending_opt_pt_thread = None
        optimizer_error = getattr(self, "_pending_opt_pt_error", None)
        self._pending_opt_pt_error = None
        if optimizer_error is not None:
            failures.append(f"optimizer shard: {optimizer_error}")

        try:
            self._wait_for_pending_dcp_save(raise_on_error=True)
        except Exception as exc:
            failures.append(f"DCP state: {exc}")

        local_failure = "; ".join(failures) or None
        if dist.is_initialized():
            rank_failures: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(rank_failures, local_failure)
            failures = [failure for failure in rank_failures if failure is not None]

        if failures:
            raise RuntimeError(
                "Trainer checkpoint writes did not complete on every rank: " + " | ".join(failures)
            )

    @contextmanager
    def _patch_deepspeed_load_for_dcp(self, dcp_checkpoint_dir: str):
        """Context manager that temporarily replaces deepspeed_load_checkpoint with DCP loading.

        This intercepts the exact call site inside HF Trainer's _inner_training_loop where
        deepspeed_load_checkpoint is invoked (after DeepSpeed engine init, before first step).
        The original function is always restored on exit.
        """
        import transformers.integrations.deepspeed as ds_module
        import transformers.trainer as trainer_module

        original_ds_load = ds_module.deepspeed_load_checkpoint
        original_trainer_load = getattr(trainer_module, "deepspeed_load_checkpoint", None)
        _trainer_ref = self

        def _dcp_load_replacement(_deepspeed_engine, _checkpoint_path, **_kw):
            print(
                f"[CP-SAVE] Replacing deepspeed_load_checkpoint with DCP load "
                f"from {dcp_checkpoint_dir}"
            )
            _trainer_ref._resume_from_dcp_checkpoint(dcp_checkpoint_dir)

        ds_module.deepspeed_load_checkpoint = _dcp_load_replacement
        if original_trainer_load is not None:
            trainer_module.deepspeed_load_checkpoint = _dcp_load_replacement
        try:
            yield
        finally:
            ds_module.deepspeed_load_checkpoint = original_ds_load
            if original_trainer_load is not None:
                trainer_module.deepspeed_load_checkpoint = original_trainer_load

    def _prewarm_pinned_offload_buffer(self, rank: int):
        """Pre-allocate pinned CPU buffer for state_dict offload so first checkpoint is fast."""
        if not torch.cuda.is_available():
            return
        if getattr(self, "_cp_cpu_offload_state_dict", None) is not None:
            return  # Already warmed
        try:
            from torch.distributed._state_dict_utils import _create_cpu_state_dict
        except Exception:
            return

        t_alloc = time.time()
        # state_dict() returns references to existing GPU tensors (no copy) —
        # we only need the shapes/dtypes to allocate the pinned buffer.
        state_dict = self.model.state_dict()
        self._cp_cpu_offload_state_dict = _create_cpu_state_dict(
            state_dict, pin_memory=True, share_memory=True
        )
        del state_dict
        self._cp_offload_stream = torch.cuda.Stream()
        print(
            f"[CP-SAVE][rank={rank}] pinned offload buffer pre-warmed in "
            f"{time.time() - t_alloc:.2f}s"
        )

    def _offload_state_dict_to_pinned_cpu(
        self, state_dict: dict[str, torch.Tensor], rank: int
    ) -> dict[str, torch.Tensor]:
        """Offload tensors to pinned CPU memory using non-blocking copies."""
        if not torch.cuda.is_available():
            return {k: v.cpu() for k, v in state_dict.items()}
        try:
            from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
        except Exception:
            # Private APIs can differ across torch versions; keep a safe fallback.
            return {k: v.cpu() for k, v in state_dict.items()}

        if not hasattr(self, "_cp_offload_stream") or self._cp_offload_stream is None:
            self._cp_offload_stream = torch.cuda.Stream()
        if (
            not hasattr(self, "_cp_cpu_offload_state_dict")
            or self._cp_cpu_offload_state_dict is None
        ):
            t_alloc = time.time()
            self._cp_cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )
            print(
                f"[CP-SAVE][rank={rank}] pinned buffer alloc done in "
                f"{time.time() - t_alloc:.2f}s"
            )

        with torch.cuda.stream(self._cp_offload_stream):
            self._cp_cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self._cp_cpu_offload_state_dict,
                non_blocking=True,
            )
        self._cp_offload_stream.synchronize()
        return self._cp_cpu_offload_state_dict

    def save_model(self, output_dir=None, _internal_call=False, **kwargs):
        """Override to write safetensors asynchronously in a dedicated subprocess."""
        rank = self.args.local_rank if self.args.local_rank != -1 else 0
        step = self.state.global_step
        use_async = getattr(self.args, "use_async_checkpoint_save", True)

        if use_async and _internal_call and self.is_deepspeed_enabled:
            print(f"[CP-SAVE][rank={rank}] save_model ASYNC BEGIN at step {step}")
            t_start = time.time()

            try:
                # Wait for any previous background safetensors worker
                self._wait_for_pending_safetensors_worker()

                use_local_state_dict = (
                    bool(getattr(self.args, "use_local_state_dict_for_zero2", True))
                    and self._get_zero_stage() == 2
                )
                state_dict = None
                if use_local_state_dict:
                    # ZeRO-2 keeps full model weights replicated per rank. Avoid
                    # collective gathering and only offload to CPU on saving rank.
                    if self.args.should_save:
                        t_local = time.time()
                        local_state_dict = self.model.state_dict()
                        print(
                            f"[CP-SAVE][rank={rank}] model.state_dict() local extract done in "
                            f"{time.time() - t_local:.2f}s"
                        )
                        t_cpu = time.time()
                        state_dict = self._offload_state_dict_to_pinned_cpu(local_state_dict, rank)
                        print(
                            f"[CP-SAVE][rank={rank}] model.state_dict() CPU offload done in "
                            f"{time.time() - t_cpu:.2f}s"
                        )
                        del local_state_dict
                else:
                    # Fallback for non-ZeRO2 or explicit override.
                    state_dict = self.accelerator.get_state_dict(self.deepspeed)
                    t_collective = time.time() - t_start
                    print(
                        f"[CP-SAVE][rank={rank}] get_state_dict (collective) done in {t_collective:.2f}s"
                    )

                if output_dir is None:
                    output_dir = self.args.output_dir
                os.makedirs(output_dir, exist_ok=True)

                if self.args.should_save:
                    if state_dict is None:
                        raise RuntimeError("state_dict is None on saving rank")
                    # Save config.json synchronously (tiny, needed by get_last_checkpoint)
                    if hasattr(self.model, "config") and hasattr(
                        self.model.config, "save_pretrained"
                    ):
                        self.model.config.save_pretrained(output_dir)

                    # Save training_args.bin (normally written by super().save_model)
                    try:
                        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    except Exception:
                        pass

                    ctx = multiprocessing.get_context(self.multiprocessing_context)
                    self._safetensors_worker = ctx.Process(
                        target=_save_safetensors_subprocess,
                        args=(output_dir, state_dict, rank, step),
                        kwargs={"max_shard_size": "5GB", "num_workers": 4},
                        daemon=False,
                    )
                    self._safetensors_worker.start()

                duration = time.time() - t_start
                print(
                    f"[CP-SAVE][rank={rank}] save_model ASYNC dispatched at step {step}, "
                    f"time={duration:.2f}s (safetensors writing in background process)"
                )

            except Exception as e:
                print(
                    f"[CP-SAVE][rank={rank}] WARNING: Async save_model failed ({e}), "
                    f"falling back to synchronous"
                )
                super().save_model(output_dir=output_dir, _internal_call=_internal_call, **kwargs)
        else:
            # Synchronous path: async disabled, or final trainer.save_model() call
            if use_async:
                self._wait_for_pending_safetensors_worker()

            try:
                print(f"[CP-SAVE][rank={rank}] save_model (safetensors) BEGIN at step {step}")
                t_start = time.time()
            except Exception:
                t_start = time.time()

            super().save_model(output_dir=output_dir, _internal_call=_internal_call, **kwargs)

            try:
                duration = time.time() - t_start
                print(
                    f"[CP-SAVE][rank={rank}] save_model (safetensors) END at step {step}, "
                    f"time={duration:.2f}s"
                )
            except Exception:
                pass

    def _save_checkpoint(self, model, trial, **kwargs):
        """Override to replace DeepSpeed synchronous checkpoint with DCP async.

        When use_async_checkpoint_save is enabled:
          1. Extract optimizer state dict from DeepSpeed (per-rank partition)
          2. Monkey-patch deepspeed.save_checkpoint → no-op (skip 105s sync I/O)
          3. Call super()._save_checkpoint() — handles safetensors, metadata, callbacks
          4. Restore deepspeed.save_checkpoint
          5. Launch DCP async_save for optimizer state (writes in background)

        When disabled: standard synchronous behavior with timing instrumentation.
        """
        rank = self.args.local_rank if self.args.local_rank != -1 else 0
        step = self.state.global_step
        use_async = getattr(self.args, "use_async_checkpoint_save", True)

        print(f"[CP-SAVE][rank={rank}] >>> _save_checkpoint BEGIN at step {step}")
        t_start = time.time()

        if use_async and self.is_deepspeed_enabled and self._init_dcp_async():
            original_ds_save = None
            try:
                # Wait for any previous async save to complete
                self._wait_for_pending_dcp_save()
                # Wait for any previous per-rank .pt write to complete
                if (
                    hasattr(self, "_pending_opt_pt_thread")
                    and self._pending_opt_pt_thread is not None
                ):
                    self._pending_opt_pt_thread.join()
                    self._pending_opt_pt_thread = None

                # Step 1: Extract optimizer state BEFORE we skip the DeepSpeed save
                # Each rank gets its own ZeRO-2 partition
                t_extract = time.time()
                optimizer_state = self.deepspeed.optimizer.state_dict()
                lr_scheduler_state = self.lr_scheduler.state_dict() if self.lr_scheduler else {}
                print(
                    f"[CP-SAVE][rank={rank}] Extracted optimizer state in "
                    f"{time.time() - t_extract:.2f}s"
                )

                # Step 2: Monkey-patch DeepSpeed save_checkpoint to no-op
                # This skips the 105s synchronous I/O (optimizer shards + model_states.pt)
                original_ds_save = self.model_wrapped.save_checkpoint
                self.model_wrapped.save_checkpoint = lambda *args, **_kw: None
                print(f"[CP-SAVE][rank={rank}] DeepSpeed save_checkpoint patched to no-op")

                # Step 3: Call parent _save_checkpoint
                # This still does: safetensors (save_model), trainer_state, rng, scheduler,
                # callbacks — everything EXCEPT the DeepSpeed I/O
                t_super = time.time()
                super()._save_checkpoint(model, trial, **kwargs)
                print(
                    f"[CP-SAVE][rank={rank}] super()._save_checkpoint (sans DeepSpeed) took "
                    f"{time.time() - t_super:.2f}s"
                )

                # Step 4: Restore original DeepSpeed save_checkpoint
                self.model_wrapped.save_checkpoint = original_ds_save

                # Step 5: Save optimizer state as per-rank .pt file + DCP async_save
                checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
                dcp_dir = os.path.join(checkpoint_dir, "dcp_state")
                os.makedirs(dcp_dir, exist_ok=True)

                # Save per-rank .pt file for fast resume without distributed coordination.
                # Each rank writes its own ZeRO-2 optimizer partition independently.
                # This avoids the second NCCL communicator that dcp.load() creates,
                # which crashes at 32+ GPUs on clusters.
                # Written in a background thread to avoid blocking training at large scale.
                dp_rank = dist.get_rank() if dist.is_initialized() else 0
                opt_pt_file = os.path.join(checkpoint_dir, f"optimizer_rank{dp_rank}.pt")
                # Both writers need a CPU snapshot without autograd history.
                opt_state_to_save = tree_map(
                    lambda value: (
                        value.detach().to(device="cpu", copy=True)
                        if isinstance(value, torch.Tensor)
                        else deepcopy(value)
                    ),
                    {"optimizer": optimizer_state, "lr_scheduler": lr_scheduler_state},
                )
                # Background-thread exceptions do not propagate to the Trainer
                # thread. Retain one so complete publication can fail before a
                # misleading SUCCESS sentinel is written remotely.
                self._pending_opt_pt_error = None

                def _save_optimizer_pt(path, state_dict, _rank, _step):
                    t = time.time()
                    try:
                        torch.save(state_dict, path)
                        print(
                            f"[CP-SAVE][rank={_rank}] Saved {path} in "
                            f"{time.time() - t:.2f}s (background)"
                        )
                    except Exception as exc:
                        self._pending_opt_pt_error = exc
                        print(f"[CP-SAVE][rank={_rank}] ERROR saving {path} at step {_step}: {exc}")

                opt_save_thread = threading.Thread(
                    target=_save_optimizer_pt,
                    args=(opt_pt_file, opt_state_to_save, rank, step),
                    daemon=True,
                )
                opt_save_thread.start()
                # Store reference so we can wait for completion before next checkpoint
                self._pending_opt_pt_thread = opt_save_thread
                print(
                    f"[CP-SAVE][rank={rank}] optimizer_rank{dp_rank}.pt write dispatched to background"
                )

                # Also launch DCP async_save for backward compatibility with older resume code.
                t_async = time.time()
                save_kwargs = {
                    "state_dict": opt_state_to_save,
                    "checkpoint_id": dcp_dir,
                }
                # Use dedicated process group to avoid conflicts with straggler detector
                if self._dcp_process_group is not None:
                    save_kwargs["process_group"] = self._dcp_process_group
                # Use process-based async to avoid GIL contention with training
                if self._async_checkpointer_type is not None:
                    save_kwargs["async_checkpointer_type"] = self._async_checkpointer_type
                self._pending_dcp_future = self._dcp_async_save(**save_kwargs)
                print(
                    f"[CP-SAVE][rank={rank}] DCP async_save staging done in "
                    f"{time.time() - t_async:.2f}s (writing optimizer state in background)"
                )

            except Exception as e:
                print(
                    f"[CP-SAVE][rank={rank}] WARNING: DCP async save failed ({e}), "
                    f"falling back to synchronous DeepSpeed optimizer save"
                )
                # Restore original save in case of failure (original_ds_save is
                # initialised to None before the try block, so it is always defined).
                if original_ds_save is not None:
                    self.model_wrapped.save_checkpoint = original_ds_save
                # Only save optimizer state via DeepSpeed — safetensors + metadata
                # were already saved in Step 3 (super()._save_checkpoint above).
                # Calling super()._save_checkpoint again would duplicate safetensors
                # writes and block ~55s waiting for the first subprocess to finish.
                try:
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
                    self.model_wrapped.save_checkpoint(checkpoint_dir)
                    print(f"[CP-SAVE][rank={rank}] DeepSpeed optimizer save (fallback) done")
                except Exception as e2:
                    print(
                        f"[CP-SAVE][rank={rank}] WARNING: DeepSpeed fallback also failed ({e2}). "
                        f"Checkpoint has model weights but no optimizer state."
                    )
        else:
            # Standard synchronous path (default)
            super()._save_checkpoint(model, trial, **kwargs)

        self._rotate_node_local_checkpoints()

        try:
            duration = time.time() - t_start
            print(
                f"[CP-SAVE][rank={rank}] <<< _save_checkpoint END at step {step}, "
                f"total_time={duration:.2f}s"
            )
        except Exception:
            pass

    def _rotate_node_local_checkpoints(self) -> None:
        """Apply HF retention on the other hosts without changing save ownership."""
        if not getattr(self.args, "node_local_output", False) or not dist.is_initialized():
            return
        rank = dist.get_rank()
        hosts = [None] * dist.get_world_size()
        hostname = None
        failure = None
        try:
            hostname = socket.gethostname()
        except OSError as exc:
            failure = f"Rank {rank}: cannot determine checkpoint host ({exc})"
        # Every rank has drained its previous checkpoint writers before reaching
        # this point. The current checkpoint is the newest and HF keeps it.
        dist.all_gather_object(hosts, (hostname, self.args.output_dir, failure))
        if any(host[2] for host in hosts):
            raise RuntimeError("; ".join(host[2] for host in hosts if host[2]))
        paths_by_host = {}
        for host, path, _failure in hosts:
            if host in paths_by_host and paths_by_host[host] != path:
                raise ValueError("Ranks on the same host must use the same checkpoint output_dir")
            paths_by_host[host] = path
        output_dir = self.args.output_dir
        host_ranks = [index for index, host in enumerate(hosts) if host[0] == hostname]
        failure = None
        # Global rank 0 already rotated its filesystem in the parent save. Ray
        # gives every actor LOCAL_RANK=0, so use host identity, not local rank.
        if rank != 0 and rank == min(host_ranks):
            try:
                rotate = getattr(self, "_rotate_checkpoints", None)
                if rotate is not None:
                    rotate(use_mtime=False, output_dir=output_dir)
                else:
                    # Transformers 5 moved retention out of Trainer.
                    from transformers.trainer_utils import rotate_checkpoints

                    rotate_checkpoints(
                        output_dir=output_dir,
                        save_total_limit=self.args.save_total_limit,
                        best_model_checkpoint=self.state.best_model_checkpoint,
                        use_mtime=False,
                    )
            except Exception as exc:  # Report cleanup errors to every rank.
                failure = f"Rank {rank}: checkpoint retention failed ({type(exc).__name__}: {exc})"
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        if any(failures):
            raise RuntimeError("; ".join(item for item in failures if item))

    # ------------------------------------------------------------------
    # Dataloader bypass for UnifiedDistributedBuffer
    # ------------------------------------------------------------------

    def get_train_dataloader(self):
        """Return an iterable that fetches entire batches from ``UnifiedDistributedBuffer``.

        If the underlying ``train_dataset`` is an instance of
        :class:`~icrtv2.data.unified_distributed_buffer.UnifiedDistributedBuffer`, we
        bypass :class:`torch.utils.data.DataLoader` completely.  Instead, we build a
        lightweight Python iterator that:

        1. Calls ``sample_batch`` (added below) on the buffer to fetch *all* samples
           for a batch with a *single* lock acquisition.
        2. Applies the configured ``data_collator`` to turn the python list into the
           usual dictionary of tensors expected by the model.

        This cuts down per-sample overhead and lowers the time spent in the
        data-loading phase when multi-process workers are not an option.
        """
        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True

        print("Creating custom train dataloader")
        if self._data_loading_log_dir:
            print(f"Data loading struct log dir: {self._data_loading_log_dir}")
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        return make_dataloader(
            dataset=self.train_dataset,
            data_collator=data_collator,
            batch_size=self._train_batch_size or 1,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            multiprocessing_context=self.multiprocessing_context,
            enable_struct_logs=self._data_loading_enable_struct_logs,
            data_loading_logging_level=self._data_loading_logging_level,
            log_dir=self._data_loading_log_dir,
            actor_id=self.args.process_index,
            prefetch_factor=self._dataloader_prefetch_factor,
        )

    @staticmethod
    def _optimizer_reset_resume_enabled() -> bool:
        return os.environ.get("GROOT_ALLOW_OPTIMIZER_RESET_RESUME", "").lower() in {
            "1",
            "true",
            "yes",
        }

    @staticmethod
    def _is_dcp_checkpoint(checkpoint_dir: str) -> bool:
        """Check if a checkpoint uses async checkpoint format (vs DeepSpeed format).

        Detects both:
        - Per-rank .pt files (optimizer_rank0.pt) — new format, no dcp.load() needed
        - DCP state directory (dcp_state/.metadata) — old format, uses dcp.load()

        TODO: Once all legacy checkpoints have been migrated, simplify to only
        check for optimizer_rank0.pt (the per-rank format). The dcp_state/
        fallback can be removed at that point.
        """
        from pathlib import Path

        # Check for per-rank .pt files (preferred new format)
        if (Path(checkpoint_dir) / "optimizer_rank0.pt").exists():
            return True
        # Check for DCP metadata (old format, backward compat)
        if (Path(checkpoint_dir) / "dcp_state" / ".metadata").exists():
            return True
        # Recovery-only fallback for durable uploads that contain the model,
        # trainer state, and scheduler but omitted async optimizer metadata.
        if GR00TTrainer._optimizer_reset_resume_enabled():
            return (
                (Path(checkpoint_dir) / TRAINER_STATE_NAME).exists()
                and (Path(checkpoint_dir) / "scheduler.pt").exists()
                and (Path(checkpoint_dir) / "model.safetensors.index.json").exists()
            )
        return False

    def _resume_from_dcp_checkpoint(self, checkpoint_dir: str):
        """Load optimizer state from DCP format and model weights from safetensors.

        This replaces deepspeed_load_checkpoint for DCP-format checkpoints.
        Model weights are loaded from safetensors shards in parallel.
        Optimizer state is loaded from dcp_state/ via torch.distributed.checkpoint.load.
        """
        from safetensors.torch import load_file

        print(f"[CP-SAVE] Resuming from DCP checkpoint: {checkpoint_dir}")
        print(f"[CP-SAVE] Model device: {next(self.model.parameters()).device}")

        # Load model weights from safetensors — parallel shard loading + incremental apply
        t_start = time.time()
        index_file = Path(checkpoint_dir) / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = _json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            num_shards = len(shard_files)

            # Phase 1: Read all shards in parallel (I/O bound on Lustre)
            from concurrent.futures import ThreadPoolExecutor

            # Load directly to model's device to avoid CPU→GPU copy in load_state_dict
            model_device = str(next(self.model.parameters()).device)

            def _load_shard(shard_name):
                t = time.time()
                shard_path = os.path.join(checkpoint_dir, shard_name)
                data = load_file(shard_path, device=model_device)
                print(
                    f"[CP-SAVE] Read {shard_name} to {model_device} in "
                    f"{time.time() - t:.1f}s ({len(data)} tensors)"
                )
                return shard_name, data

            num_workers = min(num_shards, 4)
            print(
                f"[CP-SAVE] Reading {num_shards} safetensors shards "
                f"with {num_workers} parallel workers"
            )
            t_read = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                shard_results = list(pool.map(_load_shard, shard_files))
            print(f"[CP-SAVE] All shards read in {time.time() - t_read:.2f}s")

            # Phase 2: Apply shards to model (sequential, CPU-bound)
            t_apply = time.time()
            unexpected_all = []
            for i, (shard_name, shard_state_dict) in enumerate(shard_results):
                t_shard = time.time()
                _, unexpected = self.model.load_state_dict(shard_state_dict, strict=False)
                unexpected_all.extend(unexpected)
                del shard_state_dict
                print(
                    f"[CP-SAVE] Applied shard {i + 1}/{num_shards}: {shard_name} "
                    f"in {time.time() - t_shard:.1f}s"
                )
            del shard_results
            print(f"[CP-SAVE] All shards applied in {time.time() - t_apply:.2f}s")
            missing_all = []
        else:
            # Single safetensors file
            st_file = Path(checkpoint_dir) / "model.safetensors"
            state_dict = load_file(str(st_file))
            missing_all, unexpected_all = self.model.load_state_dict(state_dict, strict=False)
            missing_all = list(missing_all)
            unexpected_all = list(unexpected_all)
            del state_dict

        if unexpected_all:
            print(
                f"[CP-SAVE] WARNING: Unexpected keys when loading model weights: "
                f"{len(unexpected_all)} keys"
            )
        print(f"[CP-SAVE] Model weights loaded from safetensors in {time.time() - t_start:.2f}s")

        # Load optimizer state — prefer per-rank .pt (no distributed coordination),
        # fall back to dcp.load() for older checkpoints that only have dcp_state/.
        t_opt = time.time()

        zero_stage = self._get_zero_stage()
        if zero_stage != 2:
            raise ValueError(f"DCP checkpoint loading requires ZeRO stage 2, got {zero_stage}")

        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        opt_pt_file = os.path.join(checkpoint_dir, f"optimizer_rank{dp_rank}.pt")

        if os.path.exists(opt_pt_file):
            # Fast path: per-rank .pt file — no NCCL, no second communicator, no dcp.load().
            # Each rank independently reads its own ZeRO-2 optimizer partition.
            print(f"[CP-SAVE] Loading optimizer from {opt_pt_file} (per-rank, no dist)")
            state_dict_to_load = torch.load(opt_pt_file, map_location="cpu", weights_only=False)
        elif (Path(checkpoint_dir) / "dcp_state" / ".metadata").exists():
            # Fallback: load from DCP format for older checkpoints.
            # WARNING: dcp.load() creates a second NCCL communicator which may fail at 32+ GPUs
            # on clusters with misconfigured EFA. Prefer per-rank .pt for new checkpoints.
            import torch.distributed.checkpoint as dcp

            print(
                "[CP-SAVE] Per-rank .pt not found, falling back to dcp.load() "
                "(may fail at 32+ GPUs)"
            )
            dcp_dir = os.path.join(checkpoint_dir, "dcp_state")
            optimizer_state = self.deepspeed.optimizer.state_dict()
            lr_scheduler_state = self.lr_scheduler.state_dict() if self.lr_scheduler else {}
            state_dict_to_load = {
                "optimizer": optimizer_state,
                "lr_scheduler": lr_scheduler_state,
            }
            dcp.load(state_dict=state_dict_to_load, checkpoint_id=dcp_dir)
        elif self._optimizer_reset_resume_enabled():
            state_dict_to_load = None
            print(
                "[CP-SAVE] WARNING: Durable checkpoint has no optimizer state; "
                "continuing with fresh optimizer moments and restored scheduler"
            )
        else:
            raise FileNotFoundError(f"No resumable optimizer state found in {checkpoint_dir}")

        if state_dict_to_load is not None:
            # DeepSpeed ZeRO-2 load_state_dict expects {dp_rank: state_dict}.
            # Each rank loads its own partition, so wrap with current rank as key.
            self.deepspeed.optimizer.load_state_dict({dp_rank: state_dict_to_load["optimizer"]})
            if self.lr_scheduler and state_dict_to_load.get("lr_scheduler"):
                self.lr_scheduler.load_state_dict(state_dict_to_load["lr_scheduler"])
            print(f"[CP-SAVE] Optimizer state loaded in {time.time() - t_opt:.2f}s")

        # Load scheduler
        scheduler_file = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_file) and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(
                torch.load(scheduler_file, map_location="cpu", weights_only=False)
            )
            print("[CP-SAVE] LR scheduler loaded from scheduler.pt")

        # Load RNG states
        rng_file = os.path.join(
            checkpoint_dir,
            # HF saves these as rng_state_{process_index}.pth (GLOBAL rank);
            # reading by local_rank made every node load rng_state_0..7, so
            # ranks sharing a local_rank across nodes restored identical RNG.
            f"rng_state_{self.args.process_index}.pth",
        )
        if os.path.exists(rng_file):
            import random

            import numpy as np

            rng_state = torch.load(rng_file, weights_only=False)
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["cpu"])
            if torch.cuda.is_available() and "cuda" in rng_state:
                torch.cuda.random.set_rng_state_all(rng_state["cuda"])
            print(f"[CP-SAVE] RNG state loaded from {rng_file}")

        print("[CP-SAVE] DCP checkpoint resume complete")

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Resume from checkpoint with support for both DeepSpeed and DCP formats."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            if getattr(self.args, "node_local_output", False):
                raise ValueError("Automatic checkpoint discovery is unsafe for node-local output")
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logging.warning(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        self._dcp_resume_checkpoint = None
        if resume_from_checkpoint is not None and self._is_dcp_checkpoint(resume_from_checkpoint):
            logging.info(
                f"Detected DCP checkpoint format at {resume_from_checkpoint}, "
                f"will use DCP resume path"
            )
            self._dcp_resume_checkpoint = resume_from_checkpoint

        if resume_from_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        # Sync save_steps/logging_steps from current args into trainer state.
        # HF Trainer's _inner_training_loop loads trainer_state.json which may have
        # stale values from the original run. This callback fires after state load,
        # ensuring current CLI args take effect.
        if resume_from_checkpoint is not None:
            _args_ref = self.args

            class _SyncArgsCallback(TrainerCallback):
                def on_train_begin(self, args, state, _control, **kwargs):
                    state.save_steps = _args_ref.save_steps
                    state.logging_steps = _args_ref.logging_steps

            self.add_callback(_SyncArgsCallback())

        # Pre-warm pinned CPU offload buffer for async checkpoint saves.
        # on_train_begin fires after DeepSpeed engine init (model on GPU)
        # but before the first training step — hides the ~14s pinned alloc cost.
        use_async = getattr(self.args, "use_async_checkpoint_save", True)
        if use_async and self.is_deepspeed_enabled:
            _trainer_ref_pw = self

            class _PrewarmPinnedBufferCallback(TrainerCallback):
                def on_train_begin(self, args, state, _control, **kwargs):
                    _rank = args.local_rank if args.local_rank != -1 else 0
                    use_local = (
                        bool(getattr(args, "use_local_state_dict_for_zero2", True))
                        and _trainer_ref_pw._get_zero_stage() == 2
                    )
                    if use_local and args.should_save:
                        _trainer_ref_pw._prewarm_pinned_offload_buffer(_rank)

            self.add_callback(_PrewarmPinnedBufferCallback())

        # Initialize straggler detector before training starts
        if self.straggler_enabled and STRAGGLER_AVAILABLE and straggler_module is not None:
            self._initialize_straggler_detector()

        if self._dcp_resume_checkpoint is not None:
            # HF Trainer's _inner_training_loop has a hardcoded sequence:
            #   DeepSpeed engine init → deepspeed_load_checkpoint() → training loop
            #
            # We need to intercept deepspeed_load_checkpoint() because:
            # 1. DeepSpeed engine must be initialized first (model on GPU, optimizer created)
            # 2. Only then can we load DCP optimizer state into the engine
            # 3. There is no HF Trainer callback between "DeepSpeed init" and "load checkpoint"
            #
            # Alternatives considered:
            # - Override _inner_training_loop: ~500-line method, fragile across HF versions
            # - on_train_begin callback: fires AFTER deepspeed_load_checkpoint already ran
            # - Skip resume + manual load: loses HF step counting/state restoration
            #
            # The monkey-patch is scoped to this train() call and restored in finally.
            with self._patch_deepspeed_load_for_dcp(self._dcp_resume_checkpoint):
                return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
        else:
            # Standard DeepSpeed resume path
            return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # ------------------------------------------------------------------------------------------
    # Training step override for timing metrics and truncated BPTT (same pattern as BaseTrainer)
    # ------------------------------------------------------------------------------------------

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch: int | None = None,
    ):
        """Override training_step to measure and log step time, with optional straggler detection."""
        with self.timer.with_label("training_step"):
            with self._straggler_section("training_step", profile_cuda=self.straggler_profile_cuda):
                if self.args.use_truncated_bptt:
                    loss = self.training_step_truncated_bptt(model, inputs, num_items_in_batch)
                else:
                    self._maybe_mark_ttt_cudagraph_step_begin(model)
                    loss = super().training_step(model, inputs, num_items_in_batch)

        self._maybe_dump_nonfinite_loss_diagnostics(
            stage="post_backward",
            model=model,
            inputs=inputs,
            outputs=None,
            loss=loss,
            include_inputs=False,
            include_outputs=False,
            include_parameters=True,
            include_gradients=True,
        )

        try:
            self._log_batch_and_timing_metrics(inputs)
        except Exception:
            pass

        return loss

    @staticmethod
    def _model_stack_contains_ttt(model: Any) -> bool:
        stack = [model]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            if current is None:
                continue
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)

            names = [type(current).__name__]
            config = getattr(current, "config", None)
            if config is not None:
                names.append(type(config).__name__)
            if any("ttt" in name.lower() for name in names):
                return True

            for attr in ("module", "_orig_mod", "model", "base_model"):
                child = getattr(current, attr, None)
                if child is not None and child is not current:
                    stack.append(child)
        return False

    def _maybe_mark_ttt_cudagraph_step_begin(self, model: Any) -> None:
        ttt_reduce_overhead = (
            os.environ.get("GROOT_TTT_INNER_COMPILE_MODE", "").lower() == "reduce-overhead"
            or os.environ.get("GROOT_TTT_TRANSFORMER_COMPILE_MODE", "").lower() == "reduce-overhead"
        )
        model_reduce_overhead = (
            getattr(self.args, "torch_compile", False)
            and getattr(self.args, "torch_compile_mode", None) == "reduce-overhead"
        )
        if not (model_reduce_overhead or ttt_reduce_overhead):
            return
        if not (
            self._model_stack_contains_ttt(model)
            or self._model_stack_contains_ttt(getattr(self, "model", None))
        ):
            return

        compiler = getattr(torch, "compiler", None)
        mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None)
        if mark_step_begin is not None:
            mark_step_begin()

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """Override get_batch_samples to log data loading stall and per-batch datapoint ids.

        Measures the time the training loop stalls waiting for data, and logs the
        ``VLAStepDataId``s of the data points in each batch. This helps diagnose
        data loading bottlenecks and trace which data points the model saw at a
        given step.
        """
        structlog.contextvars.bind_contextvars(global_step=self.state.global_step)
        t0 = time.perf_counter()
        with self.timer.with_label("data_loading_stall_seconds"):
            batch_samples, num_items = super().get_batch_samples(
                epoch_iterator, num_batches, device
            )
        stall_ms = round((time.perf_counter() - t0) * 1000, 2)

        # get_datapoint_ids walks every batch in batch_samples in one pass, so
        # we call it once here and serialize to dicts for structlog.
        ids = self.get_datapoint_ids(batch_samples)

        structlog.get_logger().info(
            "get_batch_samples",
            data_loading_stall_ms=stall_ms,
            datapoint_ids=[dataclasses.asdict(d) for d in ids],
        )

        return batch_samples, num_items

    def _log_batch_and_timing_metrics(self, inputs):
        """Log batch content and padding metrics (rank 0 local, no collective ops)."""
        if self.args.local_rank not in (-1, 0):
            return

        from collections.abc import Mapping

        batch = inputs
        for _ in range(5):
            if (
                isinstance(batch, Mapping)
                and "inputs" in batch
                and isinstance(batch["inputs"], Mapping)
            ):
                batch = batch["inputs"]
            else:
                break

        if not isinstance(batch, Mapping):
            return

        metrics = {}

        if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
            metrics["batch/tokens"] = int(batch["attention_mask"].sum())

        if "pixel_values" in batch and isinstance(batch["pixel_values"], torch.Tensor):
            metrics["batch/pixel_elements"] = batch["pixel_values"].numel()

        grid_key = "image_grid_thw" if "image_grid_thw" in batch else "image_grid_hws"
        if grid_key in batch and isinstance(batch[grid_key], torch.Tensor):
            grid = batch[grid_key]
            metrics["batch/images"] = int(grid.shape[0])
            if grid.dim() == 2 and grid.shape[1] >= 1:
                metrics["batch/frames"] = int(grid[:, 0].sum())

        reward = batch.get("reward")
        if reward is not None:
            reward_tensor = reward.detach().float()

            if reward_tensor.numel() > 0:
                metrics["batch/reward_count"] = reward_tensor.numel()
                metrics["batch/reward_mean"] = reward_tensor.mean().item()
                metrics["batch/reward_min"] = reward_tensor.min().item()
                metrics["batch/reward_max"] = reward_tensor.max().item()
                metrics["batch/reward_std"] = reward_tensor.std(unbiased=False).item()

        if metrics:
            self.log(metrics)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ttt_trajectory_scalar_metadata(inputs: Any) -> None:
        """Normalize TTT trajectory metadata before compiled model forward.

        ``GrootN2TTT.forward`` flattens trajectory batches and historically
        reduced these values with ``.item()`` inside forward. Under model-level
        torch.compile that creates scalar graph breaks, so keep the exact same
        reductions in the uncompiled trainer path.
        """
        batch = inputs
        if (
            isinstance(batch, Mapping)
            and "inputs" in batch
            and isinstance(batch["inputs"], Mapping)
        ):
            batch = batch["inputs"]
        if not isinstance(batch, Mapping) or "is_real_timestep" not in batch:
            return

        for key in ("num_cond_frames", "num_total_frames"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                batch[key] = int(value.max().item())

        has_real_action = batch.get("has_real_action")
        if isinstance(has_real_action, torch.Tensor):
            batch["has_real_action"] = bool(has_real_action.any().item())

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """
        self._normalize_ttt_trajectory_scalar_metadata(inputs)

        with self.timer.with_label("model_forward"):
            with self._straggler_section("model_forward", profile_cuda=self.straggler_profile_cuda):
                loss, outputs = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )

        self._maybe_dump_nonfinite_loss_diagnostics(
            stage="compute_loss",
            model=model,
            inputs=inputs,
            outputs=outputs,
            loss=loss,
            include_inputs=True,
            include_outputs=True,
            include_parameters=True,
            include_gradients=False,
        )
        self._maybe_dump_nonfinite_model_output_diagnostics(
            stage="compute_loss",
            model=model,
            inputs=inputs,
            outputs=outputs,
        )

        self._log_rank0_scalar_outputs(outputs)

        if model.training and getattr(self.args, "log_distributed_aux_losses", False):
            self._accumulate_aux_loss_metrics(outputs, loss.device)

        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            if self.args.local_rank in (-1, 0):
                self.log({"train_accuracy": acc_mean})

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: nn.Module,
        trial: Any,
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        should_log_global_metrics = (
            self.control.should_log and self.state.global_step > self._globalstep_last_logged
        )
        extra_metrics = {}
        if getattr(self.args, "log_distributed_grad_norm", False) and grad_norm is not None:
            extra_metrics["grad_norm"] = grad_norm
        global_metrics = (
            self._maybe_build_global_training_metrics(tr_loss.device, extra_metrics)
            if should_log_global_metrics
            else {}
        )

        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )

        if global_metrics:
            self.log(global_metrics)

    def training_step_truncated_bptt(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch=None
    ) -> torch.Tensor:
        """Perform a training step on a batch of inputs using truncated back propagation through time.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        keys_T_second = [
            "action",
            "action_mask",
            "embodiment_id",
            "state",
            "task_index",
            "is_real_timestep",
        ]

        def _sync_seq_len(inputs):
            """Sync shapes across ranks and adjust seq len dimension accordingly"""
            # "fixed" pads every sample (on every rank) to the same precomputed constant
            # in the collator, so all ranks already share an identical sequence length.
            if self.args.seq_len_handle_strategy == "fixed":
                return inputs

            local_T = inputs["inputs"]["state"].shape[1]

            # Create tensor on same device as data
            T_tensor = torch.tensor(
                [local_T], dtype=torch.long, device=inputs["inputs"]["state"].device
            )
            if self.args.seq_len_handle_strategy == "truncate":
                dist.all_reduce(T_tensor, op=dist.ReduceOp.MIN)
            else:
                dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T_global = T_tensor.item()

            # keys with T as the second dim
            # all VL related keys have T as the first dim
            if local_T != T_global and self.args.seq_len_handle_strategy == "truncate":
                for k in inputs["inputs"].keys():
                    if not isinstance(inputs["inputs"][k], torch.Tensor):
                        continue
                    inputs["inputs"][k] = (
                        inputs["inputs"][k][:, :T_global]
                        if k in keys_T_second
                        else inputs["inputs"][k][:T_global]
                    )
            elif local_T != T_global and self.args.seq_len_handle_strategy == "pad":
                T_pad = T_global - local_T
                for k in inputs["inputs"].keys():
                    if not isinstance(inputs["inputs"][k], torch.Tensor):
                        continue
                    if k in keys_T_second:
                        pad = torch.zeros_like(inputs["inputs"][k][:, :1]).repeat_interleave(
                            T_pad, dim=1
                        )
                        inputs["inputs"][k] = torch.cat([inputs["inputs"][k], pad], dim=1)
                    else:
                        # for VL related, we repeat instead of zero padding
                        pad = inputs["inputs"][k][:1].repeat_interleave(T_pad, dim=0)
                        inputs["inputs"][k] = torch.cat([inputs["inputs"][k], pad], dim=0)

            return inputs

        def _truncate_inputs(inputs):
            chunk_size: int = self.args.truncated_bptt_chunk_size
            T = inputs["inputs"]["state"].shape[1]
            N_chunks = math.ceil(T / chunk_size)
            truncated_inputs = []
            chunk_sizes = []
            total_mask_sum = inputs["inputs"]["action_mask"].sum().item()
            for i in range(N_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, T)
                chunk = {"inputs": {}}
                for k in inputs["inputs"].keys():
                    if not isinstance(inputs["inputs"][k], torch.Tensor):
                        chunk["inputs"][k] = inputs["inputs"][k]
                        continue
                    if k in keys_T_second:
                        chunk["inputs"][k] = inputs["inputs"][k][:, start:end].clone()
                    else:
                        chunk["inputs"][k] = inputs["inputs"][k][start:end].clone()
                truncated_inputs.append(chunk)
                chunk_sizes.append(end - start)
            del inputs
            return deque(truncated_inputs), chunk_sizes, total_mask_sum

        inputs = self._prepare_inputs(inputs)
        if self.args.sync_seq_len:
            inputs = _sync_seq_len(inputs)
        inputs, chunk_sizes, total_mask_sum = _truncate_inputs(inputs)
        N_chunks = len(chunk_sizes)
        if self.args.local_rank in (-1, 0):
            self.log(
                {
                    "train/bptt_num_chunks": float(N_chunks),
                    "train/bptt_seq_len": float(sum(chunk_sizes)),
                }
            )

        total_loss = 0.0
        denoise_timestep = None  # for the first chunk, we don't specify the denoise timestep to let the model sample it
        total_seqlen = None
        last_mini_batch_params_dict = None
        for chunk_idx in range(N_chunks):
            chunk_inputs = inputs.popleft()
            chunk_inputs["denoise_timestep"] = denoise_timestep
            chunk_inputs["total_seqlen"] = total_seqlen
            chunk_inputs["last_mini_batch_params_dict"] = last_mini_batch_params_dict
            self._maybe_mark_ttt_cudagraph_step_begin(model)
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(
                    model, chunk_inputs, num_items_in_batch=num_items_in_batch, return_outputs=True
                )

            kwargs = {}

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            action_mask_sum = outputs["action_mask_sum"]
            scaled_loss = loss * (action_mask_sum / total_mask_sum)

            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                scaled_loss = scaled_loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(scaled_loss, **kwargs)

            total_loss += scaled_loss.detach().item()

            # clear and update the state
            del loss, scaled_loss, chunk_inputs
            denoise_timestep = outputs.get("denoise_timestep")
            if denoise_timestep is not None:
                denoise_timestep = denoise_timestep.detach()
            total_seqlen = (
                outputs["L_token"] if chunk_idx == 0 else total_seqlen + outputs["L_token"]
            )
            last_mini_batch_params_dict = [
                {k: v.detach().clone() for k, v in layer.items()}
                for layer in outputs["state_output"]
            ]
            del outputs

        return torch.tensor(total_loss, device=self.model.device)
