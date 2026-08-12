# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import partial
import os
import warnings

import structlog
import torch
import torch.distributed as dist
import torch.utils.data

from groot.core.training.experiment import memory_monitor
from groot.core.utils.distributed import worker_logging


def make_dataloader(
    dataset,
    data_collator,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    multiprocessing_context: str | None = "fork",
    enable_struct_logs: bool = False,
    data_loading_logging_level: str = "INFO",
    log_dir: str | None = None,
    actor_id: int | None = None,
    prefetch_factor: int | None = None,
) -> torch.utils.data.DataLoader:
    persistent_workers = num_workers > 0
    params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    # How many batches each worker pre-builds (PyTorch default is 2). This only
    # applies when PyTorch worker processes exist; with num_workers=0, loading is
    # synchronous in the main process and PyTorch rejects prefetch_factor.
    if prefetch_factor is not None:
        if num_workers > 0:
            params["prefetch_factor"] = int(prefetch_factor)
        else:
            warnings.warn(
                "data.dataloader_prefetch_factor has no effect with "
                "training.dataloader_num_workers=0; ignoring it.",
                stacklevel=2,
            )
    if enable_struct_logs and num_workers == 0:
        warnings.warn(
            "enable_struct_logs=True has no effect with num_workers=0; "
            "DataLoader does not invoke worker_init_fn without workers.",
            stacklevel=2,
        )
    # Wire the worker init hook when struct logs are on OR the memory monitor is
    # enabled, so GROOT_MEMORY_MONITOR can profile the dataloader worker
    # processes (where DeltaCAT decodes/materializes episode video).
    if enable_struct_logs or memory_monitor.enabled():
        params["worker_init_fn"] = partial(
            _dataloader_worker_init_fn,
            level=data_loading_logging_level,
            log_dir=log_dir,
            actor_id=actor_id,
            struct_logs=enable_struct_logs,
        )
    if num_workers > 0:
        params["multiprocessing_context"] = multiprocessing_context
    return torch.utils.data.DataLoader(dataset, **params)


def _dataloader_worker_init_fn(worker_id, level="INFO", log_dir=None, actor_id=None, struct_logs=True):
    # Profile this dataloader worker process (where DeltaCAT decodes episode
    # video) when GROOT_MEMORY_MONITOR is set. No-op otherwise.
    memory_monitor.start()
    if not struct_logs:
        return
    log_file = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        prefix = f"training_actor_{actor_id}_" if actor_id is not None else ""
        log_file = f"{log_dir}/{prefix}data_loading_worker_{worker_id}.jsonl"
    worker_logging.configure_logging(
        worker_type="pytorch_dataloader", level=level, log_file=log_file, force=True
    )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    structlog.contextvars.bind_contextvars(rank=rank, pytorch_worker_id=worker_id)
    structlog.get_logger().info("data_loader_worker_process_ready")
