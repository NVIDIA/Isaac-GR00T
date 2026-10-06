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

# Make training work w/ and w/o distributed training.
from contextlib import contextmanager

import torch


def is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if is_dist_avail_and_initialized():
        return torch.distributed.get_rank()
    return 0


def barrier():
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


def _collective_device() -> torch.device:
    """Pick the device used for status-broadcast collectives.

    NCCL backend requires CUDA tensors; gloo / mpi accept CPU. We rely on the
    caller (typically `experiment.run` after `torch.cuda.set_device(local_rank)`)
    having already pinned the current CUDA device.
    """
    if torch.distributed.get_backend() == "nccl":
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@contextmanager
def run_or_wait_on_rank0(label: str | None = None):
    """Run the body on rank-0 only, broadcasting any rank-0 error to all ranks.

    Yields ``True`` on rank-0, ``False`` elsewhere; callers gate the body with
    ``if is_rank0:``. If rank-0 raises, every rank raises consistently
    (rank-0 re-raises the original; others get ``RuntimeError(label: ...)``).
    Without this contract a rank-0 raise inside a plain ``if get_rank() == 0:``
    block leaves other ranks hanging at the next NCCL collective until the
    30-min timeout fires.

    Degenerates to a plain ``with`` block when distributed is not initialized.

        with run_or_wait_on_rank0(label="generate_stats") as is_rank0:
            if is_rank0:
                ...
    """
    if not is_dist_avail_and_initialized():
        yield True
        return

    rank = torch.distributed.get_rank()
    is_rank0 = rank == 0
    device = _collective_device()
    status = torch.zeros(1, dtype=torch.int, device=device)
    rank0_err: BaseException | None = None
    error_summary: list[str | None] = [None]
    try:
        yield is_rank0
    except BaseException as exc:
        if is_rank0:
            status += 1
            rank0_err = exc
            error_summary[0] = f"{type(exc).__name__}: {exc}"
        else:
            # Work outside the ``if is_rank0:`` gate raised on a non-rank-0 rank;
            # propagate locally — we only broadcast rank-0 failures.
            raise
    finally:
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
        if status.item() != 0:
            torch.distributed.broadcast_object_list(error_summary, src=0)
        torch.distributed.barrier()

    if status.item() == 0:
        return
    if rank0_err is not None:
        raise rank0_err
    label_prefix = f"{label}: " if label else ""
    raise RuntimeError(
        f"{label_prefix}rank-0 failed inside run_or_wait_on_rank0() with "
        f"{error_summary[0] or '<no error info broadcast>'}; "
        "see rank-0's traceback for the full stack."
    ) from None


def run_on_rank0(fn, *args, label: str | None = None, **kwargs):
    """Call ``fn(*args, **kwargs)`` on rank-0 only; return its result there, ``None`` elsewhere.

    A rank-0 failure is broadcast to every rank (see ``run_or_wait_on_rank0``).
    ``args`` / ``kwargs`` are evaluated on every rank, so keep rank-0-only work inside ``fn``.
    """
    result = None
    with run_or_wait_on_rank0(label=label or getattr(fn, "__qualname__", None)) as is_rank0:
        if is_rank0:
            result = fn(*args, **kwargs)
    return result
