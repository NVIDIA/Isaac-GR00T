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

"""CPU-only regression tests for `gr00t.experiment.dist_utils.run_or_wait_on_rank0`.

These pin the behavioural contract that motivates the helper:

  - When rank-0 raises inside the ``with`` block, *every* rank must raise
    (so the failure surfaces synchronously instead of stalling at the next
    NCCL collective for 30 minutes).
  - When the helper is used outside a torch.distributed process group, it
    must degenerate to a plain context manager that runs once and yields
    ``True``.
  - When all ranks succeed, no spurious exception is raised.
  - The body actually runs only on rank-0.

We use the gloo backend over file:// init so the tests run on CPU-only CI
without needing GPU / NCCL. A wall-clock timeout on each subprocess pins
the "no hang" guarantee — if a rank ever stalls in dist.barrier instead
of raising, the test fails on timeout rather than passing silently.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = 3
# Generous headroom so this stays robust under parallel CI (pytest-xdist -n auto):
# each test spawns WORLD_SIZE torch-importing subprocesses, which contend for
# cores when every xdist worker is busy. Non-hang paths complete in well under
# 1s and a real barrier hang would block far longer, so a larger budget still
# reliably catches the regression this guards.
WORKER_TIMEOUT_S = 120


def _setup_pg(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "0")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _teardown_pg() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _all_rank_succeed_worker(rank: int, world_size: int, init_file: str, scratch_dir: str) -> None:
    from gr00t.experiment.dist_utils import run_or_wait_on_rank0

    _setup_pg(rank, world_size, init_file)
    try:
        with run_or_wait_on_rank0() as is_rank0:
            if is_rank0:
                Path(scratch_dir, "rank0_was_here").write_text("ok")
        # Sentinel: every rank that *exits cleanly* writes its rank file.
        # If a rank raised inside the helper or hung at the barrier, this
        # never lands on disk and the parent assertion catches it.
        Path(scratch_dir, f"rank_{rank}_done").write_text("ok")
    finally:
        _teardown_pg()


def _rank0_raises_worker(rank: int, world_size: int, init_file: str, scratch_dir: str) -> None:
    from gr00t.experiment.dist_utils import run_or_wait_on_rank0

    _setup_pg(rank, world_size, init_file)
    try:
        try:
            with run_or_wait_on_rank0(label="unit-test-label") as is_rank0:
                if is_rank0:
                    raise ValueError(f"boom-from-rank-{rank}")
        except Exception as exc:
            Path(scratch_dir, f"rank_{rank}_exc.txt").write_text(f"{type(exc).__name__}: {exc}")
            return
        # No exception observed: that's a contract violation.
        Path(scratch_dir, f"rank_{rank}_no_exc").write_text("contract-violation")
    finally:
        _teardown_pg()


def _wandb_init_failure_worker(
    rank: int, world_size: int, init_file: str, scratch_dir: str
) -> None:
    """Mimic the production call shape at ``experiment.py``:

    .. code-block:: python

        if config.training.use_wandb:
            with run_or_wait_on_rank0(label="wandb.init") as is_rank0:
                if is_rank0:
                    wandb.init(...)  # raises on auth/network failure

    The body re-raises a ``RuntimeError`` standing in for
    ``wandb.errors.AuthenticationError`` / ``CommError`` so the test does
    not require wandb. The contract is identical: a rank-0 raise must
    propagate to every rank instead of letting non-rank-0 ranks advance
    to the next NCCL collective and hang.
    """
    from gr00t.experiment.dist_utils import run_or_wait_on_rank0

    _setup_pg(rank, world_size, init_file)
    try:
        try:
            with run_or_wait_on_rank0(label="wandb.init") as is_rank0:
                if is_rank0:
                    raise RuntimeError("simulated wandb auth failure")
        except Exception as exc:
            Path(scratch_dir, f"rank_{rank}_exc.txt").write_text(f"{type(exc).__name__}: {exc}")
            return
        Path(scratch_dir, f"rank_{rank}_no_exc").write_text("contract-violation")
    finally:
        _teardown_pg()


def _spawn(target, scratch_dir: str) -> None:
    """Spawn `WORLD_SIZE` workers with a temp file:// rendezvous.

    `mp.spawn(join=True)` blocks until every child exits. Hang protection
    comes from `@pytest.mark.timeout(WORKER_TIMEOUT_S)` on each test: if a
    worker stalls in `dist.barrier`, pytest interrupts the whole test on
    its wall-clock budget rather than wedging the CI run.
    """
    init_file = os.path.join(scratch_dir, "init")
    mp.spawn(
        target,
        args=(WORLD_SIZE, init_file, scratch_dir),
        nprocs=WORLD_SIZE,
        join=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_or_wait_on_rank0_no_dist_runs_body_once_and_yields_true():
    """Helper degenerates cleanly outside a process group."""
    from gr00t.experiment.dist_utils import run_or_wait_on_rank0

    runs: list[bool] = []
    with run_or_wait_on_rank0() as is_rank0:
        runs.append(is_rank0)

    assert runs == [True], "Body must run exactly once with is_rank0=True when not distributed"


def test_run_or_wait_on_rank0_no_dist_propagates_local_error():
    """Outside distributed, exceptions still propagate (no silent swallow)."""
    from gr00t.experiment.dist_utils import run_or_wait_on_rank0

    with pytest.raises(RuntimeError, match="local-only"):
        with run_or_wait_on_rank0() as is_rank0:
            assert is_rank0
            raise RuntimeError("local-only")


def test_run_on_rank0_no_dist_calls_fn_and_returns_result():
    """``run_on_rank0`` degenerates to a plain call returning the fn result."""
    from gr00t.experiment.dist_utils import run_on_rank0

    calls: list[tuple] = []

    def fn(a, b, c=0):
        calls.append((a, b, c))
        return a + b + c

    assert run_on_rank0(fn, 1, 2, c=3) == 6
    assert calls == [(1, 2, 3)], "fn must be invoked exactly once with the forwarded args"


def test_run_on_rank0_no_dist_propagates_local_error():
    """Outside distributed, a fn raise still propagates (no silent swallow)."""
    from gr00t.experiment.dist_utils import run_on_rank0

    def boom():
        raise RuntimeError("local-only")

    with pytest.raises(RuntimeError, match="local-only"):
        run_on_rank0(boom)


@pytest.mark.timeout(WORKER_TIMEOUT_S)
def test_run_or_wait_on_rank0_runs_body_only_on_rank0_when_all_succeed():
    """Body runs on rank-0; all ranks exit cleanly."""
    with tempfile.TemporaryDirectory() as scratch_dir:
        _spawn(_all_rank_succeed_worker, scratch_dir)

        scratch = Path(scratch_dir)
        assert (scratch / "rank0_was_here").exists(), (
            "Rank-0 body should have executed and written the sentinel"
        )
        for rank in range(WORLD_SIZE):
            assert (scratch / f"rank_{rank}_done").exists(), (
                f"Rank {rank} did not exit cleanly through run_or_wait_on_rank0()"
            )


@pytest.mark.timeout(WORKER_TIMEOUT_S)
def test_run_or_wait_on_rank0_propagates_rank0_error_to_all_ranks():
    """Contract: rank-0 raise → every rank raises; no rank hangs at barrier.

    This is the load-bearing assertion. Without run_or_wait_on_rank0(), the same
    pattern (`if get_rank() == 0:` + raise) hangs non-rank-0 ranks at the
    next NCCL collective for the default 30-min timeout. Here we assert
    every rank observes an exception inside the wall-clock budget.
    """
    with tempfile.TemporaryDirectory() as scratch_dir:
        _spawn(_rank0_raises_worker, scratch_dir)

        scratch = Path(scratch_dir)
        for rank in range(WORLD_SIZE):
            exc_file = scratch / f"rank_{rank}_exc.txt"
            no_exc_file = scratch / f"rank_{rank}_no_exc"
            assert not no_exc_file.exists(), (
                f"Rank {rank} did not raise — contract violation; without "
                "broadcasting, this rank would have hung at the next "
                "NCCL collective."
            )
            assert exc_file.exists(), f"Rank {rank} did not write its exception sentinel"
            payload = exc_file.read_text()
            if rank == 0:
                # rank-0 should re-raise the *original* exception type.
                assert "ValueError" in payload, (
                    f"Rank-0 should re-raise the underlying ValueError, got: {payload}"
                )
            else:
                # Other ranks raise a synthesized RuntimeError pointing at
                # rank-0's traceback. The label and rank-0's exception
                # summary must round-trip through broadcast_object_list so
                # log scrapers can name the failing call site without
                # cross-referencing rank-0's separate traceback.
                assert "RuntimeError" in payload, (
                    f"Non-rank-0 should raise RuntimeError after status broadcast, got: {payload}"
                )
                assert "unit-test-label" in payload, (
                    f"Non-rank-0 RuntimeError must include the label arg, got: {payload}"
                )
                assert "ValueError" in payload and "boom-from-rank-0" in payload, (
                    f"Non-rank-0 RuntimeError must echo rank-0's <ExceptionType>: <msg>, got: {payload}"
                )


# ---------------------------------------------------------------------------
# Sanity: the spawn machinery itself doesn't deadlock when nothing raises.
# ---------------------------------------------------------------------------


def _smoke_worker(rank: int, world_size: int, init_file: str, scratch_dir: str) -> None:
    _setup_pg(rank, world_size, init_file)
    try:
        # Cross-rank reduction sanity check, independent of run_or_wait_on_rank0.
        t = torch.tensor([float(rank)])
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        Path(scratch_dir, f"smoke_rank_{rank}_sum").write_text(str(t.item()))
    finally:
        _teardown_pg()


@pytest.mark.timeout(WORKER_TIMEOUT_S)
def test_wandb_init_failure_propagates_to_all_ranks():
    """Regression for the ``wandb.init`` call site in ``experiment.run``.

    Pre-fix: ``wandb.init`` was guarded only by ``if global_rank == 0:``
    with no exception broadcast. A rank-0 ``AuthenticationError`` /
    ``CommError`` left non-rank-0 ranks oblivious; they advanced to the
    next NCCL collective in ``pipeline.setup()`` / ``Trainer.train`` and
    stalled there until NCCL's default 30-min timeout fired.

    Post-fix: the call sits inside ``run_or_wait_on_rank0(label="wandb.init")``,
    so rank-0 failures fail-fast cluster-wide with a label-bearing
    ``RuntimeError`` on the other ranks.
    """
    with tempfile.TemporaryDirectory() as scratch_dir:
        _spawn(_wandb_init_failure_worker, scratch_dir)

        scratch = Path(scratch_dir)
        for rank in range(WORLD_SIZE):
            no_exc_file = scratch / f"rank_{rank}_no_exc"
            exc_file = scratch / f"rank_{rank}_exc.txt"
            assert not no_exc_file.exists(), (
                f"Rank {rank} did not raise on simulated wandb.init failure — "
                "without run_or_wait_on_rank0, this rank would have hung at the next "
                "NCCL collective."
            )
            assert exc_file.exists(), f"Rank {rank} did not write its exception sentinel"
            payload = exc_file.read_text()
            if rank == 0:
                assert "RuntimeError" in payload and "wandb auth failure" in payload, (
                    f"Rank-0 should re-raise the underlying wandb error, got: {payload}"
                )
            else:
                assert "RuntimeError" in payload and "wandb.init" in payload, (
                    f"Non-rank-0 RuntimeError must name the wandb.init label, got: {payload}"
                )


def test_experiment_run_wraps_wandb_init_in_run_or_wait_on_rank0():
    """Source-level pin: ``experiment.run`` must keep ``wandb.init`` inside
    ``run_or_wait_on_rank0``. A future refactor that moves the call back outside the
    helper would silently re-introduce the NCCL-hang regression — pure
    behavioural tests cannot catch that without spinning up a real wandb
    backend, so we pin the textual structure of the call site instead.

    Read the source file directly rather than importing the module so this
    test runs even in CPU-only venvs that do not install ``wandb``.
    """
    src_path = Path(__file__).resolve().parents[3] / "gr00t" / "experiment" / "experiment.py"
    src = src_path.read_text(encoding="utf-8")
    assert 'run_or_wait_on_rank0(label="wandb.init")' in src, (
        "experiment.run must wrap wandb.init in run_or_wait_on_rank0(label='wandb.init'); "
        "the wrap was removed or its label changed, which silently re-opens the "
        "rank-0-raise → NCCL-hang regression."
    )


@pytest.mark.timeout(WORKER_TIMEOUT_S)
def test_gloo_pg_smoke():
    """Sanity-check the gloo file:// rendezvous works on this CI host."""
    with tempfile.TemporaryDirectory() as scratch_dir:
        _spawn(_smoke_worker, scratch_dir)
        expected_sum = sum(range(WORLD_SIZE))
        for rank in range(WORLD_SIZE):
            payload = Path(scratch_dir, f"smoke_rank_{rank}_sum").read_text()
            assert float(payload) == float(expected_sum)
