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

"""Regression tests for `BestMetricCheckpointCallback`.

Three layers, cheapest first:

* **Algorithm (single-rank, mocked Trainer, CPU):** improvement detection,
  directory bookkeeping, ``exp_cfg_dir`` copy, previous-best replacement.
* **Distributed decision (multi-rank gloo, CPU):** ``_broadcast_save_decision``
  routes rank-0's verdict to every rank, and the full callback drives every rank
  into ``Trainer.save_model`` (so a real ZeRO-3 gather would not deadlock).
* **Sharded save parity (multi-GPU NCCL, ``@pytest.mark.multigpu``):** a real FSDP
  full-shard run proves the saved checkpoint holds consolidated full weights rather
  than a single rank's shard — the exact corruption that motivated this fix. FSDP
  stands in for DeepSpeed ZeRO-3 (identical callback path; DeepSpeed wheels are
  x86_64-only and absent on the arm64 multi-GPU runner). The heavy lifting lives in
  ``_run_best_metric_fsdp_save.py`` (torchrun entry).
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from unittest.mock import MagicMock

from gr00t.experiment.utils import BestMetricCheckpointCallback, _broadcast_save_decision
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Lightweight stand-ins for HF dataclasses. We stick to attribute access only;
# the callback never calls methods on these objects.
# ---------------------------------------------------------------------------


@dataclass
class _FakeArgs:
    output_dir: str


@dataclass
class _FakeState:
    is_world_process_zero: bool = True
    global_step: int = 100


@dataclass
class _FakeControl:
    pass


def _make_callback(
    *,
    tmp_path: Path,
    greater_is_better: bool = True,
    exp_cfg_dir: Path | None = None,
) -> tuple[BestMetricCheckpointCallback, MagicMock]:
    """Build a callback wired to a fresh MagicMock Trainer.

    Returns the ``(callback, trainer)`` pair so tests can assert on
    ``trainer.save_model`` call counts and arguments without spinning up
    a real HF ``Trainer``.
    """
    trainer = MagicMock(name="Trainer")
    cb = BestMetricCheckpointCallback(
        metric_name="eval_accuracy",
        trainer=trainer,
        greater_is_better=greater_is_better,
        exp_cfg_dir=exp_cfg_dir,
    )
    return cb, trainer


def _invoke(
    cb: BestMetricCheckpointCallback,
    *,
    tmp_path: Path,
    metrics: dict | None,
    is_rank0: bool = True,
    global_step: int = 100,
) -> None:
    """Drive ``cb.on_evaluate`` with minimal fake HF args/state/control.

    Lets each algorithm-layer test exercise the callback at a single
    eval step without instantiating real HF dataclasses.
    """
    cb.on_evaluate(
        args=_FakeArgs(output_dir=str(tmp_path)),
        state=_FakeState(is_world_process_zero=is_rank0, global_step=global_step),
        control=_FakeControl(),
        metrics=metrics,
        model=MagicMock(name="model"),
    )


# ---------------------------------------------------------------------------
# Algorithm layer — single-rank, mocked trainer
# ---------------------------------------------------------------------------


def test_does_not_save_when_metrics_is_none(tmp_path):
    cb, trainer = _make_callback(tmp_path=tmp_path)
    _invoke(cb, tmp_path=tmp_path, metrics=None)
    trainer.save_model.assert_not_called()
    assert cb._best_checkpoint_dir is None
    assert cb.best_metric == -float("inf")


def test_does_not_save_when_metric_name_missing(tmp_path):
    cb, trainer = _make_callback(tmp_path=tmp_path)
    _invoke(cb, tmp_path=tmp_path, metrics={"other_metric": 0.9})
    trainer.save_model.assert_not_called()
    assert cb.best_metric == -float("inf")


def test_does_not_save_when_metric_does_not_improve(tmp_path):
    cb, trainer = _make_callback(tmp_path=tmp_path)
    cb.best_metric = 0.95
    _invoke(cb, tmp_path=tmp_path, metrics={"eval_accuracy": 0.80})
    trainer.save_model.assert_not_called()
    assert cb.best_metric == 0.95


def test_saves_when_metric_improves_greater_is_better(tmp_path):
    cb, trainer = _make_callback(tmp_path=tmp_path)
    _invoke(cb, tmp_path=tmp_path, metrics={"eval_accuracy": 0.85}, global_step=42)

    expected_dir = tmp_path / "checkpoint-42-best-eval_accuracy_0.85"
    trainer.save_model.assert_called_once_with(str(expected_dir))
    assert expected_dir.is_dir(), "rank-0 should have mkdir'd the output dir"
    assert cb.best_metric == pytest.approx(0.85)
    assert cb._best_checkpoint_dir == str(expected_dir)


def test_saves_when_metric_improves_lower_is_better(tmp_path):
    cb, trainer = _make_callback(tmp_path=tmp_path, greater_is_better=False)
    _invoke(cb, tmp_path=tmp_path, metrics={"eval_accuracy": 0.20}, global_step=42)

    expected_dir = tmp_path / "checkpoint-42-best-eval_accuracy_0.2"
    trainer.save_model.assert_called_once_with(str(expected_dir))
    assert cb.best_metric == pytest.approx(0.20)


def test_copies_exp_cfg_dir_on_save(tmp_path):
    exp_cfg = tmp_path / "experiment_cfg"
    exp_cfg.mkdir()
    (exp_cfg / "conf.yaml").write_text("dummy: 1\n")
    cb, _ = _make_callback(tmp_path=tmp_path / "out", exp_cfg_dir=exp_cfg)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cb.on_evaluate(
        args=_FakeArgs(output_dir=str(out_dir)),
        state=_FakeState(global_step=10),
        control=_FakeControl(),
        metrics={"eval_accuracy": 0.5},
        model=MagicMock(),
    )

    copied = out_dir / "checkpoint-10-best-eval_accuracy_0.5" / "experiment_cfg" / "conf.yaml"
    assert copied.is_file(), "exp_cfg_dir should be copied into the best-checkpoint dir"


def test_no_copy_when_exp_cfg_dir_does_not_exist(tmp_path):
    """If exp_cfg_dir is configured but absent on disk, the save still
    proceeds and the missing directory is simply not copied — same
    behavior as before this refactor."""
    cb, trainer = _make_callback(tmp_path=tmp_path, exp_cfg_dir=tmp_path / "missing")
    _invoke(cb, tmp_path=tmp_path, metrics={"eval_accuracy": 0.5})
    trainer.save_model.assert_called_once()
    assert not (tmp_path / "checkpoint-100-best-eval_accuracy_0.5" / "missing").exists()


def test_previous_best_dir_is_replaced_on_each_improvement(tmp_path):
    cb, trainer = _make_callback(tmp_path=tmp_path)

    _invoke(cb, tmp_path=tmp_path, metrics={"eval_accuracy": 0.6}, global_step=10)
    first_dir = tmp_path / "checkpoint-10-best-eval_accuracy_0.6"
    assert first_dir.is_dir()
    assert cb._best_checkpoint_dir == str(first_dir)

    _invoke(cb, tmp_path=tmp_path, metrics={"eval_accuracy": 0.8}, global_step=20)
    second_dir = tmp_path / "checkpoint-20-best-eval_accuracy_0.8"
    assert second_dir.is_dir()
    assert not first_dir.exists(), "previous best-checkpoint dir should be removed"
    assert cb._best_checkpoint_dir == str(second_dir)
    assert trainer.save_model.call_count == 2


def test_first_improvement_does_not_rmtree_anything(tmp_path):
    """The rmtree branch must short-circuit on the very first save (when
    ``_best_checkpoint_dir`` is still None) instead of crashing on
    ``Path(None)``."""
    cb, trainer = _make_callback(tmp_path=tmp_path)
    _invoke(cb, tmp_path=tmp_path, metrics={"eval_accuracy": 0.5})
    trainer.save_model.assert_called_once()
    assert cb._best_checkpoint_dir is not None


# ---------------------------------------------------------------------------
# Distributed-decision layer — multi-rank with gloo
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an unused TCP port suitable for ``init_method='tcp://...'``.

    Binds to port 0, reads back the OS-assigned port, then releases the
    socket — the standard "let the kernel pick" idiom. There is an
    unavoidable race between release and the next bind, but for spawn-
    style multiprocess tests this is the conventional approach.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_gloo_workers(worker, *worker_args, world_size: int = 2) -> dict:
    """Spawn ``world_size`` gloo ranks running ``worker`` and return their results.

    Centralizes the per-test boilerplate (free port, ``mp.Manager`` dict,
    ``mp.spawn``) so each multi-rank test only supplies its worker and the
    worker-specific arguments. ``worker`` is invoked as
    ``worker(rank, world_size, init_method, *worker_args, return_dict)`` and is
    expected to write its observable outcome into ``return_dict[rank]``.
    """
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        worker,
        args=(world_size, init_method, *worker_args, results),
        nprocs=world_size,
        join=True,
    )
    return results


def _broadcast_worker(rank, world_size, init_method, save_flag, metric_value, return_dict):
    """Per-rank worker for the ``_broadcast_save_decision`` gloo test.

    Rank-0 contributes the true ``(save_flag, metric_value)``; every
    other rank passes placeholder zeros. After the broadcast each rank
    writes its observed pair into ``return_dict[rank]`` so the parent
    test can assert all ranks landed on rank-0's truth.
    """
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    try:
        my_flag = save_flag if rank == 0 else 0
        my_value = metric_value if rank == 0 else 0.0
        out_flag, out_value = _broadcast_save_decision(my_flag, my_value)
        return_dict[rank] = (out_flag, out_value)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("save_flag", "metric_value"),
    [
        (1, 0.42),  # rank-0 says "save with metric 0.42"
        (0, 0.0),  # rank-0 says "skip this round"
    ],
)
def test_broadcast_save_decision_routes_rank0_truth_to_all_ranks(save_flag, metric_value):
    results = _run_gloo_workers(_broadcast_worker, save_flag, metric_value)
    assert set(results.keys()) == {0, 1}
    for rank in (0, 1):
        out_flag, out_value = results[rank]
        assert out_flag == save_flag, f"rank {rank} got flag {out_flag}, expected {save_flag}"
        assert out_value == pytest.approx(metric_value), (
            f"rank {rank} got value {out_value}, expected {metric_value}"
        )


def _callback_collective_worker(
    rank, world_size, init_method, tmp_dir, rank0_has_metrics, return_dict
):
    """End-to-end: every rank constructs the callback, on_evaluate is called
    with `metrics=None` everywhere except rank-0 (mirroring the HF behavior
    we're trying to be robust against), and we record whether each rank's
    `trainer.save_model` was invoked. The contract: every rank must hit
    save_model, otherwise rank-0 would deadlock in a real ZeRO-3 gather.
    """
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    try:
        trainer = MagicMock(name=f"Trainer-rank{rank}")
        cb = BestMetricCheckpointCallback(
            metric_name="eval_accuracy",
            trainer=trainer,
            greater_is_better=True,
            exp_cfg_dir=None,
        )
        # Rank-0 alone owns the metrics dict (worst case for the broadcast).
        metrics = {"eval_accuracy": 0.7} if (rank == 0 and rank0_has_metrics) else None
        cb.on_evaluate(
            args=_FakeArgs(output_dir=str(tmp_dir)),
            state=_FakeState(is_world_process_zero=(rank == 0), global_step=7),
            control=_FakeControl(),
            metrics=metrics,
            model=MagicMock(),
        )
        return_dict[rank] = trainer.save_model.call_count
    finally:
        dist.destroy_process_group()


def test_collective_save_runs_on_every_rank_when_rank0_decides_to_save(tmp_path):
    """The whole point of the broadcast: non-rank-0 ranks must still call
    `trainer.save_model` so the gather collective completes. Without the
    broadcast, rank-1 would early-return on `metrics is None` and rank-0
    would hang inside the gather."""
    results = _run_gloo_workers(_callback_collective_worker, str(tmp_path), True)
    assert results[0] == 1, "rank-0 should have called save_model exactly once"
    assert results[1] == 1, (
        "rank-1 should also have called save_model — otherwise a real "
        "ZeRO-3 gather would deadlock here"
    )


def test_no_collective_when_rank0_decides_to_skip(tmp_path):
    """Symmetry: when rank-0 sees no improvement, no rank should hit
    save_model. (Skipping the collective is fine because no rank entered it.)"""
    results = _run_gloo_workers(_callback_collective_worker, str(tmp_path), False)
    assert results[0] == 0
    assert results[1] == 0


# ---------------------------------------------------------------------------
# Pre-test environment guards
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_no_dist_state():
    """Tests may leave a dangling process group on failure paths; isolate."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    # Multiprocessing spawn requires a known start method on Linux.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Sharded-save parity layer — real FSDP full-shard across every visible GPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _visible_multigpu_count() -> int:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip(f"Need at least 2 visible GPUs for the FSDP parity test, got {num_gpus}")
    return num_gpus


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.timeout(900, func_only=True)
def test_best_metric_fsdp_save_matches_full_weights(tmp_path, _visible_multigpu_count):
    """The best-metric checkpoint saved under a real sharded backend must hold
    consolidated full weights, not rank-0's shard.

    Spawns ``_run_best_metric_fsdp_save.py`` under torchrun with one rank per
    visible GPU. That script wraps a tiny model in a real HF ``Trainer`` + FSDP
    full-shard, fires the callback's ``on_evaluate`` with metrics on rank-0 only,
    then asserts (on rank-0) that every saved parameter matches a pre-shard
    reference snapshot and that no tensor is zeroed. Pre-fix, the non-rank-0 shards
    would have been written un-consolidated and this would fail. FSDP stands in for
    DeepSpeed ZeRO-3 here — same callback path, but DeepSpeed wheels are x86_64-only
    and absent on the arm64 multi-GPU runner.
    """
    from test_support.runtime import get_root

    num_gpus = _visible_multigpu_count
    repo_root = get_root()
    runner = Path(__file__).with_name("_run_best_metric_fsdp_save.py")
    output_dir = tmp_path / "fsdp_best_metric_output"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--max-restarts=0",
        f"--nproc_per_node={num_gpus}",
        str(runner),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # Some GB200 CI nodes report disabled P2P between NVLINK-connected GPUs.
    env.setdefault("NCCL_IGNORE_DISABLED_P2P", "1")

    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=840)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, _ = proc.communicate()
        pytest.fail(f"FSDP best-metric save run timed out\n\n{stdout}")

    if proc.returncode != 0:
        pytest.fail(f"FSDP best-metric save run failed (exit {proc.returncode})\n\n{stdout}")

    assert "weight parity verified" in stdout, (
        f"runner did not report a successful parity check\n\n{stdout}"
    )
