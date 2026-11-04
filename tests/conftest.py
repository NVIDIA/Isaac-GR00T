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

"""Shared pytest configuration."""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
import sys
import tempfile

from filelock import FileLock
import pytest


def _retain_failure_output(report) -> None:
    """Keep complete failed-test output in CI artifacts and bound console sections."""
    directory = os.environ.get("PYTEST_FAILURE_LOG_DIR")
    if not directory or not report.failed:
        return
    identity = hashlib.sha256((report.nodeid + report.when).encode()).hexdigest()[:16]
    path = Path(directory) / f"{identity}-{report.when}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            stream.write(f"{report.nodeid} ({report.when})\n\n{report.longrepr}\n")
            for name, content in report.sections:
                stream.write(f"\n=== {name} ===\n{content}\n")
        note = f"Complete failure log: {path}"
    except OSError as error:
        # Artifact storage trouble must not hide the original test failure.
        note = f"Could not retain full failure log at {path}: {type(error).__name__}"
        report.sections.append(("Failure log retention", note))
    limit = 8192
    report.sections[:] = [
        (
            name,
            f"[Output truncated; {note}]\n" + content[-limit:] if len(content) > limit else content,
        )
        for name, content in report.sections
    ]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _retain_failure_output(outcome.get_result())


def _pin_xdist_worker_to_gpu() -> None:
    """Pin each pytest-xdist worker to a single GPU.

    Runs at conftest import time, which is *before* any test module
    (and therefore any ``import torch``) executes inside the worker
    subprocess.  pytest-xdist exposes the worker id as ``PYTEST_XDIST_WORKER``
    (e.g. ``gw0``, ``gw1``).  We map ``gwN`` to the Nth GPU visible to the
    parent process so each worker sees exactly one GPU and they don't
    contend for memory.

    No-op when running outside xdist (single-process pytest).
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker or not worker.startswith("gw"):
        return
    try:
        idx = int(worker[2:])
    except ValueError:
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        gpus = [g for g in visible.split(",") if g.strip()]
        if 0 <= idx < len(gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[idx]
            return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)


_pin_xdist_worker_to_gpu()


def _cgroup_cpu_quota() -> int | None:
    """Whole CPUs allowed by a cgroup v2 or v1 CFS quota, or None without one."""
    cgroup = Path("/sys/fs/cgroup")
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()
    except (OSError, ValueError):
        try:
            quota = (cgroup / "cpu/cpu.cfs_quota_us").read_text().strip()
            period = (cgroup / "cpu/cpu.cfs_period_us").read_text().strip()
        except OSError:
            return None
    if not quota.isdigit() or not period.isdigit() or int(period) == 0:
        return None  # "max" (v2) or -1 (v1) means no quota.
    return max(1, int(quota) // int(period))


def _available_cpus() -> int:
    """CPUs this process may use, including a cgroup CPU quota."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    quota = _cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus or quota, quota)
    return cpus or 1


def _limit_xdist_worker_threads() -> None:
    """Share the CPUs among pytest-xdist workers instead of oversubscribing them.

    Each worker's torch would otherwise start one OpenMP thread per host core,
    so eight workers on a CPU-limited pod thrash. Earlier collection may
    already have imported torch, so its thread pool is resized as well.
    An explicit OMP_NUM_THREADS or MKL_NUM_THREADS is kept. torch.set_num_threads
    sets both pools, so it uses the smaller explicit limit, if any. Values that
    are not a positive integer, such as nested OpenMP lists, are left to the
    runtime and not used for resizing.
    """
    workers = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if not workers or not workers.isdigit() or int(workers) == 0:
        return
    threads = max(1, _available_cpus() // int(workers))
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    explicit = [int(v) for n in names if (v := os.environ.get(n, "")).isdigit() and int(v) > 0]
    for name in names:
        os.environ.setdefault(name, str(threads))
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(min(explicit, default=threads))


_limit_xdist_worker_threads()


def _configure_shared_caches() -> None:
    """Set shared cache env vars in os.environ once for the whole test session.

    HF cache dirs are content-addressed, so all test groups safely share one
    location.  UV_PROJECT_ENVIRONMENT forwards the active venv to uv
    subprocesses so ``uv run`` uses the same installed packages as the test
    runner.  Tests that need an isolated venv (e.g. SO100's lerobot_conversion
    step) can strip UV_PROJECT_ENVIRONMENT from their local env dict.
    """
    from test_support.runtime import build_shared_hf_cache_env, resolve_shared_uv_cache_dir

    # Single shared HF cache for all test groups.
    hf_env = build_shared_hf_cache_env("shared")
    os.environ.update(hf_env)

    if hf_env:
        print(
            f"\n[conftest] shared HF cache: {hf_env.get('HF_HOME', 'default')}",
            flush=True,
        )

    uv_cache = resolve_shared_uv_cache_dir()
    if uv_cache is not None:
        os.environ["UV_CACHE_DIR"] = str(uv_cache)
        print(f"[conftest] UV_CACHE_DIR={uv_cache}", flush=True)

    # Forward the active venv to uv subprocesses.
    if not os.environ.get("UV_PROJECT_ENVIRONMENT"):
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv:
            os.environ["UV_PROJECT_ENVIRONMENT"] = venv
            print(f"[conftest] UV_PROJECT_ENVIRONMENT={venv}", flush=True)


def pytest_configure(config) -> None:  # noqa: ARG001
    # Set before any test runs so subprocesses launched via run_bash_blocks /
    # uv run inherit it — PYTEST_CURRENT_TEST alone can be cleared by uv.
    os.environ["GROOT_PATCH_MISTRAL"] = "1"
    os.environ["GROOT_HF_LOCAL_FIRST"] = "1"
    os.environ.setdefault("GROOT_SKIP_HF_MODEL_WEIGHTS", "1")
    _configure_shared_caches()


@pytest.fixture
def serialize_subprocess_spawns():
    """Let only one subprocess-spawning test run at a time, host-wide.

    Under ``-n auto`` every core is already claimed by an xdist worker, so a
    test that additionally spawns its own torch-importing processes
    oversubscribes the box and can blow its wall-clock timeout. A host-wide
    lock serializes such tests across workers; they finish in well under a
    second when not starved, so the serialization is effectively free.
    """
    lock_path = Path(tempfile.gettempdir()) / "gr00t-test-subprocess-spawn.lock"
    with FileLock(str(lock_path)):
        yield


@pytest.fixture(scope="session")
def load_hf_model_weights():
    """Temporarily opt a test into normal Hugging Face checkpoint weight loading."""

    @contextlib.contextmanager
    def _enabled():
        previous = os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS")
        os.environ["GROOT_SKIP_HF_MODEL_WEIGHTS"] = "0"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("GROOT_SKIP_HF_MODEL_WEIGHTS", None)
            else:
                os.environ["GROOT_SKIP_HF_MODEL_WEIGHTS"] = previous

    return _enabled
