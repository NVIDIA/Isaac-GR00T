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

"""pytest hooks to make CI logs easier to read."""

from __future__ import annotations

import os
import time


_test_start_times: dict[str, float] = {}


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
    _configure_shared_caches()


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    _test_start_times[nodeid] = time.perf_counter()
    print(f"\n\n{'=' * 80}\n[TEST START] {nodeid}\n", flush=True)


def pytest_runtest_logfinish(nodeid: str, location: tuple) -> None:
    elapsed = time.perf_counter() - _test_start_times.pop(nodeid, time.perf_counter())
    print(f"\n[TEST END]   {nodeid}  ({elapsed:.1f}s)\n{'=' * 80}\n\n", flush=True)
