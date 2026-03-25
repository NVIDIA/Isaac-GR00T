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

"""
Utils used to point subprocess environments to shared uv cache and venvs.
"""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import time


SHARED_DRIVE_ROOT = pathlib.Path("/shared")


def resolve_shared_uv_cache_dir(cache_key: str) -> pathlib.Path | None:
    """Return a writable shared uv cache path for the given key, or None."""
    cache_dir = SHARED_DRIVE_ROOT / f"uv-cache/{cache_key}"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except OSError:
        print(
            f"[cache] warning: shared uv cache unavailable at {cache_dir}; "
            "falling back to uv default cache dir"
        )
        return None


def build_shared_hf_cache_env(cache_key: str) -> dict[str, str]:
    """Build HF cache environment variables under shared storage for a cache key."""
    hf_cache_dir = SHARED_DRIVE_ROOT / f"hf-cache/{cache_key}"
    try:
        hub_cache_dir = hf_cache_dir / "hub"
        transformers_cache_dir = hf_cache_dir / "transformers"
        datasets_cache_dir = hf_cache_dir / "datasets"
        hub_cache_dir.mkdir(parents=True, exist_ok=True)
        transformers_cache_dir.mkdir(parents=True, exist_ok=True)
        datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(
            f"[cache] warning: shared Hugging Face cache unavailable at {hf_cache_dir}; "
            "falling back to defaults"
        )
        return {}

    return {
        "HF_HOME": str(hf_cache_dir),
        "HF_HUB_CACHE": str(hub_cache_dir),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache_dir),
        "TRANSFORMERS_CACHE": str(transformers_cache_dir),
        "HF_DATASETS_CACHE": str(datasets_cache_dir),
    }


def build_uv_runtime_env(
    *,
    uv_cache_dir: pathlib.Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a runtime env with shared uv cache and venv selection."""
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    if uv_cache_dir is not None:
        env["UV_CACHE_DIR"] = str(uv_cache_dir)

    # Prefer the currently active venv when present.
    if os.environ.get("UV_PROJECT_ENVIRONMENT"):
        env["UV_PROJECT_ENVIRONMENT"] = os.environ["UV_PROJECT_ENVIRONMENT"]
    elif os.environ.get("VIRTUAL_ENV"):
        env["UV_PROJECT_ENVIRONMENT"] = os.environ["VIRTUAL_ENV"]

    return env


def build_shared_runtime_env(
    cache_key: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build runtime env with shared uv cache and shared HF cache for a key."""
    merged_extra_env = {**build_shared_hf_cache_env(cache_key)}
    if extra_env:
        merged_extra_env.update(extra_env)
    return build_uv_runtime_env(
        uv_cache_dir=resolve_shared_uv_cache_dir(cache_key),
        extra_env=merged_extra_env,
    )


def wait_for_server_ready(
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout_s: float,
) -> None:
    """Wait until the server accepts TCP connections, or raise if it dies/times out."""
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            raise AssertionError(f"Model server failed to start.\nreturncode={proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                elapsed = time.monotonic() - deadline + timeout_s
                print(f"Model server ready after {elapsed:.1f}s.")
                return
        except OSError:
            if time.monotonic() >= deadline:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=15)
                raise AssertionError(
                    "Model server did not become ready before timeout.\n"
                    f"timeout_seconds={timeout_s}\n"
                    "Set the corresponding env var to override."
                )
            time.sleep(0.5)


def run_subprocess_step(
    cmd: list[str],
    *,
    step: str,
    cwd: pathlib.Path,
    env: dict[str, str],
    timeout_s: int | float | None = None,
    stream_output: bool = False,
    log_prefix: str = "examples",
    failure_prefix: str = "Subprocess step failed",
    output_tail_chars: int = 8000,
) -> tuple[subprocess.CompletedProcess, float]:
    """Run a subprocess step with consistent timing/logging/failure formatting."""
    print(f"[{log_prefix}] step={step} command={' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    run_kwargs = {
        "cwd": cwd,
        "env": env,
        "check": False,
    }
    if timeout_s is not None:
        run_kwargs["timeout"] = timeout_s
    if not stream_output:
        run_kwargs["capture_output"] = True
        run_kwargs["text"] = True
    result = subprocess.run(cmd, **run_kwargs)
    elapsed_s = time.perf_counter() - start
    print(f"[{log_prefix}] step={step} elapsed_s={elapsed_s:.2f}", flush=True)

    if result.returncode != 0:
        if stream_output:
            output_info = "See streamed test logs above for subprocess output."
        else:
            output = (result.stdout or "") + (result.stderr or "")
            output_info = f"output_tail=\n{output[-output_tail_chars:]}"
        raise AssertionError(
            f"{failure_prefix}: {step}\n"
            f"elapsed_s={elapsed_s:.2f}\n"
            f"returncode={result.returncode}\n"
            f"command={' '.join(cmd)}\n"
            f"{output_info}"
        )
    return result, elapsed_s
