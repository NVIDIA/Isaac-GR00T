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

"""Shared subprocess/runtime helpers for tests."""

from __future__ import annotations

import os
import pathlib
import re
import socket
import subprocess
import tempfile
import time


DEFAULT_SERVER_STARTUP_SECONDS = 600.0


def _default_cache_path() -> pathlib.Path:
    """Return the cache root directory."""
    if "TEST_CACHE_PATH" in os.environ:
        return pathlib.Path(os.environ["TEST_CACHE_PATH"])

    local_fallback = pathlib.Path.home() / ".cache" / "g00t"
    local_fallback.mkdir(parents=True, exist_ok=True)
    return local_fallback


TEST_CACHE_PATH = _default_cache_path()


def get_root() -> pathlib.Path:
    """Return the root directory of the repository."""
    return pathlib.Path(__file__).resolve().parents[1]


def resolve_shared_model_path(
    repo_id: str,
    *,
    subdir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> pathlib.Path:
    """Return a shared model path, downloading once if not present.

    Models are stored at ``SHARED_DRIVE_ROOT/models/<repo_name>/`` so all tests
    share a single copy.  If *subdir* is given the returned path points to that
    subdirectory (useful for HF repos with nested checkpoint folders).

    Args:
        repo_id: HuggingFace repo id, e.g. ``"nvidia/GR00T-N1.7-3B"``.
        subdir: Optional subdirectory within the downloaded repo.
        allow_patterns: Optional list of file patterns to download (passed to
            ``snapshot_download``).  If ``None`` the entire repo is fetched.
    """
    model_name = repo_id.split("/")[-1]
    model_root = SHARED_DRIVE_ROOT / "models" / model_name
    target = model_root / subdir if subdir else model_root

    # Quick check: already downloaded?
    if target.is_dir() and any(target.iterdir()):
        return target

    token = os.environ.get("HF_TOKEN", "")
    assert token, "HF_TOKEN is required to download gated models. Set via: export HF_TOKEN=hf_..."

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_root),
        token=token,
        **({"allow_patterns": allow_patterns} if allow_patterns else {}),
    )
    return target


EGL_VENDOR_DIRS = [
    pathlib.Path("/usr/share/glvnd/egl_vendor.d"),
    pathlib.Path("/etc/glvnd/egl_vendor.d"),
    pathlib.Path("/usr/local/share/glvnd/egl_vendor.d"),
]


def hf_hub_download_cmd(repo_id: str, filename: str, local_dir: str) -> list[str]:
    """Build a ``uv run python -c`` command that downloads a file from HuggingFace.

    Reads HF_TOKEN from the environment and passes it explicitly so gated repos
    work without requiring ``huggingface-cli login``.  Raises AssertionError if
    HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN", "")
    assert token, (
        "HF_TOKEN environment variable is not set. "
        "A HuggingFace token with access to gated repos is required. "
        "Set it via: export HF_TOKEN=hf_..."
    )
    return [
        "uv",
        "run",
        "python",
        "-c",
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id={repo_id!r}, filename={filename!r}, "
        f"local_dir={local_dir!r}, token={token!r})",
    ]


# GPU names that contain these tokens are known to have RT cores.
# Compute-only data-center GPUs (A100, H100, H200, B200, V100, etc.) do not.
_RT_CORE_GPU_PATTERNS = (
    r"\brtx\b",  # RTX 20xx/30xx/40xx/50xx, Quadro RTX, RTX Ax000
    r"\bl40\b",  # L40 / L40S
    r"\bl4\b",  # L4
)


def has_rt_core_gpu() -> bool:
    """Return True if any available GPU has RT cores (required for Vulkan ray tracing).

    Checks ``nvidia-smi`` GPU names against known RT-capable product lines.
    Returns False if nvidia-smi is unavailable or no matching GPU is found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False
        for name in result.stdout.strip().splitlines():
            if any(re.search(pat, name.strip().lower()) for pat in _RT_CORE_GPU_PATTERNS):
                return True
    except Exception:
        pass
    return False


def find_nvidia_egl_vendor_file() -> pathlib.Path:
    """Return the first NVIDIA EGL vendor JSON file found, or raise FileNotFoundError."""
    for vendor_dir in EGL_VENDOR_DIRS:
        for candidate in vendor_dir.glob("*nvidia*.json") if vendor_dir.is_dir() else []:
            return candidate
    searched = ", ".join(str(d) for d in EGL_VENDOR_DIRS)
    raise FileNotFoundError(
        f"NVIDIA EGL vendor file not found (searched: {searched}). "
        "robosuite requires EGL_PLATFORM_DEVICE_EXT which is only provided by the "
        "NVIDIA EGL implementation. Install the NVIDIA GL/EGL packages or run on a "
        "host with the full NVIDIA driver stack."
    )


def resolve_shared_uv_cache_dir() -> pathlib.Path | None:
    """Return a writable uv cache path, or None.

    Only redirects the uv cache when TEST_CACHE_PATH is set — on dev
    machines uv's default cache (~/.cache/uv) is already local and fast, so
    there is no benefit to overriding it.
    """
    if "TEST_CACHE_PATH" not in os.environ:
        return None
    cache_dir = TEST_CACHE_PATH / "uv-cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except OSError:
        print(
            f"[cache] warning: uv cache unavailable at {cache_dir}; "
            "falling back to uv default cache dir"
        )
        return None


def build_shared_hf_cache_env(cache_key: str) -> dict[str, str]:
    """Build HF cache environment variables for a cache key."""
    hf_cache_dir = TEST_CACHE_PATH / f"hf-cache/{cache_key}"
    try:
        hub_cache_dir = hf_cache_dir / "hub"
        transformers_cache_dir = hf_cache_dir / "transformers"
        datasets_cache_dir = hf_cache_dir / "datasets"
        hub_cache_dir.mkdir(parents=True, exist_ok=True)
        transformers_cache_dir.mkdir(parents=True, exist_ok=True)
        datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(
            f"[cache] warning: Hugging Face cache unavailable at {hf_cache_dir}; "
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
    """Build a runtime env with uv cache and venv selection.

    UV_PROJECT_ENVIRONMENT tells uv which venv to use when running subprocesses
    (e.g. ``uv run python ...``). We forward the currently active venv so that
    subprocesses use the same installed packages as the test runner — both in CI
    and on dev machines where the developer runs inside a local venv.
    """
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    if uv_cache_dir is not None:
        env["UV_CACHE_DIR"] = str(uv_cache_dir)

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
    """Build runtime env with uv cache and per-test HF cache."""
    uv_cache_dir = resolve_shared_uv_cache_dir()
    merged_extra_env = {**build_shared_hf_cache_env(cache_key)}
    if extra_env:
        merged_extra_env.update(extra_env)
    env = build_uv_runtime_env(uv_cache_dir=uv_cache_dir, extra_env=merged_extra_env)

    cache_source = "TEST_CACHE_PATH" if "TEST_CACHE_PATH" in os.environ else "local fallback"
    uv_cache_str = str(uv_cache_dir) if uv_cache_dir is not None else "uv default"
    uv_venv = env.get("UV_PROJECT_ENVIRONMENT", "uv default")
    uv_venv_source = (
        "UV_PROJECT_ENVIRONMENT"
        if os.environ.get("UV_PROJECT_ENVIRONMENT")
        else "VIRTUAL_ENV"
        if os.environ.get("VIRTUAL_ENV")
        else "unset"
    )
    hf_home = env.get("HF_HOME", "hf default")
    print(
        f"[cache] cache_path={TEST_CACHE_PATH} ({cache_source})"
        f" uv_cache={uv_cache_str}"
        f" uv_venv={uv_venv} ({uv_venv_source})"
        f" hf_home={hf_home} (key={cache_key})",
        flush=True,
    )
    return env


def assert_port_available(host: str, port: int) -> None:
    """Raise AssertionError if the port is already bound.

    Call this before starting a model server subprocess to catch port conflicts
    early (e.g. a leftover process from a previous test run or two tests
    inadvertently assigned the same port).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as exc:
            raise AssertionError(
                f"Port {port} on {host} is already in use. "
                "Each test file uses a unique port — check for a conflicting "
                "process or a previous test run that did not shut down cleanly."
            ) from exc


def start_server_process(
    server_code: str,
    *,
    cwd: pathlib.Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen, pathlib.Path]:
    """Start a model server subprocess with stderr captured to a temp file.

    Returns the Popen object and the path to the stderr log file.  On failure
    the caller should read and print the log so CI output includes the error.
    """
    stderr_log = pathlib.Path(tempfile.mktemp(prefix="server_stderr_", suffix=".log"))
    stderr_fh = open(stderr_log, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        ["bash", "-c", server_code],
        cwd=cwd,
        env=env,
        stdout=stderr_fh,
        stderr=stderr_fh,
    )
    return proc, stderr_log


def _dump_server_log(log_path: pathlib.Path, tail_chars: int = 8000) -> str:
    """Read the tail of a server log file and return it as a string."""
    try:
        text = log_path.read_text()
        return text[-tail_chars:] if len(text) > tail_chars else text
    except OSError:
        return "<server log not available>"


def wait_for_server_ready(
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout_s: float,
    server_log: pathlib.Path | None = None,
) -> None:
    """Wait until the server accepts TCP connections, or raise if it dies/times out."""
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            log_info = ""
            if server_log is not None:
                log_info = f"\nServer output:\n{_dump_server_log(server_log)}"
            raise AssertionError(
                f"Model server failed to start.\nreturncode={proc.returncode}{log_info}"
            )
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
                log_info = ""
                if server_log is not None:
                    log_info = f"\nServer output:\n{_dump_server_log(server_log)}"
                raise AssertionError(
                    "Model server did not become ready before timeout.\n"
                    f"timeout_seconds={timeout_s}\n"
                    f"Set the corresponding env var to override.{log_info}"
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
