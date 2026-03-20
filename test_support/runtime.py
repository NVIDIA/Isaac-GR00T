"""Shared subprocess/runtime helpers for tests."""

from __future__ import annotations

import os
import pathlib
import re
import socket
import subprocess
import time


def _default_shared_drive_root() -> pathlib.Path:
    """Return the shared drive root, falling back to a local cache dir.

    In CI, CI_SHARED_DRIVE_PATH points to a shared PVC mounted across all
    runners, so assets/models/submodules downloaded on one run are reused on
    the next without re-downloading.

    On dev machines (CI_SHARED_DRIVE_PATH unset), the fallback is
    ~/.cache/g00t — a local directory that plays the same caching role so
    tests work identically without any extra setup.
    """
    if "CI_SHARED_DRIVE_PATH" in os.environ:
        return pathlib.Path(os.environ["CI_SHARED_DRIVE_PATH"])

    local_fallback = pathlib.Path.home() / ".cache" / "g00t"
    local_fallback.mkdir(parents=True, exist_ok=True)
    return local_fallback


# In CI: shared PVC path from CI_SHARED_DRIVE_PATH.
# On dev machines: ~/.cache/g00t (created automatically).
SHARED_DRIVE_ROOT = _default_shared_drive_root()


def get_root() -> pathlib.Path:
    """Return the root directory of the repository."""
    return pathlib.Path(__file__).resolve().parents[1]


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
    """Return a writable shared uv cache path common to all tests, or None.

    Only redirects the uv cache when CI_SHARED_DRIVE_PATH is set — on dev
    machines uv's default cache (~/.cache/uv) is already local and fast, so
    there is no benefit to overriding it.
    """
    if "CI_SHARED_DRIVE_PATH" not in os.environ:
        return None
    cache_dir = SHARED_DRIVE_ROOT / "uv-cache"
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
    """Build a runtime env with shared uv cache and venv selection.

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
    """Build runtime env with shared uv cache and per-test HF cache."""
    uv_cache_dir = resolve_shared_uv_cache_dir()
    merged_extra_env = {**build_shared_hf_cache_env(cache_key)}
    if extra_env:
        merged_extra_env.update(extra_env)
    env = build_uv_runtime_env(uv_cache_dir=uv_cache_dir, extra_env=merged_extra_env)

    shared_drive_source = (
        "CI_SHARED_DRIVE_PATH" if "CI_SHARED_DRIVE_PATH" in os.environ else "local fallback"
    )
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
        f"[cache] shared_drive_root={SHARED_DRIVE_ROOT} ({shared_drive_source})"
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
