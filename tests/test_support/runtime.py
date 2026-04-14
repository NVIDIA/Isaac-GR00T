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

import json
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
    return pathlib.Path(__file__).resolve().parents[2]


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


def checkpoint_tree_ready(path: pathlib.Path) -> bool:
    """Return True if *path* looks like a HuggingFace ``transformers`` checkpoint dir."""
    if not (path / "config.json").is_file():
        return False
    index_file = path / "model.safetensors.index.json"
    if not index_file.is_file():
        return True
    shards = set(json.loads(index_file.read_text()).get("weight_map", {}).values())
    return all((path / shard).is_file() for shard in shards)


# Backward-compatible alias used by other test files.
libero_checkpoint_tree_ready = checkpoint_tree_ready


def demo_dataset_tree_ready(path: pathlib.Path) -> bool:
    """Return True if *path* looks like a bundled LeRobot demo dataset.

    Checks for ``meta/modality.json``, at least one ``.parquet`` under ``data/``,
    and at least one ``.mp4`` under ``videos/``.
    """
    if not (path / "meta" / "modality.json").is_file():
        return False
    data_dir = path / "data"
    if not data_dir.is_dir() or not any(data_dir.rglob("*.parquet")):
        return False
    videos_dir = path / "videos"
    if not videos_dir.is_dir() or not any(videos_dir.rglob("*.mp4")):
        return False
    return True


# Backward-compatible alias used by other test files.
libero_demo_tree_ready = demo_dataset_tree_ready


# ---------------------------------------------------------------------------
# Generic model checkpoint resolver
# ---------------------------------------------------------------------------


def resolve_model_checkpoint_path(
    *,
    hf_repo_id: str,
    hf_subdir: str | None = None,
    path_override_env: str | None = None,
    repo_root: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve a GR00T model checkpoint, downloading from HuggingFace if needed.

    Resolution order:

    1. Environment variable *path_override_env* (must be a complete checkpoint).
    2. ``<repo_root>/checkpoints/<model_name>/<subdir>``.
    3. Git worktree toplevel + same relative ``checkpoints/...`` path.
    4. ``SHARED_DRIVE_ROOT/models/<model_name>/<subdir>``, downloading from
       HuggingFace when missing (requires ``HF_TOKEN``).

    Args:
        hf_repo_id: HuggingFace repo id, e.g. ``"nvidia/GR00T-N1.7-LIBERO"``.
        hf_subdir: Optional subdirectory within the repo (e.g. ``"libero_10"``).
        path_override_env: Name of an env var that, when set, overrides all other
            resolution.
        repo_root: Repository root (auto-detected if ``None``).
    """
    root = repo_root if repo_root is not None else get_root()
    model_name = hf_repo_id.split("/")[-1]
    rel_path = f"checkpoints/{model_name}/{hf_subdir}" if hf_subdir else f"checkpoints/{model_name}"

    if path_override_env:
        override = os.environ.get(path_override_env, "").strip()
        if override:
            p = pathlib.Path(override).expanduser().resolve()
            assert checkpoint_tree_ready(p), (
                f"{path_override_env} does not point to a complete checkpoint directory: {p}"
            )
            return p

    local = root / rel_path
    if checkpoint_tree_ready(local):
        return local

    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(root),
            text=True,
            timeout=30,
        ).strip()
        git_cp = pathlib.Path(toplevel) / rel_path
        if checkpoint_tree_ready(git_cp):
            return git_cp
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    allow = [f"{hf_subdir}/*"] if hf_subdir else None
    target = resolve_shared_model_path(hf_repo_id, subdir=hf_subdir, allow_patterns=allow)
    assert checkpoint_tree_ready(target), (
        f"Checkpoint at {target} is incomplete after resolve/download "
        "(check HF_TOKEN, network, and Hugging Face repo layout)."
    )
    return target


# ---------------------------------------------------------------------------
# Generic demo dataset resolver
# ---------------------------------------------------------------------------


def resolve_demo_dataset(
    *,
    dataset_name: str,
    path_override_env: str | None = None,
    global_env_var: str | None = None,
    hf_download_env_var: str | None = None,
    repo_root: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve a bundled LeRobot demo dataset by *dataset_name*.

    Resolution order (first match wins):

    1. *path_override_env* environment variable (per-test override).
    2. *global_env_var* environment variable (CI / local override).
    3. ``<repo_root>/demo_data/<dataset_name>`` — normal clone with Git LFS.
    4. ``SHARED_DRIVE_ROOT/datasets/<dataset_name>`` — shared PVC / local cache.
    5. If *hf_download_env_var* is set in the environment, download from HuggingFace
       into the shared path (requires ``HF_TOKEN``).

    Args:
        dataset_name: Directory name under ``demo_data/`` (e.g. ``"libero_demo"``
            or ``"droid_sample"``).
        path_override_env: Name of an env var for per-test path override.
        global_env_var: Name of an env var for a global path override
            (e.g. ``"LIBERO_DEMO_DATASET_PATH"``).
        hf_download_env_var: Name of an env var whose value is a HuggingFace
            *dataset* repo id; when set, the dataset is downloaded into shared
            storage.
        repo_root: Repository root (auto-detected if ``None``).
    """
    root = repo_root if repo_root is not None else get_root()

    if path_override_env:
        alt = os.environ.get(path_override_env, "").strip()
        if alt:
            resolved = pathlib.Path(alt).expanduser().resolve()
            assert demo_dataset_tree_ready(resolved), (
                f"{path_override_env} does not point to a complete demo dataset: {resolved}"
            )
            return resolved

    if global_env_var:
        env_path = os.environ.get(global_env_var, "").strip()
        if env_path:
            resolved = pathlib.Path(env_path).expanduser().resolve()
            assert demo_dataset_tree_ready(resolved), (
                f"{global_env_var} does not point to a complete demo dataset tree: {resolved}"
            )
            return resolved

    in_repo = root / "demo_data" / dataset_name
    if demo_dataset_tree_ready(in_repo):
        return in_repo

    shared = SHARED_DRIVE_ROOT / "datasets" / dataset_name
    if demo_dataset_tree_ready(shared):
        return shared

    if hf_download_env_var:
        hf_dataset = os.environ.get(hf_download_env_var, "").strip()
        if hf_dataset:
            token = os.environ.get("HF_TOKEN", "")
            assert token, (
                f"HF_TOKEN is required to download {hf_download_env_var} into shared storage"
            )
            shared.parent.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=hf_dataset,
                repo_type="dataset",
                local_dir=str(shared),
                token=token,
            )
            assert demo_dataset_tree_ready(shared), (
                f"Downloaded HuggingFace dataset into {shared} but it does not match "
                "expected LeRobot demo dataset layout"
            )
            return shared

    raise AssertionError(
        f"{dataset_name} dataset not found. "
        f"It ships in-repo under demo_data/{dataset_name} (requires Git LFS). "
        f"Alternatives: set {global_env_var or path_override_env or 'a path env var'} "
        f"to an existing checkout; populate {shared} on the shared drive "
        "(CI_SHARED_DRIVE_PATH / ~/.cache/g00t)."
    )


# ---------------------------------------------------------------------------
# LIBERO-specific convenience wrappers (backward compatibility)
# ---------------------------------------------------------------------------

_LIBERO_N17_LIBERO_REPO = "nvidia/GR00T-N1.7-LIBERO"
_LIBERO_N17_LIBERO_SUBDIR = "libero_10"


def resolve_libero_n17_libero10_checkpoint_path(
    repo_root: pathlib.Path | None = None,
    *,
    path_override_env: str,
) -> pathlib.Path:
    """Resolve the LIBERO-finetuned GR00T-N1.7 checkpoint (``libero_10`` subfolder).

    Thin wrapper around :func:`resolve_model_checkpoint_path` kept for backward
    compatibility with existing callers.
    """
    return resolve_model_checkpoint_path(
        hf_repo_id=_LIBERO_N17_LIBERO_REPO,
        hf_subdir=_LIBERO_N17_LIBERO_SUBDIR,
        path_override_env=path_override_env,
        repo_root=repo_root,
    )


def resolve_libero_demo_dataset_path(
    repo_root: pathlib.Path | None = None,
    *,
    path_override_env: str | None = None,
) -> pathlib.Path:
    """Return the path to the LIBERO ``libero_demo`` dataset.

    Thin wrapper around :func:`resolve_demo_dataset` kept for backward
    compatibility with existing callers.
    """
    return resolve_demo_dataset(
        dataset_name="libero_demo",
        path_override_env=path_override_env,
        global_env_var="LIBERO_DEMO_DATASET_PATH",
        hf_download_env_var="GR00T_LIBERO_DEMO_HF_DATASET",
        repo_root=repo_root,
    )


_DROID_N17_REPO = "nvidia/GR00T-N1.7-DROID"


def resolve_droid_n17_checkpoint_path(
    repo_root: pathlib.Path | None = None,
    *,
    path_override_env: str,
) -> pathlib.Path:
    """Resolve the DROID-finetuned GR00T-N1.7 checkpoint.

    Resolution order:

    1. Environment variable named by *path_override_env* (must be a complete checkpoint).
    2. ``<repo_root>/checkpoints/GR00T-N1.7-DROID``.
    3. Git worktree toplevel + same relative ``checkpoints/...`` path.
    4. ``SHARED_DRIVE_ROOT/models/GR00T-N1.7-DROID``, downloading
       from Hugging Face when missing (requires ``HF_TOKEN``).

    Raises:
        AssertionError: if overrides are incomplete or download leaves a broken tree.
    """
    root = repo_root if repo_root is not None else get_root()

    override = os.environ.get(path_override_env, "").strip()
    if override:
        p = pathlib.Path(override).expanduser().resolve()
        assert libero_checkpoint_tree_ready(p), (
            f"{path_override_env} does not point to a complete checkpoint directory: {p}"
        )
        return p

    local = root / "checkpoints/GR00T-N1.7-DROID"
    if libero_checkpoint_tree_ready(local):
        return local

    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(root),
            text=True,
            timeout=30,
        ).strip()
        git_cp = pathlib.Path(toplevel) / "checkpoints/GR00T-N1.7-DROID"
        if libero_checkpoint_tree_ready(git_cp):
            return git_cp
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    target = resolve_shared_model_path(_DROID_N17_REPO)
    assert libero_checkpoint_tree_ready(target), (
        f"Checkpoint at {target} is incomplete after resolve/download "
        "(check HF_TOKEN, network, and Hugging Face repo layout)."
    )
    return target


def resolve_droid_demo_dataset_path(
    repo_root: pathlib.Path | None = None,
    *,
    path_override_env: str | None = None,
) -> pathlib.Path:
    """Return the path to the DROID ``droid_sample`` dataset (small LeRobot bundle).

    This is the 3-episode DROID demo described in the README under
    ``demo_data/droid_sample`` (Git LFS in the Isaac-GR00T repo).

    Resolution order (first match wins):

    0. If *path_override_env* is set and that variable is non-empty in the
       environment, its path is used (must satisfy :func:`libero_demo_tree_ready`).
    1. ``DROID_DEMO_DATASET_PATH`` — explicit directory (CI or local override).
    2. ``<repo_root>/demo_data/droid_sample`` — normal clone with Git LFS.
    3. ``SHARED_DRIVE_ROOT/datasets/droid_sample`` — shared PVC / local cache.

    Raises:
        AssertionError: if no usable tree is found.
    """
    root = repo_root if repo_root is not None else get_root()

    if path_override_env:
        alt = os.environ.get(path_override_env, "").strip()
        if alt:
            resolved = pathlib.Path(alt).expanduser().resolve()
            assert libero_demo_tree_ready(resolved), (
                f"{path_override_env} does not point to a complete droid_sample-style dataset: "
                f"{resolved}"
            )
            return resolved

    env_path = os.environ.get("DROID_DEMO_DATASET_PATH", "").strip()
    if env_path:
        resolved = pathlib.Path(env_path).expanduser().resolve()
        assert libero_demo_tree_ready(resolved), (
            f"DROID_DEMO_DATASET_PATH does not point to a complete droid_sample tree: {resolved}"
        )
        return resolved

    in_repo = root / "demo_data" / "droid_sample"
    if libero_demo_tree_ready(in_repo):
        return in_repo

    shared = SHARED_DRIVE_ROOT / "datasets" / "droid_sample"
    if libero_demo_tree_ready(shared):
        return shared

    raise AssertionError(
        "droid_sample dataset not found. It ships in-repo under demo_data/droid_sample "
        "(requires Git LFS). Alternatives: set DROID_DEMO_DATASET_PATH to an existing checkout; "
        f"populate {shared} on the shared drive (CI_SHARED_DRIVE_PATH / ~/.cache/g00t)."
    )


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
