# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import tempfile
import time

import yaml

from groot.core.utils.storage import load_msc_telemetry_settings
from groot.core.utils.storage.checkpoint import (
    node_local_checkpoint_output,
    validate_explicit_resume_checkpoint,
)

try:
    import multistorageclient as msc
    from multistorageclient.types import ExecutionMode

    MSC_AVAILABLE = True
except ImportError:
    MSC_AVAILABLE = False
    msc = None
    ExecutionMode = None

HF_MODEL_URI_PREFIX = "hf://model/"
DEFAULT_HF_ARTIFACT_CACHE_ROOT = "/tmp/groot_hf_artifacts"


class DataSyncConfigBuilder:
    def __init__(
        self,
        credentials_file: str,
        profile: str = "default",
        use_rust_client: bool = True,
        enable_telemetry: bool = True,
    ):
        self.use_rust_client = use_rust_client
        self.enable_telemetry = enable_telemetry
        self.credentials_file = os.path.expanduser(credentials_file)
        self.profile = profile
        self.config = {}
        self.config["profiles"] = {}

        if (
            profile == "default"
            and os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
            and os.environ.get("AWS_DEFAULT_REGION")
            and os.environ.get("AWS_ENDPOINT_URL")
        ):
            self.access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            self.region = os.environ.get("AWS_DEFAULT_REGION")
            self.endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        else:
            credentials_file = self.credentials_file

            if not os.path.exists(credentials_file):
                raise FileNotFoundError(f"Credentials file not found at {credentials_file}")

            # Load credentials from file
            with open(credentials_file) as f:
                creds = json.load(f)

            if profile not in creds:
                available_profiles = list(creds.keys())
                raise ValueError(
                    f"Profile '{profile}' not found in credentials file. Available profiles: {available_profiles}"
                )

            profile_creds = creds[profile]

            # Validate required fields
            required_fields = ["aws_access_key_id", "aws_secret_access_key"]
            for field in required_fields:
                if field not in profile_creds:
                    raise ValueError(f"Missing required field '{field}' in profile '{profile}'")

            self.access_key = profile_creds["aws_access_key_id"]
            self.secret_key = profile_creds["aws_secret_access_key"]
            self.region = profile_creds.get("region", "us-west-2")
            self.endpoint_url = profile_creds["endpoint_url"]

    def add_profile(self, profile: str, bucket: str):
        self.config["profiles"][profile] = {
            "storage_provider": {
                "type": "s8k",
                "options": {
                    "region_name": self.region,
                    "base_path": bucket,
                    "endpoint_url": self.endpoint_url,
                },
            },
            "credentials_provider": {
                "type": "S3Credentials",
                "options": {
                    "access_key": self.access_key,
                    "secret_key": self.secret_key,
                },
            },
        }

        if self.use_rust_client:
            self.config["profiles"][profile]["storage_provider"]["options"]["rust_client"] = {}

    def add_path_mapping(self, path_url: str, msc_url: str):
        if "path_mapping" not in self.config:
            self.config["path_mapping"] = {}
        self.config["path_mapping"][path_url] = msc_url

    def add_cache(self, profile: str, cache_path: str = "/tmp/msc_cache", size: str = "100G"):
        if "cache" not in self.config:
            self.config["cache"] = {}
        if profile not in self.config["profiles"]:
            raise ValueError(f"Profile {profile} not found in config")
        else:
            self.config["profiles"][profile]["caching_enabled"] = True
        self.config["cache"]["location"] = cache_path
        self.config["cache"]["size"] = size
        self.config["cache"]["check_source_version"] = False

    def build(self):
        """
        Build the MSC configuration file and set the environment variable.
        """
        if self.enable_telemetry:
            telemetry_settings = load_msc_telemetry_settings(
                credentials_file=self.credentials_file,
                profile=self.profile,
            )
            if telemetry_settings is not None:
                self.config["opentelemetry"] = telemetry_settings

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
            yaml.dump(self.config, temp_file)
            os.environ["MSC_CONFIG"] = temp_file.name


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse S3 URI into bucket and path components.

    Args:
        s3_uri: S3 URI in format s3://bucket/path or s3://bucket/path/

    Returns:
        tuple: (bucket, path) where path is the S3 key prefix

    Raises:
        ValueError: If the URI format is invalid
    """
    if not s3_uri.startswith("s3://"):
        error_msg = f"Invalid S3 URI format. Expected s3://bucket/path, got: {s3_uri}"
        raise ValueError(error_msg)

    # Remove s3:// prefix
    uri_parts = s3_uri[5:]

    # Split into bucket and path
    if "/" not in uri_parts:
        # Just bucket, no path
        return uri_parts, ""

    bucket, path = uri_parts.split("/", 1)

    # Remove trailing slash if present
    path = path.rstrip("/")

    return bucket, path


def sync_s3_uri_to_local(
    source_uri: str,
    target_dir: str | Path,
    *,
    profile_name: str,
    credentials_file: str = "~/.gear/data_credentials",
    credentials_profile: str = "default",
    enable_telemetry: bool = False,
) -> None:
    """Sync a full S3 URI into a local directory using MSC.

    Args:
        source_uri: S3 URI in format ``s3://bucket/prefix``.
        target_dir: Local destination directory.
        profile_name: MSC profile name to create for this sync.
        credentials_file: Gear data credentials file.
        credentials_profile: Credentials profile to read from ``credentials_file``.
        enable_telemetry: Whether to enable MSC telemetry when settings are available.
    """
    if not MSC_AVAILABLE:
        raise ImportError("multistorageclient is required to sync S3 assets")

    bucket, prefix = parse_s3_uri(source_uri)
    source_url = f"msc://{profile_name}/{prefix}" if prefix else f"msc://{profile_name}/"

    config_builder = DataSyncConfigBuilder(
        credentials_file,
        profile=credentials_profile,
        enable_telemetry=enable_telemetry,
    )
    config_builder.add_profile(profile_name, bucket)
    config_builder.build()

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    msc.sync(  # type: ignore[union-attr]
        source_url=source_url,
        target_url=str(target_path),
        execution_mode=ExecutionMode.LOCAL,  # type: ignore[union-attr]
    )


def get_remote_last_checkpoint(remote_url: str) -> str | None:
    """
    Find the latest checkpoint in a remote storage location (S3/MSC).

    Args:
        remote_url: Remote storage URL (e.g., "msc://GearCheckpoints/my-experiment")

    Returns:
        URL of the latest checkpoint, or None if no checkpoints found
    """
    if not MSC_AVAILABLE:
        print("multistorageclient not available, cannot resume from remote storage")
        return None

    try:
        # List items in the remote directory
        obj_metadata_list = list(msc.list(url=remote_url, include_directories=True))  # type: ignore[union-attr]

        # Filter for checkpoint directories (checkpoint-*)
        checkpoint_dirs = [
            obj_metadata
            for obj_metadata in obj_metadata_list
            if obj_metadata.type == "directory"
            and obj_metadata.key.rstrip("/").split("/")[-1].startswith("checkpoint-")
        ]

        if not checkpoint_dirs:
            print(f"No checkpoints found in {remote_url}")
            return None

        # Extract checkpoint numbers and find the latest
        checkpoints = []
        for obj_metadata in checkpoint_dirs:
            try:
                # Extract number from "checkpoint-1000"
                step = int(obj_metadata.key.split("-")[-1])
                checkpoints.append((step, obj_metadata.key))
            except (ValueError, IndexError):
                continue

        if not checkpoints:
            print(f"No valid checkpoints found in {remote_url}")
            return None

        # Sort by step number and get the latest
        checkpoints.sort(reverse=True)
        latest_step, latest_url = checkpoints[0]
        print(f"Found latest remote checkpoint: {latest_url} (step {latest_step})")
        return latest_url

    except Exception as e:
        print(f"Error listing remote checkpoints: {e}")
        return None


def download_remote_checkpoint_for_resume(
    remote_checkpoint_url: str,
    local_checkpoint_dir: Path,
) -> bool:
    """
    Download a checkpoint from remote storage to local directory.
    Caller must ensure only one process calls this function to avoid concurrent downloads.

    Args:
        remote_checkpoint_url: URL of the remote checkpoint (e.g., "msc://.../checkpoint-1000")
        local_checkpoint_dir: Local directory to download checkpoint to

    Returns:
        True if download successful, False otherwise
    """
    if not MSC_AVAILABLE:
        print("multistorageclient not available, cannot download from remote storage")
        return False

    try:
        start_time = time.time()
        print(f"Downloading checkpoint from {remote_checkpoint_url}")
        print(f"Destination: {local_checkpoint_dir}")

        # Create parent directory if needed
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Download using MSC
        msc.sync(  # type: ignore[union-attr]
            source_url=remote_checkpoint_url,
            target_url=str(local_checkpoint_dir),
            execution_mode=ExecutionMode.LOCAL,  # type: ignore[union-attr]
        )

        # Force filesystem sync to ensure all files are written
        os.sync()

        duration = time.time() - start_time
        print(f"Checkpoint download complete in {duration:.2f}s")

        return True

    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return False


def resolve_checkpoint_path(
    checkpoint_path: str,
    output_dir: str,
    global_rank: int | None = None,
    *,
    require_resume_state: bool = False,
) -> str:
    """
    Resolve checkpoint path, downloading from remote storage if needed.

    Only global rank 0 downloads into a shared filesystem. For a cache under
    /tmp, the lowest global rank on each host downloads into that host's local
    filesystem, independently of the training output location.

    Args:
        checkpoint_path: Local POSIX path or remote MSC URL
        output_dir: Output directory where cache will be created
        global_rank: Global rank override; defaults to the current distributed rank,
            or 0 when distributed training is not initialized.
        require_resume_state: Require a durable SUCCESS marker and valid explicit
            async ZeRO-2 recovery state, using a separate cache-completion marker.

    Returns:
        Local POSIX path to checkpoint directory
    """
    if not checkpoint_path.startswith("msc://"):
        # Already a local path, return as-is
        return checkpoint_path
    try:
        import torch.distributed as dist
    except ImportError:
        dist = None

    distributed = dist is not None and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    if global_rank is None:
        global_rank = dist.get_rank() if distributed else 0

    def check_failures(failure):
        failures = [failure]
        if distributed:
            failures = [None] * world_size
            dist.all_gather_object(failures, failure)
        if any(failures):
            raise RuntimeError("; ".join(item for item in failures if item))

    # Older cache entries may omit hidden files such as DCP .metadata. Keep
    # them untouched, but never trust their completion markers for this format.
    shared_cache_dir = Path(output_dir) / "checkpoint-cache-v2"

    # Generate deterministic cache path from URL
    url_hash = hashlib.md5(checkpoint_path.encode()).hexdigest()[:8]
    local_dir = shared_cache_dir / f"{url_hash}"

    failure = None
    node_local_cache = None
    hostname = None
    try:
        node_local_cache = node_local_checkpoint_output(output_dir)
        hostname = socket.gethostname()
    except Exception as exc:
        failure = f"Rank {global_rank}: checkpoint cache setup failed ({type(exc).__name__}: {exc})"
    check_failures(failure)
    hosts = [(hostname, node_local_cache)]
    if distributed:
        hosts = [None] * world_size
        dist.all_gather_object(hosts, (hostname, node_local_cache))
    if any(host[1] != hosts[0][1] for host in hosts):
        raise ValueError("All ranks must use the same checkpoint cache policy")
    download_leader = global_rank == 0
    leader_label = "Rank 0"
    if node_local_cache and distributed:
        host_ranks = [rank for rank, host in enumerate(hosts) if host[0] == hostname]
        download_leader = global_rank == min(host_ranks)
        leader_label = f"Node leader rank {min(host_ranks)}"
    elif node_local_cache:
        download_leader = True
        leader_label = f"Rank {global_rank}"

    marker = local_dir / (
        ".resume_download_complete" if require_resume_state else ".download_complete"
    )
    failure = None
    if download_leader:
        try:
            if not marker.exists():
                logging.info(f"{leader_label}: Downloading checkpoint from {checkpoint_path}")
                logging.info(f"{leader_label}: Destination: {local_dir}")
                local_dir.mkdir(parents=True, exist_ok=True)
                msc.sync(  # type: ignore[union-attr]
                    source_url=checkpoint_path,
                    target_url=str(local_dir),
                    execution_mode=ExecutionMode.LOCAL,  # type: ignore[union-attr]
                    ignore_hidden=False,
                )
                if require_resume_state:
                    validate_explicit_resume_checkpoint(
                        local_dir, require_success=True, world_size=world_size
                    )
                marker.touch()
                logging.info(f"{leader_label}: Checkpoint download complete")
            else:
                logging.info(f"{leader_label}: Checkpoint already cached at {local_dir}")
        except Exception as exc:
            failure = f"{leader_label}: checkpoint download/validation failed ({type(exc).__name__}: {exc})"

    # Every rank takes the same failure path instead of waiting at a barrier
    # after a download leader has already raised.
    check_failures(failure)

    # All ranks verify download completed
    failure = None
    try:
        if not marker.is_file():
            raise RuntimeError(f"Checkpoint download failed or incomplete: {local_dir}")
        if require_resume_state:
            validate_explicit_resume_checkpoint(
                local_dir, require_success=True, world_size=world_size
            )
    except Exception as exc:
        failure = (
            f"Rank {global_rank}: checkpoint cache validation failed ({type(exc).__name__}: {exc})"
        )
    check_failures(failure)

    return str(local_dir)


def list_directories_from_path(msc_url: str) -> list[str]:
    """List all directories in remote MSC storage.

    Args:
        msc_url: MSC URL to list directories from (e.g., "msc://GearCheckpoints/my-exp")

    Returns:
        List of directory basenames (e.g., ["checkpoint-100", "checkpoint-200", "experiment_cfg", ...])
    """
    if not msc_url.startswith("msc://"):
        raise ValueError(f"Expected MSC URL starting with msc://, got: {msc_url}")

    try:
        # List items in the remote directory
        obj_list = msc.list(url=msc_url, include_directories=True)  # type: ignore[union-attr]

        # Extract directory names (basenames)
        directory_names = []
        for obj in obj_list:
            if obj.type == "directory":
                basename = obj.key.rstrip("/").split("/")[-1]
                directory_names.append(basename)

        return directory_names

    except Exception as e:
        logging.error(f"Error listing MSC directories: {e}")
        raise


def upload_dir(
    local_path: str,
    remote_url: str,
) -> None:
    """Upload a local directory to remote MSC storage.

    Args:
        local_path: Local directory path to upload
        remote_url: Remote MSC URL destination (e.g., "msc://profile/path/to/dir")
        execution_mode: MSC execution mode (default: ExecutionMode.LOCAL)

    Raises:
        ValueError: If remote_url is not a valid MSC URL
    """
    if not remote_url.startswith("msc://"):
        raise ValueError(f"Expected MSC URL starting with msc://, got: {remote_url}")

    logging.info(f"Uploading directory from {local_path} to {remote_url}")
    msc.sync(  # type: ignore[union-attr]
        source_url=local_path,
        target_url=remote_url,
        execution_mode=ExecutionMode.LOCAL,  # type: ignore[union-attr]
    )


def upload_single_file(
    local_path: str,
    remote_url: str,
    attributes: dict[str, str] | None = None,
) -> None:
    """Upload a single file to remote MSC storage.

    Args:
        local_path: Local file path to upload
        remote_url: Remote MSC URL destination (e.g., "msc://profile/path/to/file")
        attributes: Optional metadata attributes to attach to the uploaded file

    Raises:
        ValueError: If remote_url is not a valid MSC URL
    """
    if not remote_url.startswith("msc://"):
        raise ValueError(f"Expected MSC URL starting with msc://, got: {remote_url}")

    logging.info(f"Uploading file from {local_path} to {remote_url}")
    msc.upload_file(  # type: ignore[union-attr]
        url=remote_url,
        local_path=local_path,
        attributes=attributes,
    )


def _resolve_cached_hf_snapshot(path: str) -> str | None:
    """Return a local Hugging Face snapshot path when a repo is already cached."""
    if not path or "://" in path:
        return None

    expanded = Path(path).expanduser()
    if expanded.exists():
        return str(expanded)

    # Treat only org/repo-style names as HF repo ids. Relative filesystem paths
    # should fall through unchanged when they are not already present.
    if path.startswith((".", "/", "~")) or path.count("/") != 1:
        return None

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None

    repo_cache = Path(HF_HUB_CACHE) / f"models--{path.replace('/', '--')}"
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.exists():
        return None

    def has_model_weights(snapshot: Path) -> bool:
        weight_names = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        )
        return (snapshot / "config.json").exists() and any(
            (snapshot / weight_name).exists() for weight_name in weight_names
        )

    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = snapshots_dir / revision
        if snapshot.exists() and has_model_weights(snapshot):
            return str(snapshot)

    snapshots = sorted(
        (snapshot for snapshot in snapshots_dir.iterdir() if snapshot.is_dir()),
        key=lambda snapshot: snapshot.stat().st_mtime,
        reverse=True,
    )
    valid_snapshots = [snapshot for snapshot in snapshots if has_model_weights(snapshot)]
    if valid_snapshots:
        snapshot = valid_snapshots[0]
        logging.info("Resolved HF repo %s to cached snapshot %s", path, snapshot)
        return str(snapshot)

    return None


def hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )


def parse_hf_model_uri(path: str) -> tuple[str, str, str]:
    """Parse a Hugging Face model URI into its repository, revision, and artifact path.

    The accepted format is
    ``hf://model/<org>/<repo>@<revision>/<optional-artifact-path>``. The
    revision is required, while the artifact path may be omitted to address the
    repository root.

    For example, ``hf://model/org/repo@abc123/checkpoint-100`` returns
    ``("org/repo", "abc123", "checkpoint-100")``.

    Raises:
        ValueError: If the URI has the wrong prefix or omits the organization,
            repository, or revision.
    """
    if not path.startswith(HF_MODEL_URI_PREFIX):
        raise ValueError(f"HF model URI must start with {HF_MODEL_URI_PREFIX!r}: {path}")
    body = path[len(HF_MODEL_URI_PREFIX) :]
    parts = body.split("/", maxsplit=2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError("HF model URI must use hf://model/<org>/<repo>@<revision>/<optional-path>")
    repo_name, at, revision = parts[1].partition("@")
    if not repo_name or not at or not revision:
        raise ValueError(f"HF model URI must include <org>/<repo>@<revision>: {path}")
    repo_id = f"{parts[0]}/{repo_name}"
    return repo_id, revision, parts[2].strip("/") if len(parts) == 3 else ""


def _resolve_hf_model_uri(path: str, *, cache_root: str | Path | None = None) -> str:
    repo_id, revision, artifact_path = parse_hf_model_uri(path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve HF model URIs. "
            f"Install huggingface-hub or use a local path instead: {path}"
        ) from exc

    cache_root_path = Path(
        cache_root
        or os.environ.get("GROOT_HF_ARTIFACT_CACHE_ROOT")
        or DEFAULT_HF_ARTIFACT_CACHE_ROOT
    )
    cache_root_path.mkdir(parents=True, exist_ok=True)
    token = hf_token() or None

    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "cache_dir": str(cache_root_path),
        "token": token,
    }
    if artifact_path:
        download_kwargs["allow_patterns"] = (artifact_path, f"{artifact_path}/**")
    snapshot = Path(snapshot_download(**download_kwargs))
    resolved = snapshot / artifact_path if artifact_path else snapshot

    if not resolved.exists():
        raise FileNotFoundError(f"HF model artifact does not exist after download: {path}")
    return str(resolved)


def resolve_model_path(
    path: str,
    param_name: str = "path",
    *,
    cache_root: str | Path | None = None,
) -> str:
    """Resolve a model path from msc://, s3://, hf://model, local, or HF Hub name.

    - msc:// or s3:// URLs: downloaded via MSC to MSC_CACHE_PATH.
    - hf://model URLs: downloaded from Hugging Face to GROOT_HF_ARTIFACT_CACHE_ROOT.
    - Absolute local paths: validated to exist.
    - HF Hub names: resolved to an existing local cache snapshot when available,
      otherwise returned as-is.
    """
    if not path:
        return path
    if path.startswith(HF_MODEL_URI_PREFIX):
        return _resolve_hf_model_uri(path, cache_root=cache_root)
    if path.startswith("msc://") or path.startswith("s3://"):
        cache_dir = os.environ.get("MSC_CACHE_PATH")
        if not cache_dir:
            raise RuntimeError(
                f"MSC_CACHE_PATH environment variable is required when {param_name} is a remote URL: {path}"
            )
        return resolve_checkpoint_path(path, cache_dir)
    if path.startswith("/") and not os.path.exists(path):
        raise RuntimeError(
            f"{param_name} points to a non-existent local path: {path}\n"
            f"This usually means the checkpoint's config was saved with a "
            f"cluster-local path. Fix by editing config.json to use an s3:// URL."
        )
    cached_hf_snapshot = _resolve_cached_hf_snapshot(path)
    if cached_hf_snapshot is not None:
        return cached_hf_snapshot
    return path
