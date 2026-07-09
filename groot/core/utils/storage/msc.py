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
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time

import yaml

try:
    import multistorageclient as msc
    from multistorageclient.types import ExecutionMode

    MSC_AVAILABLE = True
except ImportError:
    MSC_AVAILABLE = False
    msc = None
    ExecutionMode = None


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
            credentials_file = os.path.expanduser(credentials_file)

            if not os.path.exists(credentials_file):
                raise FileNotFoundError(f"Credentials file not found at {credentials_file}")

            # Load credentials from file
            with open(credentials_file) as f:
                creds = json.load(f)

            if profile not in creds:
                available_profiles = list(creds.keys())
                raise ValueError(
                    f"Profile '{profile}' not found in credentials file. "
                    f"Available profiles: {available_profiles}"
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

    def _configure_telemetry(self):
        """
        Enable OpenTelemetry metrics collection and export.
        """
        if not os.environ.get("MSC_VAULT_APPROLE_ID") or not os.environ.get(
            "MSC_VAULT_APPROLE_SECRET"
        ):
            return
        self.config["opentelemetry"] = {
            "metrics": {
                "attributes": [
                    {
                        "type": "static",
                        "options": {
                            "attributes": {
                                "msc.ppp": "GR00T",
                                "msc.cluster": "OSMO",
                                "msc.job": os.environ.get("SUBMISSION_ID", "data_sync"),
                            }
                        },
                    },
                    {"type": "host", "options": {"attributes": {"msc.node": "name"}}},
                    {"type": "process", "options": {"attributes": {"msc.process": "pid"}}},
                    {
                        "type": "msc_config",
                        "options": {
                            "attributes": {
                                "msc.vault_approle_id": {
                                    "expression": "opentelemetry.metrics.exporter.options.auth.approle_id"
                                },
                                "msc.vault_approle_secret": {
                                    "expression": "hash('sha3-224', opentelemetry.metrics.exporter.options.auth.approle_secret)"
                                },
                            }
                        },
                    },
                ],
                "reader": {
                    "options": {
                        # ≤ 100 Hz collect frequency.
                        "collect_interval_millis": 10,
                        "collect_timeout_millis": 100,
                        # ≤ 1 Hz export frequency.
                        "export_interval_millis": 1000,
                        "export_timeout_millis": 500,
                    }
                },
                "exporter": {
                    "type": "_otlp_mtls_vault",
                    "options": {
                        "exporter": {
                            "endpoint": "https://otlp.us-west-2.aws.telemetry.dgxc.ngc.nvidia.com/v1/metrics"
                        },
                        "auth": {
                            "vault_endpoint": "https://prod.vault.nvidia.com",
                            "vault_namespace": "ngc-multi-storage-client",
                            "approle_id": os.environ.get("MSC_VAULT_APPROLE_ID", ""),
                            "approle_secret": os.environ.get("MSC_VAULT_APPROLE_SECRET", ""),
                            "mount_point": "panoptes/mtls",
                            "secret_path": "certs/prod-customers",
                            "cert_key": "multi-storage-client-prod-customers-ingest-cert.crt",
                            "key_key": "multi-storage-client-prod-customers-ingest-key.key",
                            "ca_key": "multi-storage-client-prod-customers-ingest-ca-chain.crt",
                        },
                    },
                },
            }
        }

    def build(self):
        """
        Build the MSC configuration file and set the environment variable.
        """
        if self.enable_telemetry:
            self._configure_telemetry()

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
        enable_telemetry: Whether to enable MSC telemetry in the generated config.
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
    global_rank: int = 0,
) -> str:
    """
    Resolve checkpoint path, downloading from remote storage if needed.
    Only rank 0 downloads, all ranks wait and use shared location.

    Args:
        checkpoint_path: Local POSIX path or remote MSC URL
        output_dir: Output directory where cache will be created
        global_rank: Global rank in distributed training (default: 0)

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

    # Use shared cache directory (Lustre/shared filesystem)
    shared_cache_dir = Path(output_dir)

    shared_cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate deterministic cache path from URL
    url_hash = hashlib.md5(checkpoint_path.encode()).hexdigest()[:8]
    local_dir = shared_cache_dir / f"{url_hash}"

    # Rank 0 downloads, others wait
    if global_rank == 0:
        if not local_dir.exists() or not (local_dir / ".download_complete").exists():
            logging.info(f"Rank 0: Downloading checkpoint from {checkpoint_path}")
            logging.info(f"Rank 0: Destination: {local_dir}")

            local_dir.mkdir(parents=True, exist_ok=True)

            try:
                msc.sync(  # type: ignore[union-attr]
                    source_url=checkpoint_path,
                    target_url=str(local_dir),
                    execution_mode=ExecutionMode.LOCAL,  # type: ignore[union-attr]
                )

                # Create marker file to indicate download complete
                (local_dir / ".download_complete").touch()
                logging.info("Rank 0: Checkpoint download complete")
            except Exception as e:
                logging.error(f"Rank 0: Failed to download checkpoint: {e}")
                raise
        else:
            logging.info(f"Rank 0: Checkpoint already cached at {local_dir}")

    # Synchronize all ranks
    if dist is not None and dist.is_initialized():
        dist.barrier()

    # All ranks verify download completed
    if not (local_dir / ".download_complete").exists():
        raise RuntimeError(f"Checkpoint download failed or incomplete: {local_dir}")

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


def resolve_model_path(path: str, param_name: str = "path") -> str:
    """Resolve a model path from msc://, s3://, local, or HF Hub name.

    - msc:// or s3:// URLs: downloaded via MSC to MSC_CACHE_PATH.
    - Absolute local paths: validated to exist.
    - HF Hub names: resolved to an existing local cache snapshot when available,
      otherwise returned as-is.
    """
    if not path:
        return path
    if path.startswith("msc://") or path.startswith("s3://"):
        cache_dir = os.environ.get("MSC_CACHE_PATH")
        if not cache_dir:
            raise RuntimeError(
                f"MSC_CACHE_PATH environment variable is required when {param_name} "
                f"is a remote URL: {path}"
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
