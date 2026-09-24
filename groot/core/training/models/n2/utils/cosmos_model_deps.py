# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Download model dependencies from S3 and rewrite checkpoint config paths.

DreamCosmos and GrootN2 checkpoints bake absolute Lustre paths in config.json for 3
external model dependencies (text encoder, VAE, Cosmos DiT base). On machines without
Lustre, these paths don't exist. This module auto-downloads them from S3 and rewrites
configs to point to the local cache.

S3 layout (under s3://GearCheckpoints/model_deps/):
  cosmos25/2b_post.pt           - Cosmos DiT 2B base weights
  cosmos25/14b_post.pt          - Cosmos DiT 14B base weights
  cosmos25/cosmos_reason1/      - Text encoder (HF model dir)
  Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth  - Wan VAE weights
"""

import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from filelock import FileLock

logger = logging.getLogger(__name__)

# Known Lustre prefixes that contain model deps
_LUSTRE_CKPT_PREFIXES = [
    "/mnt/aws-lfs-02/shared/ckpts/",
    "/mnt/aws-lfs-03/shared/ckpts/",
    "/mnt/amlfs-02/shared/ckpts/",
    "/mnt/amlfs-05/shared/ckpts/",
    "/mnt/amlfs-07/shared/ckpts/",
]


def _find_lustre_alternative(dep_path: str) -> str | None:
    """Find an alternative Lustre path where the dep exists.

    Training checkpoints bake in the Lustre prefix of the training cluster.
    Eval pods on a different cluster may not have that mount, but often have
    the same data under a sibling mount. Swap prefixes until we find a path that
    actually exists on the current machine.
    """
    if os.path.exists(dep_path):
        return dep_path
    for from_prefix in _LUSTRE_CKPT_PREFIXES:
        if not dep_path.startswith(from_prefix):
            continue
        rel = dep_path[len(from_prefix) :]
        for to_prefix in _LUSTRE_CKPT_PREFIXES:
            if to_prefix == from_prefix:
                continue
            alt = to_prefix + rel
            if os.path.exists(alt):
                return alt
        break
    return None


# Config keys that reference model dep paths
_CONFIG_PATH_KEYS = [
    "diffusion_model_pretrained_path",  # in diffusion_model_cfg
    "text_encoder_pretrained_path",  # in text_encoder_cfg
    "vae_pretrained_path",  # in vae_cfg
    "tokenizer_path",  # top-level
]

# Config files that may contain baked Lustre paths
_CONFIG_FILES = [
    "config.json",
    "processor_config.json",
    "experiment_cfg/final_processor_config.json",
]


def _to_relative_dep_path(path: str, cache_dir: Path | None = None) -> str | None:
    """Convert a dep path (Lustre or cache) to a relative path for S3 lookup.

    Handles both original Lustre paths and previously-rewritten cache paths.
    Returns None if no known prefix matches.
    """
    for prefix in _LUSTRE_CKPT_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :].rstrip("/")
    # Also handle paths that were already rewritten to point to cache_dir
    if cache_dir is not None:
        cache_str = str(cache_dir) + "/"
        if path.startswith(cache_str):
            return path[len(cache_str) :].rstrip("/")
    return None


def _extract_dep_paths(config: dict) -> list[str]:
    """Extract all model dependency paths from a config dict (recursive search)."""
    paths = []

    def _search(obj: dict | list):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _CONFIG_PATH_KEYS and isinstance(v, str) and v:
                    paths.append(v)
                else:
                    _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)

    _search(config)
    return paths


def _is_cosmos_checkpoint(checkpoint_dir: Path) -> bool:
    """Check if this is a DreamCosmos / DreamZero / GrootN2 checkpoint by reading config.json."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
        return (
            "DreamCosmos" in model_type
            or "DreamZero" in model_type
            or model_type in {"GrootN2", "GrootN2RL"}
        )
    except (json.JSONDecodeError, KeyError):
        return False


def ensure_cosmos_model_deps(
    checkpoint_dir: str | Path,
    cache_dir: str | Path | None = None,
    *,
    cache_is_prepared: bool = False,
    lock_dir: str | Path | None = None,
) -> None:
    """Download model dependencies from S3 if not locally available.

    Reads config.json from checkpoint_dir, checks if baked Lustre paths exist locally.
    If not, downloads from S3 via boto3 and rewrites config files to point to the local cache.

    Supports DreamCosmos, DreamZero, and GrootN2 checkpoints.

    No-op if:
    - Not a supported checkpoint type
    - All baked paths already exist locally (e.g., on a cluster with Lustre)

    Args:
        checkpoint_dir: Path to the checkpoint directory containing config.json
        cache_dir: Where to find or cache model deps. Defaults to ~/checkpoints/model_deps/
        cache_is_prepared: Require every dependency to already exist instead of downloading it.
        lock_dir: Writable directory for the inter-process lock when cache_dir is read-only.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if cache_dir is None:
        cache_dir = Path("~/checkpoints/model_deps").expanduser()
    else:
        cache_dir = Path(cache_dir)
    if cache_is_prepared:
        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"prepared model dependency directory does not exist: {cache_dir}"
            )
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    lock_root = Path(lock_dir) if lock_dir is not None else cache_dir
    lock_root.mkdir(parents=True, exist_ok=True)

    # Trainer ranks on one machine share this cache. Only one of them may inspect or populate it
    # at a time: a dependency directory exists as soon as its first object lands, but it is not a
    # usable model until the complete prefix has arrived. The lock also serializes config rewrites
    # in the node-local checkpoint shared by those ranks.
    lock_name = ".model-deps.lock" if lock_dir is not None else ".download.lock"
    with FileLock(str(lock_root / lock_name)):
        _ensure_cosmos_model_deps_locked(
            checkpoint_dir,
            cache_dir,
            allow_download=not cache_is_prepared,
        )


def _ensure_cosmos_model_deps_locked(
    checkpoint_dir: Path,
    cache_dir: Path,
    *,
    allow_download: bool,
) -> None:
    """Resolve dependencies while holding the cache's inter-process lock."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        logger.warning(f"No config.json found in {checkpoint_dir}, skipping model deps check")
        return

    with open(config_path) as f:
        config = json.load(f)

    # Only applies to checkpoints that bake Lustre paths for sub-model deps
    _SUPPORTED_MODEL_TYPES = {"DreamCosmos", "DreamZero", "GrootN2", "GrootN2RL", "GrootN2TTT"}
    model_type = config.get("model_type", "")
    if not any(mt in model_type for mt in _SUPPORTED_MODEL_TYPES):
        logger.info(f"Not a supported checkpoint (model_type={model_type}), skipping")
        return

    # Extract all dependency paths from config
    dep_paths = _extract_dep_paths(config)
    if not dep_paths:
        logger.info("No model dependency paths found in config")
        return

    # Deduplicate (tokenizer_path often == text_encoder_pretrained_path)
    dep_paths = list(dict.fromkeys(dep_paths))

    # Check which paths need downloading
    paths_to_download: list[tuple[str, str]] = []  # (original_path, relative_path)
    for dep_path in dep_paths:
        # First check if the original path exists (e.g., on a cluster with Lustre)
        if os.path.exists(dep_path):
            logger.info(f"  Exists locally: {dep_path}")
            continue

        # Fall back to sibling Lustre mounts before hitting S3 (handles cross-
        # cluster evals where the checkpoint baked in the training cluster's
        # Lustre prefix but the eval cluster has the same data under a
        # different prefix — e.g. aws-lfs-02 vs amlfs-02/07).
        alt = _find_lustre_alternative(dep_path)
        if alt is not None:
            logger.info(f"  Exists on sibling Lustre: {alt}")
            continue

        rel_path = _to_relative_dep_path(dep_path, cache_dir)
        if rel_path is None:
            logger.warning(f"  Unknown prefix, cannot download: {dep_path}")
            continue

        local_cached = cache_dir / rel_path
        # For files: exists is enough. For dirs: must have content (not empty mkdir)
        if local_cached.is_file() or (local_cached.is_dir() and any(local_cached.iterdir())):
            logger.info(f"  Already cached: {local_cached}")
        else:
            paths_to_download.append((dep_path, rel_path))

    # Download missing deps
    if paths_to_download:
        if not allow_download:
            missing = ", ".join(relative for _original, relative in paths_to_download)
            raise FileNotFoundError(
                f"prepared model dependency directory {cache_dir} is incomplete; missing: {missing}"
            )
        _download_deps(paths_to_download, cache_dir)

    # Rewrite config files — replace all Lustre paths with local cache paths
    _rewrite_configs(checkpoint_dir, dep_paths, cache_dir)


def _get_s3_client():
    """Create a boto3 S3 client with PDX endpoint credentials.

    Reads credentials from environment variables, cluster credential files, or
    the standard Gear data credential source.
    """
    from gear.data import get_s3_client

    if os.environ.get("GEAR_DATA_CREDENTIALS") and not (
        os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
    ):
        import boto3

        from groot.core.data.hydrator.utils.aws import get_boto3_client_config

        profile = os.environ.get("GROOT_RL_PDX_CREDENTIALS_PROFILE", "default")
        return boto3.client("s3", **get_boto3_client_config(profile=profile))

    endpoint_url = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL")
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")

    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = os.environ.get("AWS_SESSION_TOKEN") if ak and sk else None
    if not (ak and sk):
        # Fall back to cluster credential files (GB200 clusters)
        ak_path = "/etc/gear/s3/access_key"
        sk_path = "/etc/gear/s3/secret_key"
        if os.path.exists(ak_path) and os.path.exists(sk_path):
            with open(ak_path) as f:
                ak = f.read().strip()
            with open(sk_path) as f:
                sk = f.read().strip()
    if not (ak and sk):
        try:
            from gear.common import get_s3_credentials_from_file

            creds = get_s3_credentials_from_file()
            ak = creds["aws_access_key_id"]
            sk = creds["aws_secret_access_key"]
            region = region or creds.get("region")
            endpoint_url = endpoint_url or creds.get("endpoint_url")
            session_token = creds.get("aws_session_token")
        except (ImportError, KeyError, ValueError) as exc:
            raise RuntimeError(
                "No S3 credentials found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or configure ~/.gear/data_credentials with the Gear CLI."
            ) from exc

    credentials = {
        "endpoint_url": endpoint_url or "https://pdx.s8k.io",
        "aws_access_key_id": ak,
        "aws_secret_access_key": sk,
        "region": region or "us-east-1",
    }
    if session_token:
        credentials["aws_session_token"] = session_token
    return get_s3_client(credentials=credentials)


def _download_deps(
    paths_to_download: list[tuple[str, str]],
    cache_dir: Path,
) -> None:
    """Download model deps from S3 via boto3.

    Downloads model dependencies from s3://GearCheckpoints/model_deps/.
    Uses boto3 with PDX endpoint credentials (same as checkpoint downloads).
    """
    import time

    s3 = _get_s3_client()
    bucket = "GearCheckpoints"

    for _original_path, rel_path in paths_to_download:
        local_target = cache_dir / rel_path
        s3_prefix = f"model_deps/{rel_path}"

        # Determine if this is a directory (text encoder) or single file
        is_directory = not rel_path.endswith((".pt", ".pth", ".bin", ".safetensors"))

        logger.info(f"Downloading model dep: {rel_path}")
        logger.info(f"  From: s3://{bucket}/{s3_prefix}")
        logger.info(f"  To:   {local_target}")

        local_target.parent.mkdir(parents=True, exist_ok=True)

        if is_directory:
            file_count = _download_directory_atomically(
                s3,
                bucket=bucket,
                s3_prefix=s3_prefix,
                local_target=local_target,
            )
            logger.info(f"  Done: {file_count} files downloaded")
        else:
            # Single file download
            t0 = time.time()
            # Get file size first
            head = s3.head_object(Bucket=bucket, Key=s3_prefix)
            size_mb = head["ContentLength"] / 1e6
            logger.info(f"    Size: {size_mb:.1f} MB")
            _download_file_atomically(
                s3,
                bucket=bucket,
                s3_key=s3_prefix,
                local_target=local_target,
            )
            dt = time.time() - t0
            logger.info(f"  Done in {dt:.1f}s ({size_mb / max(dt, 0.1):.1f} MB/s)")


def _download_directory_atomically(
    s3: Any,
    *,
    bucket: str,
    s3_prefix: str,
    local_target: Path,
) -> int:
    """Stage a complete S3 prefix, then publish the directory in one rename."""
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{local_target.name}.", dir=local_target.parent)
    ).resolve()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        file_count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative_path = key[len(s3_prefix) + 1 :]
                if not relative_path:
                    continue
                local_path = (temporary / relative_path).resolve()
                try:
                    local_path.relative_to(temporary)
                except ValueError as error:
                    raise ValueError(f"Model dependency key escapes its prefix: {key}") from error
                local_path.parent.mkdir(parents=True, exist_ok=True)
                size_mb = obj["Size"] / 1e6
                logger.info(f"    {relative_path} ({size_mb:.1f} MB)")
                s3.download_file(bucket, key, str(local_path))
                file_count += 1
        if file_count == 0:
            raise FileNotFoundError(f"No objects found under s3://{bucket}/{s3_prefix}/")
        try:
            os.replace(temporary, local_target)
        except OSError:
            if not (local_target.is_dir() and any(local_target.iterdir())):
                raise
        return file_count
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _download_file_atomically(
    s3: Any,
    *,
    bucket: str,
    s3_key: str,
    local_target: Path,
) -> None:
    """Stage one S3 object, then publish the file in one rename."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{local_target.name}.", dir=local_target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        s3.download_file(bucket, s3_key, str(temporary))
        os.replace(temporary, local_target)
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite_configs(
    checkpoint_dir: Path,
    dep_paths: list[str],
    cache_dir: Path,
) -> None:
    """Rewrite config files to replace Lustre paths with local cache paths.

    Only rewrites paths that don't exist locally (original path missing) but
    have been downloaded to cache_dir. Also applies cross-cluster Lustre prefix
    remapping (amlfs-02 -> aws-lfs-02).
    """
    # Build replacement map: old_path -> new_path
    replacements: dict[str, str] = {}
    for dep_path in dep_paths:
        # Skip if original path exists (no rewrite needed)
        if os.path.exists(dep_path):
            continue

        # Prefer a sibling Lustre mount over our downloaded cache — if the
        # same data is already available under a different prefix, point at
        # that instead of shuttling through ~/checkpoints/model_deps.
        alt = _find_lustre_alternative(dep_path)
        if alt is not None:
            replacements[dep_path] = alt
            continue

        rel_path = _to_relative_dep_path(dep_path, cache_dir)
        if rel_path is None:
            continue

        local_cached = cache_dir / rel_path
        # Only rewrite if the cached version exists and the path isn't already correct
        if local_cached.exists() and dep_path != str(local_cached):
            replacements[dep_path] = str(local_cached)

    # Apply cross-cluster and cache path rewrites to config files
    for config_file in _CONFIG_FILES:
        config_path = checkpoint_dir / config_file
        if not config_path.exists():
            continue

        with open(config_path) as f:
            content = f.read()

        original_content = content
        for old_path, new_path in replacements.items():
            content = content.replace(old_path, new_path)

        if content != original_content:
            _atomic_write_text(config_path, content)
            logger.info(f"  Rewrote paths in {config_file}")
        else:
            logger.info(f"No rewrites needed for {config_file}")


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a rewritten config without exposing a partial JSON document."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
