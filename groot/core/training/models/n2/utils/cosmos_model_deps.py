# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Download model dependencies and localize configs without modifying checkpoints.

Current checkpoints record durable S3 identities for their external model dependencies.
Older DreamCosmos and GrootN2 checkpoints may instead contain absolute Lustre paths.
This module resolves either representation in a small loader view with localized configs.

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

from groot.core.training.configs.model.n2 import (
    LOCAL_MODEL_DEPENDENCY_ROOTS,
    MODEL_DEPENDENCY_REMOTE_ROOT,
    relative_model_dependency_path,
)
from groot.core.training.models.n2.utils.model_dependency_integrity import (
    DEPENDENCY_RECEIPT_FILENAME,
    build_dependency_receipt,
    dependency_invalid_reason,
)
from groot.core.utils.storage.checkpoint import (
    CHECKPOINT_VIEW_CONFIG_FILES,
    CHECKPOINT_VIEW_METADATA_FILE,
    checkpoint_view_identity,
    checkpoint_view_links,
    checkpoint_view_metadata,
)

logger = logging.getLogger(__name__)

# Known Lustre prefixes that contain model deps
_LUSTRE_CKPT_PREFIXES = LOCAL_MODEL_DEPENDENCY_ROOTS


# Config keys that reference model dep paths
_CONFIG_PATH_KEYS = [
    "diffusion_model_pretrained_path",  # in diffusion_model_cfg
    "text_encoder_pretrained_path",  # in text_encoder_cfg
    "vae_pretrained_path",  # in vae_cfg
    "tokenizer_path",  # top-level
]

# Config files that may contain baked Lustre paths
_CONFIG_FILES = CHECKPOINT_VIEW_CONFIG_FILES


def _to_relative_dep_path(path: str, cache_dir: Path | None = None) -> str | None:
    """Convert a dep path (S3, Lustre, or cache) to a relative path for S3 lookup.

    Handles both original Lustre paths and previously-rewritten cache paths.
    Returns None if no known prefix matches.
    """
    return relative_model_dependency_path(path, cache_root=cache_dir)


def _extract_dep_roles(config: dict) -> dict[str, set[str]]:
    """A shared encoder/tokenizer directory must satisfy both consumers."""
    paths: dict[str, set[str]] = {}

    def _search(obj: dict | list):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _CONFIG_PATH_KEYS and isinstance(v, str) and v:
                    if k == "text_encoder_pretrained_path":
                        # CosmosTextEncoder loads both the HF model and its
                        # AutoProcessor, independently of GrootN2's tokenizer.
                        roles = {"text_encoder", "tokenizer"}
                    else:
                        roles = {"tokenizer" if k == "tokenizer_path" else "model"}
                    paths.setdefault(v, set()).update(roles)
                else:
                    _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)

    _search(config)
    return paths


def _extract_dep_paths(config: dict) -> list[str]:
    return list(_extract_dep_roles(config))


_DEPENDENCY_RECEIPT = DEPENDENCY_RECEIPT_FILENAME


def _receipt_path(path: Path) -> Path:
    return (
        path / _DEPENDENCY_RECEIPT
        if path.is_dir()
        else path.with_name(f".{path.name}{_DEPENDENCY_RECEIPT}")
    )


def _dependency_error(
    path: Path, roles: set[str], *, verification_cache_dir: Path | None = None
) -> str | None:
    receipt_path = _receipt_path(path)
    try:
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if receipt_path.exists() and not isinstance(receipt, dict):
            return "invalid dependency receipt"
    except (OSError, ValueError) as exc:
        return f"unreadable dependency receipt: {exc}"
    return dependency_invalid_reason(
        path, roles, receipt=receipt, verification_cache_dir=verification_cache_dir
    )


def _cache_target(cache_dir: Path, relative: str, *, for_write: bool = False) -> Path:
    if (
        not relative
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError(f"Invalid model dependency relative path: {relative!r}")
    target = cache_dir / relative
    # Prepared caches may link to a read-only bundle/HF snapshot outside the cache.
    # Such links are safe to read, but never repair through them into shared storage.
    if for_write and not target.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Model dependency escapes cache directory: {relative!r}")
    return target


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
    local_files_only: bool = False,
) -> str:
    """Resolve dependencies and return a loader-ready, source-immutable checkpoint path.

    Reads config.json from checkpoint_dir and resolves portable S3 identities or legacy
    Lustre paths. Missing dependencies use Hydrator's S3 backend and config files
    are localized in a small cached view. Callers must load the returned path.
    The source checkpoint, its manifest and completion markers are never changed;
    model payloads are symlinked, not copied. Runtime views are not resume checkpoints.

    Supports DreamCosmos, DreamZero, and GrootN2 checkpoints.

    No-op if:
    - Not a supported checkpoint type
    - All baked paths already exist locally (e.g., on a cluster with Lustre)

    Args:
        checkpoint_dir: Path to the checkpoint directory containing config.json
        cache_dir: Explicit cache (legacy behavior). Otherwise GROOT_N2_MODEL_DEPS_ROOT
            selects a package-owned cache: dependencies must be real files inside it,
            never links to another cache. Unset retains ~/checkpoints/model_deps/.
        cache_is_prepared: Require every dependency to already exist instead of downloading it.
        lock_dir: Writable directory for the inter-process lock when cache_dir is read-only.
        local_files_only: Resolve mounted/cached dependencies without downloading missing files.
    """
    checkpoint_dir = Path(checkpoint_dir)
    packaged_root = os.environ.get("GROOT_N2_MODEL_DEPS_ROOT") if cache_dir is None else None
    if packaged_root:
        # The root may live behind a data-disk/HOME symlink. Containment and
        # payload symlink checks below apply within this canonical directory.
        cache_dir = Path(packaged_root).expanduser().resolve()
    view_metadata = checkpoint_view_metadata(checkpoint_dir)
    if view_metadata is not None:
        # Reuse the original cache and writable view root when an upstream loader
        # already localized this checkpoint, including a prepared read-only cache.
        if cache_dir is None:
            cache_dir = view_metadata["dependency_cache"]
        if (
            not packaged_root
            and lock_dir is None
            and checkpoint_dir.resolve().parent.parent != Path(cache_dir).resolve()
        ):
            lock_dir = checkpoint_dir.resolve().parent.parent
        checkpoint_dir = Path(view_metadata["source"])
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
    # usable model until the complete prefix has arrived. The lock also serializes publication
    # of the small immutable config views shared by those ranks.
    lock_name = ".model-deps.lock" if lock_dir is not None else ".download.lock"
    with FileLock(str(lock_root / lock_name)):
        return str(
            _ensure_cosmos_model_deps_locked(
                checkpoint_dir,
                cache_dir,
                allow_download=not (cache_is_prepared or local_files_only),
                view_root=lock_root / ".checkpoint-configs",
                contained=bool(packaged_root),
            )
        )


def _ensure_cosmos_model_deps_locked(
    checkpoint_dir: Path,
    cache_dir: Path,
    *,
    allow_download: bool,
    view_root: Path,
    contained: bool = False,
) -> Path:
    """Resolve dependencies while holding the cache's inter-process lock."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        # RL feature extraction also loads processor-only directories. These
        # contain the same portable tokenizer dependency as full checkpoints.
        config_path = checkpoint_dir / "processor_config.json"
        if not config_path.exists():
            logger.warning(f"No model or processor config found in {checkpoint_dir}")
            return checkpoint_dir

    with open(config_path) as f:
        config = json.load(f)

    # Only applies to checkpoints that bake Lustre paths for sub-model deps
    _SUPPORTED_MODEL_TYPES = {"DreamCosmos", "DreamZero", "GrootN2", "GrootN2RL", "GrootN2TTT"}
    model_type = config.get("model_type", "")
    if (
        not any(mt in model_type for mt in _SUPPORTED_MODEL_TYPES)
        and config.get("processor_class") != "GrootN2Processor"
    ):
        logger.info(f"Not a supported checkpoint (model_type={model_type}), skipping")
        return checkpoint_dir

    # A legacy processor may still name the same dependency by a Lustre path while
    # config.json already stores its durable S3 identity. Resolve both spellings.
    dep_roles = _extract_dep_roles(config)
    for config_file in _CONFIG_FILES:
        if config_file == "config.json":
            continue
        additional_config = checkpoint_dir / config_file
        if additional_config.exists():
            with open(additional_config) as f:
                for path, roles in _extract_dep_roles(json.load(f)).items():
                    dep_roles.setdefault(path, set()).update(roles)
    if not dep_roles:
        logger.info("No model dependency paths found in config")
        return checkpoint_dir

    # Deduplicate (tokenizer_path often == text_encoder_pretrained_path)
    dep_paths = list(dep_roles)
    # Legacy and portable spellings of the same asset may appear in different configs.
    relative_roles: dict[str, set[str]] = {}
    for path, roles in dep_roles.items():
        relative = _to_relative_dep_path(path, cache_dir)
        if relative is not None:
            _cache_target(cache_dir, relative)
            relative_roles.setdefault(relative, set()).update(roles)

    # Check which paths need downloading
    paths_to_download: dict[str, str] = {}  # relative_path -> original_path
    resolved_paths: dict[str, str] = {}
    invalid_paths: dict[str, str] = {}
    checked: dict[tuple[Path, frozenset[str]], str | None] = {}
    verification_cache_dir = view_root.parent / ".model-dependency-verifications"
    for dep_path in dep_paths:
        rel_path = _to_relative_dep_path(dep_path, cache_dir)
        roles = relative_roles.get(rel_path, dep_roles[dep_path])
        if contained:
            if rel_path is None:
                raise ValueError(
                    f"Cannot package model dependency {dep_path!r}: use its durable "
                    "s3://GearCheckpoints/model_deps/ identity in the source checkpoint"
                )
            target = _cache_target(cache_dir, rel_path, for_write=True)
            # Reject ancestor, directory and payload links, including broken links.
            # A prepared package must not silently depend on another cache.
            ancestors = [cache_dir / path for path in target.relative_to(cache_dir).parents]
            if any(path.is_symlink() for path in [target, *ancestors]) or any(
                path.is_symlink() for path in target.rglob("*")
            ):
                raise ValueError(f"Packaged model dependency contains symlinks: {target}")
            candidates = [target]
        else:
            candidates = [Path(dep_path)] if "://" not in dep_path else []
            if rel_path is not None:
                candidates.extend(Path(prefix) / rel_path for prefix in _LUSTRE_CKPT_PREFIXES)
                candidates.append(_cache_target(cache_dir, rel_path))
        for candidate in dict.fromkeys(candidates):
            key = (candidate, frozenset(roles))
            if key not in checked:
                checked[key] = _dependency_error(
                    candidate, roles, verification_cache_dir=verification_cache_dir
                )
            error = checked[key]
            if error is None:
                resolved_paths[dep_path] = str(candidate)
                break
            if candidate.exists():
                invalid_paths[dep_path] = f"{candidate}: {error}"
        if dep_path in resolved_paths:
            continue
        if rel_path is not None:
            paths_to_download.setdefault(rel_path, dep_path)
        elif dep_path in invalid_paths:
            raise FileNotFoundError(f"Incomplete model dependency: {invalid_paths[dep_path]}")
        else:
            logger.warning(f"  Unknown prefix, cannot download: {dep_path}")

    # Download missing deps
    if paths_to_download:
        if not allow_download:
            missing = "; ".join(
                invalid_paths.get(original, relative)
                for relative, original in paths_to_download.items()
            )
            raise FileNotFoundError(
                f"prepared model dependency directory {cache_dir} is incomplete; missing: {missing}"
            )
        _download_deps(
            [(original, relative) for relative, original in paths_to_download.items()],
            cache_dir,
            roles_by_path=relative_roles,
        )
        for dep_path in dep_paths:
            if dep_path in resolved_paths:
                continue
            relative = _to_relative_dep_path(dep_path, cache_dir)
            if relative not in paths_to_download:
                continue
            local_cached = _cache_target(cache_dir, relative)
            error = _dependency_error(
                local_cached,
                relative_roles[relative],
                verification_cache_dir=verification_cache_dir,
            )
            if error is not None:
                raise ValueError(f"Downloaded model dependency {relative} is incomplete: {error}")
            resolved_paths[dep_path] = str(local_cached)

    # Rewrite config files — replace all Lustre paths with local cache paths
    return _rewrite_configs(checkpoint_dir, resolved_paths, cache_dir, view_root=view_root)


def _get_s3_client():
    """Create the lazy Hydrator-backend adapter without starting Ray workers."""
    from groot.core.training.models.n2.utils.model_dependency_download import (
        ModelDependencyDownload,
    )

    return ModelDependencyDownload()


def _download_deps(
    paths_to_download: list[tuple[str, str]],
    cache_dir: Path,
    *,
    roles_by_path: dict[str, set[str]] | None = None,
) -> None:
    """Download model deps through Hydrator's synchronous S3 backend.

    Downloads model dependencies from s3://GearCheckpoints/model_deps/.
    Inventory/validation and atomic cache publication remain resolver-owned.
    """
    import time

    s3 = _get_s3_client()
    bucket = "GearCheckpoints"

    for _original_path, rel_path in paths_to_download:
        local_target = _cache_target(cache_dir, rel_path, for_write=True)
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
                roles=(roles_by_path or {}).get(rel_path),
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
                expected_size=head["ContentLength"],
            )
            dt = time.time() - t0
            logger.info(f"  Done in {dt:.1f}s ({size_mb / max(dt, 0.1):.1f} MB/s)")


def _download_directory_atomically(
    s3: Any,
    *,
    bucket: str,
    s3_prefix: str,
    local_target: Path,
    roles: set[str] | None = None,
) -> int:
    """Stage a complete S3 prefix, then publish the directory in one rename."""
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{local_target.name}.", dir=local_target.parent)
    ).resolve()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        inventory = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.startswith(s3_prefix + "/"):
                    raise ValueError(f"Model dependency key outside prefix: {key}")
                relative_path = key[len(s3_prefix) + 1 :]
                if not relative_path or key.endswith("/"):
                    continue
                _cache_target(temporary, relative_path, for_write=True)
                if relative_path == _DEPENDENCY_RECEIPT or relative_path in inventory:
                    raise ValueError(f"Duplicate or reserved model dependency key: {key}")
                inventory[relative_path] = obj["Size"]
        if not inventory:
            raise FileNotFoundError(f"No objects found under s3://{bucket}/{s3_prefix}/")
        # Capture the entire expected inventory before starting any transfer.
        for relative_path in inventory:
            local_path = temporary / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, f"{s3_prefix}/{relative_path}", str(local_path))
        receipt = build_dependency_receipt(temporary, inventory)
        if roles:
            error = dependency_invalid_reason(temporary, roles)
            if error is not None:
                raise ValueError(f"Downloaded model dependency {s3_prefix} is incomplete: {error}")
        (temporary / _DEPENDENCY_RECEIPT).write_text(json.dumps(receipt) + "\n")
        _publish_dependency_directory(temporary, local_target)
        return len(inventory)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _publish_dependency_directory(staging: Path, target: Path) -> None:
    """Replace an incomplete cache only after its replacement has been verified.

    All resolver callers hold the cache lock. A process killed between these two
    renames leaves a missing target, never a partially downloaded public directory.
    """
    if not target.exists() and not target.is_symlink():
        os.replace(staging, target)
        return
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-replaced.", dir=target.parent) as old:
        backup = Path(old) / "dependency"
        os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            os.replace(backup, target)
            raise


def _download_file_atomically(
    s3: Any,
    *,
    bucket: str,
    s3_key: str,
    local_target: Path,
    expected_size: int,
) -> None:
    """Stage one S3 object, then publish the file in one rename."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{local_target.name}.", suffix=local_target.suffix, dir=local_target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        s3.download_file(bucket, s3_key, str(temporary))
        receipt = build_dependency_receipt(temporary, {".": expected_size})
        error = dependency_invalid_reason(temporary, {"model"})
        if error is not None:
            raise ValueError(f"Downloaded model dependency {s3_key} is incomplete: {error}")
        os.replace(temporary, local_target)
        # The cache lock covers payload+receipt publication. An interrupted receipt
        # cannot certify a partial payload: only the complete file was renamed.
        receipt_path = _receipt_path(local_target)
        temporary.write_text(json.dumps(receipt) + "\n")
        os.replace(temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite_configs(
    checkpoint_dir: Path,
    resolved_paths: dict[str, str],
    cache_dir: Path,
    *,
    view_root: Path | None = None,
) -> Path:
    """Create an inference view with localized configs, leaving the source immutable.

    Use the exact validated resolutions, never rediscover an incomplete mounted
    copy while constructing the view after a cache repair.
    """
    # Build replacement map: old_path -> new_path
    replacements = {
        original: local for original, local in resolved_paths.items() if original != local
    }

    identities = {
        local: f"{MODEL_DEPENDENCY_REMOTE_ROOT}/{relative}"
        for original, local in replacements.items()
        if (relative := _to_relative_dep_path(original, cache_dir)) is not None
    }
    configs: dict[str, str] = {}
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
            config = json.loads(content)
            owner = config.get("processor_kwargs", config)
            base_model_cfg = config.get("base_model_cfg")
            if isinstance(base_model_cfg, dict) and any(
                path in identities for path in _extract_dep_paths(base_model_cfg)
            ):
                # N2 RL-family configs serialize the actor through the nested
                # GrootN2Config.to_dict(). Keep durable identities with that
                # owner so re-saving cannot bake in this host's cache paths.
                owner = base_model_cfg
            # final_processor_config.json is a diagnostic receipt: RL producers
            # stringify every attribute, including the optional identity map.
            # It is not a loadable config and must not be interpreted as one.
            if config_file != "experiment_cfg/final_processor_config.json":
                owner["model_dependency_identities"] = {
                    **(owner.get("model_dependency_identities") or {}),
                    **identities,
                }
            configs[config_file] = json.dumps(config, indent=2) + "\n"

    if not configs:
        return checkpoint_dir
    source = checkpoint_dir.resolve()
    links = _checkpoint_view_links(source, configs)
    metadata = {"version": 1, "source": str(source), "dependency_cache": str(cache_dir.resolve())}
    identity = checkpoint_view_identity(metadata, configs, links)
    views = view_root if view_root is not None else cache_dir / ".checkpoint-configs"
    views.mkdir(parents=True, exist_ok=True)
    view = views / identity
    if view.is_dir():
        checkpoint_view_metadata(view)
        return view
    staging = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=views))
    try:
        _write_checkpoint_view(links, staging, configs)
        (staging / CHECKPOINT_VIEW_METADATA_FILE).write_text(json.dumps(metadata) + "\n")
        os.replace(staging, view)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return view


def _checkpoint_view_links(source: Path, configs: dict[str, str]) -> dict[str, str]:
    """Snapshot link topology without reading payloads or walking linked asset directories.

    Added/removed entries and retargeted source symlinks require a new view. Recurse
    only into directories expanded for localized configs; other directories remain
    live symlinks, so changes within them are already visible to loaders.
    """
    return checkpoint_view_links(source, configs)


def _write_checkpoint_view(links: dict[str, str], target: Path, configs: dict[str, str]) -> None:
    """Copy only localized JSON; link the inventoried assets without reading payloads."""
    for name, content in configs.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    for name, link in links.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(link, target_is_directory=Path(link).is_dir())
