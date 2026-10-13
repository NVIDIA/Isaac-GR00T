# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resolve or create a validated, target-local N2 TensorRT engine bundle."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import uuid

from groot.deployment.n2_trt.artifacts import resolve_checkpoint_vae_weights
from groot.deployment.n2_trt.contract import N2TensorRTContract
from groot.deployment.n2_trt.manifest import (
    MANIFEST_FILENAME,
    EngineManifest,
    RuntimeIdentity,
    SourceRevision,
    canonical_sha256,
    checkpoint_file_hashes,
    sha256_file,
)
from groot.deployment.n2_trt.policy_measurement import (
    POLICY_MEASUREMENT_FILENAME,
    validate_policy_measurement,
)

DEFAULT_BUILDER_OPTIMIZATION_LEVEL = 3
DEFAULT_CODEC = "mp4v"


def default_trt_cache_root() -> Path:
    """Return the user-local cache used for automatically built bundles."""
    configured = os.environ.get("GROOT_TRT_CACHE_ROOT")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "groot" / "tensorrt"


def resolve_or_build_engine_bundle(
    *,
    checkpoint: str | Path,
    embodiment: str,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
    device: str | None = None,
    rebuild_token: str | None = None,
    codec: str = DEFAULT_CODEC,
    num_video_steps: int = 2,
    num_action_steps: int = 4,
) -> Path:
    """Return a validated cached bundle, building it once when necessary."""
    if codec not in {"h264", "mp4v", "none"}:
        raise ValueError(f"unsupported N2 TensorRT serving codec: {codec!r}")
    cuda_device_index = _select_cuda_device(device)
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    root = Path(cache_root).expanduser() if cache_root is not None else default_trt_cache_root()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _checkpoint_lock(checkpoint_path):
        vae_weights = _prepare_checkpoint_for_identity(checkpoint_path)
        contract = N2TensorRTContract.from_checkpoint(
            checkpoint_path,
            embodiment=embodiment,
            video_denoising_steps=num_video_steps,
            action_denoising_steps=num_action_steps,
            joint_denoise=True,
            conditional_kv_cache=True,
        )
        checkpoint_files = checkpoint_file_hashes(checkpoint_path)
        vae_weights_sha256 = sha256_file(vae_weights)
    runtime = RuntimeIdentity.capture()
    source = SourceRevision.capture("auto_builder", __file__)
    if source.dirty:
        raise RuntimeError("automatic TensorRT builds require a clean, pinned source checkout")

    identity = canonical_sha256(
        {
            "contract": contract.to_dict(),
            "checkpoint_files": checkpoint_files,
            "vae_weights_sha256": vae_weights_sha256,
            "runtime": asdict(runtime),
            "source_revision": source.revision,
            "builder_optimization_level": DEFAULT_BUILDER_OPTIMIZATION_LEVEL,
            "codec": codec,
        }
    )
    bundle = root / identity
    lock_path = root / f".{identity}.lock"
    generations = root / f".{identity}.generations"
    rebuild_marker = root / f".{identity}.rebuild-token"

    with _exclusive_lock(lock_path):
        rebuild_required = rebuild and not _rebuild_token_matches(rebuild_marker, rebuild_token)
        bundle_present = bundle.exists() or bundle.is_symlink()
        if bundle_present and not rebuild_required:
            try:
                _validate_cached_bundle(
                    bundle,
                    contract=contract,
                    runtime=runtime,
                    codec=codec,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            else:
                _emit(f"Reusing validated TensorRT bundle: {bundle}")
                return bundle

        generations.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=generations))
        try:
            _emit(f"Building and validating TensorRT bundle: {bundle}")
            _run_build_pipeline(
                checkpoint=checkpoint_path,
                embodiment=embodiment,
                artifact_root=temporary,
                source_revision=source.revision,
                cuda_device_index=cuda_device_index,
                codec=codec,
                num_video_steps=num_video_steps,
                num_action_steps=num_action_steps,
            )
            _validate_cached_bundle(
                temporary,
                contract=contract,
                runtime=runtime,
                codec=codec,
            )
            retired = _promote_generation(
                bundle=bundle,
                candidate=temporary,
                generations=generations,
            )
            temporary = None
            _remove_retired_generation(retired, generations=generations)
            if rebuild and rebuild_token is not None:
                _write_rebuild_token(rebuild_marker, rebuild_token)
        except BaseException:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    _emit(f"Created validated TensorRT bundle: {bundle}")
    return bundle


def _rebuild_token_matches(marker: Path, token: str | None) -> bool:
    if token is None:
        return False
    try:
        return marker.read_text(encoding="utf-8") == token
    except FileNotFoundError:
        return False


def _write_rebuild_token(marker: Path, token: str) -> None:
    """Publish a successful rebuild token atomically under the identity lock."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{marker.name}.", dir=marker.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(token)
        os.replace(temporary_name, marker)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _select_cuda_device(device: str | None) -> int:
    """Resolve and select the CUDA target used by identity capture and engine work."""
    import torch

    if device is None:
        return torch.cuda.current_device()
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError(f"automatic TensorRT builds require a CUDA device, got {device!r}")
    if target.index is None:
        return torch.cuda.current_device()
    torch.cuda.set_device(target)
    return target.index


def _promote_generation(*, bundle: Path, candidate: Path, generations: Path) -> Path | None:
    """Atomically point ``bundle`` at a validated candidate and return its predecessor."""
    previous = bundle.resolve(strict=False) if bundle.is_symlink() else None
    link = bundle.parent / f".{bundle.name}.{uuid.uuid4().hex}.link"
    os.symlink(os.path.relpath(candidate, start=bundle.parent), link, target_is_directory=True)
    try:
        if bundle.exists() and not bundle.is_symlink():
            # Compatibility for pre-generation caches created before this layout existed.
            previous = generations / f"legacy-{uuid.uuid4().hex}"
            os.replace(bundle, previous)
            try:
                os.replace(link, bundle)
            except BaseException:
                os.replace(previous, bundle)
                raise
        else:
            os.replace(link, bundle)
    finally:
        link.unlink(missing_ok=True)
    return previous


def _remove_retired_generation(previous: Path | None, *, generations: Path) -> None:
    """Remove a superseded generation without following arbitrary symlink targets."""
    if previous is None:
        return
    try:
        previous.relative_to(generations)
    except ValueError:
        return
    shutil.rmtree(previous, ignore_errors=True)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an inter-process advisory lock for one preparation or bundle key."""
    with path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


@contextmanager
def _checkpoint_lock(checkpoint: Path) -> Iterator[None]:
    """Lock the checkpoint directory itself across cache roots and processes."""
    descriptor = os.open(checkpoint, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _prepare_checkpoint_for_identity(checkpoint: Path) -> Path:
    """Materialize dependencies before hashing the exact VAE export input."""
    localized = _ensure_checkpoint_dependencies(checkpoint)
    return resolve_checkpoint_vae_weights(localized)


def _ensure_checkpoint_dependencies(checkpoint: Path) -> Path:
    """Load the training dependency helper only when an automatic build starts."""
    from groot.core.training.models.n2.utils.cosmos_model_deps import ensure_cosmos_model_deps

    return Path(ensure_cosmos_model_deps(checkpoint))


def _validate_cached_bundle(
    bundle: Path,
    *,
    contract: N2TensorRTContract,
    runtime: RuntimeIdentity,
    codec: str,
) -> None:
    manifest_path = bundle / MANIFEST_FILENAME
    manifest = EngineManifest.read(manifest_path)
    manifest.validate(artifact_root=bundle, contract=contract, runtime=runtime)
    if manifest.build_settings.codec != codec:
        raise RuntimeError(
            "N2 TensorRT bundle codec does not match the requested serving codec: "
            f"bundle={manifest.build_settings.codec!r}, requested={codec!r}"
        )
    validate_policy_measurement(
        bundle / POLICY_MEASUREMENT_FILENAME,
        contract=contract,
        manifest_path=manifest_path,
    )


def _run_build_pipeline(
    *,
    checkpoint: Path,
    embodiment: str,
    artifact_root: Path,
    source_revision: str,
    cuda_device_index: int,
    codec: str = DEFAULT_CODEC,
    num_video_steps: int = 2,
    num_action_steps: int = 4,
) -> None:
    schedule_args = (
        "--num-video-steps",
        num_video_steps,
        "--num-action-steps",
        num_action_steps,
    )
    commands: tuple[tuple[Any, ...], ...] = (
        (
            sys.executable,
            "-P",
            "-m",
            "groot.deployment.n2_trt.tools.export_transformer",
            "--checkpoint",
            checkpoint,
            "--embodiment",
            embodiment,
            "--codec",
            codec,
            "--output-dir",
            artifact_root,
            "--source-revision",
            source_revision,
            *schedule_args,
        ),
        (
            sys.executable,
            "-P",
            "-m",
            "groot.deployment.n2_trt.tools.export_vae",
            "--checkpoint",
            checkpoint,
            "--embodiment",
            embodiment,
            "--output-dir",
            artifact_root,
            "--source-revision",
            source_revision,
            *schedule_args,
        ),
        (
            sys.executable,
            "-P",
            "-m",
            "groot.deployment.n2_trt.tools.build_engines",
            "--artifact-root",
            artifact_root,
            "--builder-optimization-level",
            DEFAULT_BUILDER_OPTIMIZATION_LEVEL,
            "--source-revision",
            source_revision,
        ),
        (
            sys.executable,
            "-P",
            "-m",
            "groot.deployment.n2_trt.validation",
            "--artifact-root",
            artifact_root,
            "--checkpoint",
            checkpoint,
            "--embodiment",
            embodiment,
            "--source-revision",
            source_revision,
            *schedule_args,
        ),
        (
            sys.executable,
            "-P",
            "-m",
            "groot.deployment.n2_trt.policy_measurement",
            "--checkpoint",
            checkpoint,
            "--embodiment",
            embodiment,
            "--engine-dir",
            artifact_root,
            "--codec",
            codec,
            "--out",
            artifact_root / POLICY_MEASUREMENT_FILENAME,
            *schedule_args,
        ),
    )
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["GROOT_N2_NUM_VIDEO_STEPS"] = str(num_video_steps)
    environment["GROOT_N2_NUM_ACTION_STEPS"] = str(num_action_steps)
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if cuda_device_index >= len(devices):
            raise ValueError(
                f"CUDA device index {cuda_device_index} is outside CUDA_VISIBLE_DEVICES={visible_devices!r}"
            )
        environment["CUDA_VISIBLE_DEVICES"] = devices[cuda_device_index]
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_device_index)
    for command in commands:
        subprocess.run(
            [str(item) for item in command],
            env=environment,
            check=True,
        )


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")
