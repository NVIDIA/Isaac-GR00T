# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Artifact resolution for N2 TensorRT conversion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from groot.core.utils.storage import msc as msc_utils
from groot.deployment.n2_trt.contract import resolve_checkpoint_actor_config


def resolve_checkpoint_vae_weights(
    checkpoint: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    """Resolve the exact external VAE weights consumed by export."""
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    if explicit is not None:
        configured = str(explicit)
    else:
        config_path = checkpoint_path / "config.json"
        model_config = resolve_checkpoint_actor_config(_read_json(config_path))
        vae_config = model_config.get("vae_cfg", {})
        configured = vae_config.get("vae_pretrained_path") if isinstance(vae_config, dict) else None
        if not isinstance(configured, str) or not configured:
            raise ValueError(f"vae_cfg.vae_pretrained_path is missing from {config_path}")

    weights = Path(configured).expanduser()
    if not weights.is_absolute():
        checkpoint_relative = checkpoint_path / weights
        if checkpoint_relative.is_file():
            weights = checkpoint_relative
        elif configured.startswith(("s3://", "msc://")):
            weights = _resolve_remote_model_file(
                configured,
                param_name="model.vae_cfg.vae_pretrained_path",
            )
        else:
            weights = Path(
                msc_utils.resolve_model_path(
                    configured,
                    "model.vae_cfg.vae_pretrained_path",
                )
            ).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(f"VAE weights not found: {weights}")
    return weights.resolve()


def _resolve_remote_model_file(path: str, *, param_name: str) -> Path:
    """Download one remote VAE object and atomically publish its cached file."""
    cache_dir = os.environ.get("MSC_CACHE_PATH")
    if not cache_dir:
        raise RuntimeError(
            f"MSC_CACHE_PATH environment variable is required when {param_name} "
            f"is a remote URL: {path}"
        )
    remote_body = path.split("://", maxsplit=1)[1]
    if "/" not in remote_body or remote_body.endswith("/"):
        raise ValueError(f"{param_name} must identify a remote file: {path}")
    filename = remote_body.rsplit("/", maxsplit=1)[1]
    if filename in {"", ".", ".."}:
        raise ValueError(f"{param_name} must identify a remote file: {path}")

    digest = hashlib.sha256(path.encode()).hexdigest()
    files_root = Path(cache_dir).expanduser() / "model_files"
    target_dir = files_root / digest
    target = target_dir / filename
    if target.is_file():
        return target.resolve()

    files_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=files_root))
    temporary_target = temporary / filename
    try:
        if not msc_utils.MSC_AVAILABLE:
            raise ImportError("multistorageclient is required to download remote VAE weights")
        if path.startswith("s3://"):
            bucket, key = msc_utils.parse_s3_uri(path)
            if not key:
                raise ValueError(f"{param_name} must identify a remote file: {path}")
            profile_name = f"model-file-{digest[:16]}"
            config_builder = msc_utils.DataSyncConfigBuilder(
                "~/.gear/data_credentials",
                profile="default",
                enable_telemetry=False,
            )
            config_builder.add_profile(profile_name, bucket)
            config_builder.build()
            download_url = f"msc://{profile_name}/{key}"
        else:
            download_url = path
        msc_utils.msc.download_file(
            url=download_url,
            local_path=str(temporary_target),
        )

        if not temporary_target.is_file():
            raise FileNotFoundError(f"Remote VAE weights were not downloaded: {path}")
        try:
            os.replace(temporary, target_dir)
        except OSError:
            if not target.is_file():
                raise
        return target.resolve()
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint metadata not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint metadata must be an object: {path}")
    return payload
