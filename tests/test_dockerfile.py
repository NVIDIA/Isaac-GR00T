# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Check that the OSS Dockerfile installs the lock around the NGC PyTorch stack."""

import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import runpy
import sys
import types

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # The monorepo unit tests collect tests/oss on Python 3.10.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "pyproject.toml").is_file():
    ROOT = ROOT.parent / "oss"

DOCKERFILE = ROOT / "docker/Dockerfile"
DOCKER_README = ROOT / "docker/README.md"
INSTALLATION = ROOT / "docs/installation.md"
IMAGE_VENV = Path("/opt/gr00t-venv")
NGC_MODULES = ("torch", "torchvision", "torchcodec", "triton", "flash_attn", "flash_attn_2_cuda")

# The OSS CI jobs run the package tests inside the image built from docker/Dockerfile
# and set GR00T_DOCKER_IMAGE_TESTS=1; anywhere else there is no image environment.
in_docker_image = pytest.mark.skipif(
    os.environ.get("GR00T_DOCKER_IMAGE_TESTS") != "1",
    reason="needs the image built from docker/Dockerfile",
)


def _gr00t_sync() -> types.SimpleNamespace:
    # run_path, unlike an import, writes no __pycache__ into docker/.
    return types.SimpleNamespace(
        **runpy.run_path(str(ROOT / "docker/gr00t-sync"), run_name="gr00t_sync")
    )


def _lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def _excluded(command: list[str]) -> set[str]:
    return {command[i + 1] for i, arg in enumerate(command) if arg == "--no-install-package"}


def test_dockerfile_uses_the_ngc_base_and_documented_system_packages():
    dockerfile = DOCKERFILE.read_text()
    installation = INSTALLATION.read_text()

    (base,) = re.findall(r"^ARG PYTORCH_IMAGE=(\S+)$", dockerfile, re.MULTILINE)
    assert base.startswith("nvcr.io/nvidia/pytorch:")
    assert f"`{base}`" in DOCKER_README.read_text()
    assert "FROM ${PYTORCH_IMAGE}\n" in dockerfile

    dockerfile_commands = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    installation_commands = "\n".join(
        re.findall(r"^```bash\n(.*?)^```", installation, re.MULTILINE | re.DOTALL)
    )
    assert _apt_install(dockerfile_commands, "Dockerfile") == _apt_install(
        installation_commands, "installation.md bash blocks"
    )


def test_dockerfile_syncs_the_lock_only_through_gr00t_sync():
    dockerfile = DOCKERFILE.read_text()

    assert 'uv venv --system-site-packages "${UV_PROJECT_ENVIRONMENT}"' in dockerfile
    assert "gr00t-sync ${UV_EXTRAS}" in dockerfile
    assert 'ARG UV_EXTRAS="--extra training"' in dockerfile
    assert "UV_NO_SYNC=1" in dockerfile
    assert "uv sync" not in dockerfile


def test_build_context_admits_only_the_copied_files():
    copy_lines = [line for line in DOCKERFILE.read_text().splitlines() if line.startswith("COPY ")]
    assert not any(line.rstrip().endswith("\\") for line in copy_lines), "multi-line COPY"
    copied = {
        token
        for line in copy_lines
        if "--from=" not in line
        for token in line.split()[1:-1]
        if not token.startswith("--")
    }
    patterns = [
        line.strip()
        for line in (ROOT / "docker/Dockerfile.dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    allowed = {pattern[1:] for pattern in patterns if pattern.startswith("!")}

    assert "*" in patterns
    assert allowed == copied == {"pyproject.toml", "uv.lock", "docker/gr00t-sync"}


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_gr00t_sync_keeps_the_ngc_pytorch_stack(machine):
    """Catch a locked torch dependency shadowing NGC's CUDA libraries in the venv."""
    lock = _lock()
    (sync, *_) = _gr00t_sync().commands(lock, machine, ["--all-extras"], "python")
    excluded = _excluded(sync)

    assert sync[:5] == ["uv", "sync", "--frozen", "--inexact", "--no-install-project"]
    assert {"torch", "torchvision", "torchcodec", "triton", "flash-attn"} <= excluded
    for torch in (p for p in lock["package"] if p["name"] == "torch"):
        for dependency in torch.get("dependencies", []):
            if dependency["name"].startswith(("nvidia-", "cuda-", "triton")):
                assert dependency["name"] in excluded, dependency["name"]
    assert "nvidia-ml-py" not in excluded


def test_gr00t_sync_installs_the_aarch64_cuda13_builds_on_x86():
    """NGC runs CUDA 13 on x86_64, where the lock pins CUDA 12 NVRx and TensorRT."""
    module = _gr00t_sync()
    lock = _lock()
    args = ["--extra", "training", "--extra", "tensorrt"]

    sync, install = module.commands(lock, "x86_64", args, "/opt/gr00t-venv/bin/python")
    assert {"nvidia-resiliency-ext", "tensorrt-cu12"} <= _excluded(sync)
    assert sync[-len(args) :] == args
    assert install[:6] == [
        "uv",
        "pip",
        "install",
        "--python",
        "/opt/gr00t-venv/bin/python",
        "--no-deps",
    ]
    assert "nvidia-resiliency-ext==0.6.0" in install
    assert "tensorrt-cu13==10.16.1.11" in install
    assert not any(arg.split("==")[0] in module.NGC_PACKAGES for arg in install[6:])

    (arm_sync,) = module.commands(lock, "aarch64", args, "python")
    assert not {"nvidia-resiliency-ext", "tensorrt-cu13"} & _excluded(arm_sync)
    _, training_install = module.commands(lock, "x86_64", ["--extra", "training"], "python")
    assert "nvidia-resiliency-ext==0.6.0" in training_install
    assert not any(arg.startswith("tensorrt-cu13==") for arg in training_install)
    assert len(module.commands(lock, "x86_64", [], "python")) == 1


@in_docker_image
def test_image_hash_record_matches_the_package():
    """docker/README.md: the mounted checkout must match the files the image was built from."""
    record = (IMAGE_VENV / ".gr00t-project.sha256").read_text().split()
    expected = dict(zip(record[1::2], record[0::2], strict=True))

    assert expected.keys() == {"pyproject.toml", "uv.lock"}
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@in_docker_image
def test_image_takes_pytorch_from_ngc_and_the_lock_from_the_venv():
    import deepspeed
    import lance
    import torch

    from groot.core.training.models.n2.components import flash_attention

    assert Path(sys.prefix).resolve() == IMAGE_VENV
    for name in NGC_MODULES:
        path = Path(importlib.import_module(name).__file__).resolve()
        assert not path.is_relative_to(IMAGE_VENV), f"{name} imported from the venv: {path}"
    for module in (deepspeed, lance):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(IMAGE_VENV), f"{module.__name__} imported from {path}"
    assert torch.version.cuda.startswith("13."), torch.version.cuda
    assert flash_attention.FLASH_ATTN_2_AVAILABLE, "N2 cannot use FlashAttention"


@pytest.mark.gpu
@in_docker_image
def test_image_gpu_stack_runs_ngc_kernels_and_the_image_ffmpeg():
    """The image runs NGC's CUDA libraries and FlashAttention on either GPU host."""
    import importlib.util

    import flash_attn
    import torch
    import torchcodec.decoders  # noqa: F401

    assert torch.cuda.is_available()
    (torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")).sum().item()
    # Fails with "no kernel image" unless flash-attn was compiled for this GPU.
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn.flash_attn_func(q, q, q, causal=True).sum().item()
    # TensorRT is an optional extra: CI images sync it, the default image does not.
    if importlib.util.find_spec("tensorrt") is not None:
        tensorrt = importlib.import_module("tensorrt")
        assert Path(tensorrt.__file__).resolve().is_relative_to(IMAGE_VENV)
        assert importlib.metadata.version("tensorrt-cu13")

    mapped = {
        Path(field).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        for field in line.split()[5:]
        if field.startswith("/")
    }
    ffmpeg = sorted(str(p) for p in mapped if p.name.startswith("libavcodec.so"))
    cublas = sorted(p for p in mapped if p.name.startswith("libcublas"))
    # arm64 builds FFmpeg with NVDEC into /usr/local; x86_64 uses the distribution's.
    ffmpeg_root = "/usr/local/lib/" if platform.machine() == "aarch64" else "/usr/lib/"
    assert ffmpeg, "TorchCodec loaded no FFmpeg"
    assert all(p.startswith(ffmpeg_root) for p in ffmpeg), f"FFmpeg from {ffmpeg}"
    assert cublas, "torch loaded no cuBLAS"
    assert not any(p.is_relative_to(IMAGE_VENV) for p in cublas), f"cuBLAS from {cublas}"


@pytest.mark.gpu
@in_docker_image
def test_training_extra_enables_straggler_detection():
    """The training extra installs NVRx with a CUPTI build that matches torch on this host."""
    from groot.core.training.experiment import trainer

    assert trainer.STRAGGLER_AVAILABLE
    # Detector.initialize loads the CUPTI extension, which fails when the wheel's
    # CUPTI major differs from torch's; the trainer only logs that failure.
    cupti = importlib.import_module(f"{trainer.straggler_module.__name__}.cupti")
    manager = cupti.CuptiManager()
    manager.initialize()
    manager.shutdown()


def _apt_install(text: str, source: str) -> list[str]:
    """Return the options and packages of the first `apt-get install` command."""
    start = text.find("apt-get install")
    assert start != -1, f"{source} has no apt-get install command"
    command = []
    for line in text[start:].splitlines():
        command.append(line.rstrip().removesuffix("\\"))
        if not line.rstrip().endswith("\\"):
            break
    return sorted(" ".join(command).split("&&")[0].split()[2:])
