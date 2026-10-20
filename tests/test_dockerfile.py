# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Check that the OSS Dockerfile runs the documented x86 training installation."""

import hashlib
import os
from pathlib import Path
import platform
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "pyproject.toml").is_file():
    ROOT = ROOT.parent / "oss"

DOCKERFILE = ROOT / "docker/Dockerfile"
INSTALLATION = ROOT / "docs/installation.md"
IMAGE_VENV = Path("/opt/gr00t-venv")

# The OSS CI jobs run the package tests inside the image built from docker/Dockerfile
# and set GR00T_DOCKER_IMAGE_TESTS=1; anywhere else there is no image environment.
in_docker_image = pytest.mark.skipif(
    os.environ.get("GR00T_DOCKER_IMAGE_TESTS") != "1",
    reason="needs the image built from docker/Dockerfile",
)


def test_dockerfile_uses_the_documented_base_and_system_packages():
    dockerfile = DOCKERFILE.read_text()
    installation = INSTALLATION.read_text()

    bases = dict(re.findall(r"^ARG BASE_IMAGE_(AMD64|ARM64)=(\S+)$", dockerfile, re.MULTILINE))
    assert bases.keys() == {"AMD64", "ARM64"}, "Dockerfile needs one default base per architecture"
    for base in bases.values():
        assert f"`{base}`" in installation
    assert "FROM base-${TARGETARCH}\n" in dockerfile

    dockerfile_commands = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    installation_commands = "\n".join(
        re.findall(r"^```bash\n(.*?)^```", installation, re.MULTILINE | re.DOTALL)
    )
    assert _apt_install(dockerfile_commands, "Dockerfile") == _apt_install(
        installation_commands, "installation.md bash blocks"
    )


def test_dockerfile_installs_the_frozen_lock_without_exclusions():
    dockerfile = DOCKERFILE.read_text()

    assert "uv sync --frozen --no-install-project ${UV_EXTRAS}" in dockerfile
    assert 'ARG UV_EXTRAS="--extra training"' in dockerfile
    assert "--system-site-packages" not in dockerfile
    assert "--no-install-package" not in dockerfile


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
    assert allowed == copied == {"pyproject.toml", "uv.lock"}


@in_docker_image
def test_image_hash_record_matches_the_package():
    """docker/README.md: the mounted checkout must match the files the image was built from."""
    record = (IMAGE_VENV / ".gr00t-project.sha256").read_text().split()
    expected = dict(zip(record[1::2], record[0::2], strict=True))

    assert expected.keys() == {"pyproject.toml", "uv.lock"}
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@in_docker_image
def test_image_venv_holds_the_locked_x86_training_stack():
    import deepspeed
    import flash_attn
    import flash_attn_2_cuda
    import lance
    import torch
    import torchcodec.decoders
    import torchvision

    from groot.core.training.models.n2.components import flash_attention

    assert Path(sys.prefix).resolve() == IMAGE_VENV
    for module in (torch, torchvision, torchcodec, flash_attn, flash_attn_2_cuda, deepspeed, lance):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(IMAGE_VENV), f"{module.__name__} imported from {path}"
    assert torch.__version__ == "2.7.1+cu128"
    assert flash_attention.FLASH_ATTN_2_AVAILABLE, "N2 cannot use FlashAttention on x86_64"


TORCH_RELEASES = {"x86_64": "2.7.1+cu128", "aarch64": "2.11.0"}


@pytest.mark.gpu
@in_docker_image
def test_image_gpu_stack_uses_the_venv_cuda_libraries_and_system_ffmpeg():
    """The image uses the venv's CUDA libraries and the distribution FFmpeg on either GPU host."""
    import importlib
    import importlib.util

    import flash_attn
    import flash_attn_2_cuda
    import torch
    import torchcodec.decoders
    import torchvision
    import triton

    modules = [torch, torchvision, torchcodec, triton, flash_attn, flash_attn_2_cuda]
    # TensorRT is an optional extra: CI images sync it, the default image does not.
    if importlib.util.find_spec("tensorrt") is not None:
        modules.append(importlib.import_module("tensorrt"))
    assert Path(sys.prefix).resolve() == IMAGE_VENV
    for module in modules:
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(IMAGE_VENV), f"{module.__name__} imported from {path}"
    assert torch.__version__.startswith(TORCH_RELEASES[platform.machine()]), torch.__version__
    assert torch.cuda.is_available()
    (torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")).sum().item()
    # Fails with "no kernel image" unless flash-attn was compiled for this GPU.
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn.flash_attn_func(q, q, q, causal=True).sum().item()

    mapped = {
        Path(field).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        for field in line.split()[5:]
        if field.startswith("/")
    }
    ffmpeg = sorted(str(p) for p in mapped if p.name.startswith("libavcodec.so"))
    cublas = sorted(p for p in mapped if p.name.startswith("libcublas"))
    assert ffmpeg, "TorchCodec loaded no FFmpeg"
    assert all(p.startswith("/usr/lib/") for p in ffmpeg), f"FFmpeg from {ffmpeg}"
    assert cublas, "torch loaded no cuBLAS"
    assert all(p.is_relative_to(IMAGE_VENV) for p in cublas), f"cuBLAS from {cublas}"


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
