# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Isaac GR00T 2 image for x86_64 Linux and Linux aarch64 (GB300), on the NGC
# PyTorch container that GR00T's internal training images use.
#
# NGC provides PyTorch, torchvision, Triton, FlashAttention and the CUDA 13 libraries.
# TorchCodec is built against that PyTorch, and docker/gr00t-sync installs uv.lock
# around it into a --system-site-packages venv. The best-effort host installation in
# docs/installation.md installs the whole lock instead. The image holds no source
# checkout. Mount the repository and run commands from its package root. See
# docker/README.md.

ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
# hadolint ignore=DL3006
FROM ${PYTORCH_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Same command as installation.md, including recommended packages.
# hadolint ignore=DL3008,DL3015
RUN apt-get update && apt-get install -y \
        build-essential ca-certificates curl git python3-dev python3-venv \
        ffmpeg yasm cmake libtool pkg-config libass-dev libfreetype6-dev \
        libvorbis-dev autoconf automake texinfo \
    && rm -rf /var/lib/apt/lists/*

# On arm64, build FFmpeg with NVDEC from source, as the internal training image
# does. sm_100 kernels also run on GB300 (compute capability 10.3).
ARG FFMPEG_VERSION=n7.1.5
ARG NV_CODEC_HEADERS_VERSION=n13.0.19.0
# hadolint ignore=DL3003
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        git clone --depth 1 --branch "${NV_CODEC_HEADERS_VERSION}" \
            https://github.com/FFmpeg/nv-codec-headers.git /opt/nv-codec-headers \
        && make -C /opt/nv-codec-headers install \
        && git clone --depth 1 --branch "${FFMPEG_VERSION}" \
            https://github.com/FFmpeg/FFmpeg.git /opt/ffmpeg \
        && cd /opt/ffmpeg \
        && ./configure \
            --extra-cflags="-I/usr/local/cuda/include" \
            --extra-ldflags="-L/usr/local/cuda/lib64" \
            --extra-libs="-lpthread -lm" \
            --ld="g++" \
            --enable-nonfree \
            --enable-cuda-nvcc \
            --enable-libass \
            --enable-libfreetype \
            --enable-libvorbis \
            --enable-gpl \
            --disable-static \
            --enable-shared \
            --nvccflags="-gencode arch=compute_100,code=sm_100 -O2" \
        && make -j"$(nproc)" \
        && make install \
        && ldconfig \
        && rm -rf /opt/ffmpeg /opt/nv-codec-headers; \
    fi \
    && ffmpeg -version

# Released TorchCodec wheels are built for PyPI PyTorch; NGC's has a different ABI.
# The cherry-picked upstream commit adds the aarch64 FFmpeg libraries to the build.
ARG TORCHCODEC_VERSION=v0.9.1
# hadolint ignore=DL3003
RUN python3 -m pip install --no-cache-dir --upgrade setuptools wheel \
    && git clone --branch "${TORCHCODEC_VERSION}" \
        https://github.com/meta-pytorch/torchcodec.git /opt/torchcodec \
    && cd /opt/torchcodec \
    && git cherry-pick --no-commit 7dbbf837d0a82 \
    && BUILD_AGAINST_ALL_FFMPEG_FROM_S3=true python3 setup.py bdist_wheel \
    && python3 -m pip install --no-cache-dir dist/torchcodec-*.whl \
    && cd / && rm -rf /opt/torchcodec \
    && python3 -c "import torchcodec.decoders"

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/

# `uv run` must not sync: a plain sync would replace NGC's PyTorch with the lock's.
ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv \
    UV_PYTHON=python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1

# The project files used for the build stay in /opt/gr00t-image so a mounted
# checkout can be compared with them.
WORKDIR /opt/gr00t-image
COPY pyproject.toml uv.lock ./
COPY --chmod=0755 docker/gr00t-sync /usr/local/bin/gr00t-sync

ARG UV_EXTRAS="--extra training"
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --system-site-packages "${UV_PROJECT_ENVIRONMENT}" \
    && gr00t-sync ${UV_EXTRAS} \
    && sha256sum pyproject.toml uv.lock > "${UV_PROJECT_ENVIRONMENT}/.gr00t-project.sha256"

ENV VIRTUAL_ENV=/opt/gr00t-venv \
    PATH="/opt/gr00t-venv/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]
