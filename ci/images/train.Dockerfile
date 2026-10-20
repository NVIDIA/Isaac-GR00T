# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# syntax=docker/dockerfile:1
# Multi-arch OSS training image. The CI jobs choose the NGC PyTorch base per
# target architecture while applying the package lock at image build time.

ARG PYTORCH_IMAGE
FROM ${PYTORCH_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${CUDA_HOME}/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspace

# OS packages used by training and video decode.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential yasm cmake libtool git pkg-config \
        libass-dev libfreetype6-dev libvorbis-dev \
        autoconf automake texinfo tmux ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ARM64 torchcodec does not have a suitable prebuilt wheel for this runtime.
# Match the legacy GB200 image by building FFmpeg and torchcodec from source.
# hadolint ignore=DL3003
RUN ARCH="$(dpkg --print-architecture)" && \
    if [ "${ARCH}" = "arm64" ]; then \
        git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /opt/nv-codec-headers && \
        make -C /opt/nv-codec-headers install && \
        mkdir -p /opt/ffmpeg/sources && \
        git clone https://git.ffmpeg.org/ffmpeg.git /opt/ffmpeg/sources/ffmpeg && \
        cd /opt/ffmpeg/sources/ffmpeg && \
        ./configure \
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
            --nvccflags="-gencode arch=compute_89,code=sm_89 -O2" && \
        make -j"$(nproc)" && \
        make install; \
    fi

# Preserve NGC's CUDA/PyTorch stack, add training packages that should not be
# resolved from the project lockfile during image build.
# hadolint ignore=DL3059
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel uv && \
    python -m pip uninstall -y apex || true && \
    sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt && \
    sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt && \
    sed -i s/filelock==3.18.0/filelock==3.25.2/g /etc/pip/constraint.txt && \
    python -m pip install --no-cache-dir deepspeed==0.18.8 && \
    ARCH="$(dpkg --print-architecture)" && \
    if [ "${ARCH}" = "amd64" ]; then \
        python -m pip install --no-cache-dir torchcodec==0.4.0 && \
        python -c "import torchcodec; import torchcodec.decoders"; \
    fi

# Remove this when a compatible upstream ARM64 wheel is available.
# hadolint ignore=DL3003
RUN ARCH="$(dpkg --print-architecture)" && \
    if [ "${ARCH}" = "arm64" ]; then \
        git clone https://github.com/meta-pytorch/torchcodec.git /opt/torchcodec && \
        cd /opt/torchcodec && \
        git checkout -b release/0.9 origin/release/0.9 && \
        git cherry-pick --no-commit 7dbbf837d0a82 && \
        BUILD_AGAINST_ALL_FFMPEG_FROM_S3=true python3 setup.py bdist_wheel && \
        python -m pip install --no-cache-dir dist/torchcodec-0.9.1-cp312-cp312-linux_aarch64.whl && \
        python -c "import torchcodec; import torchcodec.decoders"; \
    fi

COPY pyproject.toml uv.lock ./

# Use a venv with system-site-packages so NGC torch/NCCL/CUDA packages remain
# visible. Skip packages that should come from the optimized base runtime or the
# compatibility pins above.
# hadolint ignore=DL3059
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /workspace/.venv --system-site-packages --python python3 && \
    VIRTUAL_ENV=/workspace/.venv PATH="/workspace/.venv/bin:${PATH}" \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --active --no-install-project --extra training --extra dev \
        --no-install-package deepspeed \
        --no-install-package flash-attn \
        --no-install-package torch \
        --no-install-package torchcodec \
        --no-install-package torchvision \
        --no-install-package nvidia-cublas-cu12 \
        --no-install-package nvidia-cuda-cupti-cu12 \
        --no-install-package nvidia-cuda-nvrtc-cu12 \
        --no-install-package nvidia-cuda-runtime-cu12 \
        --no-install-package nvidia-cudnn-cu12 \
        --no-install-package nvidia-cufft-cu12 \
        --no-install-package nvidia-curand-cu12 \
        --no-install-package nvidia-cusolver-cu12 \
        --no-install-package nvidia-cusparse-cu12 \
        --no-install-package nvidia-cusparselt-cu12 \
        --no-install-package nvidia-nccl-cu12 \
        --no-install-package nvidia-nvjitlink-cu12 \
        --no-install-package nvidia-nvtx-cu12 \
        --no-install-package triton \
        --inexact && \
    uv pip install --python /workspace/.venv/bin/python --no-cache-dir pip wheel

ENV PATH="/workspace/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/.venv" \
    HF_HUB_OFFLINE=1 \
    MUJOCO_GL=egl

CMD ["/bin/bash"]
