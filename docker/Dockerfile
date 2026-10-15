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

# Isaac GR00T 2 training image for x86_64 Linux.
#
# Runs the x86 training installation from docs/installation.md: the documented
# public CUDA base, its system packages, and `uv sync --frozen`. The frozen uv
# environment stays the software contract; this file only records those steps.
# The image holds no source checkout. Mount the repository and run commands
# from its package root. See docker/README.md.

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Same command as installation.md, including recommended packages.
# hadolint ignore=DL3008,DL3015
RUN apt-get update && apt-get install -y \
        build-essential ca-certificates curl git python3-dev python3-venv \
        ffmpeg yasm cmake libtool pkg-config libass-dev libfreetype6-dev \
        libvorbis-dev autoconf automake texinfo \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv \
    UV_PYTHON=python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

# The project files used for the build stay in /opt/gr00t-image so a mounted
# checkout can be compared with them.
WORKDIR /opt/gr00t-image
COPY pyproject.toml uv.lock ./

ARG UV_EXTRAS="--extra training"
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${UV_EXTRAS} \
    && sha256sum pyproject.toml uv.lock > "${UV_PROJECT_ENVIRONMENT}/.gr00t-project.sha256"

ENV VIRTUAL_ENV=/opt/gr00t-venv \
    PATH="/opt/gr00t-venv/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]
