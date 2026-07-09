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
# syntax=docker/dockerfile:1
# Stable minimal base image (layers 1-3: apt/ROS/uv, s5cmd/OSMO, Node.js + agent CLIs).
# Does NOT include pyproject.toml/uv.lock deps — rebuilt only when this Dockerfile changes.
# image.minimal extends this with the cheap uv sync layer.
#
# === Layer ordering for cache stability ===
#
# Docker invalidates a layer's cache when its content changes, AND all layers
# below it. We order layers so that the most stable (rarely-changing) layers
# are at the top, and the hottest (most-frequently-changing) layers are as
# late as possible. This minimizes rebuild time on typical MRs.
#
# Order rationale:
#   1. System deps + ROS + uv        — changes only on Dockerfile edits
#   2. s5cmd / OSMO CLI / osmo_runner — pinned versions, no Python dependency
#   3. Node.js / agent CLIs          — pinned versions, no Python dependency

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Use bash as the default shell for all RUN commands
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# TARGETARCH (amd64/arm64) is a BuildKit predefined ARG but is not reliably
# propagated into heredoc shells. Layer 2 uses uname -m
# (returns x86_64 or aarch64) for architecture detection instead.
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    AWS_ENDPOINT_URL=https://pdx.s8k.io \
    S3_ENDPOINT_URL=https://pdx.s8k.io \
    ROS_DISTRO=humble \
    ASSETKIT_SOURCES='["gear-locomanip"]' \
    PYTHON=/usr/bin/python

# ── Layer 1: System dependencies + ROS + uv (very stable) ─────────────────
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOF
set -e

# Install system dependencies
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    wget \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    python-is-python3 \
    software-properties-common \
    gnupg \
    lsb-release \
    ffmpeg \
    libegl1 \
    jq \
    dnsutils \
    iproute2 \
    ipset \
    iptables \
    gettext-base \
    lcov \
    docker.io

# Install ROS 2 Humble
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list
apt-get update
apt-get install -y --no-install-recommends \
    ros-humble-ros-core \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-common-interfaces \
    ros-dev-tools

# Source ROS 2 in bashrc
echo 'source /opt/ros/humble/setup.bash' >> /etc/bash.bashrc

# Install uv (replaces pip for package management)
curl -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
EOF

WORKDIR /workspace

# ── Layer 2: s5cmd, OSMO CLI, osmo_runner, crane (stable, no Python dependency) ──
# s5cmd is installed for both amd64 and arm64.
# osmo_runner, OSMO CLI, and crane are amd64-only (used by the amd64 CI runner only).
# Uses uname -m (x86_64 / aarch64) instead of TARGETARCH for architecture detection.
# hadolint ignore=DL3003,SC2155
RUN --mount=type=secret,id=bucket_access_key \
    --mount=type=secret,id=bucket_secret_key <<EOF
set -e

ARCH=$(uname -m)  # x86_64 or aarch64

# Install s5cmd v2.3.0 (architecture-aware)
# Filenames from https://github.com/peak/s5cmd/releases/tag/v2.3.0:
#   x86_64  → Linux-64bit
#   aarch64 → Linux-arm64  (lowercase, NOT Linux-ARM64bit)
if [ "${ARCH}" = "aarch64" ]; then
  S5CMD_ARCH="Linux-arm64"
else
  S5CMD_ARCH="Linux-64bit"
fi
curl -fsSL "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_${S5CMD_ARCH}.tar.gz" -o /tmp/s5cmd.tar.gz
tar -xzf /tmp/s5cmd.tar.gz -C /tmp s5cmd
mv /tmp/s5cmd /usr/local/bin/s5cmd
rm /tmp/s5cmd.tar.gz

# OSMO CLI and osmo_runner: amd64 only.
# arm64 images don't need OSMO CLI or osmo_runner; these are only needed on the amd64 CI runner.
# No arm64 osmo_runner S3 artifact is available either.
if [ "${ARCH}" != "aarch64" ]; then
  curl -fsSL "https://urm.nvidia.com/artifactory/sw-osmo-generic-local/rc/cli/2026.1.23/osmo-client-installer-2026.1.23-linux-x86_64.sh" -o /tmp/osmo-installer.sh
  chmod +x /tmp/osmo-installer.sh
  cd /tmp && ./osmo-installer.sh
  rm /tmp/osmo-installer.sh

  export AWS_ACCESS_KEY_ID=$(cat /run/secrets/bucket_access_key)
  export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/bucket_secret_key)
  export AWS_ENDPOINT_URL=https://pdx.s8k.io
  s5cmd cp s3://GearHome/ci_cd_tests/osmo_runner_ubuntu22_amd64_v6.2 /usr/local/bin/osmo_runner
  chmod +x /usr/local/bin/osmo_runner

  # crane (google/go-containerregistry) — used by osmo_docker_build.sh for
  # registry tag operations (manifest check, retag) without a Docker daemon.
  curl -fsSL "https://github.com/google/go-containerregistry/releases/download/v0.20.2/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar -xz -C /usr/local/bin crane
  chmod +x /usr/local/bin/crane
fi
EOF

# ── Layer 3: Node.js + agent CLIs (stable, no Python dependency) ───────────
# System-level install via apt + npm. Independent of the Python venv, so it
# lives above the COPY to stay cached when pyproject.toml changes.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/root/.npm <<EOF
set -e

# Add NodeSource repository for Node.js 22 LTS
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -

# Install Node.js
apt-get install -y --no-install-recommends nodejs

# Install agent CLIs globally. Claude remains required by existing CI agents;
# Codex/devcontainer are used by the gated autorun exploration job.
npm install -g --no-audit --no-fund \
  @anthropic-ai/claude-code@2.1.126 \
  @openai/codex@0.128.0 \
  @devcontainers/cli@0.86.0
EOF

ENV GROOT_CI=1 \
    LD_LIBRARY_PATH="/opt/ros/humble/lib" \
    AMENT_PREFIX_PATH="/opt/ros/humble"

CMD ["/bin/bash"]
