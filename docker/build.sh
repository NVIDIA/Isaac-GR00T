#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Build the Isaac GR00T 2 training image from the package root.
# Extra arguments are passed to `docker build`, for example
#   bash docker/build.sh --build-arg UV_EXTRAS="--extra training --extra tensorrt"

set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    -f "$package_root/docker/Dockerfile" \
    -t "${GR00T_IMAGE:-gr00t-n2}" \
    "$@" \
    "$package_root"
