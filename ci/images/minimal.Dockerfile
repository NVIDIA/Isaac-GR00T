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
# Thin OSS layer on top of base.minimal: installs Python deps from the package root.

ARG BASE_MINIMAL_TAG=latest
FROM nvcr.io/nvidian/groot-ci-base-minimal:${BASE_MINIMAL_TAG}

ARG OSS_PACKAGE_DIR=.
WORKDIR /workspace/oss

# Dependency manifests for the future Isaac-GR00T-N2 package. In gr00t this
# package lives under oss/; in Isaac-GR00T-N2 it lives at the repository root.
COPY ${OSS_PACKAGE_DIR}/pyproject.toml ${OSS_PACKAGE_DIR}/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-install-project --extra dev

ENV PATH="/workspace/oss/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/oss/.venv"

CMD ["/bin/bash"]
