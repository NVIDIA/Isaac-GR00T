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
