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

set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=${ROBOCASA_ROOT:-"$project_dir/.robocasa365"}
revision=f970a0aac4675a7e97ed5771cade4ade7044b2af

uv sync --project "$project_dir"
test -d "$source_dir/.git" || git clone https://github.com/robocasa/robocasa.git "$source_dir"
git -C "$source_dir" fetch --depth 1 origin "$revision"
git -C "$source_dir" checkout --detach "$revision"

simulator_python="$project_dir/.venv/bin/python"
site_packages=$("$simulator_python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$source_dir" > "$site_packages/robocasa365_source.pth"
"$simulator_python" -c 'import robocasa, robosuite'
