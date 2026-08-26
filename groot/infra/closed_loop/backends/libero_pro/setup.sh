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
source_dir=${LIBERO_PRO_ROOT:-"$project_dir/.libero-pro"}
revision=13613f0cc90e3869b157d34f7a88e7d6041cd423
data_dir=${LIBERO_PRO_DATA_ROOT:-"$project_dir/.libero-pro-data"}
data_revision=c86fc3b8293185a6f373677018ff3e37f8391602

uv sync --project "$project_dir"
test -d "$source_dir/.git" || git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$source_dir"
git -C "$source_dir" fetch --depth 1 origin "$revision"
git -C "$source_dir" checkout --detach "$revision"
test -d "$data_dir/.git" || git clone https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro "$data_dir"
git -C "$data_dir" fetch --depth 1 origin "$data_revision"
git -C "$data_dir" checkout --detach "$data_revision"

simulator_python="$project_dir/.venv/bin/python"
site_packages=$("$simulator_python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$source_dir" > "$site_packages/libero_pro_source.pth"

config_dir=${LIBERO_CONFIG_PATH:-"$project_dir/.libero-config"}
mkdir -p "$config_dir"
benchmark_root="$source_dir/libero/libero"
printf '%s\n' \
  "assets: $benchmark_root/assets" \
  "bddl_files: $data_dir/bddl_files" \
  "benchmark_root: $benchmark_root" \
  "datasets: $source_dir/libero/datasets" \
  "init_states: $data_dir/init_files" \
  > "$config_dir/config.yaml"

"$simulator_python" -c 'from libero.libero import benchmark'
