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

# provision_object_store.sh — thin wrapper around provision_object_store.py
#
# Usage:
#   ./provision_object_store.sh --path /mnt/aws-lfs-02/GrootDatasets --pool groot-gb200-02
#   ./provision_object_store.sh --path /mnt/aws-lfs-02/GrootDatasets --region us-east-1
#
# All arguments are forwarded directly to the Python script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec uv run python "${SCRIPT_DIR}/../migration/provision_object_store.py" "$@"
