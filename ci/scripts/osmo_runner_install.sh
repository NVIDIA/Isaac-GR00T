#!/bin/bash
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

echo "Installing OSMO CLI and Runner..."

# Check if osmo is already installed
if command -v osmo &> /dev/null; then
  echo "OSMO CLI already installed ($(osmo --version 2> /dev/null || echo 'version unknown'))"
else
  echo "Installing OSMO CLI..."
  (cd /tmp && curl -fsSL https://urm.nvidia.com/artifactory/sw-osmo-generic-local/rc/cli/2026.1.23/osmo-client-installer-2026.1.23-linux-x86_64.sh -o osmo-installer.sh && chmod +x osmo-installer.sh && ./osmo-installer.sh)
  echo "OSMO CLI installed successfully"
fi

# Login to OSMO
echo "Logging in to OSMO..."
osmo login --method token --token "${OSMO_TOKEN}" "${OSMO_URL}"

# Always install osmo_runner (always fetch latest version)
# TODO(cuonguyen): have a proper way to release osmo_runner with versioning
echo "Installing OSMO Runner..."
s5cmd cp s3://GearHome/ci_cd_tests/osmo_runner_ubuntu22_amd64_v6.2 /usr/local/bin/osmo_runner
chmod +x /usr/local/bin/osmo_runner
echo "OSMO Runner installed successfully"
