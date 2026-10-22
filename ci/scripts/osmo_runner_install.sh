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

OSMO_CLI_VERSION="6.3.1"
OSMO_CLI_SHA256="21d9622634ee3f854c90682ca93f618e624603c22be646338199a0a01f94dd6f"

echo "Installing OSMO CLI and Runner..."

# Check if osmo is already installed
if command -v osmo &> /dev/null; then
  echo "OSMO CLI already installed ($(osmo version 2> /dev/null || echo 'version unknown'))"
else
  echo "Installing OSMO CLI..."
  curl -fsSL \
    "https://github.com/NVIDIA/OSMO/releases/download/${OSMO_CLI_VERSION}/osmo-client-installer-${OSMO_CLI_VERSION}-linux-x86_64.sh" \
    -o /tmp/osmo-installer.sh
  echo "${OSMO_CLI_SHA256}  /tmp/osmo-installer.sh" | sha256sum -c -
  chmod +x /tmp/osmo-installer.sh
  (cd /tmp && ./osmo-installer.sh)
  rm /tmp/osmo-installer.sh
  osmo version
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
