#!/bin/bash
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
