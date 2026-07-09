#!/usr/bin/env bash
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
