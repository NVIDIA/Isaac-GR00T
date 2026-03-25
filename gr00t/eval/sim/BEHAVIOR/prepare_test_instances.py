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

from pathlib import Path

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    dir_path = Path(__file__).parent

    test_instances_path = dir_path / "test_instances"
    test_instances_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="behavior-1k/2025-challenge-hidden-instances",
        repo_type="dataset",
        local_dir=test_instances_path,
    )
