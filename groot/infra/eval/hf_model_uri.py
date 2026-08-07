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

"""Hugging Face model checkpoint URI parsing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HfModelUri:
    repo_id: str
    revision: str
    checkpoint_name: str

    @classmethod
    def init_from_uri(cls, uri: str) -> HfModelUri:
        # Keep the public OSS CLI importable with base dependencies only.
        from groot.core.utils.storage.msc import parse_hf_model_uri

        repo_id, revision, checkpoint_name = parse_hf_model_uri(uri)
        if not checkpoint_name:
            raise ValueError(f"model URI must include a checkpoint path: {uri}")
        return cls(repo_id=repo_id, revision=revision, checkpoint_name=checkpoint_name)
