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

"""Hugging Face artifact cache layout."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from groot.infra.eval.oss_open_loop_eval_config import (
    DEFAULT_CACHE_ROOT,
    OssOpenLoopEvalConfig,
)


@dataclass(frozen=True)
class HfArtifactCache:
    root: Path

    @classmethod
    def init_from_config(cls, config: OssOpenLoopEvalConfig) -> HfArtifactCache:
        root = Path(
            config.cache_root
            or os.environ.get("GROOT_HF_ARTIFACT_CACHE_ROOT")
            or DEFAULT_CACHE_ROOT
        )
        return cls(root=root)

    def download(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
    ) -> Path:
        from huggingface_hub import snapshot_download

        from groot.core.utils.storage.msc import hf_token

        self.root.mkdir(parents=True, exist_ok=True)
        return Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                cache_dir=str(self.root),
                token=hf_token() or None,
            )
        )
