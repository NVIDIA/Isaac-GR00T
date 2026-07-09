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

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re

from groot.infra.eval.oss_open_loop_eval_config import (
    DEFAULT_CACHE_ROOT,
    OssOpenLoopEvalConfig,
)

SnapshotDownloadFn = Callable[..., str]


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

    def model_root(self, repo_id: str, revision: str) -> Path:
        return self.root / "models" / self.safe_path_component(repo_id) / self.safe_path_component(revision)

    def dataset_root(self, repo_id: str, revision: str, local_name: str) -> Path:
        return (
            self.root
            / "datasets"
            / self.safe_path_component(repo_id)
            / self.safe_path_component(revision)
            / local_name
        )

    def download(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        local_dir: Path,
        snapshot_download_fn: SnapshotDownloadFn | None = None,
    ) -> Path:
        if snapshot_download_fn is None:
            from huggingface_hub import snapshot_download

            snapshot_download_fn = snapshot_download
        local_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            snapshot_download_fn(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(local_dir),
                token=self.hf_token() or None,
            )
        )

    def hf_token(self) -> str:
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or ""
        )

    def safe_path_component(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "--", value).strip("-") or "artifact"
