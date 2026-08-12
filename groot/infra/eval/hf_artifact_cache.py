# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
