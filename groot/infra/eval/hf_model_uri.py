# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
