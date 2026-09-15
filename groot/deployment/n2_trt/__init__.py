# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Checkpoint-driven, fail-closed N2 TensorRT deployment contracts."""

from groot.deployment.n2_trt.artifacts import resolve_checkpoint_vae_weights
from groot.deployment.n2_trt.contract import (
    N2ModelVariant,
    N2TensorRTContract,
    TransformerPrecision,
    resolve_checkpoint_vae_type,
)
from groot.deployment.n2_trt.manifest import (
    ArtifactMember,
    ArtifactRecord,
    BuildSettings,
    EngineManifest,
    RuntimeIdentity,
    SourceRevision,
    TensorSpec,
)
from groot.deployment.n2_trt.options import N2AccelerationOptions

__all__ = [
    "ArtifactMember",
    "ArtifactRecord",
    "BuildSettings",
    "EngineManifest",
    "N2AccelerationOptions",
    "N2ModelVariant",
    "N2TensorRTContract",
    "RuntimeIdentity",
    "SourceRevision",
    "TensorSpec",
    "TransformerPrecision",
    "resolve_checkpoint_vae_type",
    "resolve_checkpoint_vae_weights",
]
