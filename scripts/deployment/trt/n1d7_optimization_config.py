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

"""Shared precision policy and artifact schema for the optional N1.7 TRT
quantization and execution optimizations.

This module is the single source of truth for how a ``--quantization`` policy
maps to per-component precision tags. Export (artifact naming, torch dtype,
calibration recipe selection), engine build (TRT boundary precision), and the
runtime (engine file resolution) all derive from :data:`COMPONENT_PRECISIONS`
so the three stages can never disagree on artifact names.
"""

from dataclasses import dataclass
import os
from typing import Any, Literal, Optional

import torch


ExecutionProfile = Literal["baseline", "optimized"]
QuantizationPolicy = Literal["none", "fp8", "mixed_nvfp4"]

ACTION_HEAD_CONSTANTS_FILENAME = "action_head_constants.pt"
ACTION_HEAD_FORMAT_VERSION = 1

# Per-component precision tags for each quantization policy. A tag names the
# quantization *recipe* of a component, not a single dtype: "nvfp4" components
# keep their accuracy-sensitive layers in FP8 (LLM o_proj/down_proj, DiT
# attention) — calibration.py holds the exact per-layer breakdown. "none"
# reproduces the baseline export exactly (FP32 ViT for TRT accuracy, BF16
# everywhere else).
COMPONENT_PRECISIONS: dict[str, dict[str, str]] = {
    "none": {"vit": "fp32", "llm": "bf16", "dit": "bf16", "aux": "bf16"},
    "fp8": {"vit": "fp8", "llm": "fp8", "dit": "fp8", "aux": "fp16"},
    "mixed_nvfp4": {"vit": "fp8", "llm": "nvfp4", "dit": "nvfp4", "aux": "fp16"},
}

QUANTIZED_TAGS = frozenset({"fp8", "nvfp4"})

# FP8/NVFP4 are Q/DQ recipes on top of an FP16 torch graph, so their tags
# resolve to float16 for export.
_PRECISION_TO_DTYPE: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8": torch.float16,
    "nvfp4": torch.float16,
}


def component_precisions(quantization: QuantizationPolicy) -> dict[str, str]:
    """Map a quantization policy to its per-component precision tags."""
    return dict(COMPONENT_PRECISIONS[quantization])


def resolve_dtype(precision: str) -> torch.dtype:
    """Torch dtype used to build/export a component with this precision tag."""
    if precision not in _PRECISION_TO_DTYPE:
        raise ValueError(
            f"Unsupported precision '{precision}', choose from {list(_PRECISION_TO_DTYPE)}"
        )
    return _PRECISION_TO_DTYPE[precision]


def trt_boundary_precision(precision: str) -> str:
    """TRT network I/O precision for an artifact precision tag.

    FP8/NVFP4 live in internal Q/DQ nodes; the network boundary stays FP16.
    """
    return "fp16" if precision in QUANTIZED_TAGS else precision


def format_quantization_summary(metadata: Optional[dict[str, Any]]) -> str:
    """One user-facing quantization summary shared by export, build, and runtime."""
    if not metadata or "quantization" not in metadata:
        return "Quantization: none (no quantization metadata)"
    quantization = metadata["quantization"]
    lines = [f"Quantization: {quantization}"]
    if quantization != "none":
        lines.append(
            "  vit={vit}, llm={llm}, dit={dit}, aux={aux}".format(
                vit=metadata.get("vit_precision", "?"),
                llm=metadata.get("llm_precision", "?"),
                dit=metadata.get("dit_precision", "?"),
                aux=metadata.get("aux_precision", "?"),
            )
        )
        calibration = metadata.get("calibration")
        if calibration:
            lines.append(
                "  Calibration: dataset={dataset}, embodiment={embodiment}, "
                "samples={samples}".format(
                    dataset=calibration.get("dataset"),
                    embodiment=calibration.get("embodiment"),
                    samples=calibration.get("samples"),
                )
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class OptimizationArtifacts:
    """Resolved optimized-action-head artifact paths for one deployment directory."""

    cross_kv: str
    constants: str
    format_version: int

    @classmethod
    def from_metadata(
        cls,
        root: str,
        metadata: Optional[dict[str, Any]],
        *,
        artifact_kind: Literal["onnx", "engine"],
    ) -> Optional["OptimizationArtifacts"]:
        if not metadata or not metadata.get("optimized_action_head", False):
            return None
        dit_precision = metadata.get("dit_precision", "bf16")
        return cls(
            cross_kv=os.path.join(root, f"dit_cross_kv_{dit_precision}.{artifact_kind}"),
            constants=os.path.join(root, ACTION_HEAD_CONSTANTS_FILENAME),
            format_version=int(metadata.get("action_head_format_version") or 0),
        )
