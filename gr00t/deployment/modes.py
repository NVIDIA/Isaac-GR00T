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

"""Single source of truth for the deployment CLIs' shared mode-flag value sets.

``scripts/deployment`` is not an importable package, so every value set
shared across the CLIs lives here and is imported via ``gr00t.*``. A CLI
never re-authors these strings — it imports the enum or fails on a name
that does not exist, so cross-file drift is not expressible.

Each mode flag is one enum here, imported by its CLI. The enums hold the
legitimate *subset* each tool supports (the tools really do run different
subsets — that is genuine capability, not duplication):

- :class:`ExportMode` — the two ``export_mode`` CLIs (``export_onnx_n1d7``,
  ``build_trt_pipeline``).
- :class:`VerifyMode` — ``verify_n1d7_trt`` ``--mode``.
- :class:`BenchmarkMode` — ``benchmark_inference`` ``--trt-mode``.
- :class:`BuildEngineMode` — ``build_tensorrt_engine`` ``--mode``.
- :class:`InferenceMode` — ``setup_tensorrt_engines(mode=...)``, shared by the
  sim-eval rollout ``--trt-mode`` and ``standalone_inference_script``.

Each member's value equals its name (via :func:`_generate_next_value_`), so
``tyro`` keeps the value-form CLI surface unchanged (``--mode full_pipeline``,
not ``--mode FULL_PIPELINE``) — no CLI/README/docstring edits needed when
switching from a ``Literal``.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum


class _StrEnum(str, enum.Enum):
    """``enum.StrEnum`` stand-in for Python 3.10 (dGPU/Orin).

    Members are ``str`` subclasses whose value equals their name, so ``==``,
    ``in``, dict-keying, JSON, f-strings, and ``tyro`` choices all see the bare
    value. ``__str__`` is restored to ``str``'s so ``str()``/``%s`` yield the
    value rather than ``ClassName.member``.
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

    __str__ = str.__str__


class ExportMode(_StrEnum):
    """Allowed values for the ``--export-mode`` flag, shared by the two
    ``export_mode`` CLIs."""

    dit_only = enum.auto()
    action_head = enum.auto()
    full_pipeline = enum.auto()


class VerifyMode(_StrEnum):
    """Allowed values for ``verify_n1d7_trt`` ``--mode``."""

    dit_only = enum.auto()
    action_head = enum.auto()
    n17_full_pipeline = enum.auto()
    vit_llm_only = enum.auto()


class BenchmarkMode(_StrEnum):
    """Allowed values for ``benchmark_inference`` ``--trt-mode``."""

    dit_only = enum.auto()
    n17_full_pipeline = enum.auto()
    vit_llm_only = enum.auto()


class InferenceMode(_StrEnum):
    """Engine subset ``setup_tensorrt_engines`` swaps onto a policy at inference
    time. Shared by the sim-eval rollout CLI (``--trt-mode``) and
    ``standalone_inference_script``; its members are exactly the modes
    ``setup_tensorrt_engines`` dispatches on."""

    dit_only = enum.auto()
    action_head = enum.auto()
    vit_llm_only = enum.auto()
    n17_full_pipeline = enum.auto()


class BuildEngineMode(_StrEnum):
    """Allowed values for ``build_tensorrt_engine`` ``--mode``."""

    single = enum.auto()
    full_pipeline = enum.auto()


# ---------------------------------------------------------------------------
# Pipeline component table (shared by the build + pipeline CLIs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineComponent:
    """One N1.7 pipeline component: its display name, the ONNX filename
    candidates the exporter may write for it, and the built engine filename."""

    name: str
    onnx_candidates: tuple[str, ...]
    engine: str


# One row per N1.7 pipeline component, in build order. ``build_full_pipeline``
# builds from this; ``build_trt_pipeline`` derives the per-export-mode expected
# subset from it; the deployment tests bind it against the verify-side loaders.
FULL_PIPELINE_COMPONENTS: tuple[PipelineComponent, ...] = (
    PipelineComponent("ViT", ("vit_fp32.onnx", "vit_bf16.onnx"), "vit.engine"),
    PipelineComponent("LLM", ("llm_bf16.onnx",), "llm_bf16.engine"),
    PipelineComponent("VL Self-Attention", ("vl_self_attention.onnx",), "vl_self_attention.engine"),
    PipelineComponent("State Encoder", ("state_encoder.onnx",), "state_encoder.engine"),
    PipelineComponent("Action Encoder", ("action_encoder.onnx",), "action_encoder.engine"),
    PipelineComponent("DiT", ("dit_bf16.onnx",), "dit_bf16.engine"),
    PipelineComponent("Action Decoder", ("action_decoder.onnx",), "action_decoder.engine"),
)

# Components each ``ExportMode`` writes ONNX for — must stay in sync with the
# export branches in ``export_onnx_n1d7.main`` (Step 4). The build stage must
# require *exactly* this subset: ``action_head`` keeps ViT/LLM/VL-SA in PyTorch,
# so demanding the full 7 makes its build fail with a misleading "missing ONNX".
EXPORT_MODE_COMPONENTS: dict[ExportMode, frozenset[str]] = {
    ExportMode.dit_only: frozenset({"DiT"}),
    ExportMode.action_head: frozenset({"State Encoder", "Action Encoder", "DiT", "Action Decoder"}),
    ExportMode.full_pipeline: frozenset(c.name for c in FULL_PIPELINE_COMPONENTS),
}

# Per ``export_mode``, the (build, verify, benchmark) mode each downstream stage
# runs in ``build_trt_pipeline``. One selector fans out to three sibling
# stage-modes that each consume a *different* engine set; the binding between
# this table and the component sets above is enforced by
# ``tests/scripts/deployment/test_trt_pipeline_modes.py``.
PIPELINE_STAGE_MODES: dict[ExportMode, tuple[BuildEngineMode, VerifyMode, BenchmarkMode]] = {
    ExportMode.full_pipeline: (
        BuildEngineMode.full_pipeline,
        VerifyMode.n17_full_pipeline,
        BenchmarkMode.n17_full_pipeline,
    ),
    ExportMode.action_head: (
        BuildEngineMode.full_pipeline,
        VerifyMode.action_head,
        BenchmarkMode.dit_only,
    ),
    ExportMode.dit_only: (
        BuildEngineMode.single,
        VerifyMode.dit_only,
        BenchmarkMode.dit_only,
    ),
}
