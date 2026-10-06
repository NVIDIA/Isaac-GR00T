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

"""Cross-stage invariant for the TRT export-mode fan-out (CPU-only).

``build_trt_pipeline._MODE_MAP`` routes one ``export_mode`` selector to a
``(build, verify, benchmark)`` tuple. Each stage produces/consumes a *set* of
engines, but nothing else binds those sets — a valid-but-wrong row (e.g. a
build mode that requires ONNX the export never wrote, or a verify mode that
loads engines the build never produced) is only discovered when the row is run.

These tests bind the rows statically:

* every component a build produces is sourced from the shared
  ``FULL_PIPELINE_COMPONENTS`` / ``EXPORT_MODE_COMPONENTS`` tables, and
* the verify and benchmark consumers' required engine sets are the documented
  contract from ``trt_model_forward.setup_tensorrt_engines`` (the ``_setup_*``
  loaders), which both stages share.

A future row that pairs a producer with a consumer (verify or benchmark)
needing more than it produces fails here instead of at deploy time.
"""

from __future__ import annotations

from gr00t.deployment.modes import (
    EXPORT_MODE_COMPONENTS,
    FULL_PIPELINE_COMPONENTS,
    PIPELINE_STAGE_MODES as _MODE_MAP,
    ExportMode,
)
import pytest


_ENGINE_OF = {c.name: c.engine for c in FULL_PIPELINE_COMPONENTS}
_FULL = EXPORT_MODE_COMPONENTS[ExportMode.full_pipeline]

# Engines each *setup mode* REQUIRES (non-fallback). Verify and benchmark use
# the same mode strings, both resolved by the trt_model_forward.py setup loaders
# (``_setup_dit_only`` / ``_setup_action_head`` / ``_setup_n17_full_pipeline``),
# so one table backs both stages. Engines that fall back to PyTorch when absent
# (LLM / VL-SA in n17_full_pipeline) are NOT required. Keep in sync if a
# ``_setup_*`` loader changes which engines it hard-requires.
SETUP_REQUIRED_ENGINES: dict[str, frozenset[str]] = {
    "dit_only": frozenset({"dit_bf16.engine"}),
    "action_head": frozenset(
        {
            "state_encoder.engine",
            "action_encoder.engine",
            "dit_bf16.engine",
            "action_decoder.engine",
        }
    ),
    "n17_full_pipeline": frozenset(
        {
            "vit.engine",
            "state_encoder.engine",
            "action_encoder.engine",
            "dit_bf16.engine",
            "action_decoder.engine",
        }
    ),
}


def _engines_built_for(export_mode: str, build_mode: str) -> frozenset[str]:
    """Engines the build stage produces for a ``_MODE_MAP`` row.

    Mirrors ``build_trt_pipeline._run_build``: the ``single`` builder emits just
    the DiT engine; the ``full_pipeline`` builder emits one engine per component
    that ``export_mode`` wrote ONNX for (``EXPORT_MODE_COMPONENTS``).
    """
    if build_mode == "single":
        return frozenset({_ENGINE_OF["DiT"]})
    produced = EXPORT_MODE_COMPONENTS[ExportMode(export_mode)]
    return frozenset(_ENGINE_OF[name] for name in produced)


def _required_for(mode: str) -> frozenset[str]:
    assert mode in SETUP_REQUIRED_ENGINES, (
        f"setup mode {mode!r} is routed by _MODE_MAP but has no documented "
        "required-engine set in this test."
    )
    return SETUP_REQUIRED_ENGINES[mode]


@pytest.mark.parametrize("export_mode", sorted(_MODE_MAP))
def test_build_produces_every_engine_verify_requires(export_mode):
    build_mode, verify_mode, _bench = _MODE_MAP[export_mode]
    built = _engines_built_for(export_mode, build_mode)
    missing = _required_for(verify_mode) - built
    assert not missing, (
        f"export_mode={export_mode!r}: build ({build_mode}) produces {sorted(built)}, "
        f"but verify ({verify_mode}) requires {sorted(missing)} it never builds."
    )


@pytest.mark.parametrize("export_mode", sorted(_MODE_MAP))
def test_build_produces_every_engine_benchmark_requires(export_mode):
    # Same fan-out hazard on the third map: benchmark loads engines via the same
    # setup_tensorrt_engines loaders, so a row whose benchmark mode needs an
    # engine the build never produced must also fail here, not at deploy time.
    build_mode, _verify, bench_mode = _MODE_MAP[export_mode]
    built = _engines_built_for(export_mode, build_mode)
    missing = _required_for(bench_mode) - built
    assert not missing, (
        f"export_mode={export_mode!r}: build ({build_mode}) produces {sorted(built)}, "
        f"but benchmark ({bench_mode}) requires {sorted(missing)} it never builds."
    )


@pytest.mark.parametrize("export_mode", sorted(_MODE_MAP))
def test_export_mode_component_names_are_valid(export_mode):
    valid = {c.name for c in FULL_PIPELINE_COMPONENTS}
    unknown = set(EXPORT_MODE_COMPONENTS[ExportMode(export_mode)]) - valid
    assert not unknown, f"{export_mode}: unknown component names {sorted(unknown)}"


def test_action_head_build_excludes_pytorch_only_components():
    # The regression guard for B-2026-06-29-003: action_head keeps ViT/LLM/VL-SA
    # in PyTorch, so the build must NOT require their ONNX.
    produced = EXPORT_MODE_COMPONENTS[ExportMode.action_head]
    for name in ("ViT", "LLM", "VL Self-Attention"):
        assert name not in produced
    assert produced == frozenset({"State Encoder", "Action Encoder", "DiT", "Action Decoder"})


def test_setup_modes_have_known_required_sets():
    # Every verify AND benchmark mode referenced by _MODE_MAP must have a
    # documented required set above — otherwise a new row escapes the invariant.
    for _build, verify_mode, bench_mode in _MODE_MAP.values():
        assert verify_mode in SETUP_REQUIRED_ENGINES, (
            f"verify mode {verify_mode!r} is routed by _MODE_MAP but has no "
            "documented required-engine set in this test."
        )
        assert bench_mode in SETUP_REQUIRED_ENGINES, (
            f"benchmark mode {bench_mode!r} is routed by _MODE_MAP but has no "
            "documented required-engine set in this test."
        )
