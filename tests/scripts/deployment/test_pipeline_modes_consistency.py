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

"""Pin the structural mode-flag SOT in :mod:`gr00t.deployment.modes`.

Every deployment CLI mode field must *be* its SOT enum (not a re-inlined
``Literal`` or ad-hoc enum). With each CLI importing its enum, cross-file drift
is no longer expressible; this test guards against a future regression that
re-inlines the choices.
"""

from __future__ import annotations

import os
import sys
from typing import get_type_hints

from gr00t.deployment.modes import BenchmarkMode, BuildEngineMode, ExportMode, VerifyMode
import pytest


@pytest.fixture(scope="module")
def deploy_imports():
    """Make ``scripts/deployment`` importable; the directory is not a
    package and relies on runtime ``sys.path`` insertion."""
    deploy_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../scripts/deployment")
    )
    if deploy_dir not in sys.path:
        sys.path.insert(0, deploy_dir)

    return deploy_dir


# ---------------------------------------------------------------------------
# Each CLI field must *be* its SOT enum (no re-inlined Literal / ad-hoc enum)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name, cls_name, field_name, mode_enum",
    [
        ("export_onnx_n1d7", "ExportConfig", "export_mode", ExportMode),
        ("build_trt_pipeline", "PipelineConfig", "export_mode", ExportMode),
        ("verify_n1d7_trt", "VerifyConfig", "mode", VerifyMode),
        ("benchmark_inference", "BenchmarkConfig", "trt_mode", BenchmarkMode),
        ("build_tensorrt_engine", "BuildConfig", "mode", BuildEngineMode),
    ],
)
def test_cli_mode_field_is_sot_enum(deploy_imports, module_name, cls_name, field_name, mode_enum):
    """A CLI whose mode field is not its SOT enum has reverted to an ad-hoc
    ``Literal``/enum — re-import the enum instead."""
    try:
        mod = __import__(module_name)
    except (ImportError, OSError) as e:  # torch/tensorrt or native CUDA libs absent on CPU CI
        pytest.skip(f"{module_name} not importable in this env: {e}")
    cfg_cls = getattr(mod, cls_name, None)
    if cfg_cls is None:
        pytest.skip(f"{module_name} has no attribute {cls_name!r}")

    resolved = get_type_hints(cfg_cls)[field_name]
    assert resolved is mode_enum, (
        f"{module_name}.{cls_name}.{field_name} is annotated {resolved!r}, not the SOT enum "
        f"{mode_enum.__name__}. Import the enum from gr00t.deployment.modes instead of "
        "re-declaring a Literal or ad-hoc enum."
    )
