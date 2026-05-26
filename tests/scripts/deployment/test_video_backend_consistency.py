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

"""Pin that the ``video_backend`` choices stay consistent across the
three deployment CLIs, the runtime dispatch in
:mod:`gr00t.utils.video_utils`, and the canonical superset in
:mod:`gr00t.deployment.modes`. Re-introducing an ad-hoc ``Literal`` in
any CLI or growing a runtime backend outside the canonical tuple fails
CI here.
"""

from __future__ import annotations

import os
import sys
from typing import get_args, get_type_hints

from gr00t.deployment.modes import VIDEO_BACKEND_CANONICAL, VideoBackend
import pytest


@pytest.fixture(scope="module")
def deploy_imports():
    """Make ``scripts/deployment`` importable. The directory is not a
    package; it relies on ``sys.path`` insertion at runtime, so we mirror
    that here so the CLI configs can be imported."""
    deploy_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../scripts/deployment")
    )
    if deploy_dir not in sys.path:
        sys.path.insert(0, deploy_dir)
    return deploy_dir


def _literal_values_of_dataclass_field(cfg_cls, field_name: str) -> set[str]:
    """Resolve a dataclass field's ``Literal`` annotation (or
    ``VideoBackend`` alias) to its allowed value set, using
    ``get_type_hints`` so ``from __future__ import annotations`` files
    work correctly."""
    hints = get_type_hints(cfg_cls)
    return set(get_args(hints[field_name]))


# ---------------------------------------------------------------------------
# SOT shape contract
# ---------------------------------------------------------------------------


def test_canonical_set_is_nonempty():
    """Guard against an empty canonical tuple silently passing every
    subset check below."""
    assert len(VIDEO_BACKEND_CANONICAL) >= 1


def test_video_backend_is_subset_of_canonical():
    """The CLI surface must not expose a backend the runtime cannot
    dispatch to."""
    cli = set(get_args(VideoBackend))
    canonical = set(VIDEO_BACKEND_CANONICAL)
    missing = cli - canonical
    assert not missing, (
        f"VideoBackend contains {missing} not present in "
        "VIDEO_BACKEND_CANONICAL. Update gr00t/deployment/modes.py to "
        "register the new backend."
    )


# ---------------------------------------------------------------------------
# CLI sites: each must resolve to the same value set as the SOT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name, cls_name",
    [
        ("export_onnx_n1d7", "ExportConfig"),
        ("build_trt_pipeline", "PipelineConfig"),
        ("standalone_inference_script", "ArgsConfig"),
    ],
)
def test_cli_video_backend_field_matches_sot(deploy_imports, module_name, cls_name):
    """A CLI whose ``video_backend`` field drifts from
    :data:`VideoBackend` has reverted to an ad-hoc ``Literal``."""
    try:
        mod = __import__(module_name)
    except Exception as e:
        pytest.skip(f"{module_name} not importable in this env: {e}")
    cfg_cls = getattr(mod, cls_name, None)
    if cfg_cls is None:
        pytest.skip(f"{module_name} has no attribute {cls_name!r}")

    cli_values = _literal_values_of_dataclass_field(cfg_cls, "video_backend")
    sot_values = set(get_args(VideoBackend))
    assert cli_values == sot_values, (
        f"{module_name}.{cls_name}.video_backend drifted from the "
        f"VideoBackend SOT: cli={sorted(cli_values)} vs "
        f"sot={sorted(sot_values)}. Re-import the alias instead of "
        "redeclaring a Literal."
    )


# ---------------------------------------------------------------------------
# Runtime dispatch: each per-function allow-list is a subset of canonical
# ---------------------------------------------------------------------------


def test_runtime_dispatch_per_function_allowlists_are_subsets_of_canonical():
    """No runtime dispatch chain may grow a backend the canonical
    tuple does not list."""
    try:
        from gr00t.utils import video_utils
    except Exception as e:
        pytest.skip(f"video_utils not importable in this env: {e}")

    canonical = set(VIDEO_BACKEND_CANONICAL)
    for attr_name in (
        "_GET_FRAMES_BY_INDICES_BACKENDS",
        "_GET_FRAMES_BY_TIMESTAMPS_BACKENDS",
        "_GET_ALL_FRAMES_BACKENDS",
    ):
        per_func = set(getattr(video_utils, attr_name))
        missing = per_func - canonical
        assert not missing, (
            f"{attr_name} contains {sorted(missing)} not present in "
            "VIDEO_BACKEND_CANONICAL. Update gr00t/deployment/modes.py "
            "to register the new backend."
        )
