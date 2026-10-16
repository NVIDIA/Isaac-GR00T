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

"""CPU-only tests for the DiT SDPA backend-selection context.

``_sdpa_context`` pins scaled-dot-product attention to the math backend on DGX
Spark (sm121), where the mem-efficient kernel dispatch is unreliable. It used to
do that via ``torch.backends.cuda.sdp_kernel(...)``, which carries a
``FutureWarning`` and is documented as slated for removal; it now uses
``torch.nn.attention.sdpa_kernel([SDPBackend.MATH])``.

The two are equivalent by construction -- the deprecated helper is a shim that
builds its backend list from exactly these flags -- so these tests lock in the
*observable* contract rather than re-deriving it:

1. the four backend flags inside the context (the legacy call produced exactly
   ``flash=False, mem_efficient=False, math=True, cudnn=False``),
2. no deprecation warning escapes, which is the regression this guards,
3. prior flag state is restored on exit, and
4. the platform/env gating that decides whether to pin at all is unchanged.

No GPU and no checkpoint download are required: backend selection is global
state that PyTorch tracks on CPU-only builds too.
"""

import warnings

from gr00t.model.modules.dit import _is_spark_sm121, _sdpa_context, _should_force_math_sdpa
import pytest
import torch


# Flag state the legacy ``sdp_kernel(enable_flash=False, enable_math=True,
# enable_mem_efficient=False, enable_cudnn=False)`` call produced.
EXPECTED_MATH_ONLY = {"flash": False, "mem_efficient": False, "math": True, "cudnn": False}


def _sdpa_flags() -> dict[str, bool]:
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def test_forced_math_selects_math_backend_only(monkeypatch):
    """Inside the pinned context, math is the only enabled SDPA backend."""
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")

    with _sdpa_context():
        assert _sdpa_flags() == EXPECTED_MATH_ONLY


def test_forced_math_emits_no_deprecation_warning(monkeypatch):
    """The pinned path must not raise FutureWarning/DeprecationWarning.

    This is the regression guard: ``torch.backends.cuda.sdp_kernel`` warns and is
    slated for removal, which would break the Spark path outright.
    """
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        with _sdpa_context():
            pass


def test_forced_math_restores_previous_flags(monkeypatch):
    """Exiting the context leaves global backend selection as it was."""
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")

    before = _sdpa_flags()
    with _sdpa_context():
        pass
    assert _sdpa_flags() == before


def test_default_mode_does_not_touch_backend_selection(monkeypatch):
    """Off the Spark path the context is inert -- no global flags are changed."""
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "default")

    before = _sdpa_flags()
    with _sdpa_context():
        assert _sdpa_flags() == before
    assert _sdpa_flags() == before


@pytest.mark.parametrize(
    "override,is_spark,expected",
    [
        ("math", False, True),  # explicit opt-in wins on any device
        ("default", True, False),  # explicit opt-out wins even on Spark
        (None, True, True),  # unset -> pin on Spark
        (None, False, False),  # unset -> inert elsewhere
    ],
)
def test_gating_precedence(monkeypatch, override, is_spark, expected):
    """Env override takes precedence over the sm121 capability probe."""
    monkeypatch.delenv("GR00T_DIT_SDPA_MODE", raising=False)
    if override is not None:
        monkeypatch.setenv("GR00T_DIT_SDPA_MODE", override)
    monkeypatch.setattr("gr00t.model.modules.dit._is_spark_sm121", lambda: is_spark)

    assert _should_force_math_sdpa() is expected


def test_capability_probe_is_false_without_cuda():
    """The probe must not raise on CPU-only builds."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; CPU-only branch not exercised")
    assert _is_spark_sm121() is False
