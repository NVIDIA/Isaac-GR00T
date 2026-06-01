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

"""Local pytest hooks for GPU policy tests."""

from __future__ import annotations

import pytest


_TEST_FILE = "test_gr00t_policy_gpu.py"
_WARMUP_NODEID_SUFFIX = f"{_TEST_FILE}::test_warmup_model_load"
_INFERENCE_NODEID_FRAGMENT = f"{_TEST_FILE}::TestGr00tPolicyGPU::"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Run ``test_warmup_model_load`` before any inference test that shares its fixture.

    The module-scoped ``policy`` fixture is initialized on first request, so a
    cold checkpoint read falls under whatever test triggers the fixture first.
    Hoisting the warmup test guarantees the load happens under its 900s budget
    even when collection order is changed by plugins like ``pytest-randomly``.
    """
    warmup_idx = next(
        (i for i, it in enumerate(items) if it.nodeid.endswith(_WARMUP_NODEID_SUFFIX)),
        None,
    )
    if warmup_idx is None:
        return

    first_inference_idx = next(
        (i for i, it in enumerate(items) if _INFERENCE_NODEID_FRAGMENT in it.nodeid),
        None,
    )
    if first_inference_idx is not None and warmup_idx > first_inference_idx:
        items.insert(first_inference_idx, items.pop(warmup_idx))
