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

"""Tests for gated-backbone error detection in qwen3_backbone."""

from gr00t.model.modules.qwen3_backbone import _GATED_BACKBONE_HINT, _is_gated_repo_error
import pytest


@pytest.mark.parametrize(
    "exc",
    [
        OSError(
            "You are trying to access a gated repo. Make sure to have access to it at "
            "https://huggingface.co/nvidia/Cosmos-Reason2-2B. 401 Client Error."
        ),
        Exception("Access to model nvidia/Cosmos-Reason2-2B is restricted."),
        RuntimeError("Cannot access gated repo for url ..."),
    ],
)
def test_detects_gated_errors(exc):
    assert _is_gated_repo_error(exc) is True


def test_detects_actual_gated_repo_error_instance():
    GatedRepoError = pytest.importorskip("huggingface_hub.errors").GatedRepoError
    # Detected via isinstance even when the message carries no gated markers.
    assert _is_gated_repo_error(GatedRepoError("403")) is True


def test_detects_gated_repo_error_wrapped_as_cause():
    GatedRepoError = pytest.importorskip("huggingface_hub.errors").GatedRepoError
    try:
        try:
            raise GatedRepoError("403")
        except GatedRepoError as inner:
            raise OSError("Could not load the model") from inner
    except OSError as exc:
        # Outer message has no markers; detection must follow __cause__.
        assert _is_gated_repo_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("some unrelated configuration error"),
        FileNotFoundError("config.json not found"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_ignores_unrelated_errors(exc):
    assert _is_gated_repo_error(exc) is False


def test_hint_mentions_repo_and_auth():
    msg = _GATED_BACKBONE_HINT.format(model_name="nvidia/Cosmos-Reason2-2B")
    assert "nvidia/Cosmos-Reason2-2B" in msg
    assert "hf auth login" in msg
