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

"""CPU-only tests for the export/TRT single-source contract helpers.

``_trt_contract`` has no heavy deps (json / os / logging only), so we import
it directly after putting ``scripts/deployment`` on ``sys.path``.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest


DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/deployment"))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

import _trt_contract as tc  # noqa: E402


def _write_metadata(d, **kwargs):
    meta = {"action_horizon": 16, "sa_seq_len": 17, "batch_size": 1}
    meta.update(kwargs)
    with open(os.path.join(d, "export_metadata.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _fake_policy(action_horizon):
    cfg = types.SimpleNamespace(action_horizon=action_horizon)
    action_head = types.SimpleNamespace(config=cfg, action_horizon=action_horizon)
    model = types.SimpleNamespace(action_head=action_head)
    return types.SimpleNamespace(model=model)


# --- load_export_metadata -------------------------------------------------


def test_load_metadata_from_engine_dir(tmp_path):
    _write_metadata(tmp_path)
    meta = tc.load_export_metadata(str(tmp_path))
    assert meta["action_horizon"] == 16


def test_load_metadata_from_engine_file(tmp_path):
    _write_metadata(tmp_path)
    meta = tc.load_export_metadata(str(tmp_path / "dit_bf16.engine"))
    assert meta["batch_size"] == 1


def test_load_metadata_from_sibling_onnx_dir(tmp_path):
    onnx = tmp_path / "onnx"
    engines = tmp_path / "engines"
    onnx.mkdir()
    engines.mkdir()
    _write_metadata(onnx)
    meta = tc.load_export_metadata(str(engines))
    assert meta["action_horizon"] == 16


def test_load_metadata_absent_returns_none(tmp_path):
    assert tc.load_export_metadata(str(tmp_path)) is None


# --- assert_engine_matches_policy -----------------------------------------


def test_engine_matches_policy_ok(tmp_path):
    _write_metadata(tmp_path, action_horizon=16, sa_seq_len=17)
    out = tc.assert_engine_matches_policy(_fake_policy(16), str(tmp_path))
    assert out["action_horizon"] == 16


def test_engine_action_horizon_mismatch_raises(tmp_path):
    _write_metadata(tmp_path, action_horizon=16, sa_seq_len=17)
    with pytest.raises(ValueError, match="disagree on chunk size"):
        tc.assert_engine_matches_policy(_fake_policy(40), str(tmp_path))


def test_engine_corrupt_sa_seq_len_raises(tmp_path):
    _write_metadata(tmp_path, action_horizon=16, sa_seq_len=99)
    with pytest.raises(ValueError, match="corrupt"):
        tc.assert_engine_matches_policy(_fake_policy(16), str(tmp_path))


def test_engine_missing_metadata_warns_returns_none(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        out = tc.assert_engine_matches_policy(_fake_policy(16), str(tmp_path))
    assert out is None
    assert any("no export_metadata.json" in r.getMessage() for r in caplog.records)


def test_corrupt_metadata_treated_as_absent(tmp_path):
    (tmp_path / "export_metadata.json").write_text("{ not valid json ")
    # Corrupt file must not crash; load returns None, validation degrades.
    assert tc.load_export_metadata(str(tmp_path)) is None
    assert tc.assert_engine_matches_policy(_fake_policy(16), str(tmp_path)) is None


def test_action_horizon_mismatch_message_without_sa_seq_len(tmp_path):
    # Metadata has action_horizon but no sa_seq_len: the error must not print
    # the "sa_seq_len=None" placeholder.
    with open(tmp_path / "export_metadata.json", "w") as f:
        json.dump({"action_horizon": 16, "batch_size": 1}, f)
    with pytest.raises(ValueError) as exc:
        tc.assert_engine_matches_policy(_fake_policy(40), str(tmp_path))
    assert "sa_seq_len=None" not in str(exc.value)


# --- resolve_batch_size ----------------------------------------------------


def test_resolve_batch_size_default_from_metadata(tmp_path):
    _write_metadata(tmp_path, batch_size=4)
    assert tc.resolve_batch_size(str(tmp_path)) == 4


def test_resolve_batch_size_matching_request(tmp_path):
    _write_metadata(tmp_path, batch_size=2)
    assert tc.resolve_batch_size(str(tmp_path), 2) == 2


def test_resolve_batch_size_mismatch_raises(tmp_path):
    _write_metadata(tmp_path, batch_size=1)
    with pytest.raises(ValueError, match="built .*for batch_size=1"):
        tc.resolve_batch_size(str(tmp_path), 4)


def test_resolve_batch_size_no_metadata_defaults_to_one(tmp_path):
    assert tc.resolve_batch_size(str(tmp_path)) == 1
    # A request with no metadata is accepted (nothing to validate against).
    assert tc.resolve_batch_size(str(tmp_path), 8) == 8


# --- assert_exec_horizon_within_model -------------------------------------


@pytest.mark.parametrize(
    "exec_h,model_h,ok",
    [(16, 16, True), (8, 16, True), (1, 40, True), (17, 16, False), (0, 16, False)],
)
def test_assert_exec_horizon_within_model(exec_h, model_h, ok):
    if ok:
        tc.assert_exec_horizon_within_model(exec_horizon=exec_h, model_action_horizon=model_h)
    else:
        with pytest.raises(ValueError, match="action-horizon"):
            tc.assert_exec_horizon_within_model(exec_horizon=exec_h, model_action_horizon=model_h)
