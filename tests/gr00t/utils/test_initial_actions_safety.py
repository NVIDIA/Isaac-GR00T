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

"""CPU-only regression tests for the pickle-free
:func:`gr00t.utils.initial_actions.save_initial_actions` /
:func:`gr00t.utils.initial_actions.load_initial_actions` round-trip:
structured nested dicts must go through the flat keyed-npz layout
(``allow_pickle=False``-compatible), and any file written by the
legacy pickle-based path must be rejected with a migration error."""

from __future__ import annotations

from gr00t.utils.initial_actions import (
    INITIAL_ACTIONS_FILENAME,
    load_initial_actions,
    save_initial_actions,
)
import numpy as np
import pytest


def _sample_payload() -> list[dict[str, dict[str, np.ndarray]]]:
    """Return a representative initial-actions structure: 2 datasets, several
    trajectories each, multiple action keys per trajectory."""
    return [
        {
            "traj_0": {
                "left_hand": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                "right_hand": np.array([[0.4, 0.5, 0.6]], dtype=np.float32),
            },
            "traj_1": {
                "left_hand": np.array([[1.0, 1.1, 1.2]], dtype=np.float32),
            },
        },
        {
            "traj_a": {
                "base": np.array([[7.0, 8.0]], dtype=np.float64),
            },
        },
    ]


def test_roundtrip_preserves_structure_and_values(tmp_path):
    payload = _sample_payload()
    path = tmp_path / INITIAL_ACTIONS_FILENAME

    save_initial_actions(payload, path)
    loaded = load_initial_actions(path)

    assert len(loaded) == len(payload)
    for original_ds, loaded_ds in zip(payload, loaded):
        assert set(loaded_ds.keys()) == set(original_ds.keys())
        for traj_name, action_dict in original_ds.items():
            assert set(loaded_ds[traj_name].keys()) == set(action_dict.keys())
            for action_key, expected in action_dict.items():
                np.testing.assert_array_equal(loaded_ds[traj_name][action_key], expected)
                assert loaded_ds[traj_name][action_key].dtype == expected.dtype


def test_roundtrip_handles_empty_dataset_list(tmp_path):
    """Empty input must round-trip to an empty list — no IndexError, no
    silent ``None``."""
    path = tmp_path / INITIAL_ACTIONS_FILENAME
    save_initial_actions([], path)
    assert load_initial_actions(path) == []


def test_roundtrip_handles_dataset_with_no_trajectories(tmp_path):
    """A dataset with zero trajectories must still survive the roundtrip
    so the per-dataset ordering is preserved across calls."""
    payload = [{}, {"traj_only_in_ds1": {"k": np.array([1.0])}}]
    path = tmp_path / INITIAL_ACTIONS_FILENAME
    save_initial_actions(payload, path)
    loaded = load_initial_actions(path)
    assert len(loaded) == 2
    assert loaded[0] == {}
    assert "traj_only_in_ds1" in loaded[1]


# ---------------------------------------------------------------------------
# Security regressions (pre-fix bug surface)
# ---------------------------------------------------------------------------


def test_save_rejects_separator_in_trajectory_name(tmp_path):
    """``::`` is a structural separator — a trajectory name containing
    it would silently mis-group on decode, so refuse at save time."""
    payload = [{"traj::with::sep": {"key": np.array([1.0])}}]
    path = tmp_path / INITIAL_ACTIONS_FILENAME
    with pytest.raises(ValueError, match="must not contain '::'"):
        save_initial_actions(payload, path)


def test_save_rejects_separator_in_action_key(tmp_path):
    payload = [{"traj": {"action::key": np.array([1.0])}}]
    path = tmp_path / INITIAL_ACTIONS_FILENAME
    with pytest.raises(ValueError, match="must not contain '::'"):
        save_initial_actions(payload, path)


def test_load_rejects_legacy_pickle_format(tmp_path):
    """Legacy files written via ``np.savez(path, list_of_dicts)`` forced
    numpy to pickle the list into ``arr_0``. Loading them today must
    raise a clear migration error — silently re-enabling
    ``allow_pickle=True`` is the exact RCE path."""
    legacy_path = tmp_path / INITIAL_ACTIONS_FILENAME
    legacy_payload = _sample_payload()
    # Mirror the pre-fix code: single positional arg → numpy pickles into arr_0.
    np.savez(str(legacy_path), legacy_payload)

    with pytest.raises(ValueError) as excinfo:
        load_initial_actions(legacy_path)

    msg = str(excinfo.value)
    # numpy may surface either the pickle-rejection branch or the missing-
    # __schema__ branch depending on how it encoded the legacy payload;
    # accept either since both are valid migration errors.
    assert "save_initial_actions" in msg or "pickle" in msg.lower()


def test_load_rejects_npz_missing_schema_marker(tmp_path):
    """An npz without the ``__schema__`` marker is either a legacy
    pickle file (after numpy fell back) or an unrelated file sharing
    the extension; either way it must not silently load."""
    path = tmp_path / INITIAL_ACTIONS_FILENAME
    np.savez(str(path), foo=np.array([1, 2, 3]), bar=np.array([4.0, 5.0]))

    with pytest.raises(ValueError, match="__schema__"):
        load_initial_actions(path)


def test_load_rejects_unrecognised_format_marker(tmp_path):
    """A file with ``__schema__`` but mismatched ``format`` must also
    fail closed — covers the case where another component writes a
    JSON schema array under the same key."""
    import json

    path = tmp_path / INITIAL_ACTIONS_FILENAME
    schema = {"format": "some.other.tool", "version": 1}
    np.savez(
        str(path),
        __schema__=np.frombuffer(json.dumps(schema).encode("utf-8"), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="unrecognised schema"):
        load_initial_actions(path)
