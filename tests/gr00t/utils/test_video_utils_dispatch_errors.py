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

"""Pin that the catch-all branch of every backend-dispatch chain in
:mod:`gr00t.utils.video_utils` raises a ``ValueError`` naming the bad
backend and the function's allowed set, so callers do not have to read
the source to diagnose an unknown ``video_backend``.
"""

from __future__ import annotations

import pytest


def _import_video_utils():
    """``video_utils`` imports ``av`` / ``cv2`` / ``torchvision``; skip
    cleanly when any of them is unavailable in a CPU-only venv."""
    try:
        from gr00t.utils import video_utils
    except Exception as e:
        pytest.skip(f"video_utils not importable in this env: {e}")
    return video_utils


# ---------------------------------------------------------------------------
# Per-function allow-lists exposed for the consistency test
# ---------------------------------------------------------------------------


def test_per_function_allowlists_match_implemented_branches():
    """Catch drift between each ``_*_BACKENDS`` tuple and the
    ``if/elif`` chain it documents — either side moving silently
    mis-names the supported set in the catch-all error."""
    mod = _import_video_utils()
    expected = {
        "_GET_FRAMES_BY_INDICES_BACKENDS": {"torchcodec", "decord", "ffmpeg", "opencv"},
        "_GET_FRAMES_BY_TIMESTAMPS_BACKENDS": {
            "torchcodec",
            "decord",
            "ffmpeg",
            "opencv",
            "torchvision_av",
        },
        "_GET_ALL_FRAMES_BACKENDS": {"torchcodec", "decord", "ffmpeg", "pyav"},
    }
    for attr, expected_set in expected.items():
        actual = set(getattr(mod, attr))
        assert actual == expected_set, (
            f"{attr} drifted from the implementation: actual={sorted(actual)} "
            f"vs expected={sorted(expected_set)}. Update either the tuple or "
            "the dispatch chain so they stay in sync."
        )


# ---------------------------------------------------------------------------
# _unsupported_backend_error helper formatting
# ---------------------------------------------------------------------------


def test_unsupported_backend_error_names_function_input_and_allowed_set():
    """The catch-all error must name function, bad input, per-function
    allowed set, and canonical superset."""
    mod = _import_video_utils()
    err = mod._unsupported_backend_error(
        "get_frames_by_indices", "median", ("torchcodec", "decord")
    )
    msg = str(err)
    assert isinstance(err, ValueError)
    assert "get_frames_by_indices" in msg
    assert "'median'" in msg
    assert "torchcodec" in msg and "decord" in msg
    assert "canonical" in msg.lower(), "error must reference the canonical superset"


# ---------------------------------------------------------------------------
# Dispatch chains fail-fast on unknown backend
# ---------------------------------------------------------------------------


def test_resolve_backend_universe_matches_canonical():
    """``_is_backend_available`` gates every dispatch via
    :func:`resolve_backend`. Its known-backend universe must match the
    canonical tuple — otherwise a backend listed in
    ``VIDEO_BACKEND_CANONICAL`` would be rejected at the pre-screen with
    a misleading "torchcodec is the only supported backend" error."""
    from gr00t.deployment.modes import VIDEO_BACKEND_CANONICAL

    mod = _import_video_utils()
    # Backends that need a lazy import — known to be ``True`` in CI but
    # may legitimately be absent in a CPU-only test venv.
    optional = {"torchcodec", "decord"}
    for backend in VIDEO_BACKEND_CANONICAL:
        if backend in optional:
            # Just exercise the branch without asserting True; the
            # return value depends on whether the package is installed.
            mod._is_backend_available(backend)
        else:
            assert mod._is_backend_available(backend), (
                f"_is_backend_available({backend!r}) must return True for "
                "every name listed in VIDEO_BACKEND_CANONICAL — drift here "
                "means the dispatch's pre-screen does not know about a "
                "backend the SOT lists."
            )

    assert mod._is_backend_available("definitely-not-a-backend") is False, (
        "_is_backend_available must reject names outside the canonical "
        "universe; otherwise the catch-all `else: raise ValueError(...)` in "
        "each dispatch chain is reachable for arbitrary user input."
    )


@pytest.fixture
def passthrough_resolve_backend(monkeypatch):
    """Skip ``resolve_backend``'s own pre-screen so the per-function
    dispatch chain's catch-all branch is reachable from a test."""
    mod = _import_video_utils()
    monkeypatch.setattr(mod, "resolve_backend", lambda _path, backend: backend)
    return mod


def _parse_allowed_set(msg: str) -> str:
    """Return just the per-function ``expected one of [...]`` substring,
    excluding the canonical superset that the error also embeds."""
    after = msg.split("expected one of")[1]
    return after.split("]")[0]


def test_get_frames_by_indices_catch_all_raises_value_error(passthrough_resolve_backend):
    """``get_frames_by_indices``'s catch-all names function + bad
    backend + allowed set, and does not falsely list ``pyav``."""
    mod = passthrough_resolve_backend
    with pytest.raises(ValueError) as excinfo:
        mod.get_frames_by_indices("/dev/null", [0], video_backend="foobar")
    msg = str(excinfo.value)
    assert "get_frames_by_indices" in msg
    assert "'foobar'" in msg
    assert "torchcodec" in msg
    allowed = _parse_allowed_set(msg)
    assert "pyav" not in allowed, (
        f"get_frames_by_indices must NOT list pyav in its allowed set "
        f"(its dispatch chain does not implement pyav); got: {allowed}"
    )


def test_get_frames_by_timestamps_catch_all_raises_value_error(passthrough_resolve_backend):
    """``get_frames_by_timestamps``'s catch-all names function + bad
    backend, and lists ``torchvision_av`` (its allowed set must
    include it)."""
    mod = passthrough_resolve_backend
    with pytest.raises(ValueError) as excinfo:
        mod.get_frames_by_timestamps("/dev/null", [0.0], video_backend="foobar")
    msg = str(excinfo.value)
    assert "get_frames_by_timestamps" in msg
    assert "'foobar'" in msg
    assert "torchvision_av" in msg, (
        "get_frames_by_timestamps's allowed set must include torchvision_av"
    )


def test_get_all_frames_catch_all_raises_value_error(passthrough_resolve_backend):
    """``get_all_frames``'s catch-all names function + bad backend,
    lists ``pyav``, and does not falsely list ``opencv``."""
    mod = passthrough_resolve_backend
    with pytest.raises(ValueError) as excinfo:
        mod.get_all_frames("/dev/null", video_backend="foobar")
    msg = str(excinfo.value)
    assert "get_all_frames" in msg
    assert "'foobar'" in msg
    assert "pyav" in msg, "get_all_frames's allowed set must include pyav"
    allowed = _parse_allowed_set(msg)
    assert "opencv" not in allowed, (
        f"get_all_frames must NOT list opencv in its allowed set "
        f"(its dispatch chain does not implement opencv); got: {allowed}"
    )
