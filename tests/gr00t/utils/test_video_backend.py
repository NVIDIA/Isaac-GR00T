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

"""Tests for the torchcodec video backend."""

import builtins
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_VIDEO = (
    REPO_ROOT
    / "demo_data"
    / "cube_to_bowl_5"
    / "videos"
    / "chunk-000"
    / "observation.images.front"
    / "episode_000000.mp4"
)


class TestImportSafety:
    """Importing video utilities must not import or load torchcodec."""

    def test_torchcodec_not_imported_at_module_level(self):
        code = textwrap.dedent("""\
            import sys
            sys.modules.pop("torchcodec", None)
            sys.modules.pop("torchcodec.decoders", None)
            import gr00t.utils.video_utils
            assert "torchcodec" not in sys.modules
            assert "torchcodec.decoders" not in sys.modules
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            # Headroom for parallel CI (pytest-xdist -n auto): this child imports
            # gr00t (torch et al.), which is slow when every worker is busy.
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout


@pytest.fixture(scope="module")
def sample_video_path() -> Path:
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"sample video fixture not found at {SAMPLE_VIDEO}")
    pytest.importorskip("torchcodec")
    return SAMPLE_VIDEO


class TestTorchcodecRoundtrip:
    """Decode the sample fixture end-to-end through the public helpers."""

    def test_get_frames_by_indices_returns_nhwc_uint8(self, sample_video_path: Path):
        from gr00t.utils.video_utils import get_frames_by_indices

        frames = get_frames_by_indices(str(sample_video_path), [0, 5, 10])
        assert frames.shape == (3, 480, 640, 3)
        assert frames.dtype == np.uint8
        assert not np.array_equal(frames[0], frames[1])

    def test_get_frames_by_timestamps_returns_distinct_frames(self, sample_video_path: Path):
        from gr00t.utils.video_utils import get_frames_by_timestamps
        from torchcodec.decoders import VideoDecoder

        fps = float(VideoDecoder(str(sample_video_path)).metadata.average_fps)
        # Timestamps at evenly-spaced frame boundaries — exercises the rounding
        # path in get_frames_by_timestamps that corrects float-precision drift.
        timestamps = [i / fps for i in (0, 5, 10)]

        frames = get_frames_by_timestamps(str(sample_video_path), timestamps)
        assert frames.shape == (3, 480, 640, 3)
        assert frames.dtype == np.uint8
        assert not np.array_equal(frames[0], frames[1])
        assert not np.array_equal(frames[1], frames[2])

    def test_get_frames_by_timestamps_rejects_off_grid(self, sample_video_path: Path):
        from gr00t.utils.video_utils import get_frames_by_timestamps
        from torchcodec.decoders import VideoDecoder

        fps = float(VideoDecoder(str(sample_video_path)).metadata.average_fps)
        # Explicitly off-grid (~50% between frames) — should fail the 1%
        # tolerance check rather than silently snap to the wrong frame.
        off_grid = [0.0, 1.0 / fps + 0.5 / fps]
        with pytest.raises(ValueError, match="invalid timestamps"):
            get_frames_by_timestamps(str(sample_video_path), off_grid)

    def test_get_all_frames_returns_full_sequence(self, sample_video_path: Path):
        from gr00t.utils.video_utils import get_all_frames
        from torchcodec.decoders import VideoDecoder

        expected_n = len(VideoDecoder(str(sample_video_path)))
        frames, pts = get_all_frames(str(sample_video_path))
        assert frames.shape == (expected_n, 480, 640, 3)
        assert pts.shape == (expected_n,)
        # Timestamps must be strictly monotonic.
        assert np.all(np.diff(pts) > 0)


class TestTorchcodecMissing:
    """When torchcodec is absent, helpers must raise ImportError with an install hint."""

    def test_build_decoder_raises_import_error(self, monkeypatch: pytest.MonkeyPatch):
        import gr00t.utils.video_utils as vu

        def raise_missing():
            raise ImportError("torchcodec is required for video decoding.")

        monkeypatch.setattr(vu, "_get_video_decoder_cls", raise_missing)
        with pytest.raises(ImportError, match="torchcodec is required"):
            vu.get_frames_by_indices("dummy.mp4", [0])

    def test_runtime_import_failure_is_wrapped(self, monkeypatch: pytest.MonkeyPatch):
        import gr00t.utils.video_utils as vu

        real_import = builtins.__import__

        def fail_torchcodec_import(name, *args, **kwargs):
            if name == "torchcodec.decoders":
                raise RuntimeError("failed to load FFmpeg")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_torchcodec_import)
        with pytest.raises(ImportError, match="torchcodec is required"):
            vu._get_video_decoder_cls()
