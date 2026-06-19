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

from collections import OrderedDict
import io
import shutil

from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper
import gymnasium as gym
import numpy as np
import pytest


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


class DummyEnv(gym.Env):
    def reset(self, **kwargs):
        return {}, {}

    def step(self, action):
        return {}, 0.0, False, False, {}


def _make_wrapper(record_video_keys=None) -> VideoRecordingWrapper:
    return VideoRecordingWrapper(
        DummyEnv(),
        video_dir=None,
        record_video_keys=record_video_keys,
    )


def test_video_recording_wrapper_uses_explicit_video_keys_in_order():
    obs = OrderedDict(
        [
            ("video.res512_image_side_0", _frame(1)),
            ("video.res256_image_side_0", _frame(2)),
            ("video.res512_image_side_1", _frame(3)),
            ("video.res256_image_side_1", _frame(4)),
            ("video.res512_image_wrist_0", _frame(5)),
            ("video.res256_image_wrist_0", _frame(6)),
        ]
    )
    wrapper = _make_wrapper(
        (
            "video.res256_image_side_0",
            "video.res256_image_side_1",
            "video.res256_image_wrist_0",
        )
    )

    selected = wrapper._get_video_frames(obs)

    assert [frame[0, 0, 0] for frame in selected] == [2, 4, 6]


def test_video_recording_wrapper_falls_back_to_all_video_keys():
    obs = OrderedDict(
        [
            ("video.image", _frame(1)),
            ("video.wrist_image", _frame(2)),
            ("state.joint_position", np.zeros(7, dtype=np.float32)),
        ]
    )
    wrapper = _make_wrapper()

    selected = wrapper._get_video_frames(obs)

    assert [frame[0, 0, 0] for frame in selected] == [1, 2]


def test_video_recording_wrapper_reports_missing_explicit_video_keys():
    obs = OrderedDict(
        [
            ("video.res256_image_side_0", _frame(1)),
        ]
    )
    wrapper = _make_wrapper(("video.res256_image_side_0", "video.res256_image_wrist_0"))

    try:
        wrapper._get_video_frames(obs)
    except KeyError as exc:
        assert "video.res256_image_wrist_0" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing explicit video key")


def test_video_recording_wrapper_configures_ffmpeg_h264_quality_options(monkeypatch, tmp_path):
    class FakeProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO()

        def wait(self):
            return 0

    popen_calls = []
    process = FakeProcess()
    monkeypatch.setattr(
        "gr00t.eval.sim.wrapper.video_recording_wrapper.subprocess.Popen",
        lambda cmd, **kwargs: popen_calls.append((cmd, kwargs)) or process,
    )

    wrapper = VideoRecordingWrapper(DummyEnv(), video_dir=None, fps=5, codec="h264")
    wrapper.file_path = tmp_path / "recording.mp4"
    wrapper._write_video_frame(_frame(0))

    cmd, kwargs = popen_calls[0]
    assert kwargs["stdin"] == -1
    assert kwargs["stderr"] == -1
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert cmd[cmd.index("-vcodec") + 1] == "rawvideo"
    assert cmd[cmd.index("-s") + 1] == "2x2"
    assert cmd[cmd.index("-r") + 1] == "5"
    assert cmd[cmd.index("-an") + 2] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-profile:v") + 1] == "high"
    assert cmd[cmd.index("-pix_fmt", cmd.index("-i")) + 1] == "yuv420p"
    assert process.stdin.getvalue() == _frame(0).tobytes()


def test_video_recording_wrapper_writes_mp4_with_ffmpeg(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")

    wrapper = VideoRecordingWrapper(DummyEnv(), video_dir=None, fps=5)
    video_path = tmp_path / "recording.mp4"

    wrapper.file_path = video_path
    wrapper._write_video_frame(_frame(0))
    wrapper._write_video_frame(_frame(255))
    wrapper._close_video_writer()

    assert video_path.exists()
    assert video_path.stat().st_size > 0
