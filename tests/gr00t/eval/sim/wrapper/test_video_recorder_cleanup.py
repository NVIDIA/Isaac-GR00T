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

"""Pin the ffmpeg recorder lifecycle: ``close()`` reaps the encoder child
and the inner env, a wedged encoder is killed within the grace window
(never blocks the caller forever), and a non-zero exit still surfaces.

These tests drive a *real* child ``subprocess`` (a tiny ``python -c`` stand-in
for ffmpeg) rather than a stub, so they exercise the actual
``Popen.communicate()`` semantics — in particular that the cleanup path must
not close stdin before ``communicate()`` flushes it (a stub that never touches
stdin would hide that bug).
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _import_module():
    """The wrapper imports ``gymnasium``/``cv2``; skip cleanly if absent."""
    try:
        from gr00t.eval.sim.wrapper import video_recording_wrapper as mod
    except (ImportError, OSError) as e:
        pytest.skip(f"video_recording_wrapper not importable in this env: {e}")
    return mod


# Child processes standing in for ffmpeg, all reading stdin like ffmpeg does:
_DRAIN_STDIN = [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]  # exit 0 on EOF
_DRAIN_THEN_FAIL = [sys.executable, "-c", "import sys; sys.stdin.buffer.read(); sys.exit(3)"]
_IGNORE_EOF_SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]  # never exits on EOF


def _make_wrapper(mod, cmd):
    # gym.Wrapper.__init__ asserts the wrapped env is a real gymnasium.Env, so
    # the inner env must subclass it; we still want a trivial close()-tracking
    # stand-in. Build it off the wrapper module's own gym to stay skip-safe.
    class _FakeEnv(mod.gym.Env):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    wrapper = mod.VideoRecordingWrapper(_FakeEnv(), video_dir=None)
    wrapper.video_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    return wrapper


def test_close_reaps_recorder_and_inner_env():
    mod = _import_module()
    wrapper = _make_wrapper(mod, _DRAIN_STDIN)
    proc = wrapper.video_process

    wrapper.close()

    assert proc.poll() == 0  # child saw EOF and exited cleanly
    assert wrapper.video_process is None
    assert wrapper.env.closed is True


def test_close_surfaces_nonzero_ffmpeg_exit():
    mod = _import_module()
    wrapper = _make_wrapper(mod, _DRAIN_THEN_FAIL)
    proc = wrapper.video_process

    with pytest.raises(RuntimeError, match="ffmpeg video recording failed"):
        wrapper.close()

    assert proc.poll() == 3
    assert wrapper.video_process is None
    assert wrapper.env.closed is True


def test_close_kills_wedged_recorder_within_grace(monkeypatch):
    mod = _import_module()
    monkeypatch.setattr(mod, "_FFMPEG_CLOSE_GRACE_SECONDS", 0.5)
    wrapper = _make_wrapper(mod, _IGNORE_EOF_SLEEP)
    proc = wrapper.video_process

    with pytest.raises(RuntimeError, match="did not exit"):
        wrapper.close()

    assert proc.poll() is not None  # was killed and reaped, not left running
    # finally-block teardown still ran: child handle dropped, inner env closed.
    assert wrapper.video_process is None
    assert wrapper.env.closed is True
