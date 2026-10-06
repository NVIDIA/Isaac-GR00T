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

"""Tests for torchcodec import-failure diagnostics in gr00t.utils.video_utils."""

from gr00t.utils.video_utils import _decoder_import_error


def test_ffmpeg_mismatch_message_is_actionable():
    """A native-load failure (RuntimeError) should be reported as an FFmpeg issue."""
    exc = RuntimeError(
        "Could not load libtorchcodec. Likely causes: FFmpeg is not properly "
        "installed. We support versions 4, 5, 6 and 7."
    )
    err = _decoder_import_error(exc)
    assert isinstance(err, ImportError)
    msg = str(err)
    assert "FFmpeg" in msg
    assert "FFmpeg 8" in msg
    # The original error is surfaced so the cause is not lost.
    assert "Could not load libtorchcodec" in msg


def test_oserror_is_treated_as_load_failure():
    err = _decoder_import_error(OSError("libavutil.so.60: cannot open shared object file"))
    assert isinstance(err, ImportError)
    assert "FFmpeg" in str(err)


def test_missing_package_message_points_to_install():
    """A plain ImportError means torchcodec is not installed."""
    err = _decoder_import_error(ImportError("No module named 'torchcodec'"))
    assert isinstance(err, ImportError)
    # Pin the install branch via text unique to _TORCHCODEC_INSTALL_HINT; "install"
    # alone also appears in the FFmpeg hint ("is installed", "conda install").
    assert "uv pip install torchcodec" in str(err)
    assert "native library" not in str(err)
