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

"""Tests for video backend lazy-loading behavior (GitHub issue #423).

Verifies that neither decord nor torchcodec is imported at module level,
preventing FFmpeg shared library conflicts and simulator crashes.
"""

import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDecordNotImportedAtModuleLevel:
    """Verify that importing video_utils does NOT import decord."""

    def test_decord_not_in_sys_modules_after_import(self):
        code = textwrap.dedent("""\
            import sys
            sys.modules.pop("decord", None)
            import gr00t.utils.video_utils
            assert "decord" not in sys.modules, (
                "decord was imported at module level by video_utils"
            )
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout


class TestTorchcodecNotImportedAtModuleLevel:
    """Verify that importing video_utils does NOT import torchcodec."""

    def test_torchcodec_not_in_sys_modules_after_import(self):
        code = textwrap.dedent("""\
            import sys
            sys.modules.pop("torchcodec", None)
            import gr00t.utils.video_utils
            assert "torchcodec" not in sys.modules, (
                "torchcodec was imported at module level by video_utils"
            )
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout


class TestTorchcodecUnavailable:
    """When torchcodec is not installed, resolve_backend should raise ImportError (no fallback)."""

    def test_raises_when_unavailable(self):
        from gr00t.utils.video_utils import resolve_backend

        with patch(
            "gr00t.utils.video_utils._lazy_import_torchcodec",
            side_effect=ImportError("torchcodec is not available."),
        ):
            with pytest.raises(ImportError, match="torchcodec"):
                resolve_backend("dummy.mp4", "torchcodec")


class TestDecordUnavailable:
    """When decord is not installed, resolve_backend should raise ImportError (no fallback)."""

    def test_raises_when_unavailable(self):
        from gr00t.utils.video_utils import resolve_backend

        with patch(
            "gr00t.utils.video_utils._lazy_import_decord",
            side_effect=ImportError("decord is not available."),
        ):
            with pytest.raises(ImportError, match="not available"):
                resolve_backend("dummy.mp4", "decord")


class TestAllBackendsUnavailable:
    """When a backend is unavailable, should raise ImportError."""

    def test_raises_import_error(self):
        from gr00t.utils.video_utils import resolve_backend

        with patch(
            "gr00t.utils.video_utils._is_backend_available",
            return_value=False,
        ):
            with pytest.raises(ImportError, match="not available"):
                resolve_backend("dummy.mp4", "nonexistent")


class TestDecordLazyImport:
    """Verify decord is only imported inside the decord backend branch."""

    def test_decord_imported_on_use(self):
        mock_decord = MagicMock()
        mock_vr = MagicMock()
        mock_vr.get_batch.return_value = MagicMock(
            asnumpy=MagicMock(return_value=np.zeros((1, 480, 640, 3), dtype=np.uint8))
        )
        mock_decord.VideoReader.return_value = mock_vr

        with patch.dict(sys.modules, {"decord": mock_decord}):
            from gr00t.utils.video_utils import get_frames_by_indices

            result = get_frames_by_indices("dummy.mp4", [0], video_backend="decord")

        mock_decord.VideoReader.assert_called_once_with("dummy.mp4")
        assert result.shape == (1, 480, 640, 3)


class TestDefaultVideoBackendConfig:
    """Verify the default video_backend config is torchcodec."""

    def test_data_config_defaults_to_torchcodec(self):
        from gr00t.configs.data.data_config import DataConfig

        config = DataConfig()
        assert config.video_backend == "torchcodec"
