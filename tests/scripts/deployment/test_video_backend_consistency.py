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

"""Pin that deployment/runtime APIs do not expose video backend selectors.

GR00T uses torchcodec as the only video decoder. User-facing CLIs and
public helper APIs must not grow a ``--video-backend`` flag or dataclass
field.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import get_type_hints

import pytest


@pytest.fixture(scope="module")
def deploy_imports():
    """Make ``scripts/deployment`` importable. The directory is not a
    package; it relies on ``sys.path`` insertion at runtime, so we mirror
    that here so the CLI configs can be imported."""
    deploy_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../scripts/deployment")
    )
    if deploy_dir not in sys.path:
        sys.path.insert(0, deploy_dir)
    return deploy_dir


# ---------------------------------------------------------------------------
# CLI sites: none may expose a video backend selector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name, cls_name",
    [
        ("export_onnx_n1d7", "ExportConfig"),
        ("build_trt_pipeline", "PipelineConfig"),
        ("standalone_inference_script", "ArgsConfig"),
    ],
)
def test_cli_config_has_no_video_backend_field(deploy_imports, module_name, cls_name):
    """Deployment CLIs should not expose a one-choice ``--video-backend`` flag."""
    try:
        mod = __import__(module_name)
    except (ImportError, OSError) as e:
        pytest.skip(f"{module_name} not importable in this env: {e}")
    cfg_cls = getattr(mod, cls_name, None)
    if cfg_cls is None:
        pytest.skip(f"{module_name} has no attribute {cls_name!r}")

    hints = get_type_hints(cfg_cls)
    assert "video_backend" not in hints, (
        f"{module_name}.{cls_name} exposes video_backend; remove the CLI selector "
        "and use torchcodec internally."
    )


def test_video_utils_public_api_has_no_video_backend_parameter():
    """Frame-loading helpers should not expose a one-choice backend selector."""
    try:
        from gr00t.utils import video_utils
    except (ImportError, OSError) as e:
        pytest.skip(f"video_utils not importable in this env: {e}")

    for func_name in (
        "get_frames_by_indices",
        "get_frames_by_timestamps",
        "get_all_frames",
    ):
        signature = inspect.signature(getattr(video_utils, func_name))
        assert "video_backend" not in signature.parameters


def test_dataset_public_apis_have_no_video_backend_parameter():
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    for cls in (LeRobotEpisodeLoader, ShardedSingleStepDataset):
        signature = inspect.signature(cls)
        assert "video_backend" not in signature.parameters
