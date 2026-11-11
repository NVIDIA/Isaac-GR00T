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

import importlib.util
from pathlib import Path
import subprocess

import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "eval" / "check_sim_eval_ready.py"
SPEC = importlib.util.spec_from_file_location("check_sim_eval_ready", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class _MissingPath:
    def __init__(self, _path):
        pass

    def exists(self):
        return False


def test_vulkan_icd_creation_failure_is_not_reported_as_ready(monkeypatch):
    monkeypatch.setattr(CHECKER, "Path", _MissingPath)

    def fake_run(command, *, shell, check=False):
        if check:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(CHECKER.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        CHECKER.check_vulkan_installation()
