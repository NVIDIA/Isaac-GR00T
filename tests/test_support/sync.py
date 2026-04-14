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

"""Sync test helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType


def load_sync_module(module_name: str):
    """Import a sync module with lightweight `git` stubs for unit tests."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    git_module = sys.modules.get("git")
    if git_module is None:
        git_module = ModuleType("git")
        sys.modules["git"] = git_module
    git_module.Repo = object
    git_module.__path__ = []

    if "git.exc" not in sys.modules:
        git_exc = ModuleType("git.exc")
        git_exc.GitCommandError = RuntimeError
        sys.modules["git.exc"] = git_exc

    return importlib.import_module(module_name)
