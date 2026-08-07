# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OSS test setup for optional training-image dependencies."""

import importlib
import importlib.machinery
import sys
from types import ModuleType


def _install_stub_if_unavailable(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except (ImportError, RuntimeError):
        module = ModuleType(module_name)
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
        sys.modules[module_name] = module


for _module_name in ("av", "cv2", "torchvision"):
    _install_stub_if_unavailable(_module_name)
