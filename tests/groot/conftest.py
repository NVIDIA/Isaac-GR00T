# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
