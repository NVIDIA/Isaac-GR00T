# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make repository-local skill infrastructure importable in focused tests."""

from pathlib import Path
import sys

OSS_ROOT = Path(__file__).resolve().parents[2]
if str(OSS_ROOT) not in sys.path:
    sys.path.insert(0, str(OSS_ROOT))
