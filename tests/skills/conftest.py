# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Make repository-local skill infrastructure importable in focused tests."""

from pathlib import Path
import sys

OSS_ROOT = Path(__file__).resolve().parents[2]
if str(OSS_ROOT) not in sys.path:
    sys.path.insert(0, str(OSS_ROOT))
