# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""E2E test utilities for the hydrator (shared by tests under tests/).

This package provides shared utilities (e.g. utils.py) for hydrator E2E tests.
The actual E2E test modules live under tests/groot/core/data/hydrator/e2e_tests/
and are run with pytest, marked with requires_s3.

Run E2E tests:
    pytest tests/groot/core/data/hydrator/e2e_tests/ -v -m requires_s3
"""
