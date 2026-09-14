# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Public N2 benchmark orchestration and report contracts."""

from groot.infra.benchmark.report import REPORT_SCHEMA_VERSION
from groot.infra.benchmark.spec import BENCHMARK_SPEC_SCHEMA_VERSION, BenchmarkSpec

__all__ = [
    "BENCHMARK_SPEC_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "BenchmarkSpec",
]
