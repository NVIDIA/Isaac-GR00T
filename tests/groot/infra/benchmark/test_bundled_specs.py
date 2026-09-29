# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Validate every benchmark specification shipped in the package."""

from pathlib import Path

from groot.infra.benchmark.spec import load_benchmark_spec

_SPEC_DIRECTORY = Path(__file__).resolve().parents[4] / "groot/infra/benchmark/specs"


def test_all_bundled_benchmark_specs_load() -> None:
    """Every bundled JSON spec must resolve by name and pass schema validation."""
    spec_paths = sorted(_SPEC_DIRECTORY.glob("*.json"))
    assert spec_paths, f"No benchmark specs found in {_SPEC_DIRECTORY}"

    for path in spec_paths:
        spec = load_benchmark_spec(path.stem, builtin_dir=_SPEC_DIRECTORY)
        assert spec.path == path.resolve()
