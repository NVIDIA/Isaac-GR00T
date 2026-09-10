# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _oss_sources():
    path = Path(__file__).resolve().parents[5] / "groot/core/data/packds/sources.py"
    spec = importlib.util.spec_from_file_location("oss_packds_sources", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return path, module


def test_oss_sources_are_standalone_and_support_public_locators(tmp_path: Path) -> None:
    _, sources = _oss_sources()

    local = sources.resolve_source(tmp_path)
    remote = sources.resolve_source("s3://bucket/dataset", storage_options={"region": "test"})
    assert local.source_key == "local"
    assert local.packds_uri == str(tmp_path)
    assert remote.source_key == "remote"
    assert remote.packds_uri == "s3://bucket/dataset"


def test_oss_sources_reject_deltacat() -> None:
    path, sources = _oss_sources()

    with pytest.raises(NotImplementedError, match="not supported in the OSS package"):
        sources.resolve_source("dc://catalog/namespace/table")

    source = path.read_text(encoding="utf-8")
    assert "gear-data-infra-" not in source
    assert "internal-gear-" not in source
    assert "swiftstack_pdx" not in source
