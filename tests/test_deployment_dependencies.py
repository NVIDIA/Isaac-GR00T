# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Check that OSS deployment selects its evaluator's locked dependencies."""

from pathlib import Path

from packaging.requirements import Requirement
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # The monorepo unit tests collect tests/oss on Python 3.10.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "pyproject.toml").is_file():
    ROOT = ROOT.parent / "oss"


@pytest.mark.parametrize("project", [".", "scripts/deployment/thor"])
def test_deployment_selects_locked_evaluation_dependencies(project):
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = {
        Requirement(dep).name: Requirement(dep)
        for dep in manifest["project"]["optional-dependencies"]["deployment"]
    }
    if project != ".":
        platform = tomllib.loads((ROOT / project / "pyproject.toml").read_text())
        package = next(
            Requirement(dep)
            for dep in platform["project"]["dependencies"]
            if Requirement(dep).name == "isaac-groot-n2"
        )
        assert "deployment" in package.extras
        requirements.update(
            {
                Requirement(dep).name: Requirement(dep)
                for dep in platform["tool"]["uv"]["override-dependencies"]
            }
        )
    lock = tomllib.loads((ROOT / project / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    selected = {
        dep["name"] for dep in packages["isaac-groot-n2"]["optional-dependencies"]["deployment"]
    }
    for name in ("matplotlib", "torchcodec"):
        assert name in requirements
        assert name in selected
        assert requirements[name].specifier.contains(packages[name]["version"], prereleases=True)
