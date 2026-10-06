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

"""CPU-only tests for ``tools/check_manifest_alignment.py``.

The tests have two roles: a detector self-check against synthetic
manifests (so a future change to the rules cannot regress the contract
silently), and a live-manifest gate that runs the detector against the
four real ``pyproject.toml`` files and fails CI on any unannotated
drift — closing the loop that would otherwise only surface as a Jetson
install error.
"""

from __future__ import annotations

from pathlib import Path
import sys

from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


@pytest.fixture(scope="module")
def check_module():
    """Import ``tools/check_manifest_alignment``. ``tools/`` is not a
    package, so add it to ``sys.path`` before importing."""
    tools_str = str(TOOLS_DIR)
    if tools_str not in sys.path:
        sys.path.insert(0, tools_str)
    import check_manifest_alignment as mod  # type: ignore[import-not-found]

    return mod


# ---------------------------------------------------------------------------
# Live manifest gate
# ---------------------------------------------------------------------------


def test_live_manifests_have_no_unannotated_drift(check_module):
    """Fail CI if any non-HW-tied dependency drifts across the four
    manifests without an entry in ``tools/manifest_alignment.toml``.
    Resolve by aligning the pin or by recording the intentional
    divergence with a reason."""
    parsed = {
        name: check_module.parse_manifest(path) for name, path in check_module.MANIFESTS.items()
    }
    allowlist = check_module.load_allowlist()
    drifts = check_module.detect_drifts(parsed)
    unannotated = [d for d in drifts if d.package not in allowlist]

    if unannotated:
        report = "\n".join(f"  - {d.package} ({d.kind}): {d.detail}" for d in unannotated)
        pytest.fail(
            f"{len(unannotated)} unannotated manifest drift(s) found:\n{report}\n"
            "Fix by aligning the pins across all four pyproject.toml files "
            "or by recording the intentional divergence in "
            "tools/manifest_alignment.toml with a reason."
        )


def test_live_manifests_carry_the_b_027_pins(check_module):
    """Pin the specific bugs this MR closes: every Jetson manifest
    declares ``huggingface-hub`` / ``jsonlines`` / ``opencv-python-headless``,
    and main declares ``onnxscript``. A future refactor that drops one
    of these fails this test instead of breaking a Jetson install."""
    parsed = {
        name: check_module.parse_manifest(path) for name, path in check_module.MANIFESTS.items()
    }

    for jetson in ("orin", "spark", "thor"):
        for pkg in ("huggingface-hub", "jsonlines", "opencv-python-headless"):
            assert pkg in parsed[jetson], (
                f"{jetson}/pyproject.toml must declare {pkg!r} (regression for "
                "the silent-missing-pin family this MR closes)."
            )

    assert "onnxscript" in parsed["main"], (
        "main pyproject.toml must declare onnxscript (regression for the "
        "ONNX-export pipeline pin family this MR closes)."
    )


def test_live_main_cryptography_is_at_least_44(check_module):
    """Pin the security-fix half of the bundle: main ``cryptography``
    must reject the CVE-affected 42.x / 43.x series (CVE-2024-26130 et
    al.). Any spec that admits a version below 44 fails this test —
    upgrades to 44+/45+/46+/… all pass."""
    parsed = check_module.parse_manifest(check_module.MANIFESTS["main"])
    pins = parsed.get("cryptography", [])
    assert pins, "main pyproject.toml must declare cryptography"
    spec_strs = [pin.spec for pin in pins]
    assert all(spec_strs), (
        f"main cryptography must carry a version spec, not a loose pin. Got: {spec_strs}"
    )
    pre44_probes = [Version("42.0.8"), Version("43.0.0"), Version("43.99.99")]
    for spec_str in spec_strs:
        spec = SpecifierSet(spec_str)
        admitted = [str(v) for v in pre44_probes if spec.contains(v)]
        assert not admitted, (
            f"main cryptography spec {spec_str!r} admits CVE-affected "
            f"version(s) {admitted}; expected lower bound >= 44.0.0."
        )


# ---------------------------------------------------------------------------
# Detector self-check (synthetic manifests)
# ---------------------------------------------------------------------------


def test_canonical_lowercases_and_collapses_separators(check_module):
    """PEP 503 normalisation: surface variants of the same name must
    compare equal."""
    canon = check_module._canonical
    assert canon("Huggingface-Hub") == "huggingface-hub"
    assert canon("hugging_face.hub") == "hugging-face-hub"
    assert canon("HUGGINGFACE_HUB") == "huggingface-hub"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("torch", True),
        ("torchvision", True),
        ("torchcodec", True),
        ("triton", True),
        ("flash-attn", True),
        ("deepspeed", True),
        ("tensorrt", True),
        ("tensorrt-cu12", True),
        ("tensorrt-cu13", True),
        ("nvidia-cudnn-cu13", True),
        ("nvidia-cudss-cu13", True),
        # NON-HW packages — must NOT be auto-skipped.
        ("transformers", False),
        ("cryptography", False),
        ("opencv-python-headless", False),
        ("huggingface-hub", False),
        ("diffusers", False),
        ("onnxscript", False),
    ],
)
def test_hw_tied_pattern_classification(check_module, name, expected):
    """Pin each known HW vs non-HW name explicitly so a typo in the
    auto-skip pattern cannot silently mute drift detection for a real
    package family."""
    assert check_module._is_hw_tied(name) is expected, (
        f"{name!r} HW-tied classification mismatch: expected {expected}"
    )


def test_parse_dep_extracts_name_extras_spec_and_marker(check_module):
    """Cover every PEP 508 surface the real manifests use: extras,
    version specs, and environment markers."""
    pin = check_module._parse_dep("huggingface-hub[cli]")
    assert pin is not None
    assert pin.name == "huggingface-hub"
    assert pin.spec == ""
    assert pin.marker is None

    pin = check_module._parse_dep("torchcodec==0.4.0; platform_machine == 'x86_64'")
    assert pin is not None
    assert pin.name == "torchcodec"
    assert pin.spec == "==0.4.0"
    assert pin.marker == "platform_machine == 'x86_64'"

    pin = check_module._parse_dep("opencv-python-headless>=4.5,<4.13")
    assert pin is not None
    assert pin.name == "opencv-python-headless"
    assert pin.spec == ">=4.5,<4.13"


def test_detect_drifts_flags_presence_and_version_drift(check_module):
    """End-to-end on a synthetic 3-manifest input covering presence
    drift, version drift, an auto-skipped HW-tied package, and a fully
    aligned package."""
    parse = check_module._parse_dep
    manifests = {
        "main": {
            "transformers": [parse("transformers==4.57.3")],
            "torch": [parse("torch==2.7.1")],
            "huggingface-hub": [parse("huggingface-hub[cli]")],
            "numpy": [parse("numpy==1.26.4")],
        },
        "jetson_a": {
            "transformers": [parse("transformers==4.57.6")],
            "torch": [parse("torch==2.10.0")],  # HW-tied, divergent — must be skipped
            "numpy": [parse("numpy==1.26.4")],
        },
        "jetson_b": {
            "transformers": [parse("transformers==4.57.3")],
            "torch": [parse("torch==2.10.0")],
            "huggingface-hub": [parse("huggingface-hub[cli]")],
            "numpy": [parse("numpy==1.26.4")],
        },
    }
    drifts = check_module.detect_drifts(manifests)
    by_pkg = {d.package: d for d in drifts}

    assert "torch" not in by_pkg, "HW-tied package must be auto-skipped"
    assert "numpy" not in by_pkg, "fully aligned package must not appear"
    assert "transformers" in by_pkg
    assert by_pkg["transformers"].kind == "version"
    assert "huggingface-hub" in by_pkg
    assert by_pkg["huggingface-hub"].kind == "presence"
    assert "missing from [jetson_a]" in by_pkg["huggingface-hub"].detail


def test_load_allowlist_round_trips_reason(check_module, tmp_path):
    """Allowlist names must be PEP 503-normalised so
    ``[allowed_drifts.dm_tree]`` matches a manifest's ``dm-tree``
    entry."""
    allow_file = tmp_path / "allow.toml"
    allow_file.write_text(
        '[allowed_drifts.dm_tree]\nreason = "loose on main, pinned on jetson"\n',
        encoding="utf-8",
    )
    allowlist = check_module.load_allowlist(allow_file)
    assert allowlist == {"dm-tree": "loose on main, pinned on jetson"}


def test_load_allowlist_missing_file_returns_empty(check_module, tmp_path):
    """A missing allowlist file must not crash the script — the check
    still works on a fresh checkout."""
    assert check_module.load_allowlist(tmp_path / "absent.toml") == {}
