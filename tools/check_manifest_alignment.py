#!/usr/bin/env python3

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

"""Cross-platform ``pyproject.toml`` manifest alignment check.

The repo ships four platform manifests (dGPU, Orin, Spark, Thor). They
are not a ``uv`` workspace, so PEP 621 has no native way to share pins
across them and a missed mirror only surfaces at install time on the
unsynced platform. This script is the CI lint that catches the gap.

It walks the four manifests, normalises the ``[project.dependencies]``
lists, and flags two classes of cross-manifest drift:

  * **presence** — a package present in some manifests but absent from
    others.
  * **version** — same package, different version constraints.

CUDA / wheel-index-tied dependencies (``torch``, ``torchvision``,
``triton``, ``flash-attn``, ``torchcodec``, ``tensorrt-*``,
``deepspeed``, ``nvidia-*``) are intentionally divergent per platform
and are auto-skipped by name pattern. Other intentional drifts are
documented in ``tools/manifest_alignment.toml``.

Usage::

    python tools/check_manifest_alignment.py            # lint
    python tools/check_manifest_alignment.py --report   # also print allow-listed drifts

Exit code 0 if no unannotated drift, 1 otherwise.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import cast


try:
    # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover — supports running under 3.10
    import tomli as _toml  # type: ignore[import-not-found, no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFESTS: dict[str, Path] = {
    "main": REPO_ROOT / "pyproject.toml",
    "orin": REPO_ROOT / "scripts" / "deployment" / "orin" / "pyproject.toml",
    "spark": REPO_ROOT / "scripts" / "deployment" / "spark" / "pyproject.toml",
    "thor": REPO_ROOT / "scripts" / "deployment" / "thor" / "pyproject.toml",
}

# CUDA / wheel-index-tied packages — intentionally divergent per platform.
# Matched by exact name OR prefix (the `*` suffix marks a prefix rule).
_HW_TIED_PATTERNS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchcodec",
    "triton",
    "flash-attn",
    "deepspeed",
    "tensorrt*",
    "nvidia-*",
)

ALLOWLIST_PATH = REPO_ROOT / "tools" / "manifest_alignment.toml"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


# Split a PEP 508 dependency line into name / extras / spec / marker.
# Extras are dropped for the comparison axis; the raw spec is kept so
# the report can quote the exact line a reviewer needs to edit.
_REQ_SPLIT_RE = re.compile(
    r"""^
    (?P<name>[A-Za-z0-9._-]+)            # canonical project name
    (?P<extras>\[[^\]]+\])?              # optional extras
    \s*
    (?P<spec>[^;]*)                      # version specifier(s); may be empty
    (;\s*(?P<marker>.+))?                # optional environment marker
    $""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Pin:
    """One dependency from a manifest's ``[project.dependencies]``.
    ``marker`` is preserved so platform-conditional pins on the same
    package do not falsely diff."""

    name: str  # canonical PEP 503 name (lowercased, dashes)
    spec: str  # version specifier
    marker: str | None
    raw: str  # original PEP 508 line, surfaced in reports


def _canonical(name: str) -> str:
    """Apply PEP 503 name normalisation: lowercase + collapse separators."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_hw_tied(name: str) -> bool:
    """True for CUDA / wheel-index-tied packages that are auto-skipped
    (intentionally divergent per platform; no allowlist entry needed)."""
    name_lc = name.lower()
    for pat in _HW_TIED_PATTERNS:
        if pat.endswith("*"):
            if name_lc.startswith(pat[:-1]):
                return True
        elif name_lc == pat:
            return True
    return False


def _parse_dep(raw: str) -> Pin | None:
    """Parse a PEP 508 dependency string; return ``None`` on a malformed
    or empty entry."""
    raw = raw.strip()
    if not raw:
        return None
    m = _REQ_SPLIT_RE.match(raw)
    if not m:
        return None
    name = _canonical(m.group("name"))
    spec = (m.group("spec") or "").strip()
    marker = m.group("marker")
    return Pin(name=name, spec=spec, marker=marker.strip() if marker else None, raw=raw)


def parse_manifest(path: Path) -> dict[str, list[Pin]]:
    """``{canonical_name: [Pin, ...]}`` for one manifest. The list is
    per-name because PEP 508 lets the same package appear multiple
    times under different environment markers (e.g. main's
    ``torchcodec`` split on ``platform_machine``)."""
    data = _toml.loads(path.read_text(encoding="utf-8"))
    deps_raw = cast(list[str], data.get("project", {}).get("dependencies", []) or [])
    out: dict[str, list[Pin]] = defaultdict(list)
    for raw in deps_raw:
        pin = _parse_dep(raw)
        if pin is None:
            continue
        out[pin.name].append(pin)
    return dict(out)


def load_allowlist(path: Path = ALLOWLIST_PATH) -> dict[str, str]:
    """``{canonical_package_name: reason}`` for documented intentional
    drifts. A missing file yields an empty allow-list."""
    if not path.exists():
        return {}
    data = _toml.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for name, body in data.get("allowed_drifts", {}).items():
        reason = body.get("reason", "")
        out[_canonical(name)] = reason
    return out


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Drift:
    """One cross-manifest discrepancy on a dependency."""

    package: str
    kind: str  # "presence" | "version"
    detail: str


def detect_drifts(
    manifests: dict[str, dict[str, list[Pin]]],
) -> list[Drift]:
    """Emit one ``Drift`` per cross-manifest discrepancy: ``presence``
    when a package is missing from at least one manifest but present in
    another, ``version`` when the spec set differs (markers are
    preserved so platform-conditional pins do not falsely diff)."""
    all_names: set[str] = set()
    for pins in manifests.values():
        all_names.update(pins.keys())

    drifts: list[Drift] = []

    for name in sorted(all_names):
        if _is_hw_tied(name):
            continue

        # Bucket by which manifest declares the name.
        present: dict[str, list[Pin]] = {
            mname: pins[name] for mname, pins in manifests.items() if name in pins
        }
        missing: list[str] = [mname for mname in manifests if mname not in present]

        if missing and present:
            present_str = ", ".join(sorted(present.keys()))
            missing_str = ", ".join(sorted(missing))
            drifts.append(
                Drift(
                    package=name,
                    kind="presence",
                    detail=(f"present in [{present_str}], missing from [{missing_str}]"),
                )
            )
            continue

        if missing:
            # Absent everywhere — not a drift, just unused.
            continue

        # Present in all manifests — compare (spec, marker) signatures;
        # more than one distinct signature is a version drift.
        sig_per_manifest: dict[str, tuple[tuple[str, str | None], ...]] = {}
        for mname, pin_list in present.items():
            sig_per_manifest[mname] = tuple(sorted((pin.spec, pin.marker) for pin in pin_list))

        unique_sigs = set(sig_per_manifest.values())
        if len(unique_sigs) > 1:
            rows = []
            for mname in sorted(sig_per_manifest):
                spec_strs = [
                    f"{pin.spec or '(unpinned)'}" + (f" ; {pin.marker}" if pin.marker else "")
                    for pin in present[mname]
                ]
                rows.append(f"{mname}={' / '.join(spec_strs)}")
            drifts.append(
                Drift(
                    package=name,
                    kind="version",
                    detail="; ".join(rows),
                )
            )

    return drifts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_drifts(drifts: Iterable[Drift], allowlist: dict[str, str]) -> str:
    lines = []
    for d in drifts:
        marker = "ALLOWED" if d.package in allowlist else "DRIFT  "
        reason = f" — allowlisted: {allowlist[d.package]}" if d.package in allowlist else ""
        lines.append(f"  [{marker}] {d.package} ({d.kind}): {d.detail}{reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report",
        action="store_true",
        help="also print allow-listed (annotated) drifts",
    )
    args = p.parse_args(argv)

    for name, path in MANIFESTS.items():
        if not path.exists():
            print(f"FATAL: {name} manifest missing at {path}", file=sys.stderr)
            return 2

    parsed: dict[str, dict[str, list[Pin]]] = {
        name: parse_manifest(path) for name, path in MANIFESTS.items()
    }
    allowlist = load_allowlist()

    drifts = detect_drifts(parsed)
    unannotated = [d for d in drifts if d.package not in allowlist]
    annotated = [d for d in drifts if d.package in allowlist]

    if args.report and annotated:
        print(f"Allow-listed drifts ({len(annotated)}):")
        print(_format_drifts(annotated, allowlist))
        print()

    if not unannotated:
        print(
            f"OK: {len(drifts)} drift(s) total, all allow-listed in "
            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}."
        )
        return 0

    print(
        f"FAIL: {len(unannotated)} unannotated drift(s) across {len(MANIFESTS)} manifests:",
        file=sys.stderr,
    )
    print(_format_drifts(unannotated, allowlist), file=sys.stderr)
    print(
        "\nResolve each drift by either:\n"
        "  (a) aligning the pin / adding the missing dep across all manifests, or\n"
        f"  (b) recording an intentional divergence in {ALLOWLIST_PATH.relative_to(REPO_ROOT)} "
        "with a reason.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
