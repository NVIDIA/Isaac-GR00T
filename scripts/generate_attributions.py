#!/usr/bin/env python3
"""
Generate ATTRIBUTIONS.md from the current uv environment.

Lists the name, version, license, URL, and full license text for every
direct dependency declared in pyproject.toml.

Prerequisites:
    uv pip install pip-licenses

Usage:
    python scripts/generate_attributions.py            # writes ATTRIBUTIONS.md
    python scripts/generate_attributions.py --check    # exits non-zero if stale
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPO_ROOT / "ATTRIBUTIONS.md"


def _get_direct_deps() -> list[str]:
    """Extract direct dependency package names from pyproject.toml."""
    deps: list[str] = []
    pyproject = REPO_ROOT / "pyproject.toml"
    in_section = False
    for line in pyproject.read_text().splitlines():
        stripped = line.strip()
        if stripped in ("dependencies = [", "dev = ["):
            in_section = True
            continue
        if in_section and stripped == "]":
            in_section = False
            continue
        if in_section:
            m = re.match(r'\s*"([a-zA-Z0-9_-]+)', stripped)
            if m:
                deps.append(m.group(1).lower())
    return sorted(set(deps))


def generate() -> str:
    """Run pip-licenses and return the output as a string."""
    deps = _get_direct_deps()
    cmd = [
        sys.executable,
        "-m",
        "piplicenses",
        "--format=plain-vertical",
        "--with-license-file",
        "--no-license-path",
        "--with-urls",
        "--packages",
        *deps,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"pip-licenses failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    # Strip ANSI escape codes and trailing whitespace
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout).rstrip() + "\n"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that ATTRIBUTIONS.md is up to date (exit 1 if stale).",
    )
    args = parser.parse_args()

    generated = generate()

    if args.check:
        if not OUTPUT_FILE.exists():
            print(f"ATTRIBUTIONS.md does not exist. Run: python {__file__}")
            sys.exit(1)
        existing = OUTPUT_FILE.read_text()
        if existing.rstrip() != generated.rstrip():
            diff = difflib.unified_diff(
                existing.splitlines(keepends=True),
                generated.splitlines(keepends=True),
                fromfile="ATTRIBUTIONS.md (on disk)",
                tofile="ATTRIBUTIONS.md (generated)",
                n=3,
            )
            print("ATTRIBUTIONS.md is stale. Differences:")
            print("".join(list(diff)[:60]))
            print(f"\nRegenerate with: python {__file__}")
            sys.exit(1)
        print("ATTRIBUTIONS.md is up to date.")
    else:
        OUTPUT_FILE.write_text(generated)
        print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
