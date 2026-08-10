# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search the published GR00T insight projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

OSS_ROOT = Path(__file__).resolve().parents[3]
if str(OSS_ROOT) not in sys.path:
    sys.path.insert(0, str(OSS_ROOT))

from skills.insights.infra.insight_projection import search_insights  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--start", type=Path, default=Path.cwd())
    parser.add_argument("--insights-root", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    result = search_insights(
        args.query,
        start=args.start,
        insights_root=args.insights_root,
        limit=args.limit,
    )
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
