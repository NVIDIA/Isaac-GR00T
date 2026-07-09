"""Summarize GrootN2 torch.compile diagnostic logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable

_DIAGNOSTIC_RE = re.compile(r"\[GrootN2 compile\] step=(?P<step>\d+) (?P<metrics>.*)$")
_METRIC_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>-?\d+)")


def parse_compile_diagnostic_line(line: str) -> dict[str, int] | None:
    match = _DIAGNOSTIC_RE.search(line)
    if match is None:
        return None

    record = {"step": int(match.group("step"))}
    for metric in _METRIC_RE.finditer(match.group("metrics")):
        record[metric.group("key")] = int(metric.group("value"))
    return record


def collect_compile_diagnostics(lines: Iterable[str]) -> list[dict[str, int]]:
    return [
        record
        for line in lines
        if (record := parse_compile_diagnostic_line(line)) is not None
    ]


def summarize_compile_diagnostics(records: list[dict[str, int]]) -> dict[str, int]:
    if not records:
        return {"records": 0}

    first = records[0]
    last = records[-1]
    summary = {
        "records": len(records),
        "first_step": first["step"],
        "last_step": last["step"],
    }
    for key in ("unique_graphs", "graph_breaks", "recompiles", "calls_captured"):
        first_value = first.get(key)
        last_value = last.get(key)
        if first_value is None or last_value is None:
            continue
        summary[f"{key}_first"] = first_value
        summary[f"{key}_last"] = last_value
        summary[f"{key}_delta"] = last_value - first_value
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", type=Path, help="Training log containing GrootN2 compile lines")
    args = parser.parse_args(argv)

    with args.logfile.open() as f:
        records = collect_compile_diagnostics(f)
    print(json.dumps(summarize_compile_diagnostics(records), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
