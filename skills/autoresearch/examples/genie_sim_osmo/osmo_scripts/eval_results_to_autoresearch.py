#!/usr/bin/env python3
"""Convert Genie Sim benchmark JSON files into the AutoResearch metric shape."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

BOOL_KEYS = {"success", "is_success", "task_success"}
RATE_KEYS = {
    "success_rate",
    "successRate",
    "aggregated_success_rate",
    "mean_success_rate",
    "score",
    "rate",
}


def collect_success_values(obj, bool_values: list[float], rate_values: list[float]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in BOOL_KEYS and isinstance(value, bool | int | float):
                bool_values.append(float(value))
            elif key in RATE_KEYS and isinstance(value, int | float) and math.isfinite(value):
                rate_values.append(float(value) / 100.0 if float(value) > 1.0 else float(value))
            collect_success_values(value, bool_values, rate_values)
    elif isinstance(obj, list):
        for item in obj:
            collect_success_values(item, bool_values, rate_values)


def main() -> None:
    benchmark = Path(sys.argv[1])
    output = Path(sys.argv[2])
    checkpoint = sys.argv[3]
    task_name = sys.argv[4]

    result_files = sorted(benchmark.rglob("evaluate_ret_*.json"))
    bool_values: list[float] = []
    rate_values: list[float] = []

    for path in result_files:
        with path.open("r", encoding="utf-8") as f:
            collect_success_values(json.load(f), bool_values, rate_values)

    if bool_values:
        success_rate = sum(bool_values) / len(bool_values)
    elif rate_values:
        success_rate = sum(rate_values) / len(rate_values)
    else:
        raise SystemExit(f"No success value found in {benchmark}")

    payload = {
        "metric_type": "closed_loop_sim_success_rate",
        "success_rate": success_rate,
        "num_results": len(bool_values) if bool_values else len(rate_values),
        "checkpoint_path": checkpoint,
        "task_name": task_name,
        "result_files": [str(path) for path in result_files],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
