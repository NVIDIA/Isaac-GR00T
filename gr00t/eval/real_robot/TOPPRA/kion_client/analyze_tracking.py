from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tracking import delay_estimate_as_dict, estimate_tracking_delay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Kion TCP tracking delay from a recorded rollout CSV."
    )
    parser.add_argument("csv", type=Path, help="Path to tcp_tracking.csv")
    parser.add_argument("--max-lag-ms", type=float, default=500.0)
    parser.add_argument("--lag-step-ms", type=float, default=1.0)
    parser.add_argument("--min-target-speed", type=float, default=0.005)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path; defaults to tracking_report.json beside the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = {
        arm: delay_estimate_as_dict(
            estimate_tracking_delay(
                args.csv,
                arm,
                max_lag_ms=args.max_lag_ms,
                lag_step_ms=args.lag_step_ms,
                min_target_speed_m_s=args.min_target_speed,
            )
        )
        for arm in ("left", "right")
    }
    output = args.output or args.csv.with_name("tracking_report.json")
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
