from __future__ import annotations

import argparse
import sys

from .plain_client import main as plain_main


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m gr00t.eval.real_robot.TOPPRA.rollout",
        description="Unified bimanual real-robot rollout entrypoint.",
    )
    parser.add_argument("mode", choices=("plain", "speed-rl", "speed-rl-baseline"))
    if not values or values[0] in {"-h", "--help"}:
        parser.print_help()
        return
    mode = parser.parse_args(values[:1]).mode
    forwarded = values[1:]
    if mode == "plain":
        plain_main(forwarded)
        return
    from ..speed_rl.client import main as speed_rl_main

    if mode == "speed-rl-baseline":
        forwarded = ["--execution-backend", "interpolation", *forwarded]
    speed_rl_main(forwarded)


if __name__ == "__main__":
    main()
