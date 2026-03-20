"""pytest hooks to make CI logs easier to read."""

from __future__ import annotations


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    print(f"\n{'=' * 80}\n[TEST START] {nodeid}\n", flush=True)


def pytest_runtest_logfinish(nodeid: str, location: tuple) -> None:
    print(f"\n[TEST END]   {nodeid}\n{'=' * 80}", flush=True)
