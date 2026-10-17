"""Record pytest's exit status the moment its session ends.

Loaded by scripts/_run_unit_tests.sh as `-p ci.scripts.pytest_session_status`, which
needs a completion signal that does not depend on the pytest process exiting:
pytest can finish a session and then block in interpreter shutdown, joining a
non-daemon thread or a multiprocessing child that a timed-out test left behind.
The runner reaps pytest once this file appears, and reports the status in it.
Completed xdist workers also need bounded shutdown: otherwise the controller can
block joining them before it writes aggregate status and JUnit.
"""

from collections.abc import Iterator
from multiprocessing import active_children
import os
from threading import Timer

import pytest

STATUS_FILE_ENV_VAR = "PYTEST_SESSION_STATUS_FILE"


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    if not os.environ.get(STATUS_FILE_ENV_VAR):
        return
    grace = float(os.environ.get("PYTEST_SHUTDOWN_GRACE_SECONDS", "60"))
    # xdist normally kills a worker after 10s. Keep it alive long enough for
    # its own timer to reap children that can retain execnet's pipe ends.
    node.nodemanager.EXIT_TIMEOUT = max(node.nodemanager.EXIT_TIMEOUT, grace + 5)


def _exit_completed_worker(exitstatus: int) -> None:
    # Forked children can retain execnet's pipe descriptors after the worker
    # exits, keeping the controller blocked. These are this worker's children.
    try:
        for child in active_children():
            child.kill()
    finally:
        os._exit(exitstatus)


@pytest.hookimpl(specname="pytest_sessionfinish", hookwrapper=True, tryfirst=True)
def pytest_sessionfinish_worker_shutdown(session, exitstatus: int) -> Iterator[None]:
    # The outer wrapper resumes after xdist's wrapper sends workerfinished.
    # Exiting a worker earlier would look like a crash and lose its result.
    yield
    if not hasattr(session.config, "workerinput") or not os.environ.get(STATUS_FILE_ENV_VAR):
        return
    grace = float(os.environ.get("PYTEST_SHUTDOWN_GRACE_SECONDS", "60"))
    timer = Timer(grace, _exit_completed_worker, args=(int(exitstatus),))
    # A healthy worker exits normally. Only a completed worker stuck joining
    # non-daemon threads reaches this deadline; other workers keep running.
    timer.daemon = True
    timer.start()


def pytest_sessionfinish(session, exitstatus: int) -> None:
    # Only the controller knows the result of the entire distributed session.
    # A worker finishing early must not start the runner's shutdown watchdog.
    if hasattr(session.config, "workerinput"):
        return
    path = os.environ.get(STATUS_FILE_ENV_VAR)
    if not path:
        return

    # Written aside and renamed: the runner polls for this file, so it must never
    # observe a partial status.
    partial_path = f"{path}.partial"
    with open(partial_path, "w") as status_file:
        status_file.write(f"{int(exitstatus)}\n")
        status_file.flush()
        os.fsync(status_file.fileno())
    os.replace(partial_path, path)
