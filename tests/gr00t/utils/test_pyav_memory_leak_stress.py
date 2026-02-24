"""
PyAV Memory Leak Stress Test — focuses on exception-path leaks.

The real danger is not the happy path (where close() is called),
but the exception path where containers are never closed.
This test simulates that by intentionally raising during decode loops.
"""

import gc
import os
import resource
import signal
import tempfile
import time
import traceback

import av
import numpy as np


# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------


def _timeout_handler(signum, frame):
    raise TimeoutError("Stress test exceeded time limit")


def set_timeout(seconds: int):
    """Set a global timeout (Linux only). Raises TimeoutError when exceeded."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)


def get_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def create_test_video(
    path: str, n_frames: int = 300, width: int = 640, height: int = 480, fps: int = 30
):
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    for i in range(n_frames):
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ---------------------------------------------------------------------------
# Scenario A: Exception during decode, container LEAKED (no try/finally)
# ---------------------------------------------------------------------------


def decode_leak_on_exception(video_path: str, fail_at_frame: int = 5):
    """Simulates codebase pattern: exception during decode → container never closed."""
    container = av.open(video_path)
    _stream = container.streams.video[0]
    count = 0
    for frame in container.decode(video=0):
        _arr = frame.to_ndarray(format="rgb24")
        count += 1
        if count >= fail_at_frame:
            # Simulate an exception (e.g. corrupt frame, shape mismatch, etc.)
            raise RuntimeError("Simulated decode error")
    container.close()  # Never reached!


# ---------------------------------------------------------------------------
# Scenario B: Exception during decode, container PROPERLY CLOSED (with)
# ---------------------------------------------------------------------------


def decode_safe_on_exception(video_path: str, fail_at_frame: int = 5):
    """Fixed pattern: context manager ensures close even on exception."""
    with av.open(video_path) as container:
        _stream = container.streams.video[0]
        count = 0
        for frame in container.decode(video=0):
            _arr = frame.to_ndarray(format="rgb24")
            count += 1
            if count >= fail_at_frame:
                raise RuntimeError("Simulated decode error")


# ---------------------------------------------------------------------------
# Scenario C: Happy path, no close (forget to call close)
# ---------------------------------------------------------------------------


def decode_forget_close(video_path: str):
    """Common mistake: just let the container go out of scope without closing."""
    container = av.open(video_path)
    _stream = container.streams.video[0]
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    # Forgot container.close() — relies on GC / __del__
    return len(frames)


# ---------------------------------------------------------------------------
# Scenario D: Happy path, proper close
# ---------------------------------------------------------------------------


def decode_proper_close(video_path: str):
    """Proper pattern with context manager."""
    with av.open(video_path) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return len(frames)


# ---------------------------------------------------------------------------


def run_scenario(label, fn, video_path, iterations, expect_exception=False):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Iterations: {iterations}")
    print(f"{'=' * 60}")

    gc.collect()
    rss_start = get_rss_mb()

    t0 = time.time()
    for i in range(iterations):
        try:
            fn(video_path)
        except RuntimeError:
            if not expect_exception:
                traceback.print_exc()

        # Periodically report without forcing GC (to see natural leak behavior)
        if (i + 1) % 50 == 0:
            rss_now = get_rss_mb()
            print(
                f"  iter {i + 1:4d}/{iterations}  RSS={rss_now:.1f} MB  (delta={rss_now - rss_start:+.1f} MB)"
            )

    # Final measurement after GC
    gc.collect()
    rss_end = get_rss_mb()
    elapsed = time.time() - t0
    print(
        f"\n  Result: start={rss_start:.1f} -> end={rss_end:.1f} MB  "
        f"(growth={rss_end - rss_start:+.1f} MB)  time={elapsed:.1f}s"
    )
    return rss_end - rss_start


def main():
    iterations = 100
    timeout = 300  # 5 minutes

    set_timeout(timeout)
    print(f"Timeout: {timeout}s")
    print(f"PyAV version: {av.__version__}")
    print(f"Stress test: {iterations} iterations with 640x480 300-frame video\n")

    video_path = tempfile.mktemp(suffix=".mp4", prefix="pyav_stress_")
    print("Creating large test video (300 frames, 640x480)...")
    create_test_video(video_path, n_frames=300, width=640, height=480)
    print("Done.\n")

    results = {}

    # Test 1: Exception path — leaked containers
    results["Exception + NO cleanup"] = run_scenario(
        "Exception during decode — container LEAKED (no try/finally)",
        lambda p: decode_leak_on_exception(p, fail_at_frame=5),
        video_path,
        iterations,
        expect_exception=True,
    )

    gc.collect()
    time.sleep(0.5)

    # Test 2: Exception path — properly closed containers
    results["Exception + WITH cleanup"] = run_scenario(
        "Exception during decode — container CLOSED (with statement)",
        lambda p: decode_safe_on_exception(p, fail_at_frame=5),
        video_path,
        iterations,
        expect_exception=True,
    )

    gc.collect()
    time.sleep(0.5)

    # Test 3: Happy path — forgot close
    results["Happy path + forgot close"] = run_scenario(
        "Happy path — forgot container.close() (relies on GC)",
        decode_forget_close,
        video_path,
        iterations,
    )

    gc.collect()
    time.sleep(0.5)

    # Test 4: Happy path — proper close
    results["Happy path + proper close"] = run_scenario(
        "Happy path — proper context manager close",
        decode_proper_close,
        video_path,
        iterations,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for label, growth in results.items():
        print(f"  {label:<40s}  growth={growth:+.1f} MB")

    if os.path.exists(video_path):
        os.remove(video_path)


if __name__ == "__main__":
    main()
