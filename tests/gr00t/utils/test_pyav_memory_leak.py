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

"""
PyAV Memory Leak Reproduction & Comparison Test

This script tests 3 scenarios:
  1. Current code pattern (no proper resource cleanup)
  2. Upgraded PyAV version (same code, newer library)
  3. Fixed code pattern (with proper resource cleanup via context managers / try-finally)

Usage:
  python tests/test_pyav_memory_leak.py [--iterations 200] [--video-path path/to/video.mp4]

It will generate a test video if none is provided, then run each scenario
and report RSS memory growth over repeated open/decode/close cycles.
"""

import argparse
import gc
import os
import resource
import signal
import tempfile
import time

import av
import numpy as np


# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------


def _timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded time limit")


def set_timeout(seconds: int):
    """Set a global timeout (Linux only). Raises TimeoutError when exceeded."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_rss_mb() -> float:
    """Return current RSS in MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB


def create_test_video(
    path: str, n_frames: int = 60, width: int = 640, height: int = 480, fps: int = 30
):
    """Create a synthetic test video for benchmarking."""
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height

    for i in range(n_frames):
        # Create a frame with some varying content
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"  Created test video: {path} ({n_frames} frames, {width}x{height})")


# ---------------------------------------------------------------------------
# Scenario 1: Current code pattern (matches codebase — no with / try-finally)
# ---------------------------------------------------------------------------


def decode_no_cleanup(video_path: str):
    """Mimics current codebase pattern: open -> decode -> close, no exception safety."""
    container = av.open(video_path)
    _ = container.streams.video[0]
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return len(frames)


def encode_no_cleanup(output_path: str, frames: list[np.ndarray], fps: int = 30):
    """Mimics current codebase VideoRecorder/VideoWriter: open -> encode -> close, no exception safety."""
    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    h, w = frames[0].shape[:2]
    stream.width = w
    stream.height = h

    for img in frames:
        vf = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ---------------------------------------------------------------------------
# Scenario 3: Fixed code pattern (with context manager / try-finally)
# ---------------------------------------------------------------------------


def decode_with_cleanup(video_path: str):
    """Fixed pattern: use av.open as context manager for guaranteed cleanup."""
    with av.open(video_path) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return len(frames)


def encode_with_cleanup(output_path: str, frames: list[np.ndarray], fps: int = 30):
    """Fixed pattern: use context manager for encoding."""
    with av.open(output_path, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        h, w = frames[0].shape[:2]
        stream.width = w
        stream.height = h

        for img in frames:
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(label: str, decode_fn, encode_fn, video_path: str, iterations: int):
    """Run decode+encode cycles and track memory."""
    print(f"\n{'=' * 60}")
    print(f"  Scenario: {label}")
    print(f"  Iterations: {iterations}")
    print(f"{'=' * 60}")

    gc.collect()
    rss_start = get_rss_mb()
    rss_samples = [rss_start]

    tmp_dir = tempfile.mkdtemp(prefix="pyav_bench_")

    t0 = time.time()
    for i in range(iterations):
        # Decode
        _ = decode_fn(video_path)

        # Encode (write a small video each iteration)
        dummy_frames = [np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(10)]
        out_path = os.path.join(tmp_dir, f"out_{i}.mp4")
        encode_fn(out_path, dummy_frames)

        # Remove temp file immediately
        if os.path.exists(out_path):
            os.remove(out_path)

        if (i + 1) % 20 == 0:
            gc.collect()
            rss_now = get_rss_mb()
            rss_samples.append(rss_now)
            elapsed = time.time() - t0
            print(
                f"  iter {i + 1:4d}/{iterations}  RSS={rss_now:.1f} MB  (delta={rss_now - rss_start:+.1f} MB)  elapsed={elapsed:.1f}s"
            )

    gc.collect()
    rss_end = get_rss_mb()
    elapsed = time.time() - t0

    # Cleanup temp dir
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    print(
        f"\n  Result: RSS start={rss_start:.1f} MB -> end={rss_end:.1f} MB  "
        f"(growth={rss_end - rss_start:+.1f} MB)  time={elapsed:.1f}s"
    )
    print(f"  RSS samples: {[f'{s:.1f}' for s in rss_samples]}")

    return {
        "label": label,
        "rss_start": rss_start,
        "rss_end": rss_end,
        "rss_growth": rss_end - rss_start,
        "rss_samples": rss_samples,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PyAV memory leak benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Number of decode/encode cycles")
    parser.add_argument(
        "--video-path", type=str, default=None, help="Path to test video (created if not provided)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Global timeout in seconds (0 to disable)"
    )
    args = parser.parse_args()

    if args.timeout > 0:
        set_timeout(args.timeout)
        print(f"Timeout: {args.timeout}s")

    print(f"PyAV version: {av.__version__}")
    print(f"Iterations: {args.iterations}")

    # Create or use test video
    if args.video_path and os.path.exists(args.video_path):
        video_path = args.video_path
        print(f"Using existing video: {video_path}")
    else:
        video_path = tempfile.mktemp(suffix=".mp4", prefix="pyav_test_")
        print("Creating test video...")
        create_test_video(video_path, n_frames=90, width=320, height=240)

    results = []

    # Scenario 1: Current code (no cleanup)
    r1 = run_benchmark(
        label="Current code (no with/try-finally)",
        decode_fn=decode_no_cleanup,
        encode_fn=encode_no_cleanup,
        video_path=video_path,
        iterations=args.iterations,
    )
    results.append(r1)

    # Force GC between scenarios
    gc.collect()
    time.sleep(1)

    # Scenario 2: Fixed code (with context manager)
    r2 = run_benchmark(
        label="Fixed code (with context manager)",
        decode_fn=decode_with_cleanup,
        encode_fn=encode_with_cleanup,
        video_path=video_path,
        iterations=args.iterations,
    )
    results.append(r2)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['label']:<45s}  growth={r['rss_growth']:+.1f} MB  time={r['elapsed']:.1f}s")

    print("\nNote: Run this script twice — once with av==15.0.0 and once with")
    print("upgraded av to compare version impact vs code fix impact.")

    # Cleanup test video
    if not args.video_path and os.path.exists(video_path):
        os.remove(video_path)


if __name__ == "__main__":
    main()
