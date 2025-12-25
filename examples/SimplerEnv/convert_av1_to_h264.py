#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess


VIDEO_EXTS = {".mp4", ".mov", ".mkv"}


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def is_av1(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        print(f"[ffprobe FAIL] {path}: {proc.stderr.strip()}")
        return False
    codec = proc.stdout.strip()
    return codec in ("av01", "av1")


def convert_file(path: Path):
    if not is_av1(path):
        print(f"[SKIP] {path}")
        return

    tmp = path.with_suffix(path.suffix + ".mp4")
    print(f"[CONVERT] {path} -> {tmp}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-vsync",
        "passthrough",
        "-copyts",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        str(tmp),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        print(f"[ffmpeg FAIL] {path}: {proc.stderr.strip()}")
        if tmp.exists():
            tmp.unlink()
        return

    tmp.replace(path)
    print(f"[DONE] {path}")


def find_videos(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in VIDEO_EXTS:
                yield p


def main():
    ap = argparse.ArgumentParser(
        description="Recursively convert AV1 videos to H.264 (lossless-ish) in place."
    )
    ap.add_argument("root", nargs="?", default=".", help="Root directory (default: .)")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel workers (default: CPU count)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = list(find_videos(root))
    print(f"Scanning {root}, found {len(files)} candidate video files")

    if not files:
        return

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for p in files:
            ex.submit(convert_file, p)


if __name__ == "__main__":
    main()
