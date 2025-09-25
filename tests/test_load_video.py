#!/usr/bin/env python3
"""
Script to extract and save a video frame at a specified time using different backends.
"""

import argparse
import os

from PIL import Image

from gr00t.utils.video import get_frames_by_timestamps


def save_frames_at_time(
    video_path: str,
    time_seconds: float,
    output_dir: str = "frame_outputs",
    video_backend: str = "decord",
):
    """Extract and save frames from a video at a specific time using different backends."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get frame using torchvision_av backend
    print(f"Extracting frame at {time_seconds} seconds using {video_backend}...")

    frame_torchvision = get_frames_by_timestamps(
        video_path, [time_seconds], video_backend=video_backend
    )[
        0
    ]  # Get first (and only) frame

    print(f"{video_backend} frame shape: {frame_torchvision.shape}")

    # Save frame
    output_path = os.path.join(output_dir, f"frame_{video_backend}_{time_seconds}s.png")
    Image.fromarray(frame_torchvision).save(output_path)
    print(f"Saved {video_backend} frame to: {output_path}")


def main():
    import sys

    parser = argparse.ArgumentParser(
        description="Extract and save video frame at a specific time using different backends"
    )
    parser.add_argument(
        "--video-path", type=str, help="Path to the video file", default="labeled_frames_video.mp4"
    )
    parser.add_argument(
        "--times",
        type=float,
        help="Time in seconds to extract the frame",
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.99],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="frame_outputs",
        help="Directory to save output frames (default: frame_outputs)",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="torchcodec",
        help="Video backend to use (default: torchcodec)",
    )
    args = parser.parse_args()

    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        sys.exit(1)

    # Save frames
    for time in args.times:
        save_frames_at_time(args.video_path, time, args.output_dir, args.video_backend)


if __name__ == "__main__":
    main()
