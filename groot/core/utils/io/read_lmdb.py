#!/usr/bin/env python3
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
import io
import json
import os
from typing import Any

import cv2
import lmdb
import numpy as np
from PIL import Image


class LMDBImageReader:
    """
    A class to efficiently read images from LMDB shards based on JSONL metadata.
    """

    def __init__(self, data_root: str):
        """
        Initialize the LMDB image reader.

        Args:
            data_root: Root directory where the actual data files are stored
        """
        # Cache opened LMDB environments to avoid repeatedly opening the same file
        self._lmdb_cache = {}
        self.data_root = data_root

    def _resolve_path(self, lmdb_file: str) -> str:
        """
        Resolve relative playground paths to absolute paths.

        Args:
            lmdb_file: Original path from JSONL (may be relative)

        Returns:
            Resolved absolute path
        """
        # If it's already an absolute path, return as-is
        if os.path.isabs(lmdb_file):
            return lmdb_file

        # If it starts with playground/, map it to the actual data root
        if lmdb_file.startswith("playground/"):
            # Remove 'playground/' prefix
            relative_path = lmdb_file[len("playground/") :]

            # Special handling for lmdb_database paths to avoid duplication
            # If the relative path starts with lmdb_database/ and data_root ends with lmdb_database,
            # strip the lmdb_database/ prefix to avoid duplication
            if relative_path.startswith("lmdb_database/") and self.data_root.endswith(
                "lmdb_database"
            ):
                relative_path = relative_path[len("lmdb_database/") :]

            resolved_path = os.path.join(self.data_root, relative_path)
            return resolved_path

        # For other relative paths, join with data_root
        return os.path.join(self.data_root, lmdb_file)

    def _get_lmdb_env(self, lmdb_file: str) -> lmdb.Environment | None:
        """
        Get or create an LMDB environment, with caching to avoid repeated opens.

        Args:
            lmdb_file: Path to the LMDB file (will be resolved if relative)

        Returns:
            LMDB environment or None if failed to open
        """
        # Resolve the path first
        resolved_path = self._resolve_path(lmdb_file)

        if resolved_path in self._lmdb_cache:
            return self._lmdb_cache[resolved_path]

        try:
            env = lmdb.open(
                resolved_path,
                readonly=True,
                lock=False,
                max_readers=10240,
                readahead=False,
                meminit=False,
            )
            self._lmdb_cache[resolved_path] = env
            return env
        except Exception as e:
            print(f"Failed to open LMDB file {resolved_path}. Error: {e}")
            return None

    def read_image_from_lmdb(self, image_data: dict[str, str]) -> Image.Image:
        """
        Read an image from LMDB given the metadata.

        Args:
            image_data: Dictionary containing 'lmdb_file' and 'lmdb_key'

        Returns:
            PIL Image

        Raises:
            ValueError: If image_data format is invalid
            FileNotFoundError: If LMDB file does not exist
            KeyError: If key not found in LMDB file
            Exception: If failed to read from LMDB or decode image data
        """
        if (
            not isinstance(image_data, dict)
            or "lmdb_file" not in image_data
            or "lmdb_key" not in image_data
        ):
            raise ValueError(f"Invalid image_data format: {image_data}")

        lmdb_file = image_data["lmdb_file"]
        lmdb_key = image_data["lmdb_key"]

        # Resolve the path
        resolved_path = self._resolve_path(lmdb_file)

        # Check if LMDB file exists
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"LMDB file does not exist: {resolved_path} (Original path: {lmdb_file})"
            )

        # Get LMDB environment
        env = self._get_lmdb_env(lmdb_file)
        if env is None:
            raise Exception(f"Failed to open LMDB environment for file: {resolved_path}")

        # Read image data from LMDB
        try:
            with env.begin(write=False) as txn:
                image_bin = txn.get(lmdb_key.encode("ascii"))
                if image_bin is None:
                    raise KeyError(f"Key {lmdb_key} not found in LMDB file {resolved_path}")

        except KeyError:
            raise  # Re-raise KeyError as-is
        except Exception as e:
            raise Exception(
                f"Failed to read from LMDB file {resolved_path}, key {lmdb_key}. Error: {e}"
            )

        # Convert binary data to PIL Image
        try:
            # Try direct PIL Image loading first (for JPEG/PNG data)
            buf = io.BytesIO(image_bin)
            image = Image.open(buf)
            image = image.convert("RGB")  # Ensure RGB format
            return image
        except Exception as e:
            try:
                # Fallback: decode using OpenCV and convert to PIL
                image_np = np.frombuffer(image_bin, dtype=np.uint8)
                image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise Exception(f"Failed to decode image data for key {lmdb_key}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
                return image
            except Exception as e2:
                raise Exception(
                    f"Failed to decode image for key {lmdb_key}. PIL error: {e}, CV2 error: {e2}"
                )

    def parse_image_data(self, image_data: Any) -> Image.Image:
        """
        Parse image data from various formats (file path, LMDB metadata, etc.).

        Args:
            image_data: Can be a string (file path), dict (LMDB metadata), or None

        Returns:
            PIL Image

        Raises:
            ValueError: If image_data is None or format is unsupported
            Exception: If failed to load the image
        """
        if image_data is None:
            raise ValueError("Image data cannot be None")
        elif isinstance(image_data, str):
            # Regular file path - also resolve relative paths
            resolved_path = self._resolve_path(image_data)
            try:
                image = Image.open(resolved_path).convert("RGB")
                return image
            except Exception as e:
                raise Exception(f"Failed to open image file {resolved_path}. Error: {e}")
        elif (
            isinstance(image_data, dict) and "lmdb_file" in image_data and "lmdb_key" in image_data
        ):
            # LMDB case
            return self.read_image_from_lmdb(image_data)
        else:
            raise ValueError(f"Unsupported image data format: {type(image_data)}")

    def close(self):
        """Close all cached LMDB environments."""
        for env in self._lmdb_cache.values():
            env.close()
        self._lmdb_cache.clear()


def save_image_to_disk(image: Image.Image, save_path: str, quality: int = 95) -> bool:
    """
    Save a PIL Image to disk.

    Args:
        image: PIL Image to save
        save_path: Full path where to save the image
        quality: JPEG quality (1-100)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save image
        image.save(save_path, "JPEG", quality=quality)
        print(f"    Saved image to: {save_path}")
        return True
    except Exception as e:
        print(f"    Failed to save image to {save_path}: {e}")
        return False


def process_jsonl_file(
    reader: LMDBImageReader,
    jsonl_path: str,
    max_samples: int | None = None,
    save_images: bool = True,
    output_dir: str = "./lmdb_images_output",
) -> None:
    """
    Process a JSONL file and demonstrate reading images from LMDB.

    Args:
        reader: LMDBImageReader instance to use for reading images
        jsonl_path: Path to the JSONL file
        max_samples: Maximum number of samples to process (None for all)
        save_images: Whether to save successfully loaded images to disk
        output_dir: Directory to save images to (if save_images is True)
    """
    if not os.path.exists(jsonl_path):
        print(f"JSONL file does not exist: {jsonl_path}")
        return

    print(f"Processing JSONL file: {jsonl_path}")
    print(f"Data root: {reader.data_root}")
    if save_images:
        print(f"Saving images: {save_images} → {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Images will be saved to: {os.path.abspath(output_dir)}")

    # Statistics
    total_samples = 0
    lmdb_samples = 0
    regular_samples = 0
    successful_reads = 0
    failed_reads = 0
    saved_images = 0
    lmdb_files = set()
    resolved_paths = {}

    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and total_samples >= max_samples:
                    break

                try:
                    sample = json.loads(line.strip())
                    total_samples += 1

                    # Look for image data
                    image_data = None
                    if "image" in sample:
                        image_data = sample["image"]
                    elif "data" in sample:
                        # Check interleaved format
                        for data_item in sample["data"]:
                            if data_item.get("type") == "image" and "image_list" in data_item:
                                image_data = data_item["image_list"][0]  # Take first image
                                break

                    if image_data is None:
                        print(f"Line {line_num}: No image data found")
                        continue

                    # Check if it's LMDB format
                    if isinstance(image_data, dict) and "lmdb_file" in image_data:
                        lmdb_samples += 1
                        original_path = image_data["lmdb_file"]
                        resolved_path = reader._resolve_path(original_path)

                        lmdb_files.add(original_path)
                        resolved_paths[original_path] = resolved_path

                        print(
                            f"Line {line_num}: LMDB image - file: {original_path}, key: {image_data['lmdb_key']}"
                        )
                        print(f"  Resolved to: {resolved_path}")

                        # Try to read the image
                        image = reader.read_image_from_lmdb(image_data)
                        if image is not None:
                            successful_reads += 1
                            print(f"  ✓ Successfully loaded image: {image.size} pixels")

                            # Save image if requested
                            if save_images:
                                save_name = f"line_{line_num:04d}_{image_data['lmdb_key']}"
                                save_path = os.path.join(output_dir, f"{save_name}.jpg")
                                if save_image_to_disk(image, save_path):
                                    saved_images += 1
                        else:
                            failed_reads += 1
                            print("  ✗ Failed to load image")
                    else:
                        regular_samples += 1
                        print(f"Line {line_num}: Regular image file: {image_data}")

                        # Also try to load regular files if save_images is enabled
                        if save_images and isinstance(image_data, str):
                            image = reader.parse_image_data(image_data)
                            if image is not None:
                                save_name = f"line_{line_num:04d}_regular"
                                save_path = os.path.join(output_dir, f"{save_name}.jpg")
                                if save_image_to_disk(image, save_path):
                                    saved_images += 1

                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON decode error: {e}")
                except Exception as e:
                    print(f"Line {line_num}: Unexpected error: {e}")

    except Exception as e:
        print(f"Error processing JSONL file: {e}")
        return

    # Print summary statistics
    print("\n=== Processing Summary ===")
    print(f"Total samples processed: {total_samples}")
    print(f"LMDB samples: {lmdb_samples}")
    print(f"Regular file samples: {regular_samples}")
    print(f"Successful LMDB reads: {successful_reads}")
    print(f"Failed LMDB reads: {failed_reads}")
    if save_images:
        print(f"Images saved to disk: {saved_images}")
    print(f"Unique LMDB files found: {len(lmdb_files)}")

    if lmdb_files:
        print("\nLMDB files status:")
        for original_path in sorted(lmdb_files):
            resolved_path = resolved_paths[original_path]
            status = "✓" if os.path.exists(resolved_path) else "✗ (missing)"
            print(f"  {status} {original_path}")
            if original_path != resolved_path:
                print(f"      → {resolved_path}")


def demo_single_image_read(
    lmdb_file: str,
    lmdb_key: str,
    data_root: str,
    save_image: bool = True,
    output_dir: str = "./lmdb_images_output",
) -> None:
    """
    Demonstrate reading a single image from LMDB.

    Args:
        lmdb_file: Path to LMDB file (can be relative)
        lmdb_key: Key of the image in LMDB
        data_root: Root directory for resolving relative paths
        save_image: Whether to save the loaded image to disk
        output_dir: Directory to save the image to
    """
    print("\n=== Demo: Reading single image ===")
    print(f"LMDB file: {lmdb_file}")
    print(f"LMDB key: {lmdb_key}")
    print(f"Data root: {data_root}")

    reader = LMDBImageReader(data_root=data_root)

    image_data = {"lmdb_file": lmdb_file, "lmdb_key": lmdb_key}

    image = reader.read_image_from_lmdb(image_data)

    if image:
        print(f"✓ Successfully loaded image: {image.size} pixels, mode: {image.mode}")

        if save_image:
            save_name = f"demo_{lmdb_key}"
            save_path = os.path.join(output_dir, f"{save_name}.jpg")
            if save_image_to_disk(image, save_path):
                print("✓ Image saved to disk")
            else:
                print("✗ Failed to save image to disk")
    else:
        print("✗ Failed to load image")

    reader.close()


def main():
    """Main function to demonstrate LMDB image reading."""
    print("=== LMDB Image Reader ===")

    # Configuration - modify these paths as needed
    jsonl_file = (
        "/mnt/amlfs-01/shared/vl_shared/users/yman/osmo-share/lmdb_jsonl/llava_instruct_150k.jsonl"
    )
    data_root = "/mnt/amlfs-01/shared/vl_shared/users/yman/osmo-share"
    output_dir = "./lmdb_images_output"

    # Check if JSONL file exists
    if not os.path.exists(jsonl_file):
        print(f"JSONL file not found: {jsonl_file}")
        print("Please update the jsonl_file variable with a valid path.")
        return

    # Create reader instance with data root mapping
    reader = LMDBImageReader(data_root=data_root)

    try:
        # Process JSONL file (limit to 10 samples for demo) and save images
        process_jsonl_file(
            reader, jsonl_file, max_samples=10, save_images=True, output_dir=output_dir
        )

        # Example of reading a single image with playground path
        print("\n=== Testing single image read ===")
        demo_single_image_read(
            "playground/lmdb_database/llava_instruct_150k/llava_instruct_150k_0.lmdb",
            "0000000000",
            data_root=data_root,
            save_image=True,
            output_dir=output_dir,
        )

        print("\n=== Check the images ===")
        print(f"All extracted images are saved in: {os.path.abspath(output_dir)}")
        if os.path.exists(output_dir):
            image_files = [
                f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            print(f"Total images saved: {len(image_files)}")
            if image_files:
                print("Sample files:")
                for f in sorted(image_files)[:5]:  # Show first 5 files
                    print(f"  {f}")
                if len(image_files) > 5:
                    print(f"  ... and {len(image_files) - 5} more")

    finally:
        # Clean up
        reader.close()


if __name__ == "__main__":
    main()
