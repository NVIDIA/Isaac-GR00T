# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import asdict, is_dataclass
from enum import Enum
import math
import random
from typing import Any

from albumentations.core.transforms_interface import DualTransform
import cv2
import numpy as np

from groot.core.training.configs.data.embodiment_configs import ModalityConfig


def apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    """Apply sin/cos encoding to values.

    Args:
        values: Array of shape (..., D) containing values to encode

    Returns:
        Array of shape (..., 2*D) with [sin, cos] concatenated

    Note: This DOUBLES the dimension. For example:
        Input:  [v₁, v₂, v₃] with shape (..., 3)
        Output: [sin(v₁), sin(v₂), sin(v₃), cos(v₁), cos(v₂), cos(v₃)] with shape (..., 6)
    """
    sin_values = np.sin(values)
    cos_values = np.cos(values)
    # Concatenate sin and cos: [sin(v1), sin(v2), ..., cos(v1), cos(v2), ...]
    return np.concatenate([sin_values, cos_values], axis=-1)


def nested_dict_to_numpy(data):
    """Recursively converts bottom-level list of lists to NumPy arrays.

    Args:
        data: A nested dictionary where bottom nodes are list of lists,
              and parent nodes are strings (keys)

    Returns:
        The same dictionary structure with bottom-level lists converted to NumPy arrays

    Example:
        >>> data = {"a": {"b": [[0, 1], [2, 3]]}}
        >>> result = nested_dict_to_numpy(data)
        >>> print(result["a"]["b"])
        [[0 1]
         [2 3]]
    """
    if isinstance(data, dict):
        return {key: nested_dict_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Convert lists to numpy arrays
        # NumPy will handle both 1D and 2D cases appropriately
        return np.array(data)
    else:
        return data


def _align_stats_bounds(values: np.ndarray, stats: np.ndarray):
    if stats.ndim == 2:
        if values.ndim < 2:
            raise ValueError(
                f"Expected values with at least 2 dims for 2D bounds, got {values.shape}"
            )
        if values.shape[-1] != stats.shape[-1]:
            raise ValueError(
                f"Feature dim mismatch: values {values.shape[-1]} vs stats {stats.shape[1]}"
            )
        value_time = values.shape[-2]
        stats_time = stats.shape[-2]
        if value_time <= stats_time:
            # Truncate stats to match value time
            stats = stats[..., :value_time, :]
    return stats


def normalize_values_minmax(values, params):
    """Normalize values using min-max normalization to [-1, 1] range.

    Args:
        values: Input values to normalize
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Can handle 2D or 3D arrays where last axis represents features
        params: Dictionary with "min" and "max" keys
            - params["min"]: Minimum values for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
            - params["max"]: Maximum values for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
        joint_group: Optional indexing for joint groups (legacy parameter)

    Returns:
        Normalized values in [-1, 1] range
            - Same shape as input values: (T, D) or (B, T, D)
            - Values are linearly mapped from [min, max] to [-1, 1]
            - For features where min == max, normalized value is 0

    Examples:
        # 1D bounds - same normalization for all steps
        values: (10, 5), params["min"]: (5,), params["max"]: (5,)

        # 2D bounds - per-step normalization
        values: (8, 4), params["min"]: (8, 4), params["max"]: (8, 4)
    """
    max_vals = _align_stats_bounds(values, params["max"])
    min_vals = _align_stats_bounds(values, params["min"])
    normalized = np.zeros_like(values)

    mask = ~np.isclose(max_vals, min_vals)

    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1

    return normalized


def unnormalize_values_minmax(normalized_values, params, clip: bool = True):
    """Min-max unnormalization from [-1, 1] range back to original range.

    Args:
        normalized_values: Normalized input values in [-1, 1] range
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Values outside [-1, 1] are automatically clipped
        params: Dictionary with "min" and "max" keys
            - params["min"]: Original minimum values used for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
            - params["max"]: Original maximum values used for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
        clip: Clamp ``normalized_values`` to [-1, 1] before the affine inverse (default True).
            With False, values past +/-1 decode past the [min, max] band.

    Returns:
        Unnormalized values in original range [min, max]
            - Same shape as input normalized_values: (T, D) or (B, T, D)
            - Values are linearly mapped from [-1, 1] back to [min, max]
            - Input values are clipped to [-1, 1] before unnormalization

    Examples:
        # 1D bounds - same unnormalization for all steps
        normalized_values: (10, 5), params["min"]: (5,), params["max"]: (5,)

        # 2D bounds - per-step unnormalization
        normalized_values: (8, 4), params["min"]: (8, 4), params["max"]: (8, 4)
    """
    max_vals = _align_stats_bounds(normalized_values, params["max"])
    min_vals = _align_stats_bounds(normalized_values, params["min"])
    range_vals = max_vals - min_vals

    # Unnormalize from [-1, 1]. With clip=False the affine inverse is applied as-is, so a
    # value past +/-1 decodes past the min/max (or q01/q99) band instead of being cut to it.
    if clip:
        normalized_values = np.clip(normalized_values, -1.0, 1.0)
    unnormalized = (normalized_values + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def normalize_values_meanstd(values, params):
    """Normalize values using mean-std (z-score) normalization.

    Args:
        values: Input values to normalize
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Can handle 2D or 3D arrays where last axis represents features
        params: Dictionary with "mean" and "std" keys
            - params["mean"]: Mean values for normalization
                * Case 1 - 1D params: Shape (D,) - same mean for all steps
                * Case 2 - 2D params: Shape (T, D) - different mean per step
            - params["std"]: Standard deviation values for normalization
                * Case 1 - 1D params: Shape (D,) - same std for all steps
                * Case 2 - 2D params: Shape (T, D) - different std per step

    Returns:
        Normalized values using z-score normalization
            - Same shape as input values: (T, D) or (B, T, D)
            - Values are transformed as: (x - mean) / std
            - For features where std == 0, normalized value equals original value

    Examples:
        # 1D params - same normalization for all steps
        values: (10, 5), params["mean"]: (5,), params["std"]: (5,)

        # 2D params - per-step normalization
        values: (8, 4), params["mean"]: (8, 4), params["std"]: (8, 4)
    """
    mean_vals = _align_stats_bounds(values, params["mean"])
    std_vals = _align_stats_bounds(values, params["std"])

    # Create mask for non-zero standard deviations
    mask = std_vals != 0

    # Initialize normalized array
    normalized = np.zeros_like(values)

    # Normalize only features with non-zero std
    normalized[..., mask] = (values[..., mask] - mean_vals[..., mask]) / std_vals[..., mask]

    # Keep original values for zero-std features
    normalized[..., ~mask] = values[..., ~mask]

    return normalized


def unnormalize_values_meanstd(normalized_values, params):
    """Mean-std unnormalization (reverse z-score normalization).

    Args:
        normalized_values: Normalized input values (z-scores)
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Can handle 2D or 3D arrays where last axis represents features
        params: Dictionary with "mean" and "std" keys
            - params["mean"]: Original mean values used for normalization
                * Case 1 - 1D params: Shape (D,) - same mean for all steps
                * Case 2 - 2D params: Shape (T, D) - different mean per step
            - params["std"]: Original standard deviation values used for normalization
                * Case 1 - 1D params: Shape (D,) - same std for all steps
                * Case 2 - 2D params: Shape (T, D) - different std per step

    Returns:
        Unnormalized values in original scale
            - Same shape as input normalized_values: (T, D) or (B, T, D)
            - Values are transformed as: x * std + mean
            - For features where std == 0, unnormalized value equals normalized value

    Examples:
        # 1D params - same unnormalization for all steps
        normalized_values: (10, 5), params["mean"]: (5,), params["std"]: (5,)

        # 2D params - per-step unnormalization
        normalized_values: (8, 4), params["mean"]: (8, 4), params["std"]: (8, 4)
    """
    mean_vals = _align_stats_bounds(normalized_values, params["mean"])
    std_vals = _align_stats_bounds(normalized_values, params["std"])

    # Create mask for non-zero standard deviations
    mask = std_vals != 0

    # Initialize unnormalized array
    unnormalized = np.zeros_like(normalized_values)

    # Unnormalize only features with non-zero std
    unnormalized[..., mask] = (
        normalized_values[..., mask] * std_vals[..., mask] + mean_vals[..., mask]
    )

    # Keep normalized values for zero-std features
    unnormalized[..., ~mask] = normalized_values[..., ~mask]

    return unnormalized


def to_json_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses and numpy arrays to JSON-serializable format.

    Args:
        obj: Object to convert (can be dataclass, numpy array, dict, list, etc.)

    Returns:
        JSON-serializable representation of the object
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict, then recursively process the dict
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, np.integer):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floats to Python float
        return float(obj)
    elif isinstance(obj, np.bool_):
        # Convert numpy bool to Python bool
        return bool(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list | tuple):
        # Recursively process list/tuple elements
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert set to list
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, str | int | float | bool | type(None)):
        # Already JSON-serializable
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    else:
        # For other types, try to convert to string as fallback
        # You might want to handle specific types differently
        return str(obj)


def parse_modality_configs(
    modality_configs: dict[str, dict[str, ModalityConfig]],
) -> dict[str, dict[str, ModalityConfig]]:
    parsed_modality_configs = {}
    for embodiment_tag, modality_config in modality_configs.items():
        parsed_modality_configs[embodiment_tag] = {}
        for modality, config in modality_config.items():
            if isinstance(config, dict):
                parsed_modality_configs[embodiment_tag][modality] = ModalityConfig(**config)
            else:
                parsed_modality_configs[embodiment_tag][modality] = config
    return parsed_modality_configs


class RandomResizedCropSameSize(DualTransform):
    """Random Resized Crop that keeps the output size equal to the *current* image size.

    This transform handles images of different original resolutions while being compatible
    with Albumentations ReplayMode by using deterministic random parameters.

    Args:
        scale (Tuple[float, float]): Range for the area of the crop relative to the input area.
        ratio (Tuple[float, float]): Range of aspect ratio of the crop (w/h).
        interpolation (int): cv2 interpolation for images.
        mask_interpolation (int): cv2 interpolation for masks.
        p (float): Albumentations probability.

    Notes:
        - BBoxes are assumed to be in 'pascal_voc' format if provided to Compose
          (i.e., (x_min, y_min, x_max, y_max)). This transform updates them accordingly.
        - Keypoints are assumed to be in 'xy' format (x, y[, ...]) and will be scaled/shifted;
          any additional fields (angle, scale, etc.) are passed through unchanged.
        - For ReplayMode compatibility, this transform generates deterministic random parameters
          that are then applied consistently across all targets.
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] | None = None,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        assert 0 < scale[0] <= scale[1] <= 1.0
        if ratio is not None:
            assert ratio[0] > 0 and ratio[1] > 0
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def get_params(self):
        """Generate random parameters that are independent of the input image.

        These will be used consistently across all targets in ReplayMode.
        """
        # Generate random scale factor
        scale_factor = random.uniform(self.scale[0], self.scale[1])

        # Generate random aspect ratio
        if self.ratio is not None:
            log_ratio_min = math.log(self.ratio[0])
            log_ratio_max = math.log(self.ratio[1])
            aspect_ratio = math.exp(random.uniform(log_ratio_min, log_ratio_max))
        else:
            # Will be determined based on input image aspect ratio
            aspect_ratio = None

        # Generate random position factors (0-1 range)
        x_factor = random.random()
        y_factor = random.random()

        return {
            "scale_factor": scale_factor,
            "aspect_ratio": aspect_ratio,
            "x_factor": x_factor,
            "y_factor": y_factor,
        }

    @property
    def targets_as_params(self):
        # We need the current image size to compute actual crop params
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        """Compute image-dependent parameters using the random factors from get_params().

        This ensures ReplayMode compatibility while handling different image sizes.
        """
        img = params["image"]
        h, w = img.shape[:2]

        # Get the random factors that were generated in get_params()
        # These will be consistent across replay
        scale_factor = params.get("scale_factor", random.uniform(self.scale[0], self.scale[1]))
        aspect_ratio = params.get("aspect_ratio")
        x_factor = params.get("x_factor", random.random())
        y_factor = params.get("y_factor", random.random())

        # If aspect_ratio is None, use the image's aspect ratio
        if aspect_ratio is None:
            aspect_ratio = w / h

        area = h * w

        # Try to compute crop dimensions
        target_area = area * scale_factor
        new_w = round(math.sqrt(target_area * aspect_ratio))
        new_h = round(math.sqrt(target_area / aspect_ratio))

        # Ensure crop fits within image bounds
        if new_w > w or new_h > h or new_w <= 0 or new_h <= 0:
            # Fallback: use image-dependent aspect ratio constraints
            if self.ratio is not None:
                in_ratio = w / h
                if in_ratio < self.ratio[0]:
                    new_w = w
                    new_h = round(new_w / self.ratio[0])
                elif in_ratio > self.ratio[1]:
                    new_h = h
                    new_w = round(new_h * self.ratio[1])
                else:
                    new_w, new_h = w, h
            else:
                new_w, new_h = w, h

        # Clamp to valid range
        new_w = max(1, min(new_w, w))
        new_h = max(1, min(new_h, h))

        # Compute crop position using the random factors
        max_x = max(0, w - new_w)
        max_y = max(0, h - new_h)
        x1 = int(x_factor * max_x)
        y1 = int(y_factor * max_y)

        return {
            "x1": x1,
            "y1": y1,
            "crop_w": new_w,
            "crop_h": new_h,
            "orig_w": w,
            "orig_h": h,
        }

    def apply(
        self, img: np.ndarray, x1=0, y1=0, crop_w=0, crop_h=0, orig_w=0, orig_h=0, **_params
    ) -> np.ndarray:
        cropped = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=self.interpolation)

    def apply_to_mask(
        self, mask: np.ndarray, x1=0, y1=0, crop_w=0, crop_h=0, orig_w=0, orig_h=0, **_params
    ) -> np.ndarray:
        cropped = mask[y1 : y1 + crop_h, x1 : x1 + crop_w]
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=self.mask_interpolation)

    def apply_to_bbox(
        self, bbox: tuple[float, float, float, float], **params
    ) -> tuple[float, float, float, float]:
        """Assumes 'pascal_voc' format: (x_min, y_min, x_max, y_max)."""
        x1, y1 = params["x1"], params["y1"]
        crop_w, crop_h = params["crop_w"], params["crop_h"]
        orig_w, orig_h = params["orig_w"], params["orig_h"]

        bx1, by1, bx2, by2 = bbox

        # Shift to crop coordinates
        bx1 -= x1
        bx2 -= x1
        by1 -= y1
        by2 -= y1

        # Clip to crop
        bx1 = np.clip(bx1, 0, crop_w)
        bx2 = np.clip(bx2, 0, crop_w)
        by1 = np.clip(by1, 0, crop_h)
        by2 = np.clip(by2, 0, crop_h)

        # If fully outside, return a degenerate bbox (Albumentations will later filter small/empty bboxes)
        if bx2 <= bx1 or by2 <= by1:
            return (0.0, 0.0, 0.0, 0.0)

        # Scale back to original size
        sx = orig_w / float(crop_w)
        sy = orig_h / float(crop_h)
        bx1 *= sx
        bx2 *= sx
        by1 *= sy
        by2 *= sy
        return (bx1, by1, bx2, by2)

    def apply_to_keypoint(self, keypoint: tuple[float, float, Any, Any], **params):
        """Assumes 'xy' keypoints: (x, y[, ...]).

        Any extra fields are preserved.
        """
        x, y, *rest = keypoint
        x1, y1 = params["x1"], params["y1"]
        crop_w, crop_h = params["crop_w"], params["crop_h"]
        orig_w, orig_h = params["orig_w"], params["orig_h"]

        # Shift into crop frame
        x -= x1
        y -= y1

        # Scale to original size
        sx = orig_w / float(crop_w)
        sy = orig_h / float(crop_h)
        x *= sx
        y *= sy
        return (x, y, *rest)

    def get_transform_init_args_names(self):
        return ("scale", "ratio", "interpolation", "mask_interpolation")
