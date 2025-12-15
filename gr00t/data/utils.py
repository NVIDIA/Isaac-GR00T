from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from gr00t.configs.data.embodiment_configs import ModalityConfig


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
    """
    Recursively converts bottom-level list of lists to NumPy arrays.

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


def normalize_values_minmax(values, params):
    """
    Normalize values using min-max normalization to [-1, 1] range.

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
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)

    mask = ~np.isclose(max_vals, min_vals)

    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1

    return normalized


def unnormalize_values_minmax(normalized_values, params):
    """
    Min-max unnormalization from [-1, 1] range back to original range.

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

    min_vals = params["min"]
    max_vals = params["max"]
    range_vals = max_vals - min_vals

    # Unnormalize from [-1, 1]
    unnormalized = (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def normalize_values_meanstd(values, params):
    """
    Normalize values using mean-std (z-score) normalization.

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
    mean_vals = params["mean"]
    std_vals = params["std"]

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
    """
    Mean-std unnormalization (reverse z-score normalization).

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
    mean_vals = params["mean"]
    std_vals = params["std"]

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
    """
    Recursively convert dataclasses and numpy arrays to JSON-serializable format.

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
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert set to list
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
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
