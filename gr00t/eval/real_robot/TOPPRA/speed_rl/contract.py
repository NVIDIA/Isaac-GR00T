from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class FeatureContract:
    feature_dim: int
    model_id: str
    layer: str
    dtype: str = "float32"
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("Only speed-RL feature contract version 1 is supported")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if not self.model_id or not self.layer:
            raise ValueError("model_id and layer must be non-empty")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"Invalid feature dtype {self.dtype!r}") from exc
        if dtype.kind != "f":
            raise ValueError("Speed-RL features must use a floating-point dtype")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FeatureContract:
        required = ("version", "model_id", "layer", "feature_dim", "dtype")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"Speed-RL contract is missing fields: {missing}")
        return cls(
            version=int(value["version"]),
            model_id=str(value["model_id"]),
            layer=str(value["layer"]),
            feature_dim=int(value["feature_dim"]),
            dtype=str(value["dtype"]),
        )

    def validate_compatible(self, other: FeatureContract) -> None:
        if self != other:
            raise ValueError(
                "Speed-RL feature contract mismatch: "
                f"expected={self.to_dict()}, received={other.to_dict()}"
            )


def extract_candidate_features(
    response: Any,
    expected_contract: FeatureContract,
    candidate_horizons: Sequence[int],
) -> list[FloatArray]:
    if not isinstance(response, (tuple, list)) or len(response) != 2:
        raise ValueError("Speed-RL requires a policy response shaped as (actions, info)")
    info = response[1]
    if not isinstance(info, Mapping):
        raise ValueError("Speed-RL policy info must be a mapping")
    speed_info = info.get("speed_rl")
    if not isinstance(speed_info, Mapping):
        raise ValueError("Policy response is missing info['speed_rl']")
    raw_contract = speed_info.get("contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("Policy response is missing info['speed_rl']['contract']")
    expected_contract.validate_compatible(FeatureContract.from_mapping(raw_contract))

    features = np.asarray(speed_info.get("action_features"))
    if features.ndim != 3:
        raise ValueError(
            f"info['speed_rl']['action_features'] must have shape (K, H, D), got {features.shape}"
        )
    candidate_count, horizon, feature_dim = features.shape
    if candidate_count != len(candidate_horizons):
        raise ValueError(
            f"Feature candidate count {candidate_count} does not match actions "
            f"{len(candidate_horizons)}"
        )
    if feature_dim != expected_contract.feature_dim:
        raise ValueError(
            f"Feature dimension {feature_dim} does not match contract "
            f"{expected_contract.feature_dim}"
        )
    if any(int(candidate_horizon) != horizon for candidate_horizon in candidate_horizons):
        raise ValueError(
            f"Feature horizon {horizon} does not match action horizons {tuple(candidate_horizons)}"
        )
    expected_dtype = np.dtype(expected_contract.dtype)
    if features.dtype != expected_dtype:
        raise ValueError(f"Feature dtype {features.dtype} does not match contract {expected_dtype}")
    if np.any(~np.isfinite(features)):
        raise ValueError("Speed-RL action features contain non-finite values")
    return [np.asarray(candidate, dtype=np.float32).copy() for candidate in features]


def trim_candidate_features(
    features: Sequence[FloatArray],
    discarded_waypoints: int,
) -> list[FloatArray]:
    discard = int(discarded_waypoints)
    if discard < 0:
        raise ValueError("discarded_waypoints must be non-negative")
    trimmed: list[FloatArray] = []
    for candidate in features:
        array = np.asarray(candidate, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Candidate feature must have shape (H, D), got {array.shape}")
        if discard >= len(array):
            raise RuntimeError(
                "Speed-RL features became stale before planning: "
                f"discard={discard}, horizon={len(array)}"
            )
        trimmed.append(array[discard:].copy())
    return trimmed


def pool_candidate_feature(feature: FloatArray) -> FloatArray:
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim != 2 or len(array) < 1:
        raise ValueError(f"Candidate feature must have non-empty shape (H, D), got {array.shape}")
    pooled = np.mean(array, axis=0, dtype=np.float32)
    if np.any(~np.isfinite(pooled)):
        raise ValueError("Pooled Speed-RL feature contains non-finite values")
    return pooled
