# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unified processor for robot state and action data.

Handles:
- State normalization (min/max, mean/std, sin/cos encoding)
- Action normalization
- Absolute <-> Relative action representation conversion
- Action processing with state dependency
"""

from copy import deepcopy
import logging

import numpy as np
from scipy.spatial.transform import Rotation

from groot.core.data.state_action.action_chunking import JointActionChunk
from groot.core.data.state_action.pose import EndEffectorPose, JointPose, invert_transformation
from groot.core.data.utils import (
    apply_sin_cos_encoding,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
from groot.core.training.configs.data.embodiment_configs import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from groot.core.utils.state_action.rotation import (
    gram_schmidt,
    rot_6d_to_rot_mat,
    rot_mat_to_rot_6d,
)

logger = logging.getLogger(__name__)


class StateActionProcessor:
    """Unified processor for robot state and action data.

    Handles:
    - State normalization (min/max, mean/std, sin/cos encoding)
    - Action normalization
    - Absolute <-> Relative action representation conversion
    - Action processing with state dependency
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
        use_q99_for_rel_action: bool | None = None,
        skip_inference_clip: bool = False,
    ):
        """Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
                where modality in ["state", "action"]
                Example: {"gr1_unified": {"state": ModalityConfig(...), "action": ModalityConfig(...)}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
                where modality in ["state", "action", "relative_action"]
                and stat_type in ["min", "max", "mean", "std", "q01", "q99"]
                Example: {"gr1_unified": {"state": {"left_arm": {"min": [...], "max": [...], ...}}}}
            use_percentiles: Whether to use percentiles (q01/q99) instead of min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Global flag to enable sin/cos encoding for states
            use_relative_action: Whether to override absolute action stats with relative-action stats.
            use_q99_for_rel_action: Whether relative-action stats should route q01/q99
                into the min/max slots when use_percentiles is enabled. None preserves
                the current default behavior (match use_percentiles).
            skip_inference_clip: Decode (unnormalize) WITHOUT clamping to [-1, 1] first.
                Off by default, so decoding always clamps regardless of clip_outliers. Set
                (via action_cfg in the checkpoint's processor config) for models trained
                with clip_outliers off whose outputs are meant to exceed the q01/q99 band.
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.skip_inference_clip = skip_inference_clip
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.use_q99_for_rel_action = (
            use_percentiles if use_q99_for_rel_action is None else use_q99_for_rel_action
        )

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        # Format: norm_params[embodiment_tag][modality][joint_group][stat_type]
        # where stat_type in ["min", "max", "mean", "std", "dim"]

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization.

        Args:
            statistics: Nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
            override: Whether to replace existing statistics for duplicate embodiment tags.
        """
        for key, value in statistics.items():
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(value)
            else:
                logger.info("Embodiment tag %s already in statistics, skipping updating", key)
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics for all embodiments and modalities."""
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    # Compute range, ensuring it's not zero
                    range_vals = max_vals - min_vals
                    range_vals = np.maximum(range_vals, 1e-8)

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[0]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            # Override absolute action stats with relative stats where specified
            if embodiment_tag not in self.modality_configs:
                logger.warning(
                    f"Embodiment tag '{embodiment_tag}' found in statistics but not in "
                    f"modality_configs — skipping relative action override."
                )
                continue
            if "action" in self.modality_configs[embodiment_tag]:
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs, strict=False):
                        if (
                            action_config.rep == ActionRepresentation.RELATIVE
                            and self.use_relative_action
                        ):
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            rel_stats = nested_dict_to_numpy(
                                self.statistics[embodiment_tag]["relative_action"][key]
                            )
                            # Route q01/q99 into the "min"/"max" slots that downstream
                            # normalize_values_minmax reads whenever use_percentiles is
                            # enabled — matching the absolute-action path. Pre-fix
                            # checkpoints (trained before this routing became
                            # unconditional, when the rel-action override silently used
                            # raw min/max) must be migrated with
                            # groot/core/training/scripts/patch_eval/patch_rel_stats_q99.py
                            # or they will normalize relative actions incorrectly at eval.
                            if self.use_percentiles and self.use_q99_for_rel_action:
                                rel_min = rel_stats["q01"]
                                rel_max = rel_stats["q99"]
                            else:
                                rel_min = rel_stats["min"]
                                rel_max = rel_stats["max"]
                            # If absolute stats were empty (dim=0), derive dim from relative stats.
                            # Relative stats may be (D,) or (H, D); use the last axis as feature dim.
                            if action_dim.item() == 0:
                                action_dim = np.array(
                                    rel_max.shape[-1] if rel_max.ndim > 1 else rel_max.shape[0]
                                )
                            self.norm_params[embodiment_tag]["action"][key] = {
                                "min": rel_min,
                                "max": rel_max,
                                "mean": rel_stats["mean"],
                                "std": rel_stats["std"],
                                "dim": action_dim,
                            }

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1_unified")

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)
        """
        normalized_values = {}
        state = deepcopy(state)  # Avoid modifying input

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding (doubles dimension)
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])

            # Strategy 2: Mean/std normalization
            elif (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Min/max normalization to [-1, 1]
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """Reverse state processing (denormalization).

        Args:
            state: Dict mapping joint_group -> processed state values
            embodiment_tag: Embodiment identifier

        Returns:
            Dict mapping joint_group -> raw state values

        Raises:
            ValueError: If attempting to reverse sin/cos encoding (not reversible)
        """
        unnormalized_values = {}

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Sin/cos encoding is not reversible
            if sin_cos_keys and joint_group in sin_cos_keys:
                raise ValueError(
                    f"Cannot unapply sin/cos encoding for joint group '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. This transformation is not reversible."
                )

            # Reverse mean/std normalization
            if (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized = unnormalize_values_meanstd(state[joint_group], params)
                unnormalized_values[joint_group] = unnormalized

            # Reverse min/max normalization
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized_values[joint_group] = unnormalize_values_minmax(
                    state[joint_group], params, clip=not self.skip_inference_clip
                )

        return unnormalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply action processing (absolute->relative conversion, normalization).

        Processing order:
        1. Convert absolute actions to relative (if configured)
        2. Normalize actions

        Args:
            action: Dict mapping joint_group -> raw action values
                Shape per group: (T, D) where T is action horizon, D is action dimension
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) where last timestep is used as reference

        Returns:
            Dict mapping joint_group -> processed action values
                Shape per group: (T, D)

        Raises:
            ValueError: If state is None but required for relative action conversion
        """
        action = deepcopy(action)  # Avoid modifying input

        # Step 1: Convert absolute actions to relative (if needed)
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs, strict=False):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative action processing of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    # Use last state as reference frame
                    reference_state = state[state_key][-1]

                    # Convert absolute to relative
                    action[key] = self._convert_to_relative_action(
                        action=action[key],
                        reference_state=reference_state,
                        action_type=action_config.type,
                        action_format=action_config.format,
                    )

        # Step 2: Normalize actions
        normalized_values = {}
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                normalized = normalize_values_meanstd(group_values, params)
            else:
                normalized = normalize_values_minmax(group_values, params)

            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Reverse action processing (denormalization, relative->absolute conversion).

        Processing order:
        1. Denormalize actions
        2. Convert relative actions to absolute (if configured)

        Args:
            action: Dict mapping joint_group -> processed action values
                Shape per group: (T, D) or (B, T, D) for batched
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) or (B, T_state, D) for batched

        Returns:
            Dict mapping joint_group -> raw absolute action values
                Shape per group: (T, D) or (B, T, D) for batched

        Raises:
            ValueError: If state is None but required for relative->absolute conversion
        """
        # Step 1: Unnormalize actions
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else:
                unnormalized = unnormalize_values_minmax(
                    group_values, params, clip=not self.skip_inference_clip
                )

            unnormalized_values[joint_group] = unnormalized

        # Step 2: Convert relative actions to absolute (if needed)
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs, strict=False):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    relative_action = unnormalized_values[key]

                    # Handle batched and unbatched cases
                    is_batched = relative_action.ndim == 3
                    if not is_batched:
                        assert relative_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]
                        relative_action = relative_action[None, :]
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]

                    # Convert batched relative actions to absolute
                    absolute_actions = []
                    for s, a in zip(reference_state, relative_action, strict=True):
                        # Use last timestep of state as reference
                        absolute_action = self._convert_to_absolute_action(
                            action=a,
                            reference_state=s[-1],
                            action_type=action_config.type,
                            action_format=action_config.format,
                        )
                        absolute_actions.append(absolute_action)

                    if is_batched:
                        unnormalized_values[key] = np.stack(absolute_actions, axis=0)
                    else:
                        unnormalized_values[key] = absolute_actions[0]

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Apply both state and action processing together.

        Convenience method that processes state and action in one call,
        automatically passing raw state to action processor for relative conversion.

        Args:
            state: Dict mapping joint_group -> raw state values
            action: Dict mapping joint_group -> raw action values
            embodiment_tag: Embodiment identifier

        Returns:
            Tuple of (processed_state, processed_action)
        """
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def unapply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Reverse both state and action processing together.

        Args:
            state: Dict mapping joint_group -> processed state values
            action: Dict mapping joint_group -> processed action values
            embodiment_tag: Embodiment identifier
            raw_state: Optional dict of raw states for relative->absolute conversion
                If None, will use unapplied state (but won't work for sin/cos encoded states)

        Returns:
            Tuple of (raw_state, raw_action)
        """
        # Unapply state first
        try:
            unapplied_state = self.unapply_state(state, embodiment_tag)
        except ValueError as e:
            if "sin/cos encoding" in str(e) and raw_state is None:
                raise ValueError(
                    "Cannot unapply sin/cos encoded state. Please provide raw_state parameter."
                ) from e
            raise

        # Use provided raw_state if available, otherwise use unapplied state
        state_for_action = raw_state if raw_state is not None else unapplied_state

        # Unapply action
        unapplied_action = self.unapply_action(action, embodiment_tag, state=state_for_action)

        return unapplied_state, unapplied_action

    def get_state_dim(self, embodiment_tag: str, include_sincos_expansion: bool = False) -> int:
        """Get total state dimension after processing.

        Args:
            embodiment_tag: Embodiment identifier
            include_sincos_expansion: If True, accounts for sin/cos encoding doubling dimensions

        Returns:
            Total state dimension across all joint groups
        """
        total_dim = 0
        state_config = self.modality_configs[embodiment_tag]["state"]

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = set()
        if self.apply_sincos_state_encoding and hasattr(state_config, "sin_cos_embedding_keys"):
            sin_cos_keys = set(state_config.sin_cos_embedding_keys)

        for joint_group in state_config.modality_keys:
            base_dim = self.norm_params[embodiment_tag]["state"][joint_group]["dim"].item()

            # Sin/cos encoding doubles the dimension
            if include_sincos_expansion and joint_group in sin_cos_keys:
                total_dim += base_dim * 2
            else:
                total_dim += base_dim

        return total_dim

    def get_action_dim(self, embodiment_tag: str) -> int:
        """Get total action dimension.

        Args:
            embodiment_tag: Embodiment identifier

        Returns:
            Total action dimension across all joint groups
        """
        total_dim = 0
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            total_dim += self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
        return total_dim

    def _convert_to_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert absolute action to relative action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        if action_type == ActionType.EEF:
            assert (
                action.shape[1] == 9
            ), f"Expected action dim 9 (xyz + rot6d) for EEF, got {action.shape[1]}"

            # Batched mirror of the per-step EndEffectorPose object path (a dataloader
            # hotspot): the same scipy/numpy ops run once at N=T instead of T times at N=1.
            # The copy is load-bearing: gram-schmidt writeback below must not mutate the
            # caller's action array.
            rot_6d = np.array(action[:, 3:]).reshape(-1, 2, 3)
            dots = np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1)
            # The object path selects gram-schmidt independently per row.
            non_orthogonal = ~np.isclose(dots, 0.0)
            if non_orthogonal.any():
                rot_6d[non_orthogonal] = gram_schmidt(rot_6d[non_orthogonal])
            rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
            rot_mat[:, :2, :] = rot_6d
            rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
            # from_matrix normalizes via quaternion and the object path rebuilds every
            # matrix from the normalized rotation, so as_matrix() is not a no-op here.
            homogeneous_actions = np.tile(np.eye(4), (action.shape[0], 1, 1))
            homogeneous_actions[:, :3, :3] = Rotation.from_matrix(rot_mat).as_matrix()
            homogeneous_actions[:, :3, 3] = action[:, :3]

            reference_pose = EndEffectorPose(
                translation=reference_state[:3],
                rotation=reference_state[3:],
                rotation_type="rot6d",
            )
            relative_transforms = (
                invert_transformation(reference_pose.homogeneous) @ homogeneous_actions
            )
            relative_rotations = Rotation.from_matrix(relative_transforms[:, :3, :3])
            translations = relative_transforms[:, :3, 3]

            if action_format == ActionFormat.XYZ_ROT6D:
                return np.concatenate(
                    [translations, rot_mat_to_rot_6d(relative_rotations.as_matrix())], axis=1
                )
            if action_format == ActionFormat.XYZ_ROTVEC:
                return np.concatenate([translations, relative_rotations.as_rotvec()], axis=1)
            if action_format == ActionFormat.DEFAULT:
                relative_homogeneous = np.tile(np.eye(4), (action.shape[0], 1, 1))
                relative_homogeneous[:, :3, :3] = relative_rotations.as_matrix()
                relative_homogeneous[:, :3, 3] = translations
                return relative_homogeneous
            raise ValueError(f"Unsupported action format: {action_format}")

        if action_type == ActionType.NON_EEF:
            action_chunking = JointActionChunk([JointPose(m) for m in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        relative_action_chunking = action_chunking.relative_chunking(
            reference_frame=reference_frame
        )
        return relative_action_chunking.to(action_format)

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert relative action to absolute action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert (
            reference_state.shape[0] == action.shape[1]
        ), f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"

        if action_type == ActionType.EEF:
            assert (
                action.shape[1] == 9
            ), f"Expected action dim 9 (xyz + rot6d) for EEF, got {action.shape[1]}"

            # Batched mirror of the per-step EndEffectorPose object path. Every
            # action composes with the same reference pose, so there is no
            # sequential dependency. The copy is load-bearing: gram-schmidt
            # writeback below must not mutate the caller's action array.
            rot_6d = np.array(action[:, 3:]).reshape(-1, 2, 3)
            dots = np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1)
            # The object path selects gram-schmidt independently per row.
            non_orthogonal = ~np.isclose(dots, 0.0)
            if non_orthogonal.any():
                rot_6d[non_orthogonal] = gram_schmidt(rot_6d[non_orthogonal])

            rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
            rot_mat[:, :2, :] = rot_6d
            rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
            relative = np.tile(np.eye(4), (action.shape[0], 1, 1))
            relative[:, :3, :3] = Rotation.from_matrix(rot_mat).as_matrix()
            relative[:, :3, 3] = action[:, :3]

            frame = np.eye(4)
            frame[:3, :3] = Rotation.from_matrix(
                rot_6d_to_rot_mat(reference_state[None, 3:])
            ).as_matrix()[0]
            frame[:3, 3] = reference_state[:3]

            absolute = frame @ relative
            translations = absolute[:, :3, 3]
            rotations = Rotation.from_matrix(absolute[:, :3, :3])

            if action_format == ActionFormat.XYZ_ROT6D:
                return np.concatenate(
                    [translations, rot_mat_to_rot_6d(rotations.as_matrix())], axis=-1
                )
            if action_format == ActionFormat.XYZ_ROTVEC:
                return np.concatenate([translations, rotations.as_rotvec()], axis=-1)
            if action_format == ActionFormat.DEFAULT:
                absolute[:, :3, :3] = rotations.as_matrix()
                return absolute
            raise ValueError(f"Unsupported action format: {action_format}")

        if action_type == ActionType.NON_EEF:
            if action_format != ActionFormat.DEFAULT:
                raise ValueError(
                    f"Unsupported action format for joint trajectories: {action_format}"
                )
            # JointPose stores float64, so preserve the object path's dtype.
            return np.asarray(reference_state, dtype=np.float64) + np.asarray(
                action, dtype=np.float64
            )

        raise ValueError(f"Unknown ActionType: {action_type}")

    def __str__(self) -> str:
        return (
            "StateActionProcessor("
            f"modality_configs={self.modality_configs}, "
            f"statistics={self.statistics}, "
            f"use_percentiles={self.use_percentiles}, "
            f"clip_outliers={self.clip_outliers}, "
            f"skip_inference_clip={self.skip_inference_clip}, "
            f"apply_sincos_state_encoding={self.apply_sincos_state_encoding}, "
            f"use_relative_action={self.use_relative_action})"
        )
