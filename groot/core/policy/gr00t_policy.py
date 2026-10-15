# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import transformers

from groot.core.data import schema, types
from groot.core.policy import base


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (some objects are not dict but have items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


class Gr00tPolicy(base.BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: schema.EmbodimentTag,
        model_path: str,
        *,
        num_inference_timesteps: int | None = None,
        device: int | str,
        strict: bool = True,
        use_torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            use_torch_compile: Whether to torch.compile the action head's get_action method
            torch_compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
        """
        # Import this to register all models. Models not yet migrated to
        # groot.core (e.g. DreamCosmos, N1, TTT) will fail to load until
        # they're ported — that's intentional.
        import groot.core.training.models  # noqa: F401

        super().__init__(strict=strict)
        model_dir = Path(model_path)

        # Load the pretrained model and move to target device with bfloat16 precision.
        model = transformers.AutoModel.from_pretrained(model_dir)
        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Load the processor for input/output transformation
        self.processor: types.BaseProcessor = transformers.AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self._load_modality_configs()
        self.collate_fn = self.processor.collator

        action_head = self.model.action_head

        if num_inference_timesteps is None:
            default_steps = getattr(
                action_head,
                "num_inference_timesteps",
                getattr(action_head, "num_inference_steps", 4),
            )
            print(f"Using default num_inference_timesteps from model config: {default_steps}")
        else:
            print(f"Using num_inference_timesteps from init args: {num_inference_timesteps}")
            if hasattr(action_head, "num_inference_timesteps"):
                action_head.num_inference_timesteps = num_inference_timesteps
            else:
                action_head.num_inference_steps = num_inference_timesteps

        # Extract and validate language configuration
        # Currently only supports single language input per timestep
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

        if use_torch_compile and action_head is not None:
            print(
                f"[Gr00tPolicy] Applying torch.compile (mode={torch_compile_mode}) to action head..."
            )
            try:
                action_head.get_action = torch.compile(
                    action_head.get_action,
                    mode=torch_compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
                print("[Gr00tPolicy] torch.compile applied to action head.")
            except Exception as e:  # noqa: BLE001
                print(
                    f"[Gr00tPolicy] WARNING: torch.compile failed: {e}, continuing without compilation..."
                )

    def _load_modality_configs(self) -> dict[str, "types.ModalityConfig"]:
        """Load the modality configurations from the processor."""
        orig_config = self.processor.get_modality_configs()[self.embodiment_tag.value]
        new_config = copy.deepcopy(orig_config)
        for modality, config in new_config.items():
            if modality != "action" and any(
                delta_index > 0 for delta_index in config.delta_indices
            ):
                orig_indices = config.delta_indices
                config.delta_indices = [i for i in config.delta_indices if i <= 0]
                print(
                    f"[Gr00tPolicy] WARNING: Observation modality {modality} has positive delta indices, truncating to negative indices"
                )
                print(
                    f"[Gr00tPolicy] WARNING: Observation modality {modality} original indices: {orig_indices}, new indices: {config.delta_indices}"
                )
        return new_config

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {}
            for modality, modality_dict in value.items():
                unbatched_value[modality] = {k: v[i] for k, v in modality_dict.items()}
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> types.VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return types.VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions=observation.get("previous_action", {}),  # Use prev action or no action
            languages=observation["language"],
            text=observation["language"][self.language_key][0],  # Deprecated
            embodiment=self.embodiment_tag,
        )

    def _postprocess_normalized_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Postprocess the normalized action

        Args:
            normalized_action: Normalized action tensor

        Returns:
            Postprocessed action tensor
        """
        return normalized_action

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)
            - previous_action (optional): dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of future action steps)
                - D: action dimension

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = observation["video"][video_key].shape[0]
            else:
                assert (
                    observation["video"][video_key].shape[0] == bs
                ), f"Video key '{video_key}' must have batch size {bs}. Got {observation['video'][video_key].shape[0]}"  # noqa: E501

            # Check that the expected video key exists in the observation
            assert (
                video_key in observation["video"]
            ), f"Video key '{video_key}' must be in observation"

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(
                batched_video, np.ndarray
            ), f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert (
                batched_video.dtype == np.uint8
            ), f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert (
                batched_video.ndim == 5
            ), f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(
                self.modality_configs["video"].delta_indices
            ), f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"  # noqa: E501

            # Verify channel dimension is 3 (RGB images)
            assert (
                batched_video.shape[-1] == 3
            ), f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = observation["state"][state_key].shape[0]
            else:
                assert (
                    observation["state"][state_key].shape[0] == bs
                ), f"State key '{state_key}' must have batch size {bs}. Got {observation['state'][state_key].shape[0]}"  # noqa: E501

            # Check that the expected state key exists in the observation
            assert (
                state_key in observation["state"]
            ), f"State key '{state_key}' must be in observation"

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(
                batched_state, np.ndarray
            ), f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"

            # Verify dtype is float32 (standard for continuous state values)
            assert (
                batched_state.dtype == np.float32
            ), f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"

            # Verify shape has 3 dimensions: (B, T, D)
            assert (
                batched_state.ndim == 3
            ), f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(
                self.modality_configs["state"].delta_indices
            ), f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"  # noqa: E501

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert (
                    len(observation["language"][language_key]) == bs
                ), f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"  # noqa: E501

            # Check that the expected language key exists in the observation
            assert (
                language_key in observation["language"]
            ), f"Language key '{language_key}' must be in observation"

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(
                batched_language, list
            ), f"Language key '{language_key}' must be a list. Got {type(batched_language)}"

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(
                    self.modality_configs["language"].delta_indices
                ), f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"  # noqa: E501

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(
                    batch_item, list
                ), f"Language batch item must be a list. Got {type(batch_item)}"

                # Current implementation expects exactly one language instruction per timestep
                assert (
                    len(batch_item) == 1
                ), f"Language batch item must have exactly one item. Got {len(batch_item)}"

                # Verify the instruction itself is a string
                assert isinstance(
                    batch_item[0], str
                ), f"Language batch item must be a string. Got {type(batch_item[0])}"

    def _post_process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Post-process the observation.

        Args:
            observation: Observation dictionary

        Returns:
            Post-processed observation dictionary
        """
        return observation

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        t0 = time.perf_counter()

        # Step 1: Split batched observation into individual observations
        observation = self._post_process_observation(observation)
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [{"type": types.MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        t_preprocess = time.perf_counter()

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs, options=options)
        t_predict = time.perf_counter()

        normalized_action = model_pred["action_pred"].float()
        normalized_action = self._postprocess_normalized_action(normalized_action)

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        t_end = time.perf_counter()
        print(
            f"[Gr00tPolicy] preprocess={t_preprocess - t0:.3f}s  "
            f"predict={t_predict - t_preprocess:.3f}s  "
            f"postprocess={t_end - t_predict:.3f}s  "
            f"total={t_end - t0:.3f}s"
        )
        return casted_action, {}

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(
                action_arr, np.ndarray
            ), f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"

            # Verify dtype is float32 (standard for continuous actions)
            assert (
                action_arr.dtype == np.float32
            ), f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"

            # Verify shape has 3 dimensions: (B, T, D)
            assert (
                action_arr.ndim == 3
            ), f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(
                self.modality_configs["action"].delta_indices
            ), f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"  # noqa: E501

    def get_modality_config(self) -> dict[str, types.ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Forwards the reset to the underlying model via ``on_env_reset`` if
        the model exposes that hook.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        if hasattr(self.model, "on_env_reset"):
            self.model.on_env_reset()
        return {}


class Gr00tFlatPolicyWrapper(base.PolicyWrapper):
    """Adapter: flat-dict (dotted-key) observations → nested Gr00tPolicy input.

    Clients like ``Gr00tPolicyClient`` (and the UMI control loop) send
    observations with dotted flat keys:
        - ``video.<cam>.rgb``:               np.uint8 (T, H, W, C)
        - ``state.<name>``:                  np.float32 (T, D)
        - ``annotation.human.task_description`` or ``annotation.task``: str/list[str]

    ``Gr00tPolicy.check_observation`` instead expects a nested dict with
    top-level keys ``video``/``state``/``language``, each a dict keyed by the
    modality key from the processor config, and shapes ``(B, T, ...)``.

    This wrapper performs:
        1. Flat → nested regrouping driven by the underlying policy's
           ``modality_configs``.
        2. Adds a leading batch dim when the incoming arrays are unbatched.
        3. Accepts common aliases for the language key so both
           ``annotation.task`` and ``annotation.human.task_description``
           checkpoints work with the same client.
        4. Flattens the nested action dict back to ``action.<key>`` and drops
           the batch dim on return.
    """

    _LANGUAGE_KEY_FALLBACKS = (
        "annotation.human.task_description",
        "annotation.task",
    )

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy

    def get_modality_config(self) -> dict[str, types.ModalityConfig]:
        return self.policy.get_modality_config()

    def check_observation(self, observation: dict[str, Any]) -> None:
        modality_configs = self.get_modality_config()
        for video_key in modality_configs["video"].modality_keys:
            flat = f"video.{video_key}"
            assert flat in observation, f"Video key '{flat}' must be in observation"
        for state_key in modality_configs["state"].modality_keys:
            flat = f"state.{state_key}"
            assert flat in observation, f"State key '{flat}' must be in observation"
        for language_key in modality_configs["language"].modality_keys:
            assert self._resolve_language_value(observation, language_key) is not None, (
                f"Language key '{language_key}' (or alias) must be in observation"
            )

    def check_action(self, action: dict[str, Any]) -> None:
        modality_configs = self.get_modality_config()
        for action_key in modality_configs["action"].modality_keys:
            flat = f"action.{action_key}"
            assert flat in action, f"Action key '{flat}' must be in action"
            arr = action[flat]
            assert isinstance(arr, np.ndarray), (
                f"Action key '{flat}' must be a numpy array. Got {type(arr)}"
            )
            assert arr.ndim in (2, 3), (
                f"Action key '{flat}' must be (T, D) or (B, T, D). Got shape {arr.shape}"
            )

    @classmethod
    def _resolve_language_value(
        cls, observation: dict[str, Any], modality_key: str
    ) -> Any | None:
        """Return the language value for ``modality_key`` from a flat observation.

        Tries the exact modality key first, then the known fallback aliases so
        a single client (which emits ``annotation.human.task_description``)
        works against checkpoints trained with either naming convention.
        """
        if modality_key in observation:
            return observation[modality_key]
        for alias in cls._LANGUAGE_KEY_FALLBACKS:
            if alias in observation:
                return observation[alias]
        return None

    @staticmethod
    def _maybe_add_batch_dim(value: Any) -> Any:
        """Add a leading batch dim to unbatched observation arrays/strings."""
        if isinstance(value, np.ndarray):
            if value.ndim == 4:  # (T, H, W, C) -> (1, T, H, W, C)
                return np.expand_dims(value, axis=0)
            if value.ndim == 2:  # (T, D) -> (1, T, D)
                return np.expand_dims(value, axis=0)
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            return [list(value)]  # (T,) -> (1, T)
        if isinstance(value, str):
            return [[value]]
        return value

    @staticmethod
    def _decode_serialized_video(value: Any) -> Any:
        """Decode JPEG-serialized video frames to a uint8 ``(T, H, W, C)`` array.

        Real-robot clients (e.g. the YAM control loop, see
        ``key_remapping_utils.map_observation_for_gr00t_server``) send each
        ``video.<cam>`` as a list of JPEG-encoded byte strings (one per history
        frame) to keep the wire payload small. The nested ``Gr00tPolicy`` input
        expects a decoded ``(T, H, W, C)`` uint8 array, so decode here. Already
        decoded ndarrays (clients that send raw arrays, or the serializer's
        ndarray-JPEG transport which round-trips to an array) pass through
        unchanged. Mirrors the legacy ``Gr00tG1RealPolicyWrapper._decode_video``
        (no channel reorder: cv2 imencode/imdecode round-trips raw channel order).
        """
        if isinstance(value, (list, tuple)):
            if len(value) == 0 or not isinstance(value[0], (bytes, bytearray)):
                return value
            frames: Any = value
        elif isinstance(value, np.ndarray) and value.dtype == np.object_:
            if value.size == 0 or not isinstance(value.flat[0], (bytes, bytearray)):
                return value
            frames = list(value)
        else:
            return value

        import cv2

        decoded = [
            cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in frames
        ]
        return np.stack(decoded, axis=0).astype(np.uint8)  # (T, H, W, C)

    def _video_decode_pool(self, num_keys: int) -> ThreadPoolExecutor:
        """Return the policy-held pool used to decode camera streams."""
        pool = getattr(self, "_decode_pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=num_keys,
                thread_name_prefix="gr00t-jpeg-decode",
            )
            self._decode_pool = pool
        return pool

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        modality_configs = self.get_modality_config()

        nested: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

        video_keys = list(modality_configs["video"].modality_keys)
        serialized_videos = [observation[f"video.{key}"] for key in video_keys]
        decoded_videos = (
            [self._decode_serialized_video(value) for value in serialized_videos]
            if len(video_keys) < 2
            else list(
                self._video_decode_pool(len(video_keys)).map(
                    self._decode_serialized_video,
                    serialized_videos,
                )
            )
        )
        for video_key, arr in zip(video_keys, decoded_videos, strict=True):
            nested["video"][video_key] = self._maybe_add_batch_dim(arr)

        for state_key in modality_configs["state"].modality_keys:
            arr = observation[f"state.{state_key}"]
            if isinstance(arr, np.ndarray) and arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            nested["state"][state_key] = self._maybe_add_batch_dim(arr)

        for language_key in modality_configs["language"].modality_keys:
            value = self._resolve_language_value(observation, language_key)
            nested["language"][language_key] = self._maybe_add_batch_dim(value)

        if "previous_action" in observation:
            nested["previous_action"] = observation["previous_action"]

        action, info = self.policy.get_action(nested, options)
        flat_action = {
            f"action.{key}": value[0] if getattr(value, "ndim", 0) == 3 else value
            for key, value in action.items()
        }
        return flat_action, info

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)
