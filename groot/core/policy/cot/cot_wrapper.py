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
"""CoT policy wrappers built on a remote CoT planner server.

These wrappers translate a high-level task into a simplified next-step
instruction via a remote VLM planner, write it into the observation under
``instruction_key``, then delegate to the wrapped (System-1) policy.

They are :class:`groot.core.policy.base.PolicyWrapper` subclasses, so they wrap
any :class:`~groot.core.policy.base.BasePolicy` (e.g. ``Gr00tPolicy``) and can
be hosted directly in a :class:`groot.core.policy.server_client.PolicyServer`.
That lets every robot runner reuse CoT through the standard ``PolicyClient``
interface without any robot-specific wrapper wiring::

    server = PolicyServer(CoTStartPolicyWrapper(Gr00tPolicy(...), ...))

The wrappers are embodiment-agnostic. Two robot-specific concerns are injected
at construction:

  * ``frame_extractor``: maps an observation to ``{camera_name: [PIL.Image, ...]}``
    in the order the VLM should see them. The first entry is the primary camera
    (memory representative on the server). Robots differ in their camera key
    conventions, so this callable encapsulates that knowledge.
  * ``instruction_key``: the observation key holding the language instruction
    that is read as the goal and overwritten with the planner's output
    (defaults to ``"annotation.task"``, the groot.core convention).

Step-0 ("fresh scene") inference — running System-2 once on the post-setup
observation so System-1 gets a real subtask from the first step — is exposed two
ways:

  * ``reset(options={"observation": obs})``: the server-friendly path. The
    fresh observation rides along in the reset options, so a remote
    ``PolicyClient`` can trigger it over the wire.
  * ``prepare_new_episode(obs)``: a convenience for in-process control loops
    that want step-0 *without* also resetting the wrapped policy/session.
"""

from concurrent import futures
import threading
import time
from typing import Any, Callable

from loguru import logger
from PIL import Image

from groot.core.policy import base
from groot.core.policy.cot.planner_client import CoTPlannerClient

# Maps an observation dict to ``{camera_name: [frame, ...]}``. Insertion order
# defines the camera order shown to the VLM; the first entry is the primary
# camera (memory representative on the server).
FrameExtractor = Callable[[dict[str, Any]], dict[str, list[Image.Image]]]

_DEFAULT_INSTRUCTION_KEY = "annotation.task"
# Key under which a fresh post-setup observation may be passed in reset options
# to trigger step-0 inference (see module docstring).
RESET_OBSERVATION_KEY = "observation"


class _CoTPolicyWrapperBase(base.PolicyWrapper):
    """Shared plumbing for the CoT policy wrappers.

    Builds the planner client, stores the injected ``frame_extractor`` /
    ``instruction_key``, and satisfies the :class:`BasePolicy` contract by
    forwarding validation, ``_get_action`` and ``get_modality_config`` to the
    wrapped policy. Concrete subclasses override ``get_action`` (to inject the
    CoT instruction and delegate) and ``reset``.
    """

    def __init__(
        self,
        policy: base.BasePolicy,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        fallback_response: str,
        frame_extractor: FrameExtractor,
        instruction_key: str = _DEFAULT_INSTRUCTION_KEY,
        planner_base_url: str = "http://localhost:8100",
        memory_type: str | None = None,
        session_id: str | None = None,
        task_type: str | None = None,
        strict: bool = True,
    ):
        super().__init__(policy, strict=strict)
        self.cot_planner = CoTPlannerClient(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            fallback_response=fallback_response,
            base_url=planner_base_url,
            memory_type=memory_type,
            session_id=session_id,
            task_type=task_type,
        )
        self._frame_extractor = frame_extractor
        self._instruction_key = instruction_key

    # --- BasePolicy abstract contract ------------------------------------
    # ``get_action`` is overridden by subclasses to inject the CoT instruction
    # and delegate to ``self.policy.get_action`` (which runs its own
    # validation), so these forwarders are off the hot path. They require the
    # wrapped policy to be a BasePolicy and exist to satisfy the ABC and to keep
    # the wrapper transparent if something calls them directly.
    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.policy._get_action(observation, options)

    def get_modality_config(self):
        """Forward modality config so a hosting PolicyServer can serve it."""
        return self.policy.get_modality_config()

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _normalize_goal(value: Any) -> str:
        """Flatten a language value to a single goal string.

        The instruction may arrive as a bare ``str`` (in-process control loops)
        or as a (possibly batched) ``list``/``tuple`` of strings — the gr00t
        flat-observation contract is ``str | list[str]``. Unwrap nested
        sequences down to the first string. A plain string is returned as-is,
        so this is an identity for callers that already pass strings.
        """
        while isinstance(value, (list, tuple)):
            if not value:
                return ""
            value = value[0]
        return value

    def _read_goal(self, observation: dict[str, Any]) -> str:
        """Read and normalize the instruction value from the observation."""
        return self._normalize_goal(observation[self._instruction_key])

    @staticmethod
    def _split_reset_options(
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Split reset ``options`` into (inner_options, step0_observation).

        Pops the CoT-specific ``RESET_OBSERVATION_KEY`` so the wrapped policy
        never sees it.
        """
        if not options or RESET_OBSERVATION_KEY not in options:
            return options, None
        inner = {k: v for k, v in options.items() if k != RESET_OBSERVATION_KEY}
        return (inner or None), options[RESET_OBSERVATION_KEY]

    def _forward_get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.policy.get_action(observation, options, **kwargs)


class CoTStartPolicyWrapper(_CoTPolicyWrapperBase):
    """CoT wrapper that re-infers only when the high-level instruction changes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_cot_input = ""
        self.last_cot_output = ""
        self.start_time = time.time()

    def prepare_new_episode(self, observation: dict[str, Any]):
        """Run the first System-2 inference with a fresh post-setup observation.

        Runs step-0 inference so System-1 gets a proper CoT subtask from step 0,
        without resetting the wrapped policy. ``reset(options={"observation":
        obs})`` calls this after resetting; in-process control loops may call it
        directly.
        """
        self.start_time = time.time()

        curr_language = self._read_goal(observation)
        self.last_cot_input = curr_language
        frames_by_camera = self._frame_extractor(observation)

        self.last_cot_output = self.cot_planner.infer_next_action(
            frames_by_camera=frames_by_camera,
            goal=curr_language,
            observation_time=0.0,
            subtask_candidates=None,
        )
        logger.info(
            f"COT first inference on fresh scene: '{curr_language}' -> '{self.last_cot_output}' "
            f"(cameras={list(frames_by_camera.keys())})"
        )

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get action from the underlying policy, with CoT-based task translation.

        The method translates the high-level task into a simplified instruction
        using the CoT planner before passing it to the underlying policy.

        Args:
            observation: Current observation from the environment.
            options: Optional options forwarded to the wrapped policy.
            **kwargs: Additional arguments (e.g. previous_action) forwarded to
                the underlying policy for RTC support.

        Returns:
            Tuple of (action, info) from the underlying policy.
        """
        curr_language = self._read_goal(observation)
        if curr_language != self.last_cot_input:
            self.last_cot_input = curr_language
            frames_by_camera = self._frame_extractor(observation)

            time_since_start = time.time() - self.start_time
            self.last_cot_output = self.cot_planner.infer_next_action(
                frames_by_camera=frames_by_camera,
                goal=curr_language,
                observation_time=time_since_start,
                subtask_candidates=None,
            )
            logger.info(f"COT translated task {curr_language} into {self.last_cot_output}")

        observation[self._instruction_key] = self.last_cot_output
        return self._forward_get_action(observation, options, kwargs)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy and clear cached CoT outputs.

        If ``options`` carries a fresh observation under ``RESET_OBSERVATION_KEY``,
        runs step-0 inference after resetting.
        """
        inner_options, step0_obs = self._split_reset_options(options)
        self.last_cot_input = ""
        self.last_cot_output = ""
        self.start_time = time.time()
        self.cot_planner.reset_episode()
        info = self.policy.reset(inner_options)
        if step0_obs is not None:
            self.prepare_new_episode(step0_obs)
        return info


class CoTSyncPolicyWrapper(_CoTPolicyWrapperBase):
    """CoT wrapper with time-based re-inference (synchronous).

    Re-infers on instruction change OR when ``max_drift`` seconds have elapsed
    since the last inference. Inference runs inline in ``get_action``.
    """

    def __init__(self, *args, max_drift: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_cot_input = ""
        self.last_cot_output = ""
        self.last_inference_time = 0.0
        self.max_drift = max_drift
        self.start_time = time.time()

    def prepare_new_episode(self, observation: dict[str, Any]):
        """Run the first System-2 inference with a fresh post-setup observation.

        See :meth:`CoTStartPolicyWrapper.prepare_new_episode`.
        """
        self.start_time = time.time()

        curr_language = self._read_goal(observation)
        self.last_cot_input = curr_language
        frames_by_camera = self._frame_extractor(observation)

        self.last_cot_output = self.cot_planner.infer_next_action(
            frames_by_camera=frames_by_camera,
            goal=curr_language,
            observation_time=0.0,
            subtask_candidates=None,
        )
        self.last_inference_time = time.time()
        logger.info(
            f"COT first inference on fresh scene: '{curr_language}' -> '{self.last_cot_output}' "
            f"(cameras={list(frames_by_camera.keys())})"
        )

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get action from the underlying policy, with CoT-based task translation.

        The method translates the high-level task into a simplified instruction
        using the CoT planner if:
        1. The language instruction has changed, OR
        2. More than max_drift seconds have passed since the last inference

        Args:
            observation: Current observation from the environment.
            options: Optional options forwarded to the wrapped policy.
            **kwargs: Additional arguments (e.g. previous_action) forwarded to
                the underlying policy for RTC support.

        Returns:
            Tuple of (action, info) from the underlying policy.
        """
        curr_language = self._read_goal(observation)
        current_time = time.time()
        time_since_last_inference = current_time - self.last_inference_time
        language_changed = curr_language != self.last_cot_input

        time_since_start = current_time - self.start_time

        if language_changed or time_since_last_inference >= self.max_drift:
            self.last_cot_input = curr_language
            self.last_inference_time = current_time
            frames_by_camera = self._frame_extractor(observation)

            self.last_cot_output = self.cot_planner.infer_next_action(
                frames_by_camera=frames_by_camera,
                goal=curr_language,
                observation_time=time_since_start,
                subtask_candidates=None,
            )

            if language_changed:
                logger.info(
                    f"COT inference triggered by language change: "
                    f"translated task '{curr_language}' into '{self.last_cot_output}'"
                )
            else:
                logger.info(
                    f"COT re-inference after {time_since_last_inference:.2f}s: "
                    f"translated task '{curr_language}' into '{self.last_cot_output}'"
                )

        observation[self._instruction_key] = self.last_cot_output
        return self._forward_get_action(observation, options, kwargs)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy and clear cached CoT outputs.

        If ``options`` carries a fresh observation under ``RESET_OBSERVATION_KEY``,
        runs step-0 inference after resetting.
        """
        inner_options, step0_obs = self._split_reset_options(options)
        self.last_cot_input = ""
        self.last_cot_output = ""
        self.last_inference_time = 0.0
        self.cot_planner.reset_episode()
        self.start_time = time.time()
        info = self.policy.reset(inner_options)
        if step0_obs is not None:
            self.prepare_new_episode(step0_obs)
        return info


class CoTAsyncPolicyWrapper(_CoTPolicyWrapperBase):
    """CoT wrapper with async VLM inference that guarantees freshness.

    VLM inference runs continuously in the background. When one inference completes,
    a new one starts immediately with the latest observation. If the latest result
    is older than max_drift seconds, get_action blocks until fresh inference completes.
    """

    def __init__(self, *args, max_drift: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_drift = max_drift
        self._start_time = time.time()

        # Thread-safe state (RLock needed because add_done_callback can invoke
        # callback synchronously if future already completed, causing re-entry)
        self._lock = threading.RLock()
        self._last_cot_output = ""
        self._last_result_time = 0.0  # When the last result was obtained
        self._pending_future: futures.Future | None = None
        self._running = True

        # Latest observation for continuous inference. ``_latest_frames`` is
        # the per-camera dict the VLM will see next; the injected
        # ``frame_extractor`` decides whether it's a one-entry primary dict
        # or every known camera in the obs.
        self._latest_frames: dict[str, list[Image.Image]] | None = None
        self._latest_goal: str = ""

        # Thread pool for async VLM inference
        self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cot_vlm")

    def prepare_new_episode(self, observation: dict[str, Any]):
        """Run the first System-2 inference with a fresh post-setup observation.

        Runs the first inference synchronously so System-1 gets a proper CoT
        subtask from step 0, without resetting the wrapped policy.
        """
        with self._lock:
            if self._pending_future and not self._pending_future.done():
                self._pending_future.cancel()
            self._pending_future = None
            self._start_time = time.time()

            curr_language = self._read_goal(observation)
            frames_by_camera = self._frame_extractor(observation)
            self._latest_frames = frames_by_camera
            self._latest_goal = curr_language

        result = self.cot_planner.infer_next_action(
            frames_by_camera=frames_by_camera,
            goal=curr_language,
            observation_time=0.0,
            subtask_candidates=None,
        )

        with self._lock:
            self._last_cot_output = result
            self._last_result_time = time.time()

        logger.info(
            f"COT first inference on fresh scene: '{curr_language}' -> '{result}' "
            f"(cameras={list(frames_by_camera.keys())})"
        )

    def _run_vlm_inference(
        self, frames_by_camera: dict[str, list[Image.Image]], goal: str
    ) -> tuple[str, float]:
        """Run VLM inference in background thread. Returns (result, completion_time)."""
        result = self.cot_planner.infer_next_action(
            frames_by_camera=frames_by_camera,
            goal=goal,
            observation_time=time.time() - self._start_time,
            subtask_candidates=None,
        )
        completion_time = time.time()
        logger.info(f"COT async inference completed: '{goal}' -> '{result}'")
        return result, completion_time

    def _start_next_inference(self) -> None:
        """Start a new inference with the latest observation. Must be called with lock held."""
        if not self._running or self._latest_frames is None:
            return

        self._pending_future = self._executor.submit(
            self._run_vlm_inference, self._latest_frames, self._latest_goal
        )
        self._pending_future.add_done_callback(self._on_inference_complete)

    def _on_inference_complete(self, future: futures.Future) -> None:
        """Callback when async inference completes - store result and start next inference."""
        with self._lock:
            if future.cancelled():
                return
            try:
                result, completion_time = future.result()
                self._last_cot_output = result
                self._last_result_time = completion_time
            except Exception as e:  # noqa: BLE001
                logger.warning(f"COT async inference failed: {e}")

            # Immediately start next inference with latest observation
            self._start_next_inference()

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get action from the underlying policy with async CoT-based task translation.

        Blocks if the latest VLM result is older than max_drift seconds.
        Otherwise returns immediately using the cached result.

        Args:
            observation: Current observation from the environment.
            options: Optional options forwarded to the wrapped policy.
            **kwargs: Additional arguments (e.g. previous_action) forwarded to
                the underlying policy for RTC support.

        Returns:
            Tuple of (action, info) from the underlying policy.
        """
        curr_language = self._read_goal(observation)
        frames_by_camera = self._frame_extractor(observation)

        with self._lock:
            # Update latest observation for continuous inference
            self._latest_frames = frames_by_camera
            self._latest_goal = curr_language

            # Start first inference if none pending
            if self._pending_future is None:
                self._start_next_inference()

            pending_future = self._pending_future
            result_age = time.time() - self._last_result_time
            need_to_wait = self._last_result_time == 0.0 or result_age >= self.max_drift

        # Block if result is stale (outside lock to avoid deadlock)
        if need_to_wait and pending_future is not None:
            logger.info(f"COT blocking for fresh inference (result age: {result_age:.2f}s)")
            try:
                pending_future.result()  # Wait for completion
            except Exception as e:  # noqa: BLE001
                logger.warning(f"COT inference failed while waiting: {e}")

        # Get the current output (may have been updated by callback)
        with self._lock:
            current_output = self._last_cot_output

        # Use COT output if available, otherwise fall back to original
        if current_output:
            observation[self._instruction_key] = current_output

        return self._forward_get_action(observation, options, kwargs)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy and clear cached CoT outputs.

        If ``options`` carries a fresh observation under ``RESET_OBSERVATION_KEY``,
        runs step-0 inference after resetting.
        """
        inner_options, step0_obs = self._split_reset_options(options)
        with self._lock:
            # Cancel pending inference if any
            if self._pending_future and not self._pending_future.done():
                self._pending_future.cancel()
            self._last_cot_output = ""
            self._last_result_time = 0.0
            self._pending_future = None
            self._latest_frames = None
            self._latest_goal = ""
            self._start_time = time.time()
        self.cot_planner.reset_episode()
        info = self.policy.reset(inner_options)
        if step0_obs is not None:
            self.prepare_new_episode(step0_obs)
        return info

    def shutdown(self) -> None:
        """Shutdown the executor. Call when done with the wrapper."""
        with self._lock:
            self._running = False
        self._executor.shutdown(wait=False)
