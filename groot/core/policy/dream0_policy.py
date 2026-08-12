# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from concurrent.futures import Future, ThreadPoolExecutor
import datetime
import logging
import os
import time
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from groot.core.data.types import ModalityConfig
from groot.core.policy.gr00t_policy import Gr00tPolicy
from groot.core.policy.base import BasePolicy

# Presets keyed by (action_horizon, video_history_mode).
# 8frame: stride 3, exact match to training video delta_indices.
# 4frame: stride ~ action_horizon/3, spans the full action horizon.
VIDEO_DELTA_PRESETS: dict[tuple[int, str], list[int]] = {
    (24, "8frame"): [-21, -18, -15, -12, -9, -6, -3, 0],
    (24, "4frame"): [-24, -16, -8, 0],
    (48, "8frame"): [-42, -36, -30, -24, -18, -12, -6, 0],
    (48, "4frame"): [-47, -32, -16, 0],
}


class DreamZeroPolicy(Gr00tPolicy):
    def __init__(
        self,
        *args,
        video_history_mode: Literal["8frame", "4frame"] = "8frame",
        action_horizon: int = 24,
        use_torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        **kwargs,
    ):

        # increase default recompile limit for dynamo (attr added in torch 2.7)
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 128

        # Check if we are in a distributed environment (e.g. launched via torchrun)
        device_mesh = None
        is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        if is_distributed and not dist.is_initialized():
            print("Initializing distributed mesh...")
            device_mesh = self.init_mesh()
        else:
            print("Running in single-process mode, skipping distributed mesh initialization")

        super().__init__(*args, **kwargs)
        key = (action_horizon, video_history_mode)
        if key not in VIDEO_DELTA_PRESETS:
            raise ValueError(
                f"No video delta preset for {key}. "
                f"Available: {list(VIDEO_DELTA_PRESETS.keys())}"
            )
        self.modality_configs["video"].delta_indices = list(VIDEO_DELTA_PRESETS[key])
        self.first_frame = True

        if hasattr(self.model, "action_head"):
            dynamic_cache = os.environ.get("DYNAMIC_CACHE_SCHEDULE", "false").lower() == "true"
            if dynamic_cache:
                self.model.action_head.dynamic_cache_schedule = True
            logging.info(
                f"Set dynamic_cache_schedule to {self.model.action_head.dynamic_cache_schedule} "
            )

        if hasattr(self.model, "action_head") and hasattr(
            self.model.action_head, "post_initialize"
        ):
            logging.info("Post-initializing the action head")
            self.model.action_head.post_initialize(
                use_torch_compile=use_torch_compile,
                torch_compile_mode=torch_compile_mode,
            )

        if (
            device_mesh is not None
            and hasattr(self.model, "action_head")
            and hasattr(self.model.action_head, "parallelize")
        ):
            logging.info("Parallelizing the action head")
            self.model.action_head.parallelize(device_mesh)

    def init_mesh(self) -> DeviceMesh:
        # env vars set by torchrun
        dist.init_process_group("nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

        mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("ip",),
        )
        print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

        return mesh

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def _post_process_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        # NOTE: first-frame trimming (v[:, -1:]) is disabled. It was designed
        # for clients that might have stale history on the first step, but all
        # current clients (YAM, DROID, Agibot) pad their frame buffer on reset,
        # so trimming just produces a 1-frame input that the model can't handle.
        return observation

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.first_frame = True
        self.model.reset()  # type: ignore
        return super().reset(options)

    def save_video(
        self,
        save_path: str,
        traj_id: int,
        video_latents: list | None = None,
        video_texts: list[str] | None = None,
    ) -> None:
        """Decode video latents, annotate frames with step/task info, and write to disk.

        Args:
            save_path: Path to save the video.
            traj_id: Trajectory ID used in the filename when *save_path* has no .mp4 suffix.
            video_latents: Pre-snapshotted latents. Falls back to model buffer.
            video_texts: Per-step language task strings for annotation.
        """
        if not hasattr(self.model, "decode_video_latents"):
            print("Warning: Model does not support decode_video_latents.")
            return

        latents = video_latents if video_latents is not None else self.model.video_latents_buffer
        if not latents:
            print("Warning: No video frames accumulated.")
            return

        decoded_frames = self.model.decode_video_latents(latents)
        if not decoded_frames:
            print("Warning: No frames decoded.")
            return

        decoded_frames = self._annotate_frames(decoded_frames, latents, task_texts=video_texts)

        if save_path.endswith(".mp4"):
            full_save_path = save_path
        else:
            full_save_path = save_path.split(".mp4")[0] + "_" + str(traj_id) + ".mp4"
        self._save_frames_as_video(decoded_frames, full_save_path, fps=8)

    @staticmethod
    def _annotate_frames(
        frames: list[np.ndarray],
        video_latents: list,
        task_texts: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Overlay per-frame step index (top-left) and language task (bottom-left)."""
        import cv2

        if not frames or not video_latents:
            return frames

        # Map each frame back to its chunk so we can pick the right task text.
        latent_temps = [lat.shape[2] for lat in video_latents]
        total_latent_t = sum(latent_temps)
        total_frames = len(frames)

        frame_to_chunk: list[int] = []
        cumulative = 0
        for chunk_idx, t in enumerate(latent_temps):
            if chunk_idx < len(latent_temps) - 1:
                n_frames = round(t / total_latent_t * total_frames)
            else:
                n_frames = total_frames - cumulative
            frame_to_chunk.extend([chunk_idx] * n_frames)
            cumulative += n_frames

        h = frames[0].shape[0]
        font_scale = max(0.35, h / 320 * 0.7)
        thickness_bg = max(2, int(font_scale * 3))
        thickness_fg = max(1, int(font_scale * 1.5))
        line_height = int(28 * font_scale)

        annotated: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            frame = frame.copy()

            step_label = f"Step {i}"
            step_pos = (8, line_height)
            for color, thickness in [
                ((0, 0, 0), thickness_bg),
                ((255, 255, 255), thickness_fg),
            ]:
                cv2.putText(
                    frame,
                    step_label,
                    step_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            if task_texts:
                chunk_idx = frame_to_chunk[i] if i < len(frame_to_chunk) else len(video_latents) - 1
                if chunk_idx < len(task_texts):
                    task_label = task_texts[chunk_idx]
                    task_pos = (8, h - int(12 * font_scale))
                    for color, thickness in [
                        ((0, 0, 0), thickness_bg),
                        ((255, 255, 255), thickness_fg),
                    ]:
                        cv2.putText(
                            frame,
                            task_label,
                            task_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            thickness,
                            cv2.LINE_AA,
                        )

            annotated.append(frame)

        return annotated

    @staticmethod
    def _save_frames_as_video(frames: list[np.ndarray], save_path: str, fps: int = 8) -> None:
        """Write a list of uint8 (H, W, C) frames to an mp4 file."""
        from torchvision.io import write_video

        if not frames:
            print("Warning: No frames to save. Skipping video save.")
            return

        sample_frame = frames[0]
        if len(sample_frame.shape) != 3 or sample_frame.shape[2] not in [1, 3, 4]:
            print(
                f"Warning: Invalid frame shape {sample_frame.shape}. "
                f"Expected (H, W, C) with C in [1, 3, 4]. Skipping video save."
            )
            return

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        write_video(save_path, torch.from_numpy(np.stack(frames)), fps=fps)
        print(f"Saved video to {save_path}")


class EnhancedPolicyWrapper(BasePolicy):
    """Wrapper around DreamZeroPolicy with enhanced monitoring and video saving.

    This wrapper adds:
    - Session tracking (automatic reset detection)
    - Request/response timing
    - Video saving with trajectory numbering
    - Statistics collection
    """

    def __init__(
        self,
        policy: DreamZeroPolicy,
        output_dir: str | None = None,
        save_videos: bool = False,
        save_every_n_steps: int = 10,
    ):
        super().__init__(strict=policy.strict)
        self._policy = policy
        self._output_dir = output_dir
        self._save_videos = save_videos
        self._save_every_n_steps = save_every_n_steps

        self._current_session_id: str | None = None
        self._session_start_time: float | None = None
        self._trajectory_id: int = 0

        self._total_requests: int = 0
        self._session_requests: int = 0
        self._inference_times: list[float] = []
        self._task_texts: list[str] = []

        self._video_save_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="video_save"
        )
        self._pending_futures: list[Future] = []

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            video_dir = os.path.join(self._output_dir, datetime.datetime.now().strftime("%Y%m%d"))
            os.makedirs(video_dir, exist_ok=True)
            self._video_dir = video_dir
            logging.info(f"Output directory: {self._output_dir}")
            logging.info(f"Videos will be saved to: {self._video_dir}")

        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _is_flat_observation(observation: dict[str, Any]) -> bool:
        """Detect if observation uses flat keys ('video.cam') vs nested dicts."""
        if "video" in observation and isinstance(observation["video"], dict):
            return False
        return any(k.startswith("video.") for k in observation)

    @staticmethod
    def _unflatten_observation(observation: dict[str, Any]) -> dict[str, Any]:
        """Convert flat-key observation to nested format.

        Flat:   {"video.cam": arr, "state.joints": arr, "lang_key": val, ...}
        Nested: {"video": {"cam": arr}, "state": {"joints": arr}, "language": {"lang_key": val}}
        """
        nested: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
        for key, value in observation.items():
            if key.startswith("video."):
                nested["video"][key.removeprefix("video.")] = value
            elif key.startswith("state."):
                nested["state"][key.removeprefix("state.")] = value
            elif key == "session_id":
                nested["session_id"] = value
            else:
                nested["language"][key] = value
        return nested

    @staticmethod
    def _normalize_observation(observation: dict[str, Any]) -> set[str]:
        """Normalize observation values for policy consumption.

        Handles observations from ZMQ/msgpack clients where video values may be
        lists of JPEG-encoded bytes instead of numpy arrays, and arrays may lack
        batch dimensions. No-op when observations are already properly formatted.

        Modifies observation in-place.
        Returns a set of modality names ("video", "state", "language") for which
        a batch dimension was added (caller should unbatch actions).
        """
        import cv2

        batched_modalities: set[str] = set()

        if "video" in observation:
            for key in list(observation["video"].keys()):
                value = observation["video"][key]
                if isinstance(value, np.ndarray) and value.ndim == 5:
                    continue  # Already (B, T, H, W, C)
                if isinstance(value, list):
                    if value and isinstance(value[0], bytes | bytearray):
                        # JPEG-encoded frames → decode to numpy uint8, convert BGR→RGB
                        frames = [
                            cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                            for frame in value
                            if (
                                decoded := cv2.imdecode(
                                    np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR
                                )
                            )
                            is not None
                        ]
                        value = np.array(frames, dtype=np.uint8)
                    else:
                        value = np.array(value)
                if isinstance(value, np.ndarray) and value.ndim == 4:
                    value = np.expand_dims(value, axis=0)
                    batched_modalities.add("video")
                observation["video"][key] = value

        if "state" in observation:
            for key in list(observation["state"].keys()):
                value = observation["state"][key]
                if isinstance(value, list):
                    value = np.array(value)
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    value = np.expand_dims(value, axis=0)
                    batched_modalities.add("state")
                observation["state"][key] = value

        if "language" in observation:
            for key in list(observation["language"].keys()):
                value = observation["language"][key]
                if isinstance(value, list | tuple) and value and isinstance(value[0], str):
                    observation["language"][key] = [[s] for s in value]
                    batched_modalities.add("language")

        return batched_modalities

    @staticmethod
    def _flatten_action(action: dict[str, Any]) -> dict[str, Any]:
        """Convert nested action dict to flat format: {'joints': arr} -> {'action.joints': arr}."""
        return {f"action.{key}": value for key, value in action.items()}

    def _extract_task_text(self, observation: dict[str, Any]) -> str | None:
        """Extract the language task string from a nested observation."""
        try:
            lang_key = self._policy.language_key
            lang_dict = observation.get("language", {})
            if not lang_dict:
                return None
            val = lang_dict.get(lang_key)
            if val is None:
                return None
            # val is typically list[list[str]] e.g. [["pick up the object"]]
            if isinstance(val, list | tuple) and val:
                inner = val[0]
                if isinstance(inner, list | tuple) and inner:
                    return str(inner[0])
                if isinstance(inner, str):
                    return inner
            if isinstance(val, str):
                return val
        except Exception as e:
            self._logger.warning(f"Could not extract task text: {e}")
        return None

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Accept both flat and nested observation formats.

        Handles observations from ZMQ clients that send JPEG-encoded video bytes
        and unbatched arrays. Normalizes to the batched numpy format expected by
        the underlying policy, then unbatches the actions on the way out.
        """
        is_flat = self._is_flat_observation(observation)
        if is_flat:
            observation = self._unflatten_observation(observation)
        batched_modalities = self._normalize_observation(observation)
        action, info = super().get_action(observation, options)
        if batched_modalities:
            action = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in action.items()}
        if is_flat:
            action = self._flatten_action(action)
        return action, info

    def get_modality_config(self) -> dict[str, Any]:
        return self._policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._save_videos and self._output_dir and self._session_requests > 0:
            self._save_video()

        self._wait_for_pending_saves()

        result = self._policy.reset(options)
        self._session_requests = 0
        self._task_texts = []
        return result

    def _handle_session_change(self, new_session_id: str):
        if self._current_session_id is not None:
            if self._save_videos and self._output_dir and self._session_requests > 0:
                self._save_video()

            self._wait_for_pending_saves()

            session_duration = (
                time.perf_counter() - self._session_start_time if self._session_start_time else 0
            )
            avg_inference = (
                sum(self._inference_times) / len(self._inference_times)
                if self._inference_times
                else 0
            )

            self._logger.info("=" * 80)
            self._logger.info(f"Session '{self._current_session_id}' ended")
            self._logger.info(f"  Duration: {session_duration:.1f}s")
            self._logger.info(f"  Requests: {self._session_requests}")
            self._logger.info(f"  Avg inference time: {avg_inference:.3f}s")
            self._logger.info("=" * 80)

            self._policy.reset()

        self._current_session_id = new_session_id
        self._session_start_time = time.perf_counter()
        self._session_requests = 0
        self._inference_times = []
        self._task_texts = []
        self._trajectory_id += 1

        self._logger.info(
            f"New session started: '{new_session_id}' (trajectory {self._trajectory_id})"
        )

    def _save_video(self):
        """Snapshot the latent buffer and submit an async video save."""
        if not hasattr(self._policy, "save_video"):
            self._logger.warning("Policy does not support save_video method")
            return

        try:
            video_dir = self._video_dir
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
            video_filename = f"traj_{self._trajectory_id:04d}_{timestamp}.mp4"
            video_path = os.path.join(video_dir, video_filename)

            latents_snapshot = None
            model = getattr(self._policy, "model", None)
            if model is not None and hasattr(model, "video_latents_buffer"):
                latents_snapshot = list(model.video_latents_buffer)
                model.video_latents_buffer.clear()
            texts_snapshot = list(self._task_texts) if self._task_texts else None
            self._task_texts.clear()

            future = self._video_save_executor.submit(
                self._save_video_worker,
                video_path,
                self._trajectory_id,
                latents_snapshot,
                texts_snapshot,
            )
            self._pending_futures.append(future)

            self._pending_futures = [f for f in self._pending_futures if not f.done()]

        except Exception as e:
            self._logger.error(f"Failed to submit video save: {e}")

    def _save_video_worker(
        self, video_path: str, trajectory_id: int, latents_snapshot, texts_snapshot=None
    ):
        """Runs in a background thread -- decodes latents, annotates, and writes video."""
        try:
            self._policy.save_video(
                video_path,
                trajectory_id,
                video_latents=latents_snapshot,
                video_texts=texts_snapshot,
            )
            self._logger.info(f"Saved video: {os.path.basename(video_path)}")
        except Exception as e:
            self._logger.error(f"Background video save failed: {e}", exc_info=True)

    def _wait_for_pending_saves(self, timeout: float = 120.0):
        """Block until all queued video saves finish."""
        if not self._pending_futures:
            return
        self._logger.info(f"Waiting for {len(self._pending_futures)} pending video save(s)...")
        for f in self._pending_futures:
            try:
                f.result(timeout=timeout)
            except Exception as e:
                self._logger.error(f"Video save future raised: {e}")
        self._pending_futures.clear()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Delegate action computation to the wrapped policy with timing/monitoring."""
        start_time = time.perf_counter()

        session_id = observation.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            self._handle_session_change(session_id)

        action, info = self._policy.get_action(observation, options)

        task_text = self._extract_task_text(observation)
        if task_text is not None:
            self._task_texts.append(task_text)

        inference_time = time.perf_counter() - start_time
        self._inference_times.append(inference_time)
        self._total_requests += 1
        self._session_requests += 1

        if self._session_requests == 1:
            self._logger.info(f"First inference: {inference_time:.3f}s (includes compilation)")
        elif self._session_requests % 10 == 0:
            avg_time = sum(self._inference_times[-10:]) / len(self._inference_times[-10:])
            self._logger.info(
                f"Step {self._session_requests}: "
                f"Inference time = {inference_time:.3f}s, "
                f"Avg (last 10) = {avg_time:.3f}s"
            )

        if (
            self._save_videos
            and self._output_dir
            and self._session_requests % self._save_every_n_steps == 0
        ):
            self._save_video()

        return action, info

    def check_observation(self, observation: dict[str, Any]) -> None:
        if self._is_flat_observation(observation):
            observation = self._unflatten_observation(observation)
        self._normalize_observation(observation)
        self._policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self._policy.check_action(action)

    def get_statistics(self) -> dict[str, Any]:
        avg_inference = (
            sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0
        )
        return {
            "total_requests": self._total_requests,
            "current_session_requests": self._session_requests,
            "current_trajectory_id": self._trajectory_id,
            "avg_inference_time": avg_inference,
            "min_inference_time": min(self._inference_times) if self._inference_times else 0,
            "max_inference_time": max(self._inference_times) if self._inference_times else 0,
            "pending_video_saves": len([f for f in self._pending_futures if not f.done()]),
        }

    def shutdown(self):
        """Drain pending video saves and shut down the thread pool."""
        self._wait_for_pending_saves()
        self._video_save_executor.shutdown(wait=True)
