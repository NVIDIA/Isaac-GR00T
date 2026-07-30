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
"""N2 Policy: joint video generation + action prediction.

Subclasses Gr00tPolicy for N2 refactor models. Calls model.generate() which
jointly denoises video and action, then decodes actions via the processor.
Also supports saving generated videos.
"""

# Pre-existing baseline for this module, surfaced because the lint job only runs on files
# changed in an MR. The per-step timing/config prints are the eval-run instrumentation;
# the broad except keeps a video-save failure from killing a rollout; `options` and
# `video_texts` are kept for base-class / public-signature parity.
# ruff: noqa: ARG002, BLE001, T201

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
import time
from typing import Any

import numpy as np
import torch
import transformers

from groot.core.data import schema, types
from groot.core.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

N2_VIEW_NAMES = ("top", "left", "right", "bottom")

# Codec used for the observation-video roundtrip at inference (see
# Gr00tN2Policy._codec_roundtrip_views). "h264" matches purchased/h264-converted
# datasets, "mp4v" matches in-house cv2.VideoWriter('mp4v') datasets, "none"
# disables the roundtrip (feed raw frames).
_CODEC_ROUNDTRIPS = ("h264", "none", "mp4v")
_MAX_CODEC_WORKERS = 4


def _decode_video(source: Any) -> np.ndarray:
    import av

    with av.open(source) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return np.stack(frames, axis=0)


def _mp4v_roundtrip(video: np.ndarray) -> np.ndarray:
    import tempfile

    import cv2

    frame_count, height, width, _ = video.shape
    bgr_video = np.ascontiguousarray(video[..., ::-1])
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/dev/shm") as tmp:
        tmp_path = tmp.name
    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        try:
            for frame_index in range(frame_count):
                writer.write(bgr_video[frame_index])
        finally:
            writer.release()
        return _decode_video(tmp_path)
    finally:
        os.unlink(tmp_path)


def _h264_roundtrip(video: np.ndarray) -> np.ndarray:
    import io

    import av

    frame_count, height, width, _ = video.shape
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=20)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "22"}
        for frame_index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(video[frame_index], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer.seek(0)
    return _decode_video(buffer)


def _codec_roundtrip_clip(video: np.ndarray, *, codec: str) -> np.ndarray:
    if codec == "mp4v":
        return _mp4v_roundtrip(video)
    if codec == "h264":
        return _h264_roundtrip(video)
    raise ValueError(f"Unsupported codec roundtrip: {codec!r}")


def _codec_roundtrip_videos(vids: dict[str, Any], codec: str) -> float:
    """Roundtrip clips concurrently and return elapsed seconds."""
    if codec == "none" or not vids:
        return 0.0

    start = time.perf_counter()
    arrays = {key: np.asarray(value) for key, value in vids.items()}
    clips = [
        (key, batch_index, array[batch_index])
        for key, array in arrays.items()
        for batch_index in range(array.shape[0])
    ]
    if not clips:
        return 0.0
    roundtrip = partial(_codec_roundtrip_clip, codec=codec)
    with ThreadPoolExecutor(max_workers=min(_MAX_CODEC_WORKERS, len(clips))) as executor:
        decoded_clips = executor.map(roundtrip, (clip[2] for clip in clips))

    decoded_by_key: dict[str, list[np.ndarray]] = {key: [] for key in arrays}
    for (key, _, _), decoded in zip(clips, decoded_clips, strict=True):
        decoded_by_key[key].append(decoded)
    for key in arrays:
        vids[key] = np.stack(decoded_by_key[key], axis=0).astype(np.uint8)
    return time.perf_counter() - start


def _env_bool(name: str, *, default: bool) -> bool:
    """Parse a boolean env var. Accepts 1/0, true/false, yes/no (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (expected 1/0/true/false/yes/no)")


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


class Gr00tN2Policy(Gr00tPolicy):
    """Policy for GrootN2: joint video + action inference.

    Calls model.generate() which produces both video latents and action predictions.
    Actions are decoded through the processor's StateActionProcessor.
    Video latents can be decoded and saved via save_video().
    """

    def __init__(
        self,
        embodiment_tag: schema.EmbodimentTag,
        model_path: str,
        *,
        num_inference_timesteps: int = 4,
        video_guidance_scale: float = 1.0,
        action_guidance_scale: float = 1.0,
        shift: float = 5.0,
        save_video_dir: str | None = None,
        enable_n2_pred_comparison: bool = False,
        seed: int = 42,
        device: int | str = "cuda:0",
        strict: bool = True,
        use_torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        codec_roundtrip: str = "mp4v",
    ):
        # Import to register the N2 model (pipeline setup runs AutoModel
        # registration as a side effect).
        import groot.core.training.models.n2
        import groot.core.training.models.rl  # noqa: F401  GrootN2RL AutoModel registration

        # Skip Gr00tPolicy.__init__ (it assumes action_head exists) — do the
        # essential setup directly.
        super(Gr00tPolicy, self).__init__(strict=strict)

        model_dir = model_path
        model = transformers.AutoModel.from_pretrained(model_dir)
        # GrootN2RL actor checkpoints only wrap the base GrootN2 actor; inference
        # uses the inner model's predict/vae/decode_latents APIs directly.
        if (
            getattr(model, "wrapped_model", None) is not None
            and type(model).__name__ == "GrootN2RL"
        ):
            model = model.wrapped_model
        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model
        if use_torch_compile:
            print(f"[Gr00tN2Policy] Compiling model with mode: {torch_compile_mode}")
            self.model.vae.model = torch.compile(
                self.model.vae.model, mode=torch_compile_mode, dynamic=False
            )
            # NOTE(FY): dynamic=True is needed for using kv cache — the action-only
            # forward after the first joint step has a different sequence length.
            self.model.transformer = torch.compile(self.model.transformer, mode=torch_compile_mode)

        self.processor: types.BaseProcessor = transformers.AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        self.embodiment_tag = embodiment_tag
        self.modality_configs = self._load_modality_configs()
        self.collate_fn = self.processor.collator

        language_keys = self.modality_configs["language"].modality_keys
        self.language_key = language_keys[0]

        # Generation parameters
        self.num_inference_steps = num_inference_timesteps
        self.video_guidance_scale = video_guidance_scale
        self.action_guidance_scale = action_guidance_scale
        # shift is read from GROOT_N2_SHIFT (env) so the YAM launcher can pass it
        # uniformly to the core server for both N1 and N2 — N1's Gr00tPolicy never
        # reads it, so the env var is a harmless no-op there. The ctor arg is the
        # fallback default when the env var is unset.
        self.shift = _env_float("GROOT_N2_SHIFT", default=shift)
        self.seed = seed

        # Denoise-schedule knobs are read from env vars so we can tune inference
        # without breaking the policy ctor API. Defaults produce the joint 1v+4a
        # hybrid (one joint forward + cached action sub-steps) which is the
        # fastest mode supported today; flip GROOT_N2_JOINT_DENOISE=0 or
        # GROOT_N2_KV_CACHE_COND=0 to compare against older behaviors.
        self.kv_cache_cond = _env_bool("GROOT_N2_KV_CACHE_COND", default=True)
        self.joint_denoise = _env_bool("GROOT_N2_JOINT_DENOISE", default=True)
        self.num_video_steps = _env_int("GROOT_N2_NUM_VIDEO_STEPS", default=2)
        # Observation-video codec roundtrip (see _codec_roundtrip_views). The env
        # var overrides the ctor arg so eval can flip it without an API change.
        self.codec_roundtrip = os.environ.get("GROOT_N2_CODEC_ROUNDTRIP", codec_roundtrip).lower()
        if self.codec_roundtrip not in _CODEC_ROUNDTRIPS:
            raise ValueError(
                f"codec_roundtrip must be one of {_CODEC_ROUNDTRIPS}, "
                f"got {self.codec_roundtrip!r}"
            )
        print(
            f"[Gr00tN2Policy] inference config: "
            f"num_inference_timesteps={num_inference_timesteps}, "
            f"num_video_steps={self.num_video_steps}, "
            f"shift={self.shift}, "
            f"joint_denoise={self.joint_denoise}, "
            f"kv_cache_cond={self.kv_cache_cond}, "
            f"codec_roundtrip={self.codec_roundtrip}"
        )

        # Video buffer for save_video
        self.video_latents_buffer: list[torch.Tensor] = []
        self.save_video_dir = save_video_dir
        self.enable_n2_pred_comparison = enable_n2_pred_comparison
        self._step_count = 0

        # Cached text embeddings keyed by the batch language strings
        # (json-dumped list). Reused when the incoming languages match;
        # invalidated on language change and on reset().
        self._cached_prompt_embs: torch.Tensor | None = None
        self._cached_language_dict: str | None = None

    def _codec_roundtrip_views(self, vids: dict[str, Any]) -> float:
        """Encode→decode each observation video through the configured codec.

        N2 ckpts are trained on codec-encoded LeRobot datasets, and the visual
        encoder overfits to codec artifacts (DCT block quantization, chroma
        subsampling). Feeding raw camera frames at deployment causes predictions
        to collapse to near-constant action targets even though the weights are
        fine. ``self.codec_roundtrip`` selects which codec to mimic:

        - ``"mp4v"``: ``cv2.VideoWriter('mp4v')`` @ 30fps with RGB->BGR — the exact
          encoder the in-house YAM recorder uses (in-house / cv2-mp4v datasets).
        - ``"h264"``: libx264 at crf=22 (purchased / h264-converted datasets).
        - ``"none"``: skip the roundtrip and feed raw frames.

        Mutates ``vids`` in place (values are (B, T, H, W, C) uint8 arrays).
        """
        return _codec_roundtrip_videos(vids, self.codec_roundtrip)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate video + action jointly, return decoded actions."""
        t0 = time.perf_counter()
        # Codec roundtrip on observation videos (no-op when codec_roundtrip="none").
        codec_time = self._codec_roundtrip_views(observation.get("video", {}))
        t_codec = time.perf_counter()

        observation = self._post_process_observation(observation)
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        states = []
        languages = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            languages.append(vla_step_data.languages)
            messages = [{"type": types.MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        t_preprocess = time.perf_counter()

        language_dict = json.dumps(languages)
        prompt_embs = (
            self._cached_prompt_embs if language_dict == self._cached_language_dict else None
        )
        if prompt_embs is None:
            print(f"[Gr00tN2Policy] Prompt cache MISS, encoding new text from {language_dict}")

        with torch.inference_mode():
            result = self.model.predict(
                collated_inputs,
                num_action_steps=self.num_inference_steps,
                num_video_steps=self.num_video_steps,
                video_guidance_scale=self.video_guidance_scale,
                action_guidance_scale=self.action_guidance_scale,
                shift=self.shift,
                seed=self.seed,
                predict_video=self.save_video_dir is not None or self.enable_n2_pred_comparison,
                joint_denoise=self.joint_denoise,
                kv_cache_cond=self.kv_cache_cond,
                prompt_embs=prompt_embs,
                step_callback=getattr(self, "step_callback", None),
            )
        t_predict = time.perf_counter()
        self._cached_prompt_embs = result.get("prompt_embs")
        self._cached_language_dict = language_dict

        video_latents = result.get("video_latents")
        normalized_action = result.get("action_pred")
        if normalized_action is not None:
            normalized_action = normalized_action.float()

        # Buffer video latents for save_video (not available in action_only mode)
        if video_latents is not None:
            self.video_latents_buffer.append(video_latents.detach())

        self._step_count += 1
        predicted_image = None
        if self.enable_n2_pred_comparison and self.video_latents_buffer:
            # Per-inference decoded frames for env-side pred vs real comparison images.
            predicted_image = self._decode_latest_video_frames()

        if normalized_action is None:
            return {}, {}

        # Decode actions
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.concatenate([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        if predicted_image is not None:
            casted_action["predicted_image"] = predicted_image
        t_end = time.perf_counter()
        action_shapes = {k: v.shape for k, v in casted_action.items() if isinstance(v, np.ndarray)}
        if predicted_image is not None:
            action_shapes["predicted_image"] = {
                view: arr.shape for view, arr in predicted_image.items()
            }
        print(
            f"[Gr00tN2Policy] codec={codec_time:.3f}s  "
            f"preprocess={t_preprocess - t_codec:.3f}s  "
            f"predict={t_predict - t_preprocess:.3f}s  "
            f"postprocess={t_end - t_predict:.3f}s  "
            f"total={t_end - t0:.3f}s  "
            f"action_shapes={action_shapes}"
        )
        return casted_action, {}

    def _decode_latest_video_frames(self) -> dict[str, np.ndarray] | None:
        """Decode latest latent chunk to per-view pixel frames."""
        if not self.video_latents_buffer:
            return None
        step_latents = self.video_latents_buffer[-1]
        decoded_views = self.model.decode_latents(step_latents)
        if not decoded_views:
            return None
        result: dict[str, np.ndarray] = {}
        for v_idx, view_tensor in enumerate(decoded_views):
            if v_idx >= len(N2_VIEW_NAMES):
                break
            result[N2_VIEW_NAMES[v_idx]] = view_tensor[0].cpu().numpy()
        return result

    def _flush_video(self):
        """Decode buffered video latents and save to disk."""
        if not self.video_latents_buffer or not self.save_video_dir:
            self.video_latents_buffer.clear()
            return
        os.makedirs(self.save_video_dir, exist_ok=True)
        traj_id = len([f for f in os.listdir(self.save_video_dir) if f.endswith(".mp4")])
        save_path = os.path.join(self.save_video_dir, f"traj_{traj_id}.mp4")
        try:
            self.save_video(save_path, traj_id=traj_id)
            print(
                f"[Gr00tN2Policy] Saved generated video to {save_path} "
                f"({len(self.video_latents_buffer)} chunks)"
            )
        except Exception as e:
            print(f"[Gr00tN2Policy] Failed to save video: {e}")
        self.video_latents_buffer.clear()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._flush_video()
        self._cached_prompt_embs = None
        self._cached_language_dict = None
        return super().reset(options)

    def save_video(
        self,
        save_path: str,
        traj_id: int,
        video_latents: list | None = None,
        video_texts: list[str] | None = None,
        fps: int = 8,
    ) -> None:
        """Decode video latents and save as a tiled multi-view mp4.

        For multiview models, all views are tiled horizontally into one frame.
        For single-view, saves the video directly.
        """
        from torchvision.io import write_video

        latents = video_latents if video_latents is not None else self.video_latents_buffer
        if not latents:
            return

        # Decode each step's latents separately (views are interleaved within each step),
        # then concatenate decoded frames per view across steps.
        n_views = self.model.max_n_cameras
        per_view_frames: list[list[np.ndarray]] = [[] for _ in range(n_views)]

        for step_latents in latents:
            decoded_views = self.model.decode_latents(step_latents)
            if not decoded_views:
                continue
            for v_idx, view_tensor in enumerate(decoded_views):
                per_view_frames[v_idx].append(view_tensor[0].cpu().numpy())

        if not per_view_frames[0]:
            return

        # Concatenate frames per view across steps: list of (T_total, H, W, C)
        view_frames = [np.concatenate(frames, axis=0) for frames in per_view_frames]

        if len(view_frames) == 1:
            # Single view: save directly
            tiled = view_frames[0]
        else:
            # Multiview: tile all views horizontally → (T, H, V*W, C)
            tiled = np.concatenate(view_frames, axis=2)

        if save_path.endswith(".mp4"):
            full_path = save_path
        else:
            full_path = f"{save_path.split('.mp4')[0]}_{traj_id}.mp4"

        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        write_video(full_path, torch.from_numpy(tiled), fps=fps)

    def get_modality_config(self) -> dict[str, Any]:
        return self.modality_configs
