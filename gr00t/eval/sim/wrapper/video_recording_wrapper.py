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

from collections.abc import Sequence
import os
from pathlib import Path
import subprocess
import uuid

import cv2
import gymnasium as gym
import numpy as np


_H264_CODECS = {"h264", "libx264"}
_H264_CRF = "18"
_H264_PROFILE = "high"
_H264_PIXEL_FORMAT = "yuv420p"


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        max_episode_steps=720,
        fps=20,
        codec="h264",
        overlay_text=True,
        record_video_keys: Sequence[str] | None = None,
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        if record_video_keys is not None and len(record_video_keys) == 0:
            raise ValueError("record_video_keys must not be empty when provided")

        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.max_episode_steps = max_episode_steps
        self.video_dir = video_dir
        self.video_fps = fps
        self.video_codec = codec
        self.video_process = None
        self.video_shape = None
        self.video_dtype = None
        self.file_path = None
        self.overlay_text = overlay_text
        self.record_video_keys = tuple(record_video_keys) if record_video_keys is not None else None

        self.step_count = 0

        self.is_success = False
        self.is_episode_finished = False

        # Caption buffer height is cached on the first overlay frame of each
        # episode so that every frame in the encoded stream has the same total
        # height; the H.264 encoder rejects mid-stream shape changes.
        self.caption_height = None

    def __del__(self):
        try:
            self._close_video_writer()
        except Exception:
            pass

    def _open_video_writer(self):
        if self.file_path is None:
            raise RuntimeError("Cannot write video before a file path is set")
        if self.video_shape is None:
            raise RuntimeError("Cannot open video writer before frame shape is known")

        height, width = self.video_shape[:2]
        codec = "libx264" if self.video_codec in _H264_CODECS else self.video_codec
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.video_fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            codec,
        ]
        if self.video_codec in _H264_CODECS:
            cmd.extend(
                [
                    "-crf",
                    _H264_CRF,
                    "-profile:v",
                    _H264_PROFILE,
                    "-pix_fmt",
                    _H264_PIXEL_FORMAT,
                ]
            )
        cmd.append(str(self.file_path))

        try:
            self.video_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is required for rollout video recording. Install ffmpeg or disable "
                "video recording by leaving video_dir unset."
            ) from exc

    def _write_video_frame(self, frame: np.ndarray):
        if self.video_process is None:
            self.video_shape = frame.shape
            self.video_dtype = frame.dtype
            self._open_video_writer()

        assert frame.shape == self.video_shape
        assert frame.dtype == self.video_dtype

        assert self.video_process is not None
        assert self.video_process.stdin is not None
        self.video_process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def _close_video_writer(self):
        process = self.video_process
        try:
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                stderr = b""
                if process.stderr is not None:
                    stderr = process.stderr.read()
                return_code = process.wait()
                if return_code != 0:
                    message = stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"ffmpeg video recording failed: {message}")
        finally:
            self.video_process = None
            self.video_shape = None
            self.video_dtype = None

    def _get_video_frames(self, obs: dict) -> list[np.ndarray]:
        if self.record_video_keys is None:
            return [frame for key, frame in obs.items() if key.startswith("video.")]

        missing_keys = [key for key in self.record_video_keys if key not in obs]
        if missing_keys:
            raise KeyError(
                f"Video observation keys missing from rollout observation: {missing_keys}"
            )
        return [obs[key] for key in self.record_video_keys]

    def _resize_frames_to_common_height(self, frames):
        """
        Resize all frames to have the same height for horizontal concatenation.
        Ensures both width and height are even numbers for H.264 compatibility.
        """
        if not frames:
            return frames

        # Use minimum height as target
        target_height = min(frame.shape[0] for frame in frames)
        # Ensure even height for H.264 compatibility
        target_height = target_height - (target_height % 2)

        resized_frames = []
        for frame in frames:
            if frame.shape[0] != target_height:
                # Calculate new width maintaining aspect ratio
                h, w = frame.shape[:2]
                new_width = int(w * target_height / h)
                # Ensure even width for H.264 compatibility
                new_width = new_width - (new_width % 2)

                resized_frame = cv2.resize(
                    frame, (new_width, target_height), interpolation=cv2.INTER_LINEAR
                )
                resized_frames.append(resized_frame)
            else:
                resized_frames.append(frame)

        return resized_frames

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        previous_step_count = self.step_count
        self.frames = list()
        self.step_count = 1
        self._close_video_writer()
        # New episode == new video file == new frame shape lock, so
        # drop the cached caption height too.
        self.caption_height = None

        if self.video_dir is not None and self.file_path is not None and self.file_path.exists():
            # rename the file to indicate success or failure
            original_filestem = self.file_path.stem
            new_filestem = f"{original_filestem}_s{int(self.is_success)}"

            # Add intermediate signals to the filename
            if "grasp_obj" in self.intermediate_signals:
                new_filestem += f"_g-o{int(self.intermediate_signals['grasp_obj'])}"
            # We temporarily disable contact metrics because they are not as indicative
            # if "contact_obj" in self.intermediate_signals:
            #     new_filestem += f"_c-o{int(self.intermediate_signals['contact_obj'])}"
            if "grasp_distractor_obj" in self.intermediate_signals:
                new_filestem += (
                    f"_not-g-d{int(not self.intermediate_signals['grasp_distractor_obj'])}"
                )
            # We temporarily disable contact metrics because they are not as indicative
            # if "contact_distractor_obj" in self.intermediate_signals:
            #     new_filestem += (
            #         f"_not-c-d{int(not self.intermediate_signals['contact_distractor_obj'])}"
            #     )
            # The distance metrics are not very informative, so we have excluded them
            # if (
            #     "gripper_obj_dist" in self.intermediate_signals
            #     and "gripper_distractor_dist" in self.intermediate_signals
            # ):
            #     min_gripper_obj_dist = self.intermediate_signals["gripper_obj_dist"]
            #     min_gripper_distractor_dist = self.intermediate_signals["gripper_distractor_dist"]
            #     gripper_obj_dist_lt_gripper_distractor_dist = (
            #         min_gripper_obj_dist < min_gripper_distractor_dist
            #     )
            #     new_filestem += (
            #         f"_o-lt-d{int(gripper_obj_dist_lt_gripper_distractor_dist)}"
            #     )
            #     new_filestem += f"_o-dist{min_gripper_obj_dist:.4f}"
            #     new_filestem += f"_d-dist{min_gripper_distractor_dist:.4f}"

            # Add language following metrics to the filename
            if (
                "grasp_obj" in self.intermediate_signals
                and "grasp_distractor_obj" in self.intermediate_signals
            ):
                success = self.is_success
                grasp_obj = self.intermediate_signals["grasp_obj"]
                not_grasp_distractor_obj = not self.intermediate_signals["grasp_distractor_obj"]

                # 6 cases in total
                cases = [False] * 6

                if success:
                    if grasp_obj and not_grasp_distractor_obj:
                        # case 1: follow language, good motion
                        cases[0] = True
                        case_semantic = "case_1_follow_lang_good_motion"
                    else:
                        # case 2: follow language and success, but probably bad motion
                        cases[1] = True
                        case_semantic = "case_2_follow_lang_success_bad_motion"
                else:
                    if grasp_obj and not_grasp_distractor_obj:
                        # case 3: follow language, but bad motion
                        cases[2] = True
                        case_semantic = "case_3_follow_lang_failed"
                    elif grasp_obj and not not_grasp_distractor_obj:
                        # case 4: touches both objects, not sure whether it follows language, but very likely bad motion
                        cases[3] = True
                        case_semantic = "case_4_touch_both_objects"
                    elif (not grasp_obj) and not_grasp_distractor_obj:
                        # case 5: grasp neither object, so very likely bad motion
                        cases[4] = True
                        case_semantic = "case_5_grasp_neither_object"
                    else:
                        # case 6: grasp distractor object, so it doesn't follow language
                        cases[5] = True
                        case_semantic = "case_6_grasp_distractor_object"

                language_following_rate = cases[0] or cases[1] or cases[2]

                # Add language following metrics to the filename
                # Because the 6 cases are mutually exclusive, we can just use the semantic meaning of the cases
                new_filestem += f"_{case_semantic}_lf-rate{int(language_following_rate)}"

            # We temporarily disable contact metrics because they are not as indicative
            # if (
            #     "contact_obj" in self.intermediate_signals
            #     and "contact_distractor_obj" in self.intermediate_signals
            # ):
            #     success = self.is_success
            #     contact_obj = self.intermediate_signals["contact_obj"]
            #     not_contact_distractor_obj = not self.intermediate_signals["contact_distractor_obj"]

            #     # 6 cases in total
            #     cases = [False] * 6

            #     if success:
            #         if contact_obj and not_contact_distractor_obj:
            #             # case 7: follow language, good motion
            #             cases[0] = True
            #             case_semantic = "case_7_follow_lang_good_motion"
            #         else:
            #             # case 8: follow language and success, but probably bad motion
            #             cases[1] = True
            #             case_semantic = "case_8_follow_lang_success_bad_motion"
            #     else:
            #         if contact_obj and not_contact_distractor_obj:
            #             # case 9: follow language, but bad motion
            #             cases[2] = True
            #             case_semantic = "case_9_follow_lang_failed"
            #         elif contact_obj and not not_contact_distractor_obj:
            #             # case 10: touches both objects, not sure whether it follows language, but very likely bad motion
            #             cases[3] = True
            #             case_semantic = "case_10_touch_both_objects"
            #         elif (not contact_obj) and not_contact_distractor_obj:
            #             # case 11: contact neither object, so very likely bad motion
            #             cases[4] = True
            #             case_semantic = "case_11_contact_neither_object"
            #         else:
            #             # case 12: contact distractor object, so it doesn't follow language
            #             cases[5] = True
            #             case_semantic = "case_12_contact_distractor_object"

            #     contact_language_following_rate = cases[0] or cases[1] or cases[2]
            #     new_filestem += f"_{case_semantic}_clf-rate{int(contact_language_following_rate)}"

            new_file_path = self.video_dir / f"{new_filestem}.mp4"
            should_keep_video = (
                self.is_episode_finished
                or previous_step_count >= self.max_episode_steps
                or self.is_success
            )
            if should_keep_video:
                os.rename(self.file_path, new_file_path)
            else:
                print(
                    f"Skipping video recording for unfinished episode {previous_step_count} / {self.max_episode_steps}"
                )
                os.remove(self.file_path)

        self.is_success = False
        self.is_episode_finished = False
        # "intermediate_signals" contain the metrics for 5DC tasks to indicate language following
        self.intermediate_signals = {}

        if self.video_dir is not None:
            self.file_path = self.video_dir / f"{uuid.uuid4()}.mp4"
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        self.is_episode_finished = bool(result[2] or result[3])
        if self.file_path is not None and ((self.step_count % self.steps_per_render) == 0):
            # frame = self.env.render()
            obs = result[0]
            video_frames = self._get_video_frames(obs)

            assert len(video_frames) > 0, "No video frame found in the observation"

            # Resize frames to common height for horizontal concatenation
            if len(video_frames) > 1:
                video_frames = self._resize_frames_to_common_height(video_frames)

            # Concatenate all video frames horizontally
            if len(video_frames) == 1:
                frame = video_frames[0]
            else:
                frame = np.concatenate(video_frames, axis=1)
            assert frame.dtype == np.uint8

            if self.overlay_text:
                # Droid dataset has "language.language_instruction"
                auto_language_key = [
                    k
                    for k in result[0].keys()
                    if k.startswith("annotation.") or k.startswith("language.")
                ][0]
                # assert auto_language_key in [
                #     "annotation.human.coarse_action",
                #     "annotation.human.task_description",
                # ], f"auto_language_key: {auto_language_key} not valid"
                language = result[0][auto_language_key]
                language = language + " (" + str(int(result[-1]["success"])) + ")"
                # Dynamic font scaling so that the text always fits
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_thickness = 2
                font_color = (255, 255, 255)  # White color in BGR
                padding = 5

                # Target width is frame width minus some padding
                target_width = frame.shape[1] - 2 * padding
                font_scale = 1.0

                # Binary search to find the right font scale
                text_size = cv2.getTextSize(language, font, font_scale, font_thickness)[0]
                if text_size[0] > target_width:
                    # Text too big, scale down
                    while text_size[0] > target_width and font_scale > 0.1:
                        font_scale *= 0.9
                        text_size = cv2.getTextSize(language, font, font_scale, font_thickness)[0]
                else:
                    # Text too small, scale up
                    while text_size[0] < target_width and font_scale < 2.0:
                        font_scale *= 1.1
                        text_size = cv2.getTextSize(language, font, font_scale, font_thickness)[0]
                    font_scale *= 0.9  # Scale back slightly to ensure fit

                _, baseline = cv2.getTextSize(language, font, font_scale, font_thickness)
                caption_height = text_size[1] + baseline + 2 * padding
                if (frame.shape[0] + caption_height) % 2:
                    caption_height += 1

                # Caption height must stay constant for the whole episode so
                # that the encoded frame shape never changes (the H.264 stream
                # rejects late shape changes and the wrapper asserts it).
                # First overlay frame fixes the height; later frames whose
                # natural caption would be taller (e.g. the success suffix
                # changes from "(0)" to "(1)" and the dynamic scaler picks a
                # slightly larger font) shrink font_scale further until they
                # fit in the cached buffer.
                if self.caption_height is None:
                    self.caption_height = caption_height
                else:
                    # If `font_scale` bottoms out at 0.05 with caption_height
                    # still > self.caption_height, we deliberately keep the
                    # buffer at the cached size: frame-shape stability outranks
                    # text completeness here. cv2.putText below will silently
                    # clip the few overflowing pixels, which is preferable to
                    # tripping the wrapper shape-lock assert.
                    while caption_height > self.caption_height and font_scale > 0.05:
                        font_scale *= 0.9
                        text_size, baseline = cv2.getTextSize(
                            language, font, font_scale, font_thickness
                        )
                        caption_height = text_size[1] + baseline + 2 * padding
                        if (frame.shape[0] + caption_height) % 2:
                            caption_height += 1

                caption = np.zeros(
                    (self.caption_height, frame.shape[1], frame.shape[2]), dtype=frame.dtype
                )

                cv2.putText(
                    caption,
                    language,
                    (padding, padding + text_size[1]),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                frame = np.concatenate([frame, caption], axis=0)

            self._write_video_frame(frame)

        info = result[-1]
        self.is_success |= info["success"]

        # Update intermediate signals
        if "intermediate_signals" in info:
            for key, value in info["intermediate_signals"].items():
                if key in [
                    "grasp_obj",
                    "grasp_distractor_obj",
                    "contact_obj",
                    "contact_distractor_obj",
                ]:
                    # For grasp_obj and grasp_distractor_obj, they are boolean metrics
                    # We use |= to accumulate the results
                    initial_value = False
                elif key in ["gripper_obj_dist", "gripper_distractor_dist"]:
                    # For gripper_obj_dist and gripper_distractor_dist, they are float metrics
                    # We use min to accumulate the results
                    initial_value = 1e9  # a large number
                elif key.startswith("_"):
                    # there's a _ duplicate for each of the original keys, which are their masks
                    continue
                else:
                    raise ValueError(f"Unknown key: {key}")

                if key not in self.intermediate_signals:
                    self.intermediate_signals[key] = initial_value

                if key in [
                    "grasp_obj",
                    "grasp_distractor_obj",
                    "contact_obj",
                    "contact_distractor_obj",
                ]:
                    self.intermediate_signals[key] |= value
                elif key in ["gripper_obj_dist", "gripper_distractor_dist"]:
                    original_value = self.intermediate_signals[key]
                    self.intermediate_signals[key] = min(original_value, value)
                else:
                    raise ValueError(f"Unknown key: {key}")

        return result

    def render(self, mode="rgb_array", **kwargs):
        self._close_video_writer()
        return self.file_path
