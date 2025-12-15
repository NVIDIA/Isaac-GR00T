import os
from pathlib import Path
import uuid

import av
import cv2
from gr00t.utils.video_utils import get_accumulate_timestamp_idxs
import gymnasium as gym
import numpy as np


class VideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs,
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile:v": "high"},
            **kwargs,
        )
        return obj

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)

        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recorder: VideoRecorder,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        max_episode_steps=720,
        overlay_text=True,
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.max_episode_steps = max_episode_steps
        self.video_dir = video_dir
        self.video_recorder = video_recorder
        self.file_path = None
        self.overlay_text = overlay_text

        self.step_count = 0

        self.is_success = False

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
        self.video_recorder.stop()

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
            if previous_step_count >= self.max_episode_steps or self.is_success:
                os.rename(self.file_path, new_file_path)
            else:
                print(
                    f"Skipping video recording for unfinished episode {previous_step_count} / {self.max_episode_steps}"
                )
                os.remove(self.file_path)

        self.is_success = False
        # "intermediate_signals" contain the metrics for 5DC tasks to indicate language following
        self.intermediate_signals = {}

        if self.video_dir is not None:
            self.file_path = self.video_dir / f"{uuid.uuid4()}.mp4"
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)

            # frame = self.env.render()
            obs = result[0]
            video_frames = []
            for k, v in obs.items():
                if "video" in k:
                    video_frames.append(v)

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

                # Calculate position
                text_x = padding
                text_y = frame.shape[0] - 20

                # Add dark background rectangle
                cv2.rectangle(
                    frame,
                    (text_x - padding, text_y - text_size[1] - padding),
                    (text_x + text_size[0] + padding, text_y + padding),
                    (0, 0, 0),
                    -1,
                )

                # Add text
                cv2.putText(
                    frame, language, (text_x, text_y), font, font_scale, font_color, font_thickness
                )

            self.video_recorder.write_frame(frame)

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
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_path
