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

# ruff: noqa
# NOTE: this requires installation of the droid repo.
# Adapted from https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from collections import deque

import cv2
import imageio
import numpy as np
import pandas as pd
import tqdm
import tyro
from moviepy.editor import ImageSequenceClip
from PIL import Image

from droid.robot_env import RobotEnv
from server_client import PolicyClient
from utils import resize_with_pad
from scipy.spatial.transform import Rotation

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15
RESOLUTION = (180, 320)  # resize images to this resolution before sending to the policy server

# Matches OXE DROID dataset conversion: R_euler is post-multiplied by this matrix
DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)


def compute_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
    """Convert cartesian_position (XYZ + euler 3D) to eef_9d (XYZ + rot6d).

    Uses extrinsic xyz Euler convention and applies the egocentric rotation correction
    to match the OXE DROID training data convention.
    """
    c = np.asarray(cartesian_position, dtype=np.float64).reshape(6)
    xyz = c[:3]
    euler = c[3:6]
    rot_robot = Rotation.from_euler("xyz", euler).as_matrix()
    rot_mat = rot_robot @ DROID_EEF_ROTATION_CORRECT
    rot6d = rot_mat[:2, :].reshape(6)
    return np.concatenate([xyz, rot6d]).astype(np.float32)


@dataclasses.dataclass
class Args:
    # Hardware parameters

    left_camera_id: str = "<SET THIS>"  # e.g., "24259877"
    right_camera_id: str = "<SET THIS>"  # e.g., "24514023"
    wrist_camera_id: str = "<SET THIS>"  # e.g., "13062452"

    # Policy parameters
    policy_host: str = "localhost"
    policy_port: int = 5555
    policy_api_token: str = None

    results_dir: str = None  # if None, will use the current timestamp as the results directory

    # Rollout parameters
    max_timesteps: int = 600  # how many steps to run each rollout

    # How many actions to execute from a predicted action chunk before querying policy server again
    open_loop_horizon: int = 15
    external_camera: str = (
        "left"  # which exterior camera to use for the policy server, choose from ["left", "right"]
    )
    render_camera: str = "left"  # which camera to render saved video from
    render_fps: int = 50

    debug: bool = False
    vis_cameras: bool = False

    delay_seconds: int = 5


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    assert args.external_camera in ["left", "right"], (
        f"Invalid exterior camera: {args.exterior_camera}"
    )

    if args.results_dir is None:
        results_dir = f"results_gr00t_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    else:
        results_dir = args.results_dir

    # Initialize the Panda environment.
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    os.makedirs(results_dir, exist_ok=True)

    policy_client = PolicyClient(
        host=args.policy_host, port=args.policy_port, api_token=args.policy_api_token
    )

    modality_config = policy_client.get_modality_config()
    video_delta = modality_config["video"].delta_indices
    video_T = len(video_delta)
    video_history_len = max(-min(video_delta), 0) + 1 if video_delta else 1
    video_keys = modality_config["video"].modality_keys
    state_keys = modality_config["state"].modality_keys
    state_T = len(modality_config["state"].delta_indices)
    print(
        f"Model config — video T={video_T} (delta={video_delta}), "
        f"state T={state_T}, keys: video={video_keys}, state={state_keys}"
    )

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    if args.debug:
        debug_dir = os.path.join(results_dir, "debug_data")
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "videos/wrist_image/"), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "videos/exterior_image_1_left/"), exist_ok=True)

    instruction = None
    while True:
        if instruction is None:
            instruction = input("Enter instruction: ")
        else:
            if input("Change instruction? (enter y or n) ").lower() == "y":
                instruction = input("Enter instruction: ")

        time.sleep(args.delay_seconds)

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        if args.debug:
            model_wrist_image_writer = imageio.get_writer(
                os.path.join(
                    debug_dir, "videos/wrist_image/", f"model_wrist_image_{timestamp}.mp4"
                ),
                fps=5,
            )
            model_exterior_image_1_left_writer = imageio.get_writer(
                os.path.join(
                    debug_dir,
                    "videos/exterior_image_1_left/",
                    f"model_exterior_image_1_left_{timestamp}.mp4",
                ),
                fps=5,
            )

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        # Profiling variables (reset for each rollout)
        rollout_start_time = time.time()
        obs_times = deque(maxlen=50)  # Track observation collection times
        server_times = deque(maxlen=50)  # Track server response times
        action_count = 0
        frame_buffer = deque(maxlen=video_history_len)

        for t_step in bar:
            step_start_time = time.time()
            try:
                # Get the current observation
                obs_start_time = time.time()
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )
                obs_time = time.time() - obs_start_time
                obs_times.append(obs_time)

                video.append(curr_obs[f"{args.render_camera}_image"])

                # Resize every step so the rolling frame buffer stays current.
                left_image = resize_with_pad(curr_obs["left_image"], RESOLUTION[0], RESOLUTION[1])
                right_image = resize_with_pad(curr_obs["right_image"], RESOLUTION[0], RESOLUTION[1])
                wrist_image = resize_with_pad(curr_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])

                if args.external_camera == "left":
                    ext_image = left_image
                elif args.external_camera == "right":
                    ext_image = right_image

                frame_buffer.append({"ext": ext_image, "wrist": wrist_image})

                # Send websocket request to policy server if it's time to predict a new chunk
                if (
                    actions_from_chunk_completed == 0
                    or actions_from_chunk_completed >= args.open_loop_horizon
                ):
                    actions_from_chunk_completed = 0

                    if args.debug:
                        model_wrist_image_writer.append_data(wrist_image)
                        model_exterior_image_1_left_writer.append_data(ext_image)

                    # Build video tensor with T frames derived from the model's
                    # delta_indices (e.g. [-15, 0] -> T=2, [0] -> T=1).
                    if video_T == 1:
                        video_dict = {
                            "exterior_image_1_left": ext_image[None, None, ...],
                            "wrist_image_left": wrist_image[None, None, ...],
                        }  # (B=1, T=1, H, W, C)
                    else:
                        hist_frame = frame_buffer[0]
                        cur_frame = frame_buffer[-1]
                        video_dict = {
                            "exterior_image_1_left": np.stack(
                                [hist_frame["ext"], cur_frame["ext"]]
                            )[None, ...],
                            "wrist_image_left": np.stack([hist_frame["wrist"], cur_frame["wrist"]])[
                                None, ...
                            ],
                        }  # (B=1, T=video_T, H, W, C)

                    # Build state dict from the model's reported state keys.
                    state_dict = {}
                    state_source = {
                        "eef_9d": curr_obs["eef_9d"],
                        "gripper_position": curr_obs["gripper_position"],
                        "joint_position": curr_obs["joint_position"],
                    }
                    for key in state_keys:
                        state_dict[key] = state_source[key][None, None, ...].astype(
                            np.float32
                        )  # (B=1, T=1, D)

                    lang_key = modality_config["language"].modality_keys[0]
                    request_data = {
                        "video": video_dict,
                        "state": state_dict,
                        "language": {lang_key: [[instruction]]},
                    }

                    if args.vis_cameras:
                        # viz the left image 1 and wrist image and use cv2 to display them side by side
                        left_image_display = cv2.resize(
                            left_image, (wrist_image.shape[1], wrist_image.shape[0])
                        )
                        combined_display = np.concatenate([left_image_display, wrist_image], axis=1)
                        # convert to bgr
                        combined_display = combined_display[..., ::-1]
                        cv2.imshow("Camera Views", combined_display)
                        cv2.waitKey(1)

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    server_start_time = time.time()
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [N, 8] of joint position actions (7) + gripper position (1)
                        response = policy_client.get_action(request_data)
                    server_time = time.time() - server_start_time
                    server_times.append(server_time)

                    pred_action_chunk = np.concatenate(
                        (
                            response[0]["joint_position"][0],
                            response[0]["gripper_position"][0],
                        ),
                        axis=1,
                    )

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                env.step(action)
                action_count += 1

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - step_start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

                #  profiling stats
                if obs_times:
                    avg_obs_time = np.mean(obs_times) * 1000
                    min_obs_time = np.min(obs_times) * 1000
                    max_obs_time = np.max(obs_times) * 1000
                else:
                    avg_obs_time = min_obs_time = max_obs_time = 0

                if server_times:
                    avg_server_time = np.mean(server_times) * 1000
                    min_server_time = np.min(server_times) * 1000
                    max_server_time = np.max(server_times) * 1000
                else:
                    avg_server_time = min_server_time = max_server_time = 0

                total_elapsed = time.time() - rollout_start_time
                actions_per_sec = action_count / total_elapsed if total_elapsed > 0 else 0

                bar.set_description(
                    f"Obs: {avg_obs_time:.1f}ms [{min_obs_time:.1f}-{max_obs_time:.1f}] | "
                    f"Server: {avg_server_time:.1f}ms [{min_server_time:.1f}-{max_server_time:.1f}] | "
                    f"Actions/sec: {actions_per_sec:.2f}"
                )
            except KeyboardInterrupt:
                break

        os.makedirs(os.path.join(results_dir, "videos"), exist_ok=True)
        video = np.stack(video)
        # replace whitespace with underscores in instruction
        sanitized_instruction = instruction.replace(" ", "_")
        save_filename = os.path.join(
            results_dir, "videos", f"{sanitized_instruction}_video_" + timestamp
        )
        ImageSequenceClip(list(video), fps=args.render_fps).write_videofile(
            save_filename + ".mp4", codec="libx264"
        )

        if args.debug:
            model_wrist_image_writer.close()
            model_exterior_image_1_left_writer.close()

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        new_row = {
            "success": success,
            "duration": t_step,
            "video_filename": save_filename,
        }
        new_index = len(df)
        df.loc[new_index] = new_row

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset(randomize=False)

    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join(results_dir, f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, stereo_camera="left", save_to_disk=False):
    image_observations = obs_dict["image"]
    key_left = f"{args.left_camera_id}_{stereo_camera}"
    key_right = f"{args.right_camera_id}_{stereo_camera}"
    key_wrist = f"{args.wrist_camera_id}_{stereo_camera}"

    left_image = image_observations.get(key_left)
    right_image = image_observations.get(key_right)
    wrist_image = image_observations.get(key_wrist)

    available = list(image_observations.keys())
    assert left_image is not None, (
        f"Left camera not found for key {key_left!r}. Available keys: {available}. "
        "Set --left-camera-id to the ZED serial used in observation keys."
    )
    assert right_image is not None, (
        f"Right camera not found for key {key_right!r}. Available keys: {available}. "
        "Set --right-camera-id to the ZED serial used in observation keys."
    )
    assert wrist_image is not None, (
        f"Wrist camera not found for key {key_wrist!r}. Available keys: {available}. "
        "Set --wrist-camera-id to the ZED serial used in observation keys."
    )

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])
    eef_9d = compute_eef_9d(cartesian_position)

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "eef_9d": eef_9d,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
