# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# LeKiwi Mobile Robot
import argparse
import logging
import time
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# LeRobot imports
try:
    from lerobot.common.robot_devices.control_utils import (
        busy_wait,
        init_keyboard_listener,
        is_headless,
        log_control_info,
        stop_recording,
    )
    from lerobot.common.robot_devices.robots.configs import LeKiwiRobotConfig
    from lerobot.common.robot_devices.robots.utils import make_robot_from_config
    from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError
except ImportError as e:
    print(f"LeRobot import error: {e}. Make sure lerobot library is installed and accessible.")
    print("Try running: pip install -e .")
    exit()
except FileNotFoundError as e:
    print(f"Control utils import error: {e}. Ensure lerobot common code is accessible.")
    exit()


# GR00T / Inference imports
try:
    from service import ExternalRobotInferenceClient
except ImportError:
    print("Could not import ExternalRobotInferenceClient. Make sure service.py is accessible.")
    print("Using Dummy ExternalRobotInferenceClient.")

    class ExternalRobotInferenceClient:
        def __init__(self, host, port):
            pass

        def ping(self):
            return False

        def get_action(self, obs):
            return {}  # Return empty dict for dummy


# Import tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Progress bars will not be shown.")

    # Define tqdm as identity function if not found
    def tqdm(iterable, **kwargs):
        yield from iterable


#################################################################################


class LewikiRobotClient:
    """Client-side interface for LeKiwi using MobileManipulator."""

    def __init__(self, calibrate_leader_on_connect=True):
        self.config = LeKiwiRobotConfig()
        self.robot = make_robot_from_config(self.config)
        self.robot_type = self.config.type
        self.leader_arms = self.robot.leader_arms
        self.follower_arms = self.robot.follower_arms
        self.camera_names = list(self.config.cameras.keys())
        self.cameras = self.config.cameras
        self.calibrate_leader_on_connect = calibrate_leader_on_connect
        self.logs = self.robot.logs
        print(f"LewikiRobotClient initialized for cameras: {self.camera_names}")

    @property
    def is_connected(self):
        return hasattr(self.robot, "is_connected") and self.robot.is_connected

    def connect(self):
        if self.is_connected:
            print("Warning: Robot client might already be connected.")
            return
        try:
            print("Connecting LeKiwi client (leader arm and ZMQ)...")
            self.robot.connect()
            print("================> LeKiwi Robot Client is connected =================")
        except Exception as e:
            print(f"Error connecting LewikiRobotClient: {e}")
            raise

    def disconnect(self):
        if self.is_connected:
            print("Disconnecting LeKiwi client...")
            try:
                self.robot.disconnect()
                print("================> LeKiwi Robot Client disconnected")
            except Exception as e:
                print(f"Error during disconnection: {e}")

    def capture_observation(self):
        return self.robot.capture_observation()

    def send_action(self, action_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(action_tensor, torch.Tensor):
            action_tensor = torch.tensor(action_tensor, dtype=torch.float32)
        if action_tensor.shape != (9,):
            raise ValueError(f"ACTION ERROR: Expected action shape (9,), got {action_tensor.shape}")
        return self.robot.send_action(action_tensor)

    def get_current_state_numpy(self) -> np.ndarray:
        obs = self.capture_observation()
        state_tensor = obs.get("observation.state")
        if state_tensor is None or not isinstance(state_tensor, torch.Tensor):
            print("Warning: 'observation.state' not found or invalid. Returning zeros.")
            return np.zeros(9, dtype=np.float32)
        return state_tensor.cpu().data.numpy()

    def get_camera_images_rgb(self) -> dict[str, np.ndarray]:
        obs = self.capture_observation()
        images_rgb = {}
        for name in self.camera_names:
            key = f"observation.images.{name}"
            default_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cam_config = self.cameras.get(name)
            if cam_config:
                default_img = np.zeros((cam_config.height, cam_config.width, 3), dtype=np.uint8)

            img_tensor = obs.get(key)
            if (
                img_tensor is not None
                and isinstance(img_tensor, torch.Tensor)
                and img_tensor.numel() > 0
            ):
                img_rgb = img_tensor.cpu().data.numpy()
                if img_rgb is not None and img_rgb.ndim == 3:
                    try:
                        # No conversion needed if already RGB
                        # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        images_rgb[name] = img_rgb
                    except Exception as e:
                        print(f"W: Error processing '{name}': {e}. Using placeholder.")
                        images_rgb[name] = default_img
                else:
                    images_rgb[name] = default_img
            else:
                images_rgb[name] = default_img
        return images_rgb

    def __del__(self):
        if hasattr(self, "robot") and self.robot:
            self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:
    """Wraps ExternalRobotInferenceClient for LeKiwi observations."""

    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Perform the task.",
        img_size=(480, 640),
    ):
        self.language_instruction = language_instruction
        self.img_size = img_size
        try:
            self.policy = ExternalRobotInferenceClient(host=host, port=port)
            print(f"GR00T Inference Client trying to connect to {host}:{port}")
            if not self.policy.ping():
                print(f"W: Could not ping inference server at {host}:{port}.")
            else:
                print("Inference server ping successful.")
        except Exception as e:
            print(f"Error creating/pinging inference client: {e}")
            raise

    def get_action_chunk(self, images: dict[str, np.ndarray], state: np.ndarray) -> dict:
        """Formats observations and calls the GR00T server."""
        if state.shape != (9,):
            raise ValueError(f"STATE ERROR: Expected state shape (9,), got {state.shape}")

        default_img = np.zeros(self.img_size + (3,), dtype=np.uint8)
        # Observation keys sent to the server still use the grouped state keys
        # as defined in LekiwiDataConfig (assuming that wasn't changed for input)
        obs_dict = {
            "video.front": images.get("front", default_img)[np.newaxis, ...],
            "video.wrist": images.get("wrist", default_img)[np.newaxis, ...],
            "state.single_arm": state[0:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "state.mobile_base": state[6:9][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }

        try:
            res = self.policy.get_action(obs_dict)
            if not isinstance(res, dict) or not res:
                raise RuntimeError("Invalid/empty response from inference server")
            # Check if the *returned* action keys match the new expected structure
            # This depends on the inference server using a LekiwiDataConfig with updated action_keys
            expected_action_keys = {"action.single_arm", "action.gripper", "action.mobile_base"}
            if not expected_action_keys.issubset(res.keys()):
                print(f"Warning: Action chunk missing expected keys. Got: {list(res.keys())}")
                # Might need error handling depending on strictness
            #  raise RuntimeError(f"Missing expected action keys in response. Got {res.keys()}")
            return res
        except Exception as e:
            print(f"Error during inference request: {e}")
            raise


#################################################################################


_figs = {}


def view_img_mpl(img, window_name="MPL Robot View"):
    """Uses Matplotlib for non-blocking image viewing."""
    global _figs
    if window_name not in _figs:
        try:
            fig, ax = plt.subplots(1, 1)
            fig.canvas.manager.set_window_title(window_name)
            _figs[window_name] = (fig, ax)
            plt.ion()
            plt.show(block=False)
        except Exception as e:
            print(f"Matplotlib display error for '{window_name}': {e}. Display disabled.")
            _figs[window_name] = (None, None)
            return
    fig, ax = _figs[window_name]
    if fig is None or ax is None:
        return
    try:
        ax.clear()
        ax.imshow(img)
        ax.axis("off")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception as e:
        print(f"Matplotlib plotting error for '{window_name}': {e}")


def display_images_cv2(obs_dict, robot_camera_names):
    """Displays images from observation dict using OpenCV."""
    for name in robot_camera_names:
        key = f"observation.images.{name}"
        img_tensor = obs_dict.get(key)
        if (
            img_tensor is not None
            and isinstance(img_tensor, torch.Tensor)
            and img_tensor.numel() > 0
        ):
            img_rgb_np = img_tensor.cpu().numpy()
            if img_rgb_np.ndim == 3:
                try:
                    img_bgr = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"CV2_{name}", img_bgr)
                except cv2.error as e:
                    print(f"Error showing CV2 image for {name}: {e}")
    cv2.waitKey(1)


#################################################################################


def main(args):
    """Main function to run the GR00T policy on LeKiwi."""

    CONTROL_DT = 1.0 / args.control_freq
    ACTION_KEYS_ORDER = ["action.single_arm", "action.gripper", "action.mobile_base"]
    ACTION_DIMS = {"action.single_arm": 5, "action.gripper": 1, "action.mobile_base": 3}
    EXPECTED_TOTAL_ACTION_DIM = sum(ACTION_DIMS.values())  # This should still be 9

    # --- Initialization ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    listener, events = init_keyboard_listener()
    headless_mode = is_headless()

    robot = None
    client = None

    try:
        logging.info("Mode: Running GR00T Policy")
        logging.info("Initializing GR00T policy client...")
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction=args.lang,
        )

        logging.info("Initializing LeKiwi robot client...")
        robot = LewikiRobotClient(calibrate_leader_on_connect=(not args.skip_leader_calib))

        logging.info("Connecting robot...")
        robot.connect()

        logging.info("Starting policy execution loop...")
        executed_steps = 0
        action_chunk = None
        chunk_step_index = 0
        start_time = time.perf_counter()
        timestamp = 0.0
        control_time_s = (
            float("inf") if args.control_duration_s is None else args.control_duration_s
        )

        # --- Control Loop ---
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if events.get("stop_recording", False) or events.get("exit_early", False):
                logging.warning("Exit signal received.")
                break
            # 1. Get Observations
            try:
                obs_dict_raw = robot.capture_observation()
                state_np = robot.get_current_state_numpy()
                images_rgb = robot.get_camera_images_rgb()
            except Exception as e:
                logging.error(f"Error getting observation at step {executed_steps}: {e}")
                traceback.print_exc()
                break

            # Display Cameras
            if args.display_mpl:
                if "front" in images_rgb:
                    view_img_mpl(images_rgb["front"], window_name="MPL LeKiwi Front")
                if "wrist" in images_rgb:
                    view_img_mpl(images_rgb["wrist"], window_name="MPL LeKiwi Wrist")
            if args.display_cv2 and not headless_mode:
                display_images_cv2(obs_dict_raw, robot.camera_names)

            # 2. Get Action Chunk from Policy if needed
            if action_chunk is None or chunk_step_index >= args.action_horizon:
                try:
                    action_chunk = client.get_action_chunk(images_rgb, state_np)
                    # Check if the received chunk contains the new keys
                    missing_keys = [key for key in ACTION_KEYS_ORDER if key not in action_chunk]
                    if missing_keys:
                        logging.error(
                            f"Inference server response missing required action keys: {missing_keys}"
                        )
                        logging.error(f"Available keys: {list(action_chunk.keys())}")
                        logging.error(
                            "Ensure the inference server is using a LekiwiDataConfig with matching action_keys."
                        )
                        break
                    chunk_step_index = 0
                except Exception as e:
                    logging.error(
                        f"Stopping execution due to inference error at step {executed_steps}: {e}"
                    )
                    traceback.print_exc()
                    break

            # 3. Extract and Execute Single Action Step from Chunk
            try:
                # --- This block now iterates through the new ACTION_KEYS_ORDER ---
                action_step_list = []
                for key in ACTION_KEYS_ORDER:  # Iterate through individual action keys
                    # Get the action value for the current step in the chunk for this specific key
                    action_part = action_chunk[key][chunk_step_index]
                    expected_dim = ACTION_DIMS[key]  # Should always be 1 now
                    action_part_np = np.atleast_1d(action_part)  # Ensure it's an array
                    if action_part_np.shape[0] != expected_dim:
                        # This check might be less critical if dim is always 1, but keep for safety
                        raise ValueError(
                            f"Action part '{key}' dim {action_part_np.shape[0]} != expected {expected_dim}"
                        )
                    action_step_list.append(
                        action_part_np
                    )  # Append the single value (as a 1-element array)

                # Concatenate all single values into the 9-element action tensor
                action_tensor = torch.from_numpy(np.concatenate(action_step_list)).float()
                # --- End of updated action processing block ---

                if action_tensor.shape[0] != EXPECTED_TOTAL_ACTION_DIM:
                    # This check should still pass if concatenation is correct
                    raise ValueError(
                        f"Concat action shape {action_tensor.shape} != expected ({EXPECTED_TOTAL_ACTION_DIM},)"
                    )

                robot.send_action(action_tensor)
                chunk_step_index += 1
                executed_steps += 1
                print("executed action:", executed_steps)
            except Exception as e:
                logging.error(
                    f"Error processing/executing action step {executed_steps} (chunk index {chunk_step_index}): {e}"
                )
                traceback.print_exc()
                break

            # 5. Maintain Control Frequency & Logging
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(CONTROL_DT - dt_s)
            loop_duration = time.perf_counter() - start_loop_t
            log_control_info(
                robot, loop_duration, frame_index=executed_steps, fps=args.control_freq
            )

            timestamp = time.perf_counter() - start_time

            if args.total_steps is not None and executed_steps >= args.total_steps:
                logging.info(f"Reached total step limit ({args.total_steps}).")
                break

        logging.info(
            f"Finished execution loop after {executed_steps} steps and {timestamp:.2f} seconds."
        )

    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt received.")
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")
        traceback.print_exc()
    finally:
        logging.info("Cleaning up...")
        stop_recording(robot, listener, args.display_cv2)
        if args.display_mpl:
            plt.close("all")
        logging.info("Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GR00T policy inference on LeKiwi robot using control_utils."
    )
    # --- Policy Arguments ---
    parser.add_argument(
        "--host", type=str, default="localhost", help="Inference server hostname/IP."
    )
    parser.add_argument("--port", type=int, default=5557, help="Inference server port.")
    parser.add_argument(
        "--lang",
        type=str,
        default="Drive forward, pickup the object and put it in the red box. Drive back.'",
        help="Language instruction.",
    )  # Example Lang
    # --- Control Arguments ---
    parser.add_argument(
        "--action_horizon", type=int, default=16, help="Steps to execute from action chunk."
    )
    parser.add_argument(
        "--control_freq", type=float, default=30.0, help="Target control frequency (Hz)."
    )
    parser.add_argument(
        "--control_duration_s",
        type=float,
        default=None,
        help="Max duration to run (seconds). Default: run indefinitely or until total_steps.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Max number of steps to execute. Default: run indefinitely or until control_duration_s.",
    )
    # --- Robot Arguments ---
    parser.add_argument(
        "--skip_leader_calib", action="store_true", help="Skip local leader arm calibration."
    )
    # --- Display Arguments ---
    parser.add_argument(
        "--display_mpl", action="store_true", help="Show camera feeds using Matplotlib."
    )
    parser.add_argument(
        "--display_cv2", action="store_true", help="Show camera feeds using OpenCV."
    )

    parsed_args = parser.parse_args()

    if parsed_args.control_duration_s is None and parsed_args.total_steps is None:
        print(
            "Warning: No duration or step limit set. Running indefinitely until keyboard interrupt (Ctrl+C or Esc)."
        )
    if parsed_args.control_freq <= 0:
        raise ValueError("--control_freq must be positive.")
    if parsed_args.action_horizon <= 0:
        raise ValueError("--action_horizon must be positive.")

    main(parsed_args)
