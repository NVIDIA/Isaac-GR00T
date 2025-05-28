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

# SO100 Real Robot
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# Import tqdm for progress bar
from tqdm import tqdm

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
# from service import ExternalRobotInferenceClient
from gr00t.eval.service import ExternalRobotInferenceClient

#################################################################################


class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False, cam_idx=9):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.cam_idx = cam_idx
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {"webcam": OpenCVCameraConfig(cam_idx, 30, 640, 480, "bgr")}

        # Set the robot arms
        if True:
            from lerobot.common.robot_devices.motors.configs import (
                FeetechMotorsBusConfig,
            )

            self.config.follower_arms = {
                "main": FeetechMotorsBusConfig(
                    port="/dev/ttyACM0",
                    motors={
                        # name: (index, model)
                        "shoulder_pan": [1, "sts3215"],
                        "shoulder_lift": [2, "sts3215"],
                        "elbow_flex": [3, "sts3215"],
                        "wrist_flex": [4, "sts3215"],
                        "wrist_roll": [5, "sts3215"],
                        "gripper": [6, "sts3215"],
                    },
                ),
            }

        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        self.camera = self.robot.cameras["webcam"] if self.enable_camera else None
        if self.camera is not None:
            self.camera.connect()
        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        # print("current_state", current_state)
        # print all keys of the observation
        # print("observation keys:", self.robot.capture_observation().keys())
        current_state = torch.tensor([90, 90, 90, 90, -70, 30])
        self.robot.send_action(current_state)
        time.sleep(2)
        print("-------------------------------- moving to initial pose")

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("-------------------------------- moving to home pose")
        home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        img = self.get_observation()["observation.images.webcam"].data.numpy()
        # convert bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO100 Robot disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


# policy = DiffusionPolicy.from_pretrained(
#     "/home/youliang/lerobot/outputs/train/so100_dp001/checkpoints/100000/pretrained_model"
# )


# # policy = PI0Policy.from_pretrained("/home/youliang/lerobot/outputs/train/so100_pi0/checkpoints/100000/pretrained_model")


# # policy = PI0Policy.from_pretrained("lerobot/pi0")

# print(policy.config)
# print(policy.config.input_features)
# print(policy.config.output_features)


# action_horizon = policy.config.n_action_steps
# print(f"action_horizon: {action_horizon}")

# # {'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)), 'observation.images.webcam': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))}
# # {'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(6,))}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# obs_dict = {
#     "observation.state": torch.randn(1, 6).to(device),
#     "observation.images.webcam": torch.randn(1, 3, 480, 640).to(device),
#     # "task": ["push"],
# }

# with torch.inference_mode():
#     for i in range(10):
#         print(f"iteration {i}")
#         for i in range(action_horizon):
#             action = policy.select_action(obs_dict)
#             print(action)


import torch


class DiffusionPolicy:
    def __init__(self, model_path, device="cuda"):
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        self.policy = DiffusionPolicy.from_pretrained(model_path)
        self.device = device
        self.horizon = self.policy.config.n_action_steps
        self.language_instruction = None

    def get_action(self, img, state) -> np.ndarray:
        # add batch dimension
        img = img[np.newaxis, :, :, :]
        state = state[np.newaxis, :]
        # convert to float32
        img = torch.from_numpy(img).to(self.device)
        img = img.to(torch.float32) / 255.0
        img = img.permute(0, 3, 1, 2)
        state = torch.from_numpy(state).to(self.device)
        state = state.to(torch.float32)
        obs_dict = {
            "observation.images.webcam": img,
            "observation.state": state,
        }
        print(img.shape, state.shape)
        actions = []
        for i in range(self.horizon):
            start_time = time.time()
            action = self.policy.select_action(obs_dict)
            print(f"iteration {i} time taken {time.time() - start_time:.2f} seconds")
            # convert to numpy
            # action = action.squeeze(0)
            action = action.cpu().numpy()
            actions.append(action)
        # return (horizon, action_dim)
        actions = np.concatenate(actions, axis=0)
        assert actions.shape == (self.horizon, 6), actions.shape
        return actions


class Pi0Policy:
    def __init__(self, model_path, language_instruction, device="cuda"):
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

        self.policy = PI0Policy.from_pretrained(model_path)
        self.device = device
        self.horizon = self.policy.config.n_action_steps
        self.language_instruction = language_instruction

    def get_action(self, img, state) -> np.ndarray:
        img = torch.from_numpy(img).to(self.device)
        img = img.to(torch.float32) / 255.0
        img = img.permute(2, 0, 1).contiguous()

        # add batch dimension
        img = img[np.newaxis, :, :, :]
        state = state[np.newaxis, :]
        # convert to float32
        state = torch.from_numpy(state).to(torch.float32).to(self.device)
        obs_dict = {
            "observation.images.webcam": img,
            "observation.state": state,
            "task": [self.language_instruction],
        }
        # print(img.shape, state.shape)
        actions = []
        for i in range(self.horizon):
            action = self.policy.select_action(obs_dict)
            # convert to numpy
            # action = action.squeeze(0)
            action = action.cpu().numpy()
            actions.append(action)
        # return (horizon, action_dim)
        actions = np.concatenate(actions, axis=0)
        assert actions.shape == (self.horizon, 6), actions.shape
        # print(actions.shape)
        return actions


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img, state):
        # print(self.language_instruction)
        obs_dict = {
            "video.webcam": img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }
        res = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


def get_language(language_instruction=None):
    if language_instruction is None:
        # get the language instruction from the user
        language_instruction = input(
            "Please enter the language instruction (default: Pick up the fruits and place them on the plate.): "
        )

        if language_instruction == "":
            language_instruction = "Pick up the fruits and place them on the plate."

    print("lang_instruction: ", language_instruction)

    # check if lang is a number
    if language_instruction.isdigit():
        # convert to int
        language_instruction = int(language_instruction)
        print("lang_instruction converted to int: ", language_instruction)

        from tictac_bot import TaskToString

        num = int(language_instruction)
        if num == 1:
            language_instruction = TaskToString.TOP_RIGHT
        elif num == 2:
            language_instruction = TaskToString.CENTER_TOP
        elif num == 3:
            language_instruction = TaskToString.TOP_LEFT
        elif num == 4:
            language_instruction = TaskToString.CENTER_RIGHT
        elif num == 5:
            language_instruction = TaskToString.CENTER
        elif num == 6:
            language_instruction = TaskToString.CENTER_LEFT
        elif num == 7:
            language_instruction = TaskToString.BOTTOM_RIGHT
        elif num == 8:
            language_instruction = TaskToString.CENTER_BOTTOM
        elif num == 9:
            language_instruction = TaskToString.BOTTOM_LEFT
        else:
            print("Invalid lang_instruction number. Please enter a number between 1 and 9.")
            exit(1)
        # convert to string
        language_instruction = str(language_instruction)
        print("lang_instruction converted to string: ", language_instruction)
    print("lang_instruction: ", language_instruction)
    return language_instruction


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="10.110.17.183")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=550)
    parser.add_argument("--cam_idx", type=int, default=1)
    parser.add_argument(
        "--lang_instruction", type=str, default="Pick up the fruits and place them on the plate."
    )
    parser.add_argument("--record_imgs", action="store_true")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")  # TIMEOUT
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--use_pi0", action="store_true")
    args = parser.parse_args()

    # print lang_instruction
    language_instruction = args.lang_instruction
    language_instruction = get_language(language_instruction)

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["single_arm", "gripper"]
    if USE_POLICY:
        if args.use_dp:
            client = DiffusionPolicy(
                # model_path="/home/youliang/lerobot/outputs/train/so100_dp001/checkpoints/100000/pretrained_model",
                model_path="/home/youliang/lerobot/outputs/train/so100_dp_b256/checkpoints/last/pretrained_model",
                device="cuda",
            )
        elif args.use_pi0:
            client = Pi0Policy(
                # TICTAC TOE
                # model_path="/home/youliang/lerobot/outputs/train/tictac_pi0/checkpoints/last/pretrained_model",
                # SO100 FRUITS
                model_path="/home/youliang/lerobot/outputs/train/dp_so100_pi0_b16/checkpoints/last/pretrained_model",
                language_instruction=language_instruction,
                device="cuda",
            )
        else:
            client = Gr00tRobotInferenceClient(
                host=args.host,
                port=args.port,
                language_instruction=language_instruction,
            )

        if args.record_imgs:
            # create a folder to save the images and delete all the images in the folder
            os.makedirs("eval_images", exist_ok=True)
            for file in os.listdir("eval_images"):
                os.remove(os.path.join("eval_images", file))

        robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=args.cam_idx)
        image_count = 0
        # for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
        with robot.activate():
            while True:
                eps_start_time = time.time()
                # check if timeout is reached
                while time.time() - eps_start_time < args.timeout:
                    # get the realtime image
                    print_yellow(
                        f" Current elapsed time: {time.time() - eps_start_time:.2f} seconds"
                    )
                    img = robot.get_current_img()
                    view_img(img)
                    state = robot.get_current_state()
                    action = client.get_action(img, state)
                    start_time = time.time()
                    for i in range(ACTION_HORIZON):
                        if args.use_dp or args.use_pi0:
                            concat_action = action[i]
                        else:
                            concat_action = np.concatenate(
                                [
                                    np.atleast_1d(action[f"action.{key}"][i])
                                    for key in MODALITY_KEYS
                                ],
                                axis=0,
                            )
                        assert concat_action.shape == (6,), concat_action.shape
                        robot.set_target_state(torch.from_numpy(concat_action))
                        time.sleep(0.02)

                        # get the realtime image
                        img = robot.get_current_img()
                        view_img(img)

                        if args.record_imgs:
                            # resize the image to 320x240
                            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (320, 240))
                            cv2.imwrite(f"eval_images/img_{image_count}.jpg", img)
                            image_count += 1

                        # 0.05*16 = 0.8 seconds
                        # print("executing action", i, "time taken", time.time() - start_time)
                    # print("Action chunk execution time taken", time.time() - start_time)

                robot.move_to_initial_pose()
                language_instruction = get_language(None)
                # if language_instruction is in client
                client.language_instruction = language_instruction
                print("Language instruction updated to: ", language_instruction)

    else:
        # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
        dataset = LeRobotDataset(
            repo_id="",
            root=args.dataset_path,
            episodes=[24],
        )

        robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=args.cam_idx)

        with robot.activate():
            print("Run replay of the dataset")
            actions = []
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
                # while True:
                #     i = 0
                action = dataset[i]["action"]
                img = dataset[i]["observation.images.webcam"].data.numpy()
                # original shape (3, 480, 640) for image data
                realtime_img = robot.get_current_img()

                img = img.transpose(1, 2, 0)
                view_img(img, realtime_img)
                actions.append(action)
                robot.set_target_state(action)
                time.sleep(0.05)

            # plot the actions
            plt.plot(actions)
            plt.show()

            print("Done all actions")
            robot.go_home()
            print("Done home")
