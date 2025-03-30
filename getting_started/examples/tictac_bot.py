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


"""
################################################################
This showcase the example of using a VLM as a high-level task
planner (system 2) to plan the next action in a tic-tac-toe game,
and GR00T N1 as the low-level action executor (system 1).

 High-level task
  language description ------>    |---------------------|
 Observation (e.g. image) ----->  |  VLM (e.g. chatgpt) |----|
                                  |---------------------|    |
                                                             |
                                                             |
  robot observation(images, proprio)            language instruction
                  |                                     |
                  v                                     v
           |-------------------------------------------------|
           |  GR00T N1 Vision-Language Action Model          |
           |-------------------------------------------------|
                                |
                                v
                            Robot Action

##################################################################
"""

import cv2
import numpy as np
import torch
import random
import time
from eval_gr00t_so100 import Gr00tRobotInferenceClient, SO100Robot, view_img
from enum import Enum
from pynput import keyboard
import queue
import base64


# Use OpenAI VLM to get the prompt for gr00t
USE_OPENAI_VLM = True

ACTIONS_TO_EXECUTE = 10
ACTION_HORIZON = 16
MODALITY_KEYS = ["single_arm", "gripper"]
HOST = "10.110.17.183"
PORT = 5555
CAM_IDX = 9


#################################################################################


class TaskToString(Enum):
    CENTER_LEFT = "Place the circle to the center left box"
    CENTER_RIGHT = "Place the circle to the center right box"
    CENTER = "Place the circle to the center box"
    CENTER_TOP = "Place the circle to the center top box"
    CENTER_BOTTOM = "Place the circle to the center bottom box"
    BOTTOM_LEFT = "Place the circle to the bottom left corner box"
    BOTTOM_RIGHT = "Place the circle to the bottom right corner box"
    TOP_LEFT = "Place the circle to the top left corner box"
    TOP_RIGHT = "Place the circle to the top right corner box"

    def __str__(self):
        return self.value


#################################################################################


class TicTacToeOpenAIClient:
    def __init__(self):
        # NOTE: Please write your own openai client. User's TODO
        from openai_client import OpenAIClient

        self.openai_client = OpenAIClient()
        self.prompt = self._get_prompt()

    def _get_prompt(self):
        member_names = [member.name for member in TaskToString]  # Use name instead of value
        member_names = [name.replace("_", " ") for name in member_names]
        prompt = "The image shows a robotic setup for playing tic-tac-toe. "
        prompt += "There's a 3x3 grid representing a tic-tac-toe board with some positions already filled. "
        prompt += "Orange circles represent 'O' and blue 'X' pieces represent 'X'. "
        prompt += "You are playing as 'O' (the orange circles). "
        prompt += f"Based on the current board state, what is your best next move to win or block your opponent? "
        prompt += f"Please choose one of the following positions: {', '.join(member_names)}. "
        prompt += "Only choose an empty position that is not currently occupied by 'X' or 'O'. "
        prompt += (
            "Only respond with the position name (e.g., 'CENTER', 'TOP RIGHT', 'BOTTOM LEFT')."
        )
        return prompt

    def get_gr00t_prompt(self, img: np.ndarray) -> str:
        """ "
        This gets the valid prompt for gr00t to execute the VLA tasks
        """
        # Convert numpy array to JPEG bytes
        success, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")

        # Encode the JPEG bytes to base64
        encoded_image = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ]

        response = self.openai_client(
            messages,
            max_tokens=4096,
            temperature=0,
            max_api_call_attempts=2,
            model="gpt-4o",
        )

        if response is None:
            print_green("Error: No response from OpenAI API")
            return random.choice(list(TaskToString))

        response = response.choices[0].message.content
        response = self._filter_response(response)

        try:
            # Try to convert the response to an enum member
            task = TaskToString[response]
            return task
        except KeyError:
            print_green(f"Error: Invalid response from OpenAI: {response}")
            return random.choice(list(TaskToString))

    def _filter_response(self, response: str) -> str:
        # replace space with underscore
        response = response.replace(" ", "_")
        # do a strip, remove all punctuations, and make sure it is uppercase
        selection = response.strip().replace("\n", "").replace(".", "").replace(",", "").upper()
        return selection


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


#################################################################################


if __name__ == "__main__":

    current_task = random.choice(list(TaskToString))
    print_green(f"task: {current_task}")

    if USE_OPENAI_VLM:
        vlm_client = TicTacToeOpenAIClient()

    client_instance = Gr00tRobotInferenceClient(
        host=HOST,
        port=PORT,
        language_instruction=current_task.__str__(),
    )

    robot_instance = SO100Robot(calibrate=False, enable_camera=True, cam_idx=CAM_IDX)

    #####################################################################
    # Keyboard Listener, click space to pause/resume the robot
    #####################################################################
    # Define global variables that will be used in the callback
    # Create a queue for thread-safe communication
    paused = False
    command_queue = queue.Queue()
    # Define command types
    COMMAND_PAUSE = "PAUSE"
    COMMAND_RESUME = "RESUME"

    # Function to handle keyboard events with pynput
    def on_key_press(key):
        global paused, command_queue
        try:
            # Check if space key is pressed
            if key == keyboard.Key.space:
                paused = not paused
                print(f"Execution {'paused' if paused else 'resumed'}")
                if paused:
                    # Put pause command in queue instead of directly calling functions
                    command_queue.put(COMMAND_PAUSE)
                else:
                    command_queue.put(COMMAND_RESUME)
        except AttributeError:
            pass

    # Register the keyboard listener with pynput
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    #####################################################################

    with robot_instance.activate():

        print("--------------------------------")
        print_green(" 🤖 Tic-Tac-Toe Bot is running")
        print("--------------------------------")

        # get the initial image
        if USE_OPENAI_VLM:
            img = robot_instance.get_current_img()
            prompt = vlm_client.get_gr00t_prompt(img)
            print_green(f" 🤖 Robot decided the move (0penAI VLM): \n -> '{prompt}'")
            print_green(f" 🦾 GR00T VLA is executing the move")
            client_instance.set_lang_instruction(prompt.__str__())

        while True:  # Run indefinitely until manually stopped
            # Process any commands in the queue
            try:
                while not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    if cmd == COMMAND_PAUSE:
                        robot_instance.go_home()
                        img = robot_instance.get_current_img()
                        view_img(img)
                        print_yellow(" 👨 User's turn to make a move! ")

                    elif cmd == COMMAND_RESUME:
                        # Generate a new task when resuming
                        if USE_OPENAI_VLM:
                            print_green(f" 🤖 Robot is thinking........ (aka openai vlm)")

                            img = robot_instance.get_current_img()
                            current_task = vlm_client.get_gr00t_prompt(img)
                            print_green(f" 🤖 Robot decided the move: \n -> '{current_task}'")

                        else:
                            current_task = random.choice(list(TaskToString))
                            print_green(
                                f" 🤖 Robot selected new random move: \n -> '{current_task}'"
                            )
                        print_green(f" 🦾 GR00T VLA is executing the move")
                        client_instance.set_lang_instruction(current_task.__str__())

                        # TODO(YL) remove this. this makes it easier to be in picking state
                        # target_state = torch.tensor([130, 100, 90, 100, -80, 20])
                        # robot_instance.set_target_state(target_state)
                        # time.sleep(1)
            except queue.Empty:
                pass

            # When the process is not paused. (robot is executing)
            if not paused:
                for i in range(ACTIONS_TO_EXECUTE):
                    if paused:  # Check if paused before starting a new action
                        break

                    img = robot_instance.get_current_img()
                    view_img(img)

                    state = robot_instance.get_current_state()
                    action = client_instance.get_action(img, state)
                    start_time = time.time()

                    for j in range(ACTION_HORIZON):
                        if paused:  # Check if paused during action execution
                            break

                        concat_action = np.concatenate(
                            [np.atleast_1d(action[f"action.{key}"][j]) for key in MODALITY_KEYS],
                            axis=0,
                        )
                        assert concat_action.shape == (6,), concat_action.shape
                        robot_instance.set_target_state(torch.from_numpy(concat_action))
                        time.sleep(0.025)

                        # get the realtime image
                        img = robot_instance.get_current_img()
                        view_img(img)

            else:
                # When paused, just wait a bit to avoid busy waiting
                time.sleep(0.1)
                img = robot_instance.get_current_img()
                view_img(img)
