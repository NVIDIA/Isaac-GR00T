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


import time

import numpy as np

from gr00t.eval.rtc_wrapper import RTCPolicyWrapper

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--max_rtc_overlap_factor", type=float, default=0.75)
    parser.add_argument("--http_server", type=bool, default=False)
    args = parser.parse_args()

    if args.http_server:
        """This requires: pip install requests json-numpy"""
        from gr00t.eval.http_service import HttpClientPolicy  # noqa: F401

        policy = HttpClientPolicy(args.host, args.port)
    else:
        from gr00t.eval.robot import RobotInferenceClient  # noqa: F401

        policy = RobotInferenceClient(args.host, args.port)

    # wrap the policy with RTCPolicyWrapper
    policy = RTCPolicyWrapper(
        policy, args.control_freq, args.denoising_steps, args.max_rtc_overlap_factor
    )

    for i in range(20):
        t = time.time()
        obs = {
            "video.test": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "video.ego_view": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }
        action = policy.get_action(obs)

        # NOTE(youliang): you will execute your action here in the real robot
        # env.step(action)

        # lazy sleep to simulate the inference frequency
        time.sleep(0.5)
        print(f"{i}:  used time {time.time() - t}")
