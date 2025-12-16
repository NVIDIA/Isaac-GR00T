"""
SO100 Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the SO100 / SO101 robots
using the GR00T Policy API.

Major responsibilities:
    • Initialize robot hardware from a RobotConfig (LeRobot)
    • Convert robot observations into GR00T VLA inputs
    • Query the GR00T policy server (PolicyClient)
    • Decode multi-step (temporal) model actions back into robot motor commands
    • Stream actions to the real robot in real time

This file is meant to be a simple, readable reference
for real-world policy debugging and demos.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import time
from typing import Any, Dict, List

import draccus
from gr00t.policy.server_client import PolicyClient

# Importing various robot configs ensures CLI autocompletion works
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import init_logging, log_say
import numpy as np


def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
    return obs


class So100Adapter:
    """
    Adapter between:
        • Raw robot observation dictionary
        • GR00T VLA input format
        • GR00T action chunk → robot joint commands

    Responsible for:
        • Packaging camera frames as obs["video"]
        • Building obs["state"] for arm + gripper
        • Adding language instruction
        • Adding batch/time dimensions
        • Decoding model action chunks into real robot actions
    """

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client

        # SO100 joint ordering used for BOTH training + robot execution
        self.robot_state_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.camera_keys = ["front", "wrist"]

    # -------------------------------------------------------------------------
    # Observation → Model Input
    # -------------------------------------------------------------------------
    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """
        Convert raw robot observation dict into the structured GR00T VLA input.
        """
        model_obs = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Arm + gripper state
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:5],  # (5,)
            "gripper": state[5:6],  # (1,)
        }

        # (3) Language
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    # -------------------------------------------------------------------------
    # Model Action Chunk → Robot Motor Commands
    # -------------------------------------------------------------------------
    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        """
        chunk["single_arm"]: (B, T, 5)
        chunk["gripper"]:    (B, T, 1)

        Convert to:
            {
                "shoulder_pan.pos": val,
                ...
            }
        for timestep t.
        """
        single_arm = chunk["single_arm"][0][t]  # (5,)
        gripper = chunk["gripper"][0][t]  # (1,)

        full = np.concatenate([single_arm, gripper], axis=0)  # (6,)

        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        """
        Returns a list of robot motor commands (one per model timestep).
        """
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        # Determine horizon
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Evaluation Config
# =============================================================================


@dataclass
class EvalConfig:
    """
    Command-line configuration for real-robot policy evaluation.
    """

    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "Grab markers and place into pen holder."
    play_sounds: bool = False
    timeout: int = 60


# =============================================================================
# Main Eval Loop
# =============================================================================


@draccus.wrap()
def eval(cfg: EvalConfig):
    """
    Main entry point for real-robot policy evaluation.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # -------------------------------------------------------------------------
    # 1. Initialize Robot Hardware
    # -------------------------------------------------------------------------
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    # -------------------------------------------------------------------------
    # 2. Initialize Policy Wrapper + Client
    # -------------------------------------------------------------------------
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = So100Adapter(policy_client)

    log_say(
        f'Policy ready with instruction: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )

    # -------------------------------------------------------------------------
    # 3. Main real-time control loop
    # -------------------------------------------------------------------------
    while True:
        obs = robot.get_observation()
        obs["lang"] = cfg.lang_instruction  # insert language

        # obs = {
        #     "front": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "shoulder_pan.pos": 0.0,
        #     "shoulder_lift.pos": 0.0,
        #     "elbow_flex.pos": 0.0,
        #     "wrist_flex.pos": 0.0,
        #     "wrist_roll.pos": 0.0,
        #     "gripper.pos": 0.0,
        #     "lang": cfg.lang_instruction,
        # }

        actions = policy.get_action(obs)

        for i, action_dict in enumerate(actions[: cfg.action_horizon]):
            tic = time.time()
            print(f"action[{i}]: {action_dict}")
            # action_dict = {
            #     "shoulder_pan.pos":    5.038022994995117,
            #     "shoulder_lift.pos":  17.09104347229004,
            #     "elbow_flex.pos":    -18.519847869873047,
            #     "wrist_flex.pos":     86.86847686767578,
            #     "wrist_roll.pos":      1.0669738054275513,
            #     "gripper.pos":        36.83877944946289,
            # }
            robot.send_action(action_dict)
            toc = time.time()
            if toc - tic < 1.0 / 30:
                time.sleep(1.0 / 30 - (toc - tic))


if __name__ == "__main__":
    eval()
