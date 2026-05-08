"""
StarAI Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the StarAI Cello
(single arm) robot using the GR00T Policy API.

Key differences from eval_so100.py:
    - 6 arm motors (Motor_0..Motor_5) + 1 gripper = 7 control channels
    - 3 cameras (side, rear, onhand) to match starai_single_arm_config.py
    - Language key is ``annotation.human.action.task_description``
      (data_config style used for StarAI finetuning), not
      ``annotation.human.task_description``.
    - The stock StarAiCello.connect() calls ``move_to_initial_position()``
      with a fixed home pose [0, -100, 60, 0, 30, 0, 50] that does NOT
      match the training-data distribution. We override that home pose
      with the per-motor median of episode-start poses from the training
      dataset so that the policy starts inside the training distribution.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass, field
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
    make_robot_from_config,
)
from lerobot.utils.utils import init_logging, log_say
import numpy as np

# Register StarAI Cello robot type (side-effect import)
import lerobot_robot_cello  # noqa: F401
from lerobot_robot_cello.starai_cello import StaraiCello


# Training-data episode-start pose from
# /home/kazu/data/original_data/20260427_data_groot_v2. Used to override the
# hard-coded StaraiCello.move_to_initial_position() home pose so that
# evaluation starts inside the training distribution.
# Values are the denser start-pose mode over all 300 episode-start frames
# (see src/verify_initial_pose.py). Several axes are bimodal, so a simple
# median can land between modes and start the policy outside its usual pose.
TRAINING_INITIAL_POSE: Dict[str, float] = {
    "Motor_0": -0.39,
    "Motor_1": -100.0,
    "Motor_2": 71.57,
    "Motor_3": 7.28,
    "Motor_4": 17.07,
    "Motor_5": -1.48,
    "gripper": 1.0,
}

MAX_ACTION_STEP: Dict[str, float] = {
    "Motor_0.pos": 5.0,
    "Motor_1.pos": 15.0,
    "Motor_2.pos": 10.0,
    "Motor_3.pos": 3.0,
    "Motor_4.pos": 10.0,
    "Motor_5.pos": 0.15,
    "gripper.pos": 20.0,
}


def _patched_move_to_initial_position(self: StaraiCello) -> Dict[str, Any]:
    """Replacement for StaraiCello.move_to_initial_position().

    Drives all 7 motors to ``TRAINING_INITIAL_POSE`` instead of the
    hard-coded home pose. This keeps the robot near the training-data
    episode-start distribution, which is required for the policy to
    behave sensibly from the very first step.
    """
    position = self.get_action()
    goal_pos = {
        key.removesuffix(".pos"): val
        for key, val in position.items()
        if key.endswith(".pos")
    }
    goal_pos.update(TRAINING_INITIAL_POSE)
    self.bus.sync_write("Goal_Position", goal_pos, motion_time=1500)
    time.sleep(1.5)
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}


# Monkey-patch once at import time so the override is in place before
# robot.connect() (which calls move_to_initial_position) is invoked.
StaraiCello.move_to_initial_position = _patched_move_to_initial_position


def clamp_action_to_step(
    action: Dict[str, float],
    reference_state: Dict[str, float],
    max_step: Dict[str, float],
) -> Dict[str, float]:
    """Limit absolute joint targets to a small per-cycle move from current state."""
    clamped: Dict[str, float] = {}
    for key, target in action.items():
        ref = reference_state.get(key)
        limit = max_step.get(key)
        if ref is None or limit is None:
            clamped[key] = target
            continue
        clamped[key] = float(np.clip(target, ref - limit, ref + limit))
    return clamped


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


class StarAIAdapter:
    """
    Adapter between:
        • Raw StarAI robot observation dictionary
        • GR00T VLA input format (matches starai_single_arm_config.py)
        • GR00T action chunk → robot joint commands

    Layout:
        state.single_arm: (6,)  Motor_0..Motor_5
        state.gripper:    (1,)  gripper
        video.{side, rear, onhand}
        language.annotation.human.action.task_description
    """

    def __init__(
        self,
        policy_client: PolicyClient,
        camera_keys: List[str],
        language_key: str = "annotation.human.action.task_description",
    ):
        self.policy = policy_client

        # StarAI joint ordering used for BOTH training + robot execution.
        # First 6 entries form ``single_arm``; last entry is ``gripper``.
        self.robot_state_keys = [
            "Motor_0.pos",
            "Motor_1.pos",
            "Motor_2.pos",
            "Motor_3.pos",
            "Motor_4.pos",
            "Motor_5.pos",
            "gripper.pos",
        ]
        self.camera_keys = camera_keys
        self.language_key = language_key

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        model_obs: Dict[str, Any] = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Arm + gripper state (7ch: 6 arm motors + 1 gripper)
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:6],  # (6,)
            "gripper": state[6:7],    # (1,)
        }

        # (3) Language — StarAI finetuning uses the ``action`` sub-key
        model_obs["language"] = {self.language_key: obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        """
        chunk["single_arm"]: (B, T, 6)
        chunk["gripper"]:    (B, T, 1)
        """
        single_arm = chunk["single_arm"][0][t]  # (6,)
        gripper = chunk["gripper"][0][t]        # (1,)

        full = np.concatenate([single_arm, gripper], axis=0)  # (7,)
        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Evaluation Config
# =============================================================================


@dataclass
class EvalConfig:
    """CLI configuration for StarAI real-robot policy evaluation."""

    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    # Execute only the first predicted step by default. The 20260428 log showed
    # large discontinuities inside a single predicted chunk, so receding-horizon
    # execution is the safer default for real hardware.
    action_horizon: int = 1
    # Must match the task string used during finetuning
    # (tasks.jsonl of the training dataset).
    lang_instruction: str = "pick_and_place"
    language_key: str = "annotation.human.action.task_description"
    camera_keys: List[str] = field(default_factory=lambda: ["side", "rear", "onhand"])
    clamp_action_step: bool = True
    play_sounds: bool = False
    timeout: int = 60


# =============================================================================
# Main Eval Loop
# =============================================================================


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # -------------------------------------------------------------------------
    # 1. Initialize Robot Hardware
    #    NOTE: StaraiCello.connect() will call the (patched)
    #    move_to_initial_position(), driving the arm to the training-data
    #    episode-start pose defined in TRAINING_INITIAL_POSE.
    # -------------------------------------------------------------------------
    robot = make_robot_from_config(cfg.robot)
    try:
        robot.connect()

        missing = [k for k in cfg.camera_keys if k not in robot.cameras]
        if missing:
            raise RuntimeError(
                f"Camera keys {missing} not found on robot. Configure cameras "
                f"under --robot.cameras with names matching {cfg.camera_keys}."
            )

        log_say("Initializing robot (StarAI Cello)", cfg.play_sounds, blocking=True)

        # ---------------------------------------------------------------------
        # 2. Initialize Policy Wrapper + Client
        # ---------------------------------------------------------------------
        policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
        policy = StarAIAdapter(
            policy_client,
            camera_keys=cfg.camera_keys,
            language_key=cfg.language_key,
        )

        log_say(
            f'Policy ready with instruction: "{cfg.lang_instruction}"',
            cfg.play_sounds,
            blocking=True,
        )

        # ---------------------------------------------------------------------
        # 3. Main real-time control loop
        # ---------------------------------------------------------------------
        started = time.monotonic()
        deadline = started + cfg.timeout if cfg.timeout and cfg.timeout > 0 else None

        while deadline is None or time.monotonic() < deadline:
            obs = robot.get_observation()
            obs["lang"] = cfg.lang_instruction

            state_dbg = {k: obs[k] for k in policy.robot_state_keys}
            print(f"state: {state_dbg}")

            actions = policy.get_action(obs)

            for i, action_dict in enumerate(actions[: cfg.action_horizon]):
                tic = time.time()
                if cfg.clamp_action_step:
                    action_dict = clamp_action_to_step(action_dict, state_dbg, MAX_ACTION_STEP)
                print(f"action[{i}]: {action_dict}")
                robot.send_action(action_dict)
                toc = time.time()
                if toc - tic < 1.0 / 30:
                    time.sleep(1.0 / 30 - (toc - tic))

        if deadline is not None:
            logging.info("Reached timeout=%ss; returning robot to initial position.", cfg.timeout)
            robot.move_to_initial_position()
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    eval()
