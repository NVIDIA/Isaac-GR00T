import cv2
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d import euler as te, quaternions as tq


class GoogleFractalEnv(gym.Env):
    def __init__(self, env_name: str, image_size: tuple[int, int]):
        env = simpler_env.make(env_name)
        env._max_episode_steps = 10000
        self.env = env
        agent_space = env.observation_space["agent"]
        print("[SimplerEnv] agent space keys:", list(agent_space.spaces.keys()))
        # assert False
        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(
                    low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(low=obs_low[0], high=obs_high[0], shape=(1,)),
                "state.y": gym.spaces.Box(low=obs_low[1], high=obs_high[1], shape=(1,)),
                "state.z": gym.spaces.Box(low=obs_low[2], high=obs_high[2], shape=(1,)),
                "state.rx": gym.spaces.Box(low=obs_low[3], high=obs_high[3], shape=(1,)),
                "state.ry": gym.spaces.Box(low=obs_low[4], high=obs_high[4], shape=(1,)),
                "state.rz": gym.spaces.Box(low=obs_low[5], high=obs_high[5], shape=(1,)),
                "state.rw": gym.spaces.Box(low=obs_low[6], high=obs_high[6], shape=(1,)),
                "state.gripper": gym.spaces.Box(low=obs_low[7], high=obs_high[7], shape=(1,)),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(
            {
                "action.x": gym.spaces.Box(low=action_low[0], high=action_high[0], shape=(1,)),
                "action.y": gym.spaces.Box(low=action_low[1], high=action_high[1], shape=(1,)),
                "action.z": gym.spaces.Box(low=action_low[2], high=action_high[2], shape=(1,)),
                "action.roll": gym.spaces.Box(low=action_low[3], high=action_high[3], shape=(1,)),
                "action.pitch": gym.spaces.Box(low=action_low[4], high=action_high[4], shape=(1,)),
                "action.yaw": gym.spaces.Box(low=action_low[5], high=action_high[5], shape=(1,)),
                "action.gripper": gym.spaces.Box(
                    low=action_low[6], high=action_high[6], shape=(1,)
                ),
            }
        )
        self.image_size = image_size
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

    def reset(self, seed=None, options=None):
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                self._postprocess_gripper(action["action.gripper"]),
            ],
            axis=0,
        )
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]
        qunat_xyzw = np.roll(proprio[3:7], -1)
        gripper_closedness = 1 - proprio[7]
        return {
            "video.image": cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.x": [proprio[0]],
            "state.y": [proprio[1]],
            "state.z": [proprio[2]],
            "state.rx": [qunat_xyzw[0]],
            "state.ry": [qunat_xyzw[1]],
            "state.rz": [qunat_xyzw[2]],
            "state.rw": [qunat_xyzw[3]],
            "state.gripper": [gripper_closedness],
            "annotation.human.action.task_description": self.env.unwrapped.get_language_instruction(),
        }

    def _postprocess_gripper(self, current_gripper_action: float) -> float:
        current_gripper_action = (current_gripper_action * 2) - 1  # [0,1] -> [-1,1]
        relative_gripper_action = -current_gripper_action
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        return relative_gripper_action


class WidowXBridgeEnv(gym.Env):
    def __init__(self, env_name: str, image_size: tuple[int, int]):
        env = simpler_env.make(env_name)
        env._max_episode_steps = 10000
        self.env = env
        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(
            {
                "video.image_0": gym.spaces.Box(
                    low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(low=obs_low[0], high=obs_high[0], shape=(1,)),
                "state.y": gym.spaces.Box(low=obs_low[1], high=obs_high[1], shape=(1,)),
                "state.z": gym.spaces.Box(low=obs_low[2], high=obs_high[2], shape=(1,)),
                "state.roll": gym.spaces.Box(low=obs_low[3], high=obs_high[3], shape=(1,)),
                "state.pitch": gym.spaces.Box(low=obs_low[4], high=obs_high[4], shape=(1,)),
                "state.yaw": gym.spaces.Box(low=obs_low[5], high=obs_high[5], shape=(1,)),
                "state.pad": gym.spaces.Box(low=obs_low[6], high=obs_high[6], shape=(1,)),
                "state.gripper": gym.spaces.Box(low=obs_low[7], high=obs_high[7], shape=(1,)),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(
            {
                "action.x": gym.spaces.Box(low=action_low[0], high=action_high[0], shape=(1,)),
                "action.y": gym.spaces.Box(low=action_low[1], high=action_high[1], shape=(1,)),
                "action.z": gym.spaces.Box(low=action_low[2], high=action_high[2], shape=(1,)),
                "action.roll": gym.spaces.Box(low=action_low[3], high=action_high[3], shape=(1,)),
                "action.pitch": gym.spaces.Box(low=action_low[4], high=action_high[4], shape=(1,)),
                "action.yaw": gym.spaces.Box(low=action_low[5], high=action_high[5], shape=(1,)),
                "action.gripper": gym.spaces.Box(
                    low=action_low[6], high=action_high[6], shape=(1,)
                ),
            }
        )
        self.image_size = image_size
        # Bridge orientation adjustment
        self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                self._postprocess_gripper(action["action.gripper"]),
            ],
            axis=0,
        )
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge_converted = te.mat2euler(rm_bridge @ self.default_rot.T)
        return {
            "video.image_0": cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.x": [proprio[0]],
            "state.y": [proprio[1]],
            "state.z": [proprio[2]],
            "state.roll": [rpy_bridge_converted[0]],
            "state.pitch": [rpy_bridge_converted[1]],
            "state.yaw": [rpy_bridge_converted[2]],
            "state.pad": [0],
            "state.gripper": [proprio[7]],
            "annotation.human.action.task_description": self.env.unwrapped.get_language_instruction(),
        }

    def _postprocess_gripper(self, action):
        # trained with [0, 1], 0 close, 1 open -> convert to SimplerEnv [-1, 1]
        return 2.0 * (action > 0.5) - 1.0


def register_simpler_envs():
    # Google/Fractal
    for env_name in [
        "google_robot_pick_coke_can",
        "google_robot_pick_object",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_in_closed_drawer",
    ]:
        register(
            id=f"simpler_env_google/{env_name}",
            entry_point="gr00t.eval.sim.SimplerEnv.simpler_env:GoogleFractalEnv",
            kwargs={"env_name": env_name, "image_size": (256, 320)},
        )

    # WidowX/Bridge
    for env_name in [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
        "widowx_put_eggplant_in_sink",
        "widowx_open_drawer",
        "widowx_close_drawer",
    ]:
        register(
            id=f"simpler_env_widowx/{env_name}",
            entry_point="gr00t.eval.sim.SimplerEnv.simpler_env:WidowXBridgeEnv",
            kwargs={"env_name": env_name, "image_size": (256, 256)},
        )
