from collections import OrderedDict
import csv
import json
import logging
from pathlib import Path

from gr00t.eval.sim.BEHAVIOR.og_teleop_utils import (
    generate_basic_environment_config,
    generate_robot_config,
    load_available_tasks,
)
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import omnigibson as og
from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.learning.wrappers import TaskProgressWrapper
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.transition_rules import CookingSystemRule, MixingToolRule, ToggleableMachineRule
import torch as th


gm.HEADLESS = True

# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info

PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[0:28],
            "joint_qpos_sin": np.s_[28:56],
            "joint_qpos_cos": np.s_[56:84],
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],
            "robot_ori_cos": np.s_[143:146],
            "robot_ori_sin": np.s_[146:149],
            "robot_2d_ori": np.s_[149:150],
            "robot_2d_ori_cos": np.s_[150:151],
            "robot_2d_ori_sin": np.s_[151:152],
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "grasp_left": np.s_[193:194],
            "gripper_left_qpos": np.s_[194:196],
            "gripper_left_qvel": np.s_[196:198],
            "arm_right_qpos": np.s_[198:205],
            "arm_right_qpos_sin": np.s_[205:212],
            "arm_right_qpos_cos": np.s_[212:219],
            "arm_right_qvel": np.s_[219:226],
            "eef_right_pos": np.s_[226:229],
            "eef_right_quat": np.s_[229:233],
            "grasp_right": np.s_[233:234],
            "gripper_right_qpos": np.s_[234:236],
            "gripper_right_qvel": np.s_[236:238],
            "trunk_qpos": np.s_[238:242],
            "trunk_qvel": np.s_[242:246],
            "base_qpos": np.s_[246:249],
            "base_qpos_sin": np.s_[249:252],
            "base_qpos_cos": np.s_[252:255],
            "base_qvel": np.s_[255:258],
        }
    ),
}

ROBOT_CAMERA_NAMES = {
    "R1Pro": {
        "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
        "head": "robot_r1::robot_r1:zed_link:Camera:0",
    },
}

HEAD_RESOLUTION = (720, 720)
WRIST_RESOLUTION = (480, 480)

ACTION_MAP = {
    "base": np.s_[0:3],
    "torso": np.s_[3:7],
    "left_arm": np.s_[7:14],
    "left_gripper": np.s_[14:15],
    "right_arm": np.s_[15:22],
    "right_gripper": np.s_[22:23],
}

DISABLED_TRANSITION_RULES = [ToggleableMachineRule, MixingToolRule, CookingSystemRule]


TASK_NAMES_TO_INDICES = {
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}
TASK_NAMES_TO_INSTRUCTIONS = {k: k.replace("_", " ") for k in TASK_NAMES_TO_INDICES.keys()}
# Capitalize the first letter of each word and add a period at the end
TASK_NAMES_TO_INSTRUCTIONS = {
    k: v.capitalize() + "." for k, v in TASK_NAMES_TO_INSTRUCTIONS.items()
}


def recursively_convert_to_torch(state):
    # For all the lists in state dict, convert to torch tensor
    for key, value in state.items():
        if isinstance(value, dict):
            state[key] = recursively_convert_to_torch(value)
        elif isinstance(value, list):
            # Convert to torch tensor if all elements are numeric and have consistent shapes
            try:
                state[key] = th.tensor(value, dtype=th.float32)
            except:  # noqa: E722
                pass

    return state


def load_task_instance_for_env(env, robot, instance_id: int) -> None:
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    tro_file = (
        Path(__file__).parent
        / "test_instances"
        / f"{env.task.activity_name}"
        / f"{tro_filename}-tro_state.json"
    )
    tro_file_path = str(tro_file)

    assert tro_file.exists(), (
        f"Could not find TRO file at {tro_file_path}, did you run ./populate_behavior_tasks.sh?"
    )
    with open(tro_file_path, "r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    for tro_key, tro_state in tro_state.items():
        if tro_key == "robot_poses":
            presampled_robot_poses = tro_state
            robot_pos = presampled_robot_poses[robot.model_name][0]["position"]
            robot_quat = presampled_robot_poses[robot.model_name][0]["orientation"]
            robot.set_position_orientation(robot_pos, robot_quat)
            # Write robot poses to scene metadata
            env.scene.write_task_metadata(key=tro_key, data=tro_state)
        else:
            env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

    # Try to ensure that all task-relevant objects are stable
    # They should already be stable from the sampled instance, but there is some issue where loading the state
    # causes some jitter (maybe for small mass / thin objects?)
    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()
    env.scene.update_initial_file()
    env.scene.reset()


def flatten_obs_dict(obs: dict, parent_key: str = "") -> dict:
    """
    Process the observation dictionary by recursively flattening the keys.
    so obs["robot_r1"]["camera"]["rgb"] will become obs["robot_r1::camera:::rgb"].
    """
    processed_obs = {}
    for key, value in obs.items():
        new_key = f"{parent_key}::{key}" if parent_key else key
        if isinstance(value, dict) or isinstance(value, gym.spaces.Dict):
            processed_obs.update(flatten_obs_dict(value, parent_key=new_key))
        else:
            processed_obs[new_key] = value
    return processed_obs


def preprocess_obs(env, obs: dict) -> dict:
    """
    Preprocess the observation dictionary before passing it to the policy.
    """
    obs = flatten_obs_dict(obs["robot_r1"])
    # these rgb images have alpha channel, so we need to remove it
    obs["video.observation.images.rgb.left_wrist_256_256"] = (
        obs["robot_r1:left_realsense_link:Camera:0::rgb"].cpu().numpy()[..., :3]
    )
    obs["video.observation.images.rgb.right_wrist_256_256"] = (
        obs["robot_r1:right_realsense_link:Camera:0::rgb"].cpu().numpy()[..., :3]
    )
    obs["video.observation.images.rgb.head_256_256"] = (
        obs["robot_r1:zed_link:Camera:0::rgb"].cpu().numpy()[..., :3]
    )
    obs.pop("robot_r1:left_realsense_link:Camera:0::rgb")
    obs.pop("robot_r1:right_realsense_link:Camera:0::rgb")
    obs.pop("robot_r1:zed_link:Camera:0::rgb")
    # convert all to numpy arrays
    for key, value in obs.items():
        if isinstance(value, th.Tensor):
            obs[key] = value.cpu().numpy()
    obs["annotation.human.coarse_action"] = env.task_instruction
    # process proprio
    proprio_obs = obs.pop("proprio")
    for key, value in PROPRIOCEPTION_INDICES["R1Pro"].items():
        obs[f"state.{key}"] = proprio_obs[value]
    return obs


def preprocess_action(action: dict):
    th_action = th.zeros(23, dtype=th.float32)
    for action_name, action_indices in ACTION_MAP.items():
        action_name = f"action.{action_name}"
        th_action[action_indices] = (
            th.from_numpy(action[action_name])
            if isinstance(action[action_name], np.ndarray)
            else action[action_name]
        )

    return {"robot_r1": th_action}


def postprocess_info(info: dict):
    info["success"] = False if info["done"]["success"] is None else info["done"]["success"]
    return info


class RGBLowResWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # Note that from eval.py we already set the robot to include rgb + depth + seg_instance_id modalities
        # Here, we modify the robot observation to include only rgb modalities, and use 224 * 224 resolution
        # For a complete list of available modalities, see VisionSensor.ALL_MODALITIES
        robot = env.robots[0]
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[
                    sensor_name
                ].horizontal_aperture = 40.0  # this is what we used in data collection
            robot.sensors[sensor_name].image_height = 256
            robot.sensors[sensor_name].image_width = 256
        # reload observation space
        env.load_observation_space()
        logger.info("Reloaded observation space!")


class BEHAVIORGr00tEnv(gym.Wrapper):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, task_name: str, env_idx: int = 0, total_n_envs: int = 1):
        self.task_name = task_name
        self.task_instruction = TASK_NAMES_TO_INSTRUCTIONS[self.task_name]
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        human_stats_path = (
            Path(gm.DATA_PATH) / "2025-challenge-task-instances" / "metadata" / "episodes.jsonl"
        )
        with open(human_stats_path, "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])
        self.env = self.load_env()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()
        super().__init__(self.env)
        original_observation_space = self.env.observation_space
        flat_obs_dict = flatten_obs_dict(original_observation_space["robot_r1"])
        flat_obs_dict.pop("robot_r1:left_realsense_link:Camera:0::rgb")
        flat_obs_dict.pop("robot_r1:right_realsense_link:Camera:0::rgb")
        flat_obs_dict.pop("robot_r1:zed_link:Camera:0::rgb")
        flat_obs_dict["video.observation.images.rgb.left_wrist_256_256"] = gym.spaces.Box(
            0, 255, (256, 256, 3), np.uint8
        )
        flat_obs_dict["video.observation.images.rgb.right_wrist_256_256"] = gym.spaces.Box(
            0, 255, (256, 256, 3), np.uint8
        )
        flat_obs_dict["video.observation.images.rgb.head_256_256"] = gym.spaces.Box(
            0, 255, (256, 256, 3), np.uint8
        )
        flat_obs_dict["annotation.human.coarse_action"] = gym.spaces.Text(max_length=512)
        # replace `proprio` with fine-grained state obs
        proprio_space = flat_obs_dict.pop("proprio")
        fine_grained_proprio_space = OrderedDict()
        for key, value in PROPRIOCEPTION_INDICES["R1Pro"].items():
            fine_grained_proprio_space[f"state.{key}"] = gym.spaces.Box(
                low=proprio_space.low[value],
                high=proprio_space.high[value],
                shape=(value.stop - value.start,),
                dtype=np.float32,
            )
        flat_obs_dict.update(fine_grained_proprio_space)
        self.observation_space = gym.spaces.Dict(flat_obs_dict)
        original_robot_action_space = self.env.action_space["robot_r1"]
        action_space_dict = OrderedDict()
        for action_name, action_indices in ACTION_MAP.items():
            action_space_dict[f"action.{action_name}"] = gym.spaces.Box(
                low=original_robot_action_space.low[action_indices],
                high=original_robot_action_space.high[action_indices],
                shape=(action_indices.stop - action_indices.start,),
            )
        self.action_space = gym.spaces.Dict(action_space_dict)

        self.obs, self.info = None, None
        # manually reset environment episode number
        self.env._current_episode = 0

        # determine task instance
        instances_to_run = list(range(10))
        assert env_idx < len(instances_to_run)
        assert env_idx < total_n_envs, "env_idx must be less than total_n_envs"
        n_instances_per_env = len(instances_to_run) // total_n_envs
        self._instance_indices_this_env = (
            instances_to_run[env_idx * n_instances_per_env : (env_idx + 1) * n_instances_per_env]
            if env_idx < (total_n_envs - 1)
            else instances_to_run[env_idx * n_instances_per_env :]
        )
        self._instance_idx_pointer = 0
        # load csv file
        task_instance_csv_path = (
            Path(gm.DATA_PATH) / "2025-challenge-task-instances" / "metadata" / "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        assert lines[TASK_NAMES_TO_INDICES[self.task_name]][1] == self.task_name, (
            f"Task name from args {self.task_name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[self.task_name]][1]}"
        )
        self._all_test_instances = list(range(10))

        self._task_progress_dict = {}
        self._physx_crashed = False

    def reset(self, *args, **kwargs):
        # if physx crashed, calling env reset will raise an error
        if self._physx_crashed:
            self.info["valid"] = False
            return self.obs, self.info

        instance_id = self._all_test_instances[
            self._instance_indices_this_env[self._instance_idx_pointer]
        ]
        self._instance_idx_pointer = (self._instance_idx_pointer + 1) % len(
            self._instance_indices_this_env
        )
        # the correct way to do: first reset then load task instance
        self.env.reset()
        load_task_instance_for_env(self.env, self.robot, instance_id)

        obs, info = self.env.reset()
        obs = preprocess_obs(self, obs)
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)

        self._task_progress_dict = {}
        return obs, info

    def step(self, action: dict):
        if self._physx_crashed:
            self.info["success"] = False
            self.info["valid"] = False
            return self.obs, 0, True, True, self.info

        action = preprocess_action(action)

        # avoid PhysX errors crashing the evaluator
        try:
            obs, _, terminated, truncated, info = self.env.step(action, n_render_iterations=1)
        except Exception as e:
            self._physx_crashed = True
            terminated = True
            truncated = True
            self.info["success"] = False
            self.info["valid"] = False
            logger.error(f"Error in OGEnv.step: {e}")
        else:
            self.obs = preprocess_obs(self, obs)
            self.info = postprocess_info(info)
            for metric in self.metrics:
                metric.step_callback(self.env)

            if len(self._task_progress_dict) == 0:
                self._task_progress_dict = {k: False for k in self.info["task_progress"]}
            self._task_progress_dict = {
                k: old_progress or self.info["task_progress"][k]
                for k, old_progress in self._task_progress_dict.items()
            }

        if terminated or truncated:
            if self._physx_crashed:
                # if physx crashed, don't call the end_callback as it will raise an error
                self.info["q_score"] = 0.0
                self.info["valid"] = False
            else:
                for metric in self.metrics:
                    metric.end_callback(self.env)
                self.info["q_score"] = self.metrics[1].final_q_score
                self.info["valid"] = True

        # get task progress
        if self._physx_crashed:
            task_progress = 0.0
        else:
            task_progress = (
                sum(self._task_progress_dict.values()) / len(self._task_progress_dict) * 100
            )
        self.info["task_progress"] = task_progress

        return self.obs, 0, terminated, truncated, self.info

    def load_env(self) -> og.Environment:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        # Load config file
        available_tasks = load_available_tasks()
        assert self.task_name in available_tasks, (
            f"Got invalid OmniGibson task name: {self.task_name}"
        )
        # Load the seed instance by default
        task_cfg = available_tasks[self.task_name][0]
        cfg = generate_basic_environment_config(task_name=self.task_name, task_cfg=task_cfg)
        cfg["robots"] = [
            generate_robot_config(
                task_name=self.task_name,
                task_cfg=task_cfg,
            )
        ]
        # Update observation modalities
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        logger.info(
            f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
        )
        cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        env = RGBLowResWrapper(env)
        env = TaskProgressWrapper(env)
        return env

    def load_robot(self) -> BaseRobot:
        robot = self.env.scene.object_registry("name", "robot_r1")
        # Set a big mass to robot base to prevent it from tipping over
        with og.sim.stopped():
            robot.base_footprint_link.mass = 250.0
        return robot

    def load_metrics(self) -> list[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]


def register_behavior_envs():
    register(
        id="sim_behavior_r1_pro/turning_on_radio",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "turning_on_radio",
        },
    )

    register(
        id="sim_behavior_r1_pro/picking_up_trash",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "picking_up_trash",
        },
    )

    register(
        id="sim_behavior_r1_pro/putting_away_Halloween_decorations",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "putting_away_Halloween_decorations",
        },
    )

    register(
        id="sim_behavior_r1_pro/cleaning_up_plates_and_food",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "cleaning_up_plates_and_food",
        },
    )

    register(
        id="sim_behavior_r1_pro/can_meat",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "can_meat",
        },
    )

    register(
        id="sim_behavior_r1_pro/setting_mousetraps",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "setting_mousetraps",
        },
    )

    register(
        id="sim_behavior_r1_pro/hiding_Easter_eggs",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "hiding_Easter_eggs",
        },
    )

    register(
        id="sim_behavior_r1_pro/picking_up_toys",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "picking_up_toys",
        },
    )

    register(
        id="sim_behavior_r1_pro/rearranging_kitchen_furniture",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "rearranging_kitchen_furniture",
        },
    )

    register(
        id="sim_behavior_r1_pro/putting_up_Christmas_decorations_inside",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "putting_up_Christmas_decorations_inside",
        },
    )

    register(
        id="sim_behavior_r1_pro/set_up_a_coffee_station_in_your_kitchen",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "set_up_a_coffee_station_in_your_kitchen",
        },
    )

    register(
        id="sim_behavior_r1_pro/putting_dishes_away_after_cleaning",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "putting_dishes_away_after_cleaning",
        },
    )

    register(
        id="sim_behavior_r1_pro/preparing_lunch_box",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "preparing_lunch_box",
        },
    )

    register(
        id="sim_behavior_r1_pro/loading_the_car",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "loading_the_car",
        },
    )

    register(
        id="sim_behavior_r1_pro/carrying_in_groceries",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "carrying_in_groceries",
        },
    )

    register(
        id="sim_behavior_r1_pro/bringing_in_wood",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "bringing_in_wood",
        },
    )

    register(
        id="sim_behavior_r1_pro/moving_boxes_to_storage",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "moving_boxes_to_storage",
        },
    )

    register(
        id="sim_behavior_r1_pro/bringing_water",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "bringing_water",
        },
    )

    register(
        id="sim_behavior_r1_pro/tidying_bedroom",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "tidying_bedroom",
        },
    )

    register(
        id="sim_behavior_r1_pro/outfit_a_basic_toolbox",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "outfit_a_basic_toolbox",
        },
    )

    register(
        id="sim_behavior_r1_pro/sorting_vegetables",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "sorting_vegetables",
        },
    )

    register(
        id="sim_behavior_r1_pro/collecting_childrens_toys",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "collecting_childrens_toys",
        },
    )

    register(
        id="sim_behavior_r1_pro/putting_shoes_on_rack",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "putting_shoes_on_rack",
        },
    )

    register(
        id="sim_behavior_r1_pro/boxing_books_up_for_storage",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "boxing_books_up_for_storage",
        },
    )

    register(
        id="sim_behavior_r1_pro/storing_food",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "storing_food",
        },
    )

    register(
        id="sim_behavior_r1_pro/clearing_food_from_table_into_fridge",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "clearing_food_from_table_into_fridge",
        },
    )

    register(
        id="sim_behavior_r1_pro/assembling_gift_baskets",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "assembling_gift_baskets",
        },
    )

    register(
        id="sim_behavior_r1_pro/sorting_household_items",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "sorting_household_items",
        },
    )

    register(
        id="sim_behavior_r1_pro/getting_organized_for_work",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "getting_organized_for_work",
        },
    )

    register(
        id="sim_behavior_r1_pro/clean_up_your_desk",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "clean_up_your_desk",
        },
    )

    register(
        id="sim_behavior_r1_pro/setting_the_fire",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "setting_the_fire",
        },
    )

    register(
        id="sim_behavior_r1_pro/clean_boxing_gloves",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "clean_boxing_gloves",
        },
    )

    register(
        id="sim_behavior_r1_pro/wash_a_baseball_cap",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "wash_a_baseball_cap",
        },
    )

    register(
        id="sim_behavior_r1_pro/wash_dog_toys",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "wash_dog_toys",
        },
    )

    register(
        id="sim_behavior_r1_pro/hanging_pictures",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "hanging_pictures",
        },
    )

    register(
        id="sim_behavior_r1_pro/attach_a_camera_to_a_tripod",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "attach_a_camera_to_a_tripod",
        },
    )

    register(
        id="sim_behavior_r1_pro/clean_a_patio",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "clean_a_patio",
        },
    )

    register(
        id="sim_behavior_r1_pro/clean_a_trumpet",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "clean_a_trumpet",
        },
    )

    register(
        id="sim_behavior_r1_pro/spraying_for_bugs",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "spraying_for_bugs",
        },
    )

    register(
        id="sim_behavior_r1_pro/spraying_fruit_trees",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "spraying_fruit_trees",
        },
    )

    register(
        id="sim_behavior_r1_pro/make_microwave_popcorn",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "make_microwave_popcorn",
        },
    )

    register(
        id="sim_behavior_r1_pro/cook_cabbage",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "cook_cabbage",
        },
    )

    register(
        id="sim_behavior_r1_pro/chop_an_onion",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "chop_an_onion",
        },
    )

    register(
        id="sim_behavior_r1_pro/slicing_vegetables",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "slicing_vegetables",
        },
    )

    register(
        id="sim_behavior_r1_pro/chopping_wood",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "chopping_wood",
        },
    )

    register(
        id="sim_behavior_r1_pro/cook_hot_dogs",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "cook_hot_dogs",
        },
    )

    register(
        id="sim_behavior_r1_pro/cook_bacon",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "cook_bacon",
        },
    )

    register(
        id="sim_behavior_r1_pro/freeze_pies",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "freeze_pies",
        },
    )

    register(
        id="sim_behavior_r1_pro/canning_food",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "canning_food",
        },
    )

    register(
        id="sim_behavior_r1_pro/make_pizza",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={
            "task_name": "make_pizza",
        },
    )
