# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Gymnasium adapter for RoboDojo."""

from copy import deepcopy
import json
import os
from pathlib import Path
import string
import sys
import tempfile
from typing import ClassVar

import gymnasium as gym
import numpy as np

from groot.infra.closed_loop.backends.gymnasium.errors import ResetSeedReplacementError

ROBODOJO_ROOT = Path(os.environ["ROBODOJO_ROOT"])
IMAGE_SHAPE = (480, 640, 3)
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SIDES = ("left", "right")
_APP_LAUNCHER = None


def discover_task_names():
    tasks = {path.stem for path in (ROBODOJO_ROOT / "task/RoboDojo/tasks").glob("*.py")}
    configs = {path.stem for path in (ROBODOJO_ROOT / "task/RoboDojo/config").glob("*.yml")}
    return tuple(sorted((tasks & configs) - {"__init__", "_task"}))


ROBODOJO_TASKS = discover_task_names()


def _launch_app():
    global _APP_LAUNCHER  # noqa: PLW0603
    if _APP_LAUNCHER is None:
        portable_root = tempfile.mkdtemp(prefix="groot-robodojo-")
        os.environ["XDG_DATA_HOME"] = str(Path(portable_root) / "data")
        os.environ["XDG_CACHE_HOME"] = str(Path(portable_root) / "cache")
        sys.path.insert(0, str(ROBODOJO_ROOT))
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.setdefault("ACCEPT_EULA", "Y")
        os.environ.setdefault("PRIVACY_CONSENT", "Y")
        from env.camera_manager.capture.render_sync import zero_delay_kit_args
        from isaaclab.app import AppLauncher

        _APP_LAUNCHER = AppLauncher(
            headless=True,
            enable_cameras=True,
            kit_args=f"--portable-root {portable_root} {zero_delay_kit_args()}",
        )
    return _APP_LAUNCHER.app


def _load_config(task_name, seeds):
    from env.global_configs import ENV_CONFIG_PATH
    from omegaconf import OmegaConf

    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    eval_cfg = load_yaml(Path(ENV_CONFIG_PATH) / "arx_x5.yml")
    num_envs = len(seeds)
    eval_cfg.update(task_name=task_name, num_envs=num_envs, device_id=0)
    config = OmegaConf.create(
        {
            "sim": load_yaml(Path(ENV_CONFIG_PATH) / "sim" / f"{eval_cfg['config']['sim']}.yml"),
            "scene": load_yaml(
                Path(ENV_CONFIG_PATH) / "scene" / f"{eval_cfg['config']['scene']}.yml"
            ),
            "camera": load_yaml(
                Path(ENV_CONFIG_PATH) / "camera" / f"{eval_cfg['config']['camera']}.yml"
            ),
            "robot": load_yaml(
                Path(ENV_CONFIG_PATH) / "robot" / f"{eval_cfg['config']['robot']}.yml"
            ),
            "task_env": load_yaml(ROBODOJO_ROOT / "task/RoboDojo/config" / f"{task_name}.yml"),
            "eval_cfg": eval_cfg,
        }
    )
    config.sim.scene.num_envs = num_envs
    config.sim.seed = seeds
    config.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
    config = process_randomization(config)
    config = process_config(config, task_name)[0]
    for robot in config.robot.robots:
        robot.need_planner = False
    return config


class RoboDojoEnv(gym.Env):
    metadata: ClassVar = {"render_modes": ["rgb_array"]}

    def __init__(self, task_name, num_envs=1):
        if task_name not in ROBODOJO_TASKS:
            raise ValueError(f"Unknown RoboDojo task: {task_name}")

        app = _launch_app()
        from env.observation_manager.obs_manager import ObsManager
        from task.RoboDojo.task_registry import load_task_class

        self.task_name = task_name
        self._num_envs = num_envs
        self._app = app
        self._task_class = load_task_class(task_name)[1]
        self._config = _load_config(task_name, [0] * num_envs)
        self._task = self._task_class(self._config, app)
        self._task.success = [True] * num_envs
        self._task.eval_seed = self._config.eval_cfg.get("seed", 0)
        self._layout_dir = ROBODOJO_ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
        if not (self._layout_dir / f"{task_name}_0.json").is_file():
            raise FileNotFoundError(f"No evaluation layout found for {task_name}")
        self._ObsManager = ObsManager
        self._obs_manager = None
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self._closed = False

        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"video.{camera}": gym.spaces.Box(0, 255, shape=IMAGE_SHAPE, dtype=np.uint8)
                    for camera in CAMERAS
                },
                **{
                    f"state.{side}_joints": gym.spaces.Box(
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    )
                    for side in SIDES
                },
                **{
                    f"state.{side}_gripper": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
                    for side in SIDES
                },
                "annotation.human.action.task_description": gym.spaces.Text(
                    max_length=512, charset=string.printable
                ),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                **{
                    f"action.{side}_joints": gym.spaces.Box(
                        -np.pi, np.pi, shape=(6,), dtype=np.float32
                    )
                    for side in SIDES
                },
                **{
                    f"action.{side}_gripper": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
                    for side in SIDES
                },
            }
        )
        self.num_envs = num_envs
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    @property
    def episode_horizon(self):
        return self._task.step_lim

    def _observation(self):
        observations = [
            {"annotation.human.action.task_description": self._obs_manager.instruction[env_idx]}
            for env_idx in range(self._num_envs)
        ]
        capture = self._task.capture_manager
        self._obs_manager.render_for_capture()
        env_ids = list(range(self._num_envs))
        camera_data = capture.step(env_ids=env_ids)
        for camera_id, data in enumerate(camera_data):
            for env_idx in env_ids:
                image = np.asarray(data["rgb"][env_idx]["data"], dtype=np.uint8)
                name = capture.camera_names[env_idx][camera_id].replace("cam_head", "cam_high")
                observations[env_idx][f"video.{name}"] = image[:, :, :3]

        manager = self._task.robot_manager
        for robot in manager.robot_list:
            if robot.type != "target":
                continue
            side = robot.arm_name.split("_")[0]
            joints = manager.get_joint(robot=robot, env_idx_list=env_ids)
            key = manager.process_name(robot.gripper_name)
            for env_idx in env_ids:
                observations[env_idx][f"state.{side}_joints"] = np.asarray(
                    joints[env_idx], dtype=np.float32
                )
                gripper = manager.control_manager.prev_control[env_idx][key]["position"][0]
                if robot.gripper_move["sign"] == 1:
                    gripper = (gripper - robot.gripper_scale[0]) / (
                        robot.gripper_scale[1] - robot.gripper_scale[0]
                    )
                else:
                    gripper = (robot.gripper_scale[1] - gripper) / (
                        robot.gripper_scale[1] - robot.gripper_scale[0]
                    )
                observations[env_idx][f"state.{side}_gripper"] = np.asarray(
                    [gripper], dtype=np.float32
                )
        if self._num_envs == 1:
            return observations[0]
        return {
            key: (
                np.asarray([obs[key] for obs in observations], dtype=object)
                if key.startswith("annotation")
                else np.stack([obs[key] for obs in observations])
            )
            for key in observations[0]
        }

    def reset(self, *, seed=None, options=None):
        reset_seeds = (options or {}).get("reset_seeds")
        if reset_seeds is not None:
            seeds = [int(value) for value in reset_seeds]
        elif isinstance(seed, list | tuple | np.ndarray):
            seeds = [int(value) for value in seed]
        else:
            super().reset(seed=seed)
            seeds = [
                int(self.np_random.integers(50) if seed is None else seed + env_idx)
                for env_idx in range(self._num_envs)
            ]
        if len(seeds) != self._num_envs:
            raise ValueError(f"Expected {self._num_envs} reset seeds, got {len(seeds)}")
        self._task.success = [True] * self._num_envs
        self._task.eval_seed = self._config.eval_cfg.get("seed", 0)
        self._task.end_flag = [False] * self._num_envs
        for env_idx, layout_seed in enumerate(seeds):
            layout_path = self._layout_dir / f"{self.task_name}_{layout_seed}.json"
            if not layout_path.is_file():
                raise FileNotFoundError(
                    f"No evaluation layout {layout_seed} found for {self.task_name}"
                )
            self._task.scene_manager.layout_manager.set_saved_layout(
                env_idx, json.loads(layout_path.read_text())
            )
        self._task.reset(seed=seeds, options=options)

        if self._obs_manager is None:
            observation = self._config.eval_cfg.observation
            self._obs_manager = self._ObsManager(
                observation,
                self._num_envs,
                self._task.dt,
                self.task_name,
                self._config.eval_cfg.get("description", {}),
                seeds,
            )
            self._obs_manager.initialize(self._task)
        else:
            self._obs_manager.reset()

        env_ids = list(range(self._num_envs))
        self._task.scene_manager.apply_saved_poses(env_idx_list=env_ids)
        _, unstable_envs = self._task.scene_manager.layout_manager.check_layout_stability(
            self._task
        )
        if unstable_envs:
            raise ResetSeedReplacementError(sorted(set(unstable_envs)))
        for _ in range(10):
            self._task.render()
        for index in range(200):
            self._task.sim_step()
            if index % 5 == 0:
                self._task.render()
                self._task.capture_manager.step(env_ids=env_ids)

        self._task.robot_manager.set_origin_endpose()
        self._task.robot_manager.set_robot_init_state()
        self._task.reward_manager.init_state()
        self._task.run_reward()
        if hasattr(self._task, "get_score"):
            self._task.get_score()
        self._query_support_arm_traj(env_ids)
        self._steps[:] = 0
        success = np.zeros(self._num_envs, dtype=bool)
        if self._num_envs == 1:
            return self._observation(), {"success": False}
        return self._observation(), {"success": success}

    def _query_support_arm_traj(self, env_ids):
        if not getattr(self._task, "interact", False) or not hasattr(
            self._task, "query_support_arm_traj"
        ):
            return
        for env_idx in env_ids:
            self._task.query_support_arm_traj(env_idx=env_idx)

    def step(self, action):
        env_ids = list(range(self._num_envs))
        controls = [{} for _ in env_ids]
        for robot in self._task.robot_manager.robot_list:
            if robot.type != "target":
                continue
            side = robot.arm_name.split("_")[0]
            arm_key = self._task.robot_manager.process_name(robot.arm_name)
            gripper_key = self._task.robot_manager.process_name(robot.gripper_name)
            for env_idx in env_ids:
                select = (
                    (lambda value: value)
                    if self._num_envs == 1
                    else (lambda value, idx=env_idx: value[idx])
                )
                grip = float(np.clip(select(action[f"action.{side}_gripper"])[0], 0, 1))
                if robot.gripper_move["sign"] == 1:
                    grip = grip * (robot.gripper_scale[1] - robot.gripper_scale[0])
                    grip += robot.gripper_scale[0]
                else:
                    grip = (1 - grip) * (robot.gripper_scale[1] - robot.gripper_scale[0])
                    grip += robot.gripper_scale[0]
                controls[env_idx][arm_key] = {"position": select(action[f"action.{side}_joints"])}
                controls[env_idx][gripper_key] = {
                    "position": [
                        grip,
                        grip * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]
                }

        repeat = max(1, int(self._obs_manager.collect_interval))
        control_sequences = [[deepcopy(control) for _ in range(repeat)] for control in controls]
        interpolation_steps = int(np.floor(repeat * 0.8))
        for robot in self._task.robot_manager.robot_list:
            if robot.type != "target":
                continue
            arm_key = self._task.robot_manager.process_name(robot.arm_name)
            current_joints = self._task.robot_manager.get_joint(robot=robot, env_idx_list=env_ids)
            gripper_key = self._task.robot_manager.process_name(robot.gripper_name)
            current_grippers = self._task.robot_manager.get_end_effector_real_val(
                robot, env_idx_list=env_ids
            )
            for env_idx in env_ids:
                target_joints = np.asarray(controls[env_idx][arm_key]["position"])
                target_gripper = controls[env_idx][gripper_key]["position"][0]
                for index in range(interpolation_steps):
                    alpha = (index + 1) / (interpolation_steps + 1)
                    joints = (1 - alpha) * np.asarray(
                        current_joints[env_idx]
                    ) + alpha * target_joints
                    gripper = (1 - alpha) * current_grippers[env_idx][0] + alpha * target_gripper
                    gripper = np.clip(gripper, *robot.gripper_scale)
                    control_sequences[env_idx][index][arm_key]["position"] = joints.tolist()
                    control_sequences[env_idx][index][gripper_key]["position"] = [
                        gripper,
                        gripper * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]

        if getattr(self._task, "interact", False) and hasattr(self._task, "support_arm_action"):
            for env_idx in env_ids:
                support_actions = self._task.support_arm_action[env_idx]
                for control in control_sequences[env_idx]:
                    if not support_actions:
                        break
                    control.update(support_actions.pop(0))

        self._task.robot_manager.control_manager.push(env_ids, control_sequences)
        for _ in range(repeat):
            meta_control = self._task.robot_manager.control_manager.pop(env_ids)
            self._task.step(meta_control)
            self._task.sim_step(render=False)

        self._task.reward_manager.step(env_ids)
        if getattr(self._task, "interact", False):
            self._query_support_arm_traj(env_ids)
            if hasattr(self._task, "check_support_arm_stable"):
                for env_idx in env_ids:
                    self._task.check_support_arm_stable(env_idx=env_idx)
        rewards = np.asarray(
            self._task.reward_manager.get_reward(final_check=False), dtype=np.float32
        )
        self._steps += 1
        successes = rewards.astype(bool)
        truncations = self._steps >= self._task.step_lim
        if self._num_envs == 1:
            reward = float(rewards[0])
            success = bool(successes[0])
            return (
                self._observation(),
                reward,
                success,
                bool(truncations[0]),
                {"success": success},
            )
        return (
            self._observation(),
            rewards,
            np.zeros(self._num_envs, dtype=bool),
            truncations,
            {"success": successes},
        )

    def render(self):
        return self._observation()["video.cam_high"]

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.close()
            self._task = None
        _launch_app().close()


def register_robodojo_envs():
    _launch_app()
    for task_name in ROBODOJO_TASKS:
        env_id = f"robodojo/{task_name}"
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point=RoboDojoEnv,
                kwargs={"task_name": task_name},
            )
    return ROBODOJO_TASKS


def _shapes(values):
    return {key: np.shape(value) for key, value in values.items()}


def main():
    from PIL import Image

    output_dir = ROBODOJO_ROOT / "robodojo_test_images"
    output_dir.mkdir(exist_ok=True)
    failures = []
    tasks = register_robodojo_envs()
    for task_name in tasks:
        env = None
        try:
            env = gym.make(f"robodojo/{task_name}")
            observation, _ = env.reset(seed=0)
            for step in range(3):
                action = env.action_space.sample()
                for side in SIDES:
                    joints = observation[f"state.{side}_joints"]
                    action[f"action.{side}_joints"] = np.clip(
                        joints + env.np_random.uniform(-0.02, 0.02, 6),
                        -np.pi,
                        np.pi,
                    ).astype(np.float32)
                observation, _, terminated, truncated, _ = env.step(action)
                for key in CAMERAS:
                    Image.fromarray(observation[f"video.{key}"]).save(
                        output_dir / f"{task_name}_{step}_{key}.png"
                    )
                if terminated or truncated:
                    break
            print(  # noqa: T201
                f"PASS {task_name}: action={_shapes(action)} "
                f"observation={_shapes(observation)}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            failures.append(task_name)
            message = str(error).split(" was raised from", 1)[0].splitlines()[0]
            print(f"FAIL {task_name}: {type(error).__name__}: {message}", flush=True)  # noqa: T201
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001, S110
                    pass

    print(f"RoboDojo tasks: {len(tasks) - len(failures)}/{len(tasks)} passed")  # noqa: T201
    if failures:
        raise SystemExit(f"Failed tasks: {', '.join(failures)}")
    sys.stdout.flush()
    _APP_LAUNCHER.app.close()


if __name__ == "__main__":
    main()
