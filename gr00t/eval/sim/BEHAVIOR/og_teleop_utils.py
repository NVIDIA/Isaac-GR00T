from pathlib import Path

from bddl.activity import Conditions
from gr00t.eval.sim.BEHAVIOR.og_teleop_cfg import (
    DEFAULT_TRUNK_TRANSLATE,
    R1_CONTROLLER_CONFIG,
    R1_DOWNWARD_TORSO_JOINT_POS,
    R1_GROUND_TORSO_JOINT_POS,
    R1_UPRIGHT_TORSO_JOINT_POS,
    RESOLUTION,
    ROBOT_NAME,
    ROBOT_RESET_JOINT_POS,
    ROBOT_TYPE,
    ROOM_DEPENDENCIES,
    TASK_SPECIFIC_EXTRA_ROOMS,
)
import yaml


def get_task_relevant_room_types(activity_name):
    activity_conditions = Conditions(
        activity_name,
        0,
        simulator_name="omnigibson",
        predefined_problem=None,
    )
    init_conds = activity_conditions.parsed_initial_conditions
    room_types = set()
    for init_cond in init_conds:
        if len(init_cond) == 3:
            if "inroom" == init_cond[0]:
                room_types.add(init_cond[2])

    return list(room_types)


def augment_rooms(relevant_rooms, scene_model, task_name):
    """
    Augment the list of relevant rooms by adding dependent rooms that need to be loaded together.

    Args:
        relevant_rooms: List of room types that are initially relevant
        scene_model: The scene model being used
        task_name: Name of the task being used

    Returns:
        Augmented list of room types including all dependencies
    """

    # Get dependencies for current scene
    scene_dependencies = ROOM_DEPENDENCIES[scene_model]

    # Create a copy of the original list to avoid modifying it during iteration
    augmented_rooms = relevant_rooms.copy()

    # Check each relevant room for dependencies
    for room in relevant_rooms:
        if room in scene_dependencies:
            # Add dependent rooms if they're not already in the list
            for dependent_room in scene_dependencies[room]:
                if dependent_room not in augmented_rooms:
                    augmented_rooms.append(dependent_room)

    # Additionally add any task-specific rooms
    augmented_rooms += TASK_SPECIFIC_EXTRA_ROOMS.get(task_name, dict()).get(scene_model, [])
    # Remove redundancies
    augmented_rooms = list(set(augmented_rooms))

    return augmented_rooms


def load_available_tasks():
    """
    Load available tasks from configuration file

    Returns:
        dict: Dictionary of available tasks
    """
    # Get directory of current file
    dir_path = Path(__file__).parent
    task_cfg_path = dir_path / "available_tasks.yaml"

    try:
        with open(task_cfg_path, "r") as file:
            available_tasks = yaml.safe_load(file)
        return available_tasks
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading available tasks: {e}")
        return {}


def infer_torso_qpos_from_trunk_translate(translate):
    """
    Convert from trunk translate value to torso joint positions

    Args:
        translate (float): Trunk translate value between 0.0 and 2.0

    Returns:
        torch.Tensor: Torso joint positions
    """
    translate = min(max(translate, 0.0), 2.0)

    # Interpolate between the three pre-determined joint positions
    if translate <= 1.0:
        # Interpolate between upright and down positions
        interpolation_factor = translate
        interpolated_trunk_pos = (
            1 - interpolation_factor
        ) * R1_UPRIGHT_TORSO_JOINT_POS + interpolation_factor * R1_DOWNWARD_TORSO_JOINT_POS
    else:
        # Interpolate between down and ground positions
        interpolation_factor = translate - 1.0
        interpolated_trunk_pos = (
            1 - interpolation_factor
        ) * R1_DOWNWARD_TORSO_JOINT_POS + interpolation_factor * R1_GROUND_TORSO_JOINT_POS

    return interpolated_trunk_pos


def generate_robot_config(task_name=None, task_cfg=None):
    """
    Generate robot configuration

    Args:
        task_name: Name of the task (optional)
        task_cfg: Dictionary of task config (optional)

    Returns:
        dict: Robot configuration
    """
    # Create a copy of the controller config to avoid modifying the original
    controller_config = {k: v.copy() for k, v in R1_CONTROLLER_CONFIG.items()}

    robot_config = {
        "type": ROBOT_TYPE,
        "name": ROBOT_NAME,
        "action_normalize": False,
        "controller_config": controller_config,
        "self_collisions": True,
        "obs_modalities": [],
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "grasping_mode": "assisted",
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": RESOLUTION[0],
                    "image_width": RESOLUTION[1],
                },
            },
        },
    }

    # Override position and orientation for tasks
    if task_name is not None and task_cfg is not None:
        robot_config["position"] = task_cfg["robot_start_position"]
        robot_config["orientation"] = task_cfg["robot_start_orientation"]

    # Add reset joint positions
    joint_pos = ROBOT_RESET_JOINT_POS[ROBOT_TYPE].clone()

    # NOTE: Fingers MUST start open, or else generated AG spheres will be spawned incorrectly
    joint_pos[-4:] = 0.05

    # Update trunk qpos as well
    joint_pos[6:10] = infer_torso_qpos_from_trunk_translate(DEFAULT_TRUNK_TRANSLATE)

    robot_config["reset_joint_pos"] = joint_pos

    return robot_config


def get_camera_config(name, relative_prim_path, position, orientation, resolution, modalities=[]):
    """
    Generate a camera configuration dictionary

    Args:
        name (str): Camera name
        relative_prim_path (str): Relative path to camera in the scene
        position (List[float]): Camera position [x, y, z]
        orientation (List[float]): Camera orientation [x, y, z, w]
        resolution (List[int]): Camera resolution [height, width]
        modalities (List[str]): List of modalities for the camera

    Returns:
        dict: Camera configuration dictionary
    """
    return {
        "sensor_type": "VisionSensor",
        "name": name,
        "relative_prim_path": relative_prim_path,
        "modalities": modalities,
        "sensor_kwargs": {
            "viewport_name": "Viewport",
            "image_height": resolution[0],
            "image_width": resolution[1],
        },
        "position": position,
        "orientation": orientation,
        "pose_frame": "parent",
        "include_in_obs": False,
    }


def generate_basic_environment_config(task_name, task_cfg):
    """
    Generate a basic environment configuration

    Args:
        task_name (str): Name of the task
        task_cfg: Dictionary of task config

    Returns:
        dict: Environment configuration
    """
    cfg = {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": task_cfg["scene_model"],
            "load_room_types": None,
            "load_room_instances": task_cfg.get("load_room_instances", None),
            "include_robots": False,
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": False,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {
                "max_steps": 5000,
            },
            "reward_config": {
                "r_potential": 1.0,
            },
            "include_obs": False,
        },
    }
    return cfg
