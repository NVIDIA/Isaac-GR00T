from enum import Enum

from omnigibson.transition_rules import CookingSystemRule, MixingToolRule, ToggleableMachineRule
from omnigibson.utils import transform_utils as T
import torch as th


# Define room dependencies by scene
# Format: {scene_name: {room_A: [room_B, room_C], ...}}
ROOM_DEPENDENCIES = {
    "house_single_floor": {
        "dining_room": ["kitchen", "living_room"],
        "kitchen": ["living_room", "dining_room"],
        "living_room": ["kitchen", "dining_room"],
    },
    "house_double_floor_lower": {
        "kitchen": ["living_room", "corridor"],
        "living_room": ["kitchen", "corridor"],
        "garage": ["corridor"],
    },
    "house_double_floor_upper": {},
}

TASK_SPECIFIC_EXTRA_ROOMS = {
    "bringing_in_kindling": {
        "house_double_floor_lower": ["corridor"],
    },
    "bringing_newspaper_in": {
        "house_double_floor_lower": ["corridor"],
    },
    "chopping_wood": {
        "house_double_floor_lower": ["garage"],
    },
}

VALIDATED_TASKS = [
    "picking_up_trash",  # difficulty: 2
    "putting_away_Halloween_decorations",  # difficulty: 3
    "cleaning_up_plates_and_food",  # difficulty: 3.5
    "setting_mousetraps",  # difficulty: 2
    "hiding_Easter_eggs",  # difficulty: 2
    "set_up_a_coffee_station_in_your_kitchen",  # difficulty: 3
    "putting_dishes_away_after_cleaning",  # difficulty: 3
    "preparing_lunch_box",  # difficulty: 3
    "loading_the_car",  # difficulty: 3.5
    "carrying_in_groceries",  # difficulty: 3.5
    "turning_on_radio",  # difficulty: 1
    "picking_up_toys",  # difficulty: 3.5
    "can_meat",  # difficulty: 3.5
    "rearranging_kitchen_furniture",  # difficulty: 3
    "putting_up_Christmas_decorations_inside",  # difficulty: 2
    "bringing_in_wood",  # difficulty: 1.5
    "moving_boxes_to_storage",  # difficulty: 1.5
    "bringing_water",  # difficulty: 1.5
    "tidying_bedroom",  # difficulty: 2
    "outfit_a_basic_toolbox",  # difficulty: 2,
    "sorting_vegetables",
    "collecting_childrens_toys",
    "putting_shoes_on_rack",
    "boxing_books_up_for_storage",
    "storing_food",
    "clearing_food_from_table_into_fridge",
    "assembling_gift_baskets",
    "sorting_household_items",
    "getting_organized_for_work",
    "clean_up_your_desk",
    "setting_the_fire",
    "clean_boxing_gloves",
    "wash_a_baseball_cap",
    "wash_dog_toys",
    "hanging_pictures",
    "attach_a_camera_to_a_tripod",
    "clean_a_patio",
    "clean_a_trumpet",
    "spraying_for_bugs",
    "spraying_fruit_trees",
    "make_microwave_popcorn",
    "cook_cabbage",
    "make_pizza",
    "chop_an_onion",
    "slicing_vegetables",
    "chopping_wood",
    "canning_food",
    "cook_hot_dogs",
    "cook_bacon",
    "freeze_pies",
]


# Viewing mode configuration
class ViewingMode(str, Enum):
    SINGLE_VIEW = "single_view"
    VR = "vr"
    MULTI_VIEW_1 = "multi_view_1"


# Feature flags
USE_FLUID = False
USE_CLOTH = False
USE_ARTICULATED = False
FULL_SCENE = False
VIEWING_MODE = ViewingMode.MULTI_VIEW_1
SIMPLIFIED_TRUNK_CONTROL = True
APPLY_EXTRA_GRIP = False

# Robot parameters
SUPPORTED_ROBOTS = ["R1", "R1Pro"]
ROBOT_TYPE = "R1Pro"  # This should always be our robot generally since GELLO is designed for this specific robot
ROBOT_NAME = "robot_r1"

# R1 robot-specific configurations
R1_UPRIGHT_TORSO_JOINT_POS = th.tensor(
    [0.45, -0.4, 0.0, 0.0], dtype=th.float32
)  # For upper cabinets, shelves, etc.
R1_DOWNWARD_TORSO_JOINT_POS = th.tensor(
    [1.6, -2.5, -0.94, 0.0], dtype=th.float32
)  # For bottom cabinets, dishwashers, etc.
R1_GROUND_TORSO_JOINT_POS = th.tensor(
    [1.735, -2.57, -2.1, 0.0], dtype=th.float32
)  # For ground object pick up
R1_WRIST_CAMERA_LOCAL_POS = th.tensor(
    [0.1, 0.0, -0.1], dtype=th.float32
)  # Local position of the wrist camera relative to eef
R1_WRIST_CAMERA_LOCAL_ORI = th.tensor(
    [0.6830127018922194, 0.6830127018922193, 0.18301270189221927, 0.18301270189221946],
    dtype=th.float32,
)  # Local orientation of the wrist camera relative to eef

# R1 Pro robot-specific configurations
R1PRO_HEAD_CAMERA_LOCAL_POS = th.tensor(
    [0.06, 0.0, 0.01], dtype=th.float32
)  # Local position of the head camera relative to head link
R1PRO_HEAD_CAMERA_LOCAL_ORI = th.tensor(
    [-1.0, 0.0, 0.0, 0.0], dtype=th.float32
)  # Local orientation of the head camera relative to head link

# Default parameters
DEFAULT_TRUNK_TRANSLATE = 0.5
DEFAULT_RESET_DELTA_SPEED = 10.0  # deg / sec
N_COOLDOWN_SECS = 1.5
FLASHLIGHT_INTENSITY = 2000.0

# Visualization settings
RESOLUTION = [1080, 1080]  # [H, W]
USE_VISUAL_SPHERES = False
USE_VERTICAL_VISUALIZERS = False
GHOST_APPEAR_THRESHOLD = 0.1  # Threshold for showing ghost
GHOST_APPEAR_TIME = 10  # Number of frames to wait before showing ghost
USE_REACHABILITY_VISUALIZERS = True
AUTO_CHECKPOINTING = False  # checkpoint when 1) a new termination condition is met 2) some fixed amount of time has passed
STEPS_TO_AUTO_CHECKPOINT = 6000  # Assuming 20 fps, this is about 5 minutes

# Visualization cylinder configs
VIS_GEOM_COLORS = {
    False: [
        th.tensor([1.0, 0, 0]),  # Red
        th.tensor([0, 1.0, 0]),  # Green
        th.tensor([0, 0, 1.0]),  # Blue
    ],
    True: [
        th.tensor([1.0, 0.5, 0.5]),  # Light Red
        th.tensor([0.5, 1.0, 0.5]),  # Light Green
        th.tensor([0.5, 0.5, 1.0]),  # Light Blue
    ],
}
BEACON_LENGTH = 5.0

# Global whitelist of visual-only objects
VISUAL_ONLY_CATEGORIES = {
    # "bush",
    # "tree",
    # "pot_plant",
}

# Global whitelist of task-relevant objects
EXTRA_TASK_RELEVANT_CATEGORIES = {
    "floors",
    "driveway",
    "lawn",
}

# OmniGibson simulator settings
OMNIGIBSON_MACROS = {
    "USE_NUMPY_CONTROLLER_BACKEND": True,
    "USE_GPU_DYNAMICS": (USE_FLUID or USE_CLOTH),
    "ENABLE_FLATCACHE": True,
    "ENABLE_OBJECT_STATES": True,  # True (FOR TASKS!)
    "ENABLE_TRANSITION_RULES": True,
    "ENABLE_CCD": True,
    "ENABLE_HQ_RENDERING": USE_FLUID,
    "GUI_VIEWPORT_ONLY": True,
}

# Controller configuration for R1 robot
R1_CONTROLLER_CONFIG = {
    "arm_left": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "arm_right": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "gripper_left": {
        "name": "MultiFingerGripperController",
        "mode": "smooth",
        "command_input_limits": "default",
        "command_output_limits": "default",
    },
    "gripper_right": {
        "name": "MultiFingerGripperController",
        "mode": "smooth",
        "command_input_limits": "default",
        "command_output_limits": "default",
    },
    "base": {
        "name": "HolonomicBaseJointController",
        "motor_type": "velocity",
        "vel_kp": 150,
        "command_input_limits": [-th.ones(3), th.ones(3)],
        "command_output_limits": [
            -th.tensor([0.75, 0.75, 1.0]),
            th.tensor([0.75, 0.75, 1.0]),
        ],
        "use_impedances": False,
    },
    "trunk": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "camera": {
        "name": "NullJointController",
    },
}

ROBOT_RESET_JOINT_POS = {
    "R1": th.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,  # 6 virtual base joints
            0,
            0,
            0,
            0,  # 4 trunk joints -- these will be programmatically added
            33,
            -33,  # L, R arm joints
            162,
            162,
            -108,
            -108,
            34,
            -34,
            73,
            -73,
            -65,
            65,
            0,
            0,  # 2 L gripper
            0,
            0,  # 2 R gripper
        ]
    )
    * th.pi
    / 180,
    "R1Pro": th.zeros(28) * th.pi / 180,
}

WRIST_CAMERA_LINK_NAME = {
    "R1": {
        "left": "left_eef_link",
        "right": "right_eef_link",
    },
    "R1Pro": {
        "left": "left_realsense_link",
        "right": "right_realsense_link",
    },
}

HEAD_CAMERA_LINK_NAME = {
    "R1": "eyes",
    "R1Pro": "zed_link",
}

FINGER_LINK_NAME = {
    "R1": {
        "left": "left_gripper_axis",
        "right": "right_gripper_axis",
    },
    "R1Pro": {
        "left": "left_gripper_finger_joint",
        "right": "right_gripper_finger_joint",
    },
}

# Reachability visualizer settings
REACHABILITY_VISUALIZER_CONFIG = {
    "beam_width": 0.005,
    "square_distance": 0.6,
    "square_width": 0.4,
    "square_height": 0.3,
    "beam_color": [0.7, 0.7, 0.7],
}

# Visualization cylinder configurations
VIS_CYLINDER_CONFIG = {
    "width": 0.01,
    "lengths": [0.25, 0.25, 0.5],  # x,y,z
    "proportion_offsets": [0.0, 0.0, 0.5],  # x,y,z
    "quat_offsets": [
        T.euler2quat(th.tensor([0.0, th.pi / 2, 0.0])),
        T.euler2quat(th.tensor([-th.pi / 2, 0.0, 0.0])),
        T.euler2quat(th.tensor([0.0, 0.0, 0.0])),
    ],
}

# Frame visualizer cylinder configurations
ATTACHMENT_FRAME_CONFIG = {
    "width": 0.008,
    "lengths": [0.15, 0.15, 0.15],
    "quat_offsets": [
        T.euler2quat(th.tensor([0.0, th.pi / 2, 0.0])),  # X-axis (red)
        T.euler2quat(th.tensor([-th.pi / 2, 0.0, 0.0])),  # Y-axis (green)
        T.euler2quat(th.tensor([0.0, 0.0, 0.0])),  # Z-axis (blue)
    ],
    "colors": [
        th.tensor([1.0, 0.0, 0.0]),  # Red for X-axis
        th.tensor([0.0, 1.0, 0.0]),  # Green for Y-axis
        th.tensor([0.0, 0.0, 1.0]),  # Blue for Z-axis
    ],
}

# Camera and viewport configuration
CAMERA_VIEWPORT_POSITIONS = {
    "left_shoulder": {"parent": "DockSpace", "position": "LEFT", "ratio": 0.25},
    "left_wrist": {
        "parent": "viewport_left_shoulder",
        "position": "BOTTOM",
        "ratio": 0.5,
    },
    "right_shoulder": {"parent": "DockSpace", "position": "RIGHT", "ratio": 0.2},
    "right_wrist": {
        "parent": "viewport_right_shoulder",
        "position": "BOTTOM",
        "ratio": 0.5,
    },
}

# External camera parameters
EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor0": {
        "position": [-0.4, 0, 2.0],
        "orientation": [0.2706, -0.2706, -0.6533, 0.6533],
    },
    "external_sensor1": {
        "position": [-0.2, 0.6, 2.0],
        "orientation": [-0.1930, 0.4163, 0.8062, -0.3734],
    },
    "external_sensor2": {
        "position": [-0.2, -0.6, 2.0],
        "orientation": [0.4164, -0.1929, -0.3737, 0.8060],
    },
}

# UI visual settings
UI_SETTINGS = {
    "goal_satisfied_color": 0xFF00FF00,  # Green (ABGR)
    "goal_unsatisfied_color": 0xFF0000FF,  # Red (ABGR)
    "font_size": 25,
    "top_margin": 50,
    "left_margin": 50,
}

# Status display settings
STATUS_DISPLAY_SETTINGS = {
    "event_duration": 3.0,  # seconds
    "persistent_duration": 0.1,  # For persistent events - very short
    "persistent_states": ["in_cooldown", "waiting_to_resume"],
    "event_colors": {
        "checkpoint": 0xFF00FF00,  # Green
        "rollback": 0xFFFF00FF,  # Magenta
        "cooldown": 0xFF00FFFF,  # Yellow
        "waiting": 0xFFFF0000,  # White
        "reset": 0xFF00AAFF,  # Orange
    },
    "font_size": 20,
    "bottom_margin": 50,
    "right_margin": 50,
    "line_spacing": 5,
}

# Visual sphere settings for object highlighting
OBJECT_HIGHLIGHT_SPHERE = {
    "opacity": 0.5,
    "emissive_intensity": 10000.0,
}

INCLUDE_TRUNK_CONTACT_OBS = True
INCLUDE_BASE_CONTACT_OBS = True
INCLUDE_ARM_CONTACT_OBS = False
INCLUDE_FINGER_CONTACT_OBS = False

INCLUDE_JACOBIAN_OBS = False
GHOST_UPDATE_FREQ = 3

BLINK_WHEN_IN_CONTACT = True

DISABLED_TRANSITION_RULES = [ToggleableMachineRule, MixingToolRule, CookingSystemRule]
