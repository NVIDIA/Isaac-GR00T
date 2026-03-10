from enum import Enum


"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""


class EmbodimentTag(Enum):
    ##### Pretrain embodiment tags #####
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    The RoboCasa Panda robot with omron mobile base.
    """

    GR1 = "gr1_unified"
    """
    The Fourier GR1 robot.
    """

    XDOF = "xdof"
    """
    The generic X-DOF robot.
    """

    AGIBOT = "agibot"
    """
    The AgiBot robot.
    """

    ##### Pre-registered posttrain embodiment tags #####
    UNITREE_G1 = "unitree_g1_full_body_with_waist_height_nav_cmd"
    """
    The Unitree G1 robot.
    """

    SIMPLER_ENV_GOOGLE = "simpler_env_google"
    """
    The SimplerEnv Google robot.
    """

    SIMPLER_ENV_WIDOWX = "simpler_env_widowx"
    """
    The SimplerEnv WidowX robot.
    """

    OXE_DROID = "oxe_droid_joint_position_relative"
    """
    The Open-X-Embodiment DROID robot with relative joint position actions.
    """

    BEHAVIOR_R1_PRO = "sim_behavior_r1_pro"
    """
    The Behavior R1 Pro robot.
    """

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment.
    """
