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

    GR1 = "gr1"
    """
    The Fourier GR1 robot.
    """

    ##### Pre-registered posttrain embodiment tags #####
    UNITREE_G1 = "unitree_g1"
    """
    The Unitree G1 robot.
    """

    LIBERO_PANDA = "libero_panda"
    """
    The Libero panda robot.
    """

    OXE_GOOGLE = "oxe_google"
    """
    The Open-X-Embodiment Google robot.
    """

    OXE_WIDOWX = "oxe_widowx"
    """
    The Open-X-Embodiment WidowX robot.
    """

    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    """
    The Behavior R1 Pro robot.
    """

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment.
    """
