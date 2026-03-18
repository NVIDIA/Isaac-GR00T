from enum import Enum


"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""


class EmbodimentTag(Enum):
    """Embodiment tags supported by the GR00T N1.7 checkpoint.

    Pretrain tags (baked into the base model, inference-ready):
    - OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT -> "oxe_droid_relative_eef_relative_joint"  (DROID relative EEF + joint)
    - ROBOCASA_PANDA_OMRON                  -> "robocasa_panda_omron"                    (RoboCasa Panda + Omron base)
    - XDOF                                  -> "xdof"                                    (Generic X-DOF robot)
    - AGIBOT                                -> "agibot"                                  (AgiBot robot)

    Pre-registered posttrain tags (for finetuning):
    - UNITREE_G1           -> "unitree_g1_full_body_with_waist_height_nav_cmd"   (Unitree G1 full-body)
    - SIMPLER_ENV_GOOGLE   -> "simpler_env_google"                               (SimplerEnv Google Robot)
    - SIMPLER_ENV_WIDOWX   -> "simpler_env_widowx"                               (SimplerEnv WidowX)
    - BEHAVIOR_R1_PRO      -> "sim_behavior_r1_pro"                              (Behavior R1 Pro sim)
    - LIBERO_PANDA         -> "libero_sim"                                       (LIBERO Panda robot)

    Finetuning tag (for custom robots):
    - NEW_EMBODIMENT       -> "new_embodiment"                                   (Any new embodiment)

    Use ``EmbodimentTag.resolve(s)`` to look up a tag by name or value,
    case-insensitively.
    """

    ##### Pretrain embodiment tags #####
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    The RoboCasa Panda robot with omron mobile base.
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

    OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT = "oxe_droid_relative_eef_relative_joint"
    """
    The Open-X-Embodiment DROID robot with relative EEF and relative joint position actions.
    """

    BEHAVIOR_R1_PRO = "sim_behavior_r1_pro"
    """
    The Behavior R1 Pro robot.
    """

    LIBERO_PANDA = "libero_sim"
    """
    The LIBERO Panda robot (used for LIBERO-Goal, LIBERO-Object, LIBERO-Spatial, LIBERO-10).
    """

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment.
    """

    @classmethod
    def resolve(cls, tag: "str | EmbodimentTag") -> "EmbodimentTag":
        """Resolve a string to an EmbodimentTag, case-insensitively.

        Matches by enum **name** first (e.g. ``"gr1"`` -> ``GR1``), then by
        enum **value** (e.g. ``"xdof"`` -> ``XDOF``).

        Raises:
            ValueError: If *tag* does not match any known embodiment.
        """
        if isinstance(tag, cls):
            return tag
        key = tag.strip()
        key_lower = key.lower()
        # Match by enum name (case-insensitive)
        for member in cls:
            if member.name.lower() == key_lower:
                return member
        # Match by enum value (case-insensitive)
        for member in cls:
            if member.value.lower() == key_lower:
                return member
        known = "\n".join(f"  {m.name:30s} -> {m.value}" for m in cls if not m.name.startswith("_"))
        raise ValueError(f"Unknown embodiment tag: {tag!r}\nKnown tags (name -> value):\n{known}")
