from enum import Enum


"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""


class EmbodimentTag(Enum):
    """Embodiment tags supported by the GR00T N1.7 checkpoint.

    Pretrain tags (baked into the base model nvidia/GR00T-N1.7-3B, inference-ready):
    - OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT -> "oxe_droid_relative_eef_relative_joint"
    - XDOF                                  -> "xdof_relative_eef_relative_joint"
    - XDOF_SUBTASK                          -> "xdof_relative_eef_relative_joint_subtask"
    - REAL_G1                               -> "real_g1_relative_eef_relative_joints"
    - REAL_R1_PRO_SHARPA                    -> "real_r1_pro_sharpa_relative_eef"
    - REAL_R1_PRO_SHARPA_HUMAN              -> "real_r1_pro_sharpa_relative_eef_human"
    - REAL_R1_PRO_SHARPA_MAXINSIGHTS        -> "real_r1_pro_sharpa_relative_eef_maxinsights"
    - REAL_R1_PRO_SHARPA_MECKA              -> "real_r1_pro_sharpa_relative_eef_mecka"

    Pre-registered posttrain tags (require finetuned checkpoint):
    - UNITREE_G1           -> "unitree_g1_full_body_with_waist_height_nav_cmd"
    - SIMPLER_ENV_GOOGLE   -> "simpler_env_google"
    - SIMPLER_ENV_WIDOWX   -> "simpler_env_widowx"
    - BEHAVIOR_R1_PRO      -> "sim_behavior_r1_pro"
    - LIBERO_PANDA         -> "libero_sim"

    Finetuning tag (for custom robots):
    - NEW_EMBODIMENT       -> "new_embodiment"

    Use ``EmbodimentTag.resolve(s)`` to look up a tag by name or value,
    case-insensitively.
    """

    ##### Pretrain embodiment tags (in base model processor_config.json) #####

    OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT = "oxe_droid_relative_eef_relative_joint"
    """
    The Open-X-Embodiment DROID robot with relative EEF and relative joint position actions.
    """

    XDOF = "xdof_relative_eef_relative_joint"
    """
    The generic X-DOF robot with relative EEF and relative joint position actions.
    """

    XDOF_SUBTASK = "xdof_relative_eef_relative_joint_subtask"
    """
    The generic X-DOF robot (subtask variant).
    """

    REAL_G1 = "real_g1_relative_eef_relative_joints"
    """
    Real-world Unitree G1 with relative EEF and relative joint actions.
    """

    REAL_R1_PRO_SHARPA = "real_r1_pro_sharpa_relative_eef"
    """
    Real-world R1 Pro Sharpa with relative EEF actions.
    """

    REAL_R1_PRO_SHARPA_HUMAN = "real_r1_pro_sharpa_relative_eef_human"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (human teleop data).
    """

    REAL_R1_PRO_SHARPA_MAXINSIGHTS = "real_r1_pro_sharpa_relative_eef_maxinsights"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (MaxInsights data, single-cam).
    """

    REAL_R1_PRO_SHARPA_MECKA = "real_r1_pro_sharpa_relative_eef_mecka"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (Mecka data, single-cam).
    """

    ##### Pre-registered posttrain embodiment tags #####

    UNITREE_G1 = "unitree_g1_full_body_with_waist_height_nav_cmd"
    """
    The Unitree G1 robot (sim, full-body with waist height and nav commands).
    """

    SIMPLER_ENV_GOOGLE = "simpler_env_google"
    """
    The SimplerEnv Google robot.
    """

    SIMPLER_ENV_WIDOWX = "simpler_env_widowx"
    """
    The SimplerEnv WidowX robot.
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

        Matches by enum **name** first (e.g. ``"xdof"`` -> ``XDOF``), then by
        enum **value** (e.g. ``"xdof_relative_eef_relative_joint"`` -> ``XDOF``).

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

        def _fmt(tags):
            return "\n".join(f"    {m.name:40s} -> {m.value}" for m in tags)

        msg = (
            f"Unknown embodiment tag: {tag!r}\n\n"
            f"  Base model tags (work with nvidia/GR00T-N1.7-3B):\n"
            f"{_fmt(PRETRAIN_TAGS)}\n\n"
            f"  Posttrain tags (require a finetuned checkpoint):\n"
            f"{_fmt(POSTTRAIN_TAGS)}\n\n"
            f"  Finetuning-only tags (for custom robots):\n"
            f"{_fmt(FINETUNE_ONLY_TAGS)}"
        )
        raise ValueError(msg)

    @classmethod
    def reverse_lookup(cls, value: str) -> "str":
        """Map a tag value string back to its enum name, or return the value as-is."""
        for member in cls:
            if member.value == value:
                return member.name
        return value


# Module-level tag category sets (cannot be Enum class attributes).
PRETRAIN_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
        EmbodimentTag.XDOF,
        EmbodimentTag.XDOF_SUBTASK,
        EmbodimentTag.REAL_G1,
        EmbodimentTag.REAL_R1_PRO_SHARPA,
        EmbodimentTag.REAL_R1_PRO_SHARPA_HUMAN,
        EmbodimentTag.REAL_R1_PRO_SHARPA_MAXINSIGHTS,
        EmbodimentTag.REAL_R1_PRO_SHARPA_MECKA,
    }
)
"""Tags baked into the base model (nvidia/GR00T-N1.7-3B) — usable without finetuning."""

POSTTRAIN_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.UNITREE_G1,
        EmbodimentTag.SIMPLER_ENV_GOOGLE,
        EmbodimentTag.SIMPLER_ENV_WIDOWX,
        EmbodimentTag.BEHAVIOR_R1_PRO,
        EmbodimentTag.LIBERO_PANDA,
    }
)
"""Tags that require a finetuned checkpoint."""

FINETUNE_ONLY_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.NEW_EMBODIMENT,
    }
)
"""Tags for custom robots (finetuning only, not in any shipped checkpoint)."""
