from gr00t.data.embodiment_tags import EmbodimentTag


# Mapping from gym-registered env_name prefix to EmbodimentTag.
# The prefix is the part before "/" in env_name (e.g. "libero_sim" from "libero_sim/task").
# Add new entries here when supporting a new benchmark.
ENV_PREFIX_TO_EMBODIMENT_TAG: dict[str, EmbodimentTag] = {
    # Pretrain benchmarks
    "robocasa_panda_omron": EmbodimentTag.ROBOCASA_PANDA_OMRON,
    "gr1_unified": EmbodimentTag.GR1,
    # Locomanipulation
    "gr00tlocomanip_g1": EmbodimentTag.UNITREE_G1,
    "gr00tlocomanip_g1_sim": EmbodimentTag.UNITREE_G1,
    "gr00tlocomanip_g1_new": EmbodimentTag.UNITREE_G1,
    # Posttrain benchmarks
    "sim_behavior_r1_pro": EmbodimentTag.BEHAVIOR_R1_PRO,
    "simpler_env_google": EmbodimentTag.SIMPLER_ENV_GOOGLE,
    "simpler_env_widowx": EmbodimentTag.SIMPLER_ENV_WIDOWX,
}


def get_embodiment_tag_from_env_name(env_name: str) -> EmbodimentTag:
    """Get the EmbodimentTag for a gym-registered environment name.

    Looks up the env_name prefix (before "/") in ENV_PREFIX_TO_EMBODIMENT_TAG.
    Falls back to using the prefix directly as an EmbodimentTag value.
    """
    prefix = env_name.split("/")[0]
    if prefix in ENV_PREFIX_TO_EMBODIMENT_TAG:
        return ENV_PREFIX_TO_EMBODIMENT_TAG[prefix]
    return EmbodimentTag(prefix)
