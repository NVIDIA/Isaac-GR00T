# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gr00t.data.embodiment_tags import EmbodimentTag


# Mapping from gym-registered env_name prefix to EmbodimentTag.
# The prefix is the part before "/" in env_name (e.g. "libero_sim" from "libero_sim/task").
# Add new entries here when supporting a new benchmark.
ENV_PREFIX_TO_EMBODIMENT_TAG: dict[str, EmbodimentTag] = {
    # Locomanipulation
    "gr00tlocomanip_g1": EmbodimentTag.UNITREE_G1,
    "gr00tlocomanip_g1_sim": EmbodimentTag.UNITREE_G1,
    "gr00tlocomanip_g1_new": EmbodimentTag.UNITREE_G1,
    # Posttrain benchmarks
    "simpler_env_google": EmbodimentTag.SIMPLER_ENV_GOOGLE,
    "simpler_env_widowx": EmbodimentTag.SIMPLER_ENV_WIDOWX,
    "libero_sim": EmbodimentTag.LIBERO_PANDA,
    "robocasa_panda_omron": EmbodimentTag.ROBOCASA_PANDA_OMRON,
    "robocasa365_panda_omron": EmbodimentTag.ROBOCASA_PANDA_OMRON,
    "gr1_unified": EmbodimentTag.ROBOCASA_GR1_TABLETOP,
}


def get_embodiment_tag_from_env_name(env_name: str) -> EmbodimentTag:
    """Get the EmbodimentTag for a gym-registered environment name.

    Looks up the env_name prefix (before "/") in ENV_PREFIX_TO_EMBODIMENT_TAG.
    Falls back to using the prefix directly as an EmbodimentTag value (most
    prefixes are deliberately equal to their tag's value; the dict above only
    patches the prefixes that diverge).

    Raises:
        ValueError: If the prefix is neither a key in
            ENV_PREFIX_TO_EMBODIMENT_TAG nor a valid EmbodimentTag value. This
            is the expected failure when a new benchmark is registered under a
            new prefix but no mapping entry was added — the message points at
            the exact fix so the failure is actionable at the call boundary
            instead of surfacing as a cryptic enum error deeper in eval.
    """
    prefix = env_name.split("/")[0]
    if prefix in ENV_PREFIX_TO_EMBODIMENT_TAG:
        return ENV_PREFIX_TO_EMBODIMENT_TAG[prefix]
    try:
        return EmbodimentTag(prefix)
    except ValueError:
        known_prefixes = sorted(ENV_PREFIX_TO_EMBODIMENT_TAG)
        valid_tag_values = [tag.value for tag in EmbodimentTag]
        raise ValueError(
            f"env_name prefix {prefix!r} (from env_name {env_name!r}) maps to no "
            f"EmbodimentTag. A gym environment is registered under this prefix, but "
            f"it is neither a key in ENV_PREFIX_TO_EMBODIMENT_TAG nor a valid "
            f"EmbodimentTag value. Add an entry to ENV_PREFIX_TO_EMBODIMENT_TAG in "
            f"gr00t/eval/sim/env_utils.py.\n"
            f"  known prefixes:    {known_prefixes}\n"
            f"  valid tag values:  {valid_tag_values}"
        ) from None
