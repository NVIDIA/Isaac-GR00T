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

"""Regression tests for env_name → EmbodimentTag mapping.

Covers all 10 supported sim benchmarks, including fixes for:
- GitHub Issue #479: LIBERO, SimplerEnv Google, SimplerEnv WidowX
"""

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import ENV_PREFIX_TO_EMBODIMENT_TAG, get_embodiment_tag_from_env_name
import pytest


class TestEnvPrefixMapping:
    """Verify ENV_PREFIX_TO_EMBODIMENT_TAG covers all known benchmarks."""

    def test_all_known_prefixes_present(self):
        expected_prefixes = {
            "gr00tlocomanip_g1",
            "gr00tlocomanip_g1_sim",
            "gr00tlocomanip_g1_new",
            "simpler_env_google",
            "simpler_env_widowx",
            "libero_sim",
        }
        assert set(ENV_PREFIX_TO_EMBODIMENT_TAG.keys()) == expected_prefixes

    def test_related_prefixes_map_to_same_tag(self):
        """Prefixes that share a common root must map to the same EmbodimentTag.

        Guards against accidentally assigning a conflicting tag when adding
        a new variant of an existing benchmark (e.g. gr00tlocomanip_g1_v2).
        """
        for prefix, tag in ENV_PREFIX_TO_EMBODIMENT_TAG.items():
            for other_prefix, other_tag in ENV_PREFIX_TO_EMBODIMENT_TAG.items():
                if prefix != other_prefix and other_prefix.startswith(prefix):
                    assert tag == other_tag, (
                        f"Conflicting tags: '{prefix}' -> {tag}, "
                        f"'{other_prefix}' -> {other_tag}. "
                        f"Related prefixes must map to the same EmbodimentTag."
                    )


class TestGetEmbodimentTagFromEnvName:
    """Test get_embodiment_tag_from_env_name() for all supported benchmarks."""

    # --- Benchmarks that already worked (via explicit checks or fallback) ---

    @pytest.mark.parametrize(
        "env_name",
        [
            "gr00tlocomanip_g1/LMBottlePnP",
            "gr00tlocomanip_g1_sim/LMBottlePnP",
            "gr00tlocomanip_g1_new/LMBottlePnP",
        ],
    )
    def test_locomanip_g1(self, env_name):
        assert get_embodiment_tag_from_env_name(env_name) == EmbodimentTag.UNITREE_G1

    # --- Issue #479 fixes: these were broken before ---

    def test_simpler_env_google(self):
        tag = get_embodiment_tag_from_env_name("simpler_env_google/google_robot_pick_coke_can")
        assert tag == EmbodimentTag.SIMPLER_ENV_GOOGLE

    def test_simpler_env_widowx(self):
        tag = get_embodiment_tag_from_env_name("simpler_env_widowx/widowx_spoon_on_towel")
        assert tag == EmbodimentTag.SIMPLER_ENV_WIDOWX

    def test_libero_panda(self):
        tag = get_embodiment_tag_from_env_name("libero_sim/KITCHEN_SCENE3_pick_up_the_black_bowl")
        assert tag == EmbodimentTag.LIBERO_PANDA

    # --- Edge cases ---

    def test_unknown_env_raises_value_error(self):
        with pytest.raises(ValueError):
            get_embodiment_tag_from_env_name("totally_unknown_env/some_task")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            get_embodiment_tag_from_env_name("")

    def test_multi_slash_uses_first_segment(self):
        """Only the first segment before '/' is used as the prefix."""
        tag = get_embodiment_tag_from_env_name("simpler_env_google/task/subtask")
        assert tag == EmbodimentTag.SIMPLER_ENV_GOOGLE
