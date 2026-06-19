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

from __future__ import annotations

import ast
from pathlib import Path
import re

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import ENV_PREFIX_TO_EMBODIMENT_TAG, get_embodiment_tag_from_env_name
import pytest


class TestEnvPrefixMapping:
    """Verify ENV_PREFIX_TO_EMBODIMENT_TAG covers all known benchmarks."""

    def test_all_known_prefixes_present(self):
        # NOTE: this is an author-vs-author reminder (hand-written set vs the
        # map), not the closure guard — it only catches an *accidental* edit to
        # the map. The author-vs-truth closure check (every actually-registered
        # prefix resolves) lives in TestRegisteredPrefixClosure below.
        expected_prefixes = {
            "gr00tlocomanip_g1",
            "gr00tlocomanip_g1_sim",
            "gr00tlocomanip_g1_new",
            "gr1_unified",
            "robocasa365_panda_omron",
            "robocasa_panda_omron",
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

    def test_gr1_unified_maps_to_robocasa_gr1_tabletop(self):
        env_name = "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
        assert get_embodiment_tag_from_env_name(env_name) == EmbodimentTag.ROBOCASA_GR1_TABLETOP

    def test_robocasa_panda_omron_maps_to_dedicated_tag(self):
        env_name = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
        assert get_embodiment_tag_from_env_name(env_name) == EmbodimentTag.ROBOCASA_PANDA_OMRON

    def test_robocasa365_panda_omron_maps_to_dedicated_tag(self):
        env_name = "robocasa365_panda_omron/CloseFridge_PandaOmron_Env"
        assert get_embodiment_tag_from_env_name(env_name) == EmbodimentTag.ROBOCASA_PANDA_OMRON

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


class TestRegisteredPrefixClosure:
    """Author-vs-truth closure check for the env-prefix -> EmbodimentTag mapping.

    Binds the mapping to the *real* gym registration sites (the ground truth)
    instead of a hand-written prefix list: it scans ``gr00t/eval/sim/**/*.py``
    for ``register(id="<prefix>/...")`` call sites and asserts that every
    statically-registered prefix resolves via
    :func:`get_embodiment_tag_from_env_name` without raising.

    This catches the common cross-layer categorical drift: a new benchmark adds
    ``register(id="newbench/...")`` but nobody adds the matching
    ``ENV_PREFIX_TO_EMBODIMENT_TAG`` entry, which would otherwise only surface
    as a ``ValueError`` deep in an eval run.

    Coverage note: prefixes registered in *external* dependencies (e.g. the
    ``gr00tlocomanip_*`` envs live outside this repo) or built fully
    dynamically are invisible to a static scan; those remain covered by the
    actionable fail-fast in ``get_embodiment_tag_from_env_name`` and the
    eval-time integration path.

    Implementation note: the scan walks the parsed AST rather than matching raw
    source text, so prefixes mentioned only in comments or docstrings are never
    picked up — the binding has to be a real ``id=`` keyword argument or an
    ``id``/``id_name`` assignment to count.
    """

    # The static leading text of a literal must start with a gym-style prefix.
    _PREFIX_RE = re.compile(r"^([a-z][a-z0-9_]*)/")

    @classmethod
    def _prefix_from_node(cls, node: ast.AST) -> str | None:
        """Extract the leading ``<prefix>/`` from a str or f-string AST node.

        Returns ``None`` for non-string nodes or f-strings whose text starts
        with an interpolation (no statically knowable prefix).
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal = node.value
        elif isinstance(node, ast.JoinedStr) and node.values:
            first = node.values[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                return None
            literal = first.value
        else:
            return None
        match = cls._PREFIX_RE.match(literal)
        return match.group(1) if match else None

    def _registered_prefixes(self) -> set[str]:
        sim_dir = Path(__file__).resolve().parents[4] / "gr00t" / "eval" / "sim"
        assert sim_dir.is_dir(), f"sim source dir not found: {sim_dir}"
        prefixes: set[str] = set()
        for py in sim_dir.rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                # register(id="<prefix>/...") / register(id=f"<prefix>/...")
                if isinstance(node, ast.Call):
                    candidates = [kw.value for kw in node.keywords if kw.arg == "id"]
                # id_name = f"<prefix>/..." (later passed as register(id=id_name))
                elif isinstance(node, ast.Assign):
                    candidates = (
                        [node.value]
                        if any(
                            isinstance(t, ast.Name) and t.id in {"id", "id_name"}
                            for t in node.targets
                        )
                        else []
                    )
                else:
                    continue
                for value in candidates:
                    prefix = self._prefix_from_node(value)
                    if prefix:
                        prefixes.add(prefix)
        return prefixes

    def test_every_registered_prefix_resolves(self):
        prefixes = self._registered_prefixes()
        # Guard against a vacuous pass if the scan ever stops finding call sites.
        assert prefixes, "found no register(id=...) prefixes to check; the scan likely broke"
        unresolved = []
        for prefix in sorted(prefixes):
            try:
                get_embodiment_tag_from_env_name(f"{prefix}/__closure_probe__")
            except ValueError:
                unresolved.append(prefix)
        assert not unresolved, (
            "These env-prefixes are registered via register(id=...) but resolve to no "
            f"EmbodimentTag: {unresolved}. Add them to ENV_PREFIX_TO_EMBODIMENT_TAG in "
            "gr00t/eval/sim/env_utils.py."
        )
