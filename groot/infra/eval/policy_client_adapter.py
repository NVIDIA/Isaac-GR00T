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

"""Policy client adapter for open-loop evaluator observations."""

from __future__ import annotations

from typing import Any


class PolicyClientAdapter:
    def __init__(self, client: Any):
        self.client = client

    def get_modality_config(self) -> dict[str, Any]:
        return self.client.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.client.reset(options)

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action, info = self.client.get_action(self.flat_observation(observation), options)
        return self.nested_action(action), info

    def flat_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for modality in ("video", "state"):
            for key, value in observation.get(modality, {}).items():
                flat[f"{modality}.{key}"] = value
        for key, value in observation.get("language", {}).items():
            flat[key] = value
        return flat

    def nested_action(self, action: dict[str, Any]) -> dict[str, Any]:
        nested: dict[str, Any] = {}
        for key, value in action.items():
            nested_key = key.removeprefix("action.")
            nested_value = value
            if getattr(value, "ndim", 0) == 2:
                nested_value = value[None, ...]
            nested[nested_key] = nested_value
        return nested

    def close(self) -> None:
        self.client.close()
