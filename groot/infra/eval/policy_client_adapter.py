# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Policy client adapter for open-loop evaluator observations."""

from __future__ import annotations

from typing import Any


class PolicyClientAdapter:
    def __init__(self, client: Any, *, ready: bool = True):
        self.client = client
        self._ready = ready

    def get_modality_config(self) -> dict[str, Any]:
        return self.client.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.client.reset(options)

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        flat_observation = self.flat_observation(observation)
        if not self._ready:
            self.client.warmup_and_require_ready(flat_observation, options)
            self._ready = True
        action, info = self.client.get_action(flat_observation, options)
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
