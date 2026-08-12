# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""HTTP client for a remote chain-of-thought (CoT) planner server.

This module is embodiment-agnostic: it only speaks the planner wire protocol and
manages per-session statistics. Memory is managed on the server side; the client
only sends current images and the server maintains history and builds context.
"""

import base64
import io
import sys
from dataclasses import dataclass
from typing import Any

from loguru import logger
from PIL import Image
import requests


def _encode_pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def check_cot_endpoint(base_url: str = "http://localhost:8100", timeout: int = 5) -> None:
    """Check if the COT planner endpoint is available.

    Args:
        base_url: Base URL of the COT planner server.
        timeout: Request timeout in seconds.
    """
    health_url = f"{base_url}/health"
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            logger.info(f"COT planner endpoint is available at {base_url}")
            return
        else:
            logger.error(f"COT planner endpoint returned status {response.status_code}")
            logger.error(f"Response: {response.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to COT planner at {base_url}")
        logger.error("Make sure the planner server is running.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to COT planner at {base_url}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to check COT planner endpoint: {e}")
        sys.exit(1)


@dataclass
class MemoryConfig:
    """Configuration for memory mechanisms."""

    max_history_images: int = 10
    max_history_steps: int = 10000
    image_sample_strategy: str = "uniform"


class CoTPlannerClient:
    """Client for calling a remote CoT planner server via HTTP.

    Memory is managed on the server side. The client only sends current images,
    and the server maintains history and builds context.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        base_url: str = "http://localhost:8100",
        memory_type: str | None = None,
        memory_config: MemoryConfig | None = None,
        session_id: str | None = None,
        task_type: str | None = None,
        request_timeout: int = 300,
        next_action_template: str = "v2",
        fallback_response: str,
    ):
        self.base_url = base_url.rstrip("/")
        self.memory_type = memory_type
        self.memory_config = memory_config or MemoryConfig()
        self.task_type = task_type
        self.request_timeout = request_timeout
        self.next_action_template = next_action_template
        self.fallback_response = fallback_response
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        if session_id is None:
            import uuid

            session_id = str(uuid.uuid4())
        self.session_id = session_id
        self._session_created = False

        self._step_index = 0

        self.fallback_count = 0
        self.total_inferences = 0

        self.last_raw_response: str | None = None
        self.last_metadata: dict[str, Any] | None = None
        self.last_used_fallback: bool = False

    def _ensure_session_created(self):
        """Create session on server if not already done."""
        if not self._session_created:
            payload = {
                "session_id": self.session_id,
                "memory_config": {
                    "memory_type": self.memory_type,
                    "max_history_images": self.memory_config.max_history_images,
                    "max_history_steps": self.memory_config.max_history_steps,
                    "image_sample_strategy": self.memory_config.image_sample_strategy,
                },
                "task_type": self.task_type,
            }
            response = requests.post(
                f"{self.base_url}/create_session",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            self._session_created = True

    def reset_episode(self):
        """Reset state for a new episode."""
        self._ensure_session_created()
        response = requests.post(
            f"{self.base_url}/reset_session",
            json={"session_id": self.session_id},
            timeout=10,
        )
        response.raise_for_status()
        self._step_index = 0

    def infer_next_action(
        self,
        frames_by_camera: dict[str, list[Image.Image]],
        goal: str,
        observation_time: float,
        subtask_candidates: list[str] = None,  # noqa: RUF013
        current_action: str | None = None,
        ground_truth_label: str | None = None,
    ) -> str:
        """Call remote server to infer next action.

        ``frames_by_camera`` maps camera name -> list of current frames.
        Insertion order defines the camera order shown to the model. For
        single-camera robot eval this is just a one-entry dict; the server
        renders the legacy single-camera prompt sentence in that case, so
        wire behaviour matches pre-multi-cam runs byte-for-byte.
        """
        if not frames_by_camera:
            raise ValueError("frames_by_camera must be a non-empty dict")
        self._ensure_session_created()

        camera_names = list(frames_by_camera.keys())
        current_images_by_camera_b64 = {
            name: [_encode_pil_to_base64(img) for img in frames_by_camera[name]]
            for name in camera_names
        }

        payload = {
            "session_id": self.session_id,
            "current_images_by_camera": current_images_by_camera_b64,
            "camera_names": camera_names,
            "goal": goal,
            "subtask_candidates": subtask_candidates,
            "observation_time": observation_time,
            "current_action": current_action,
            "ground_truth_label": ground_truth_label,
            "next_action_template": self.next_action_template,
            "fallback_response": self.fallback_response,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if not self._session_created:
            payload["memory_config"] = {
                "memory_type": self.memory_type,
                "max_history_images": self.memory_config.max_history_images,
                "max_history_steps": self.memory_config.max_history_steps,
                "image_sample_strategy": self.memory_config.image_sample_strategy,
            }

        response = requests.post(
            f"{self.base_url}/infer_next_action",
            json=payload,
            timeout=self.request_timeout,
        )
        if not response.ok:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:  # noqa: BLE001
                error_detail = response.text
            print(f"Server error ({response.status_code}): {error_detail}")  # noqa: T201
        response.raise_for_status()

        result = response.json()
        parsed_response = result["parsed_response"]

        self.last_raw_response = result.get("raw_response")
        self.last_metadata = result.get("metadata")

        self.total_inferences += 1
        self.last_used_fallback = result.get("used_fallback", False)
        if self.last_used_fallback:
            self.fallback_count += 1

        self._step_index += 1
        return parsed_response

    def get_stats(self) -> dict:
        """Get statistics about inference and fallback usage."""
        fallback_rate = (
            self.fallback_count / self.total_inferences if self.total_inferences > 0 else 0.0
        )
        return {
            "total_inferences": self.total_inferences,
            "fallback_count": self.fallback_count,
            "fallback_rate": fallback_rate,
            "memory_type": self.memory_type,
            "session_id": self.session_id,
        }

    def reset_stats(self):
        """Reset inference statistics."""
        self.fallback_count = 0
        self.total_inferences = 0
