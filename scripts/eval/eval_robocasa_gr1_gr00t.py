# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run RoboCasa GR1 tabletop evaluation against a GR00T policy server.

This entry point reuses DiT4DiT's RoboCasa rollout harness, but talks to
gr00t/eval/run_gr00t_server.py through GR00T's ZeroMQ PolicyClient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tyro

from gr00t.policy.server_client import PolicyClient


ENV_ACTION_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
ENV_STATE_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
    "waist": 3,
}


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 6398
    env_name: str = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    n_episodes: int = 1
    n_envs: int = 1
    max_episode_steps: int = 720
    n_action_steps: int = 12
    video_out_path: str | None = (
        "/home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state/robocasa_eval_videos"
    )
    timeout_ms: int = 120000


class Gr00tRoboCasaPolicy:
    """Adapter from DiT4DiT RoboCasa observations to GR00T sim policy format."""

    def __init__(self, host: str, port: int, timeout_ms: int, n_action_steps: int) -> None:
        self.client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)
        self.n_action_steps = n_action_steps

    def _prepare_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs = dict(obs)

        if "video.ego_view" not in obs:
            for key in (
                "video.ego_view_pad_res256_freq20",
                "video.ego_view_bg_crop_pad_res256_freq20",
            ):
                if key in obs:
                    obs["video.ego_view"] = obs[key]
                    break

        for key, dim in ENV_STATE_DIMS.items():
            flat_key = f"state.{key}"
            if flat_key not in obs:
                raise KeyError(f"RoboCasa observation is missing required key: {flat_key}")
            obs[flat_key] = obs[flat_key].astype(np.float32, copy=False)
            if obs[flat_key].shape[-1] != dim:
                raise ValueError(
                    f"{flat_key} has dim {obs[flat_key].shape[-1]}, expected {dim}"
                )

        return obs

    def step(self, observations: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
        action, _ = self.client.get_action(self._prepare_observation(observations))
        env_actions = {
            f"action.{key}": action[f"action.{key}"][:, : self.n_action_steps].astype(
                np.float32, copy=False
            )
            for key in ENV_ACTION_KEYS
        }
        return {"actions": env_actions}

    def get_modality_config(self) -> dict[str, Any]:
        return self.client.get_modality_config()


def main(args: Args) -> None:
    from examples.Robocasa_tabletop.eval_files.simulation_env import run_evaluation

    policy = Gr00tRoboCasaPolicy(
        host=args.host,
        port=args.port,
        timeout_ms=args.timeout_ms,
        n_action_steps=args.n_action_steps,
    )
    run_evaluation(
        env_name=args.env_name,
        model=policy,
        video_dir=args.video_out_path,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )


if __name__ == "__main__":
    tyro.cli(main)
