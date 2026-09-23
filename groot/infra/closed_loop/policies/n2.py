# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""N2 policy adapter for the shared closed-loop rollout contract."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import os
from typing import Any

import numpy as np


def _resolve_automatic_trt(
    *,
    checkpoint: str,
    embodiment: str,
    denoising_steps: int | None,
    cache_root: str | None,
    rebuild: bool,
    rebuild_token: str | None,
    device: str,
    codec: str,
) -> tuple[str, str]:
    """Resolve one explicit device and its compatible automatic bundle."""
    from groot.deployment.n2_trt.resolver import resolve_n2_trt_bundle, resolve_n2_trt_device

    resolved_device = resolve_n2_trt_device(device)
    bundle = resolve_n2_trt_bundle(
        checkpoint=checkpoint,
        embodiment=embodiment,
        denoising_steps=denoising_steps,
        cache_root=cache_root,
        rebuild=rebuild,
        device=resolved_device,
        codec=codec,
        rebuild_token=rebuild_token,
    )
    return resolved_device, str(bundle)


class TemporalObservationHistory:
    """Stack raw observations at the temporal offsets requested by N2."""

    def __init__(self, observation: Mapping[str, Any], policy: Any) -> None:
        modality_configs = policy.get_modality_config()
        self._delta_indices = {
            modality: tuple(int(index) for index in modality_configs[modality].delta_indices)
            for modality in ("video", "state")
        }
        positive = {
            index for indices in self._delta_indices.values() for index in indices if index > 0
        }
        if positive:
            raise ValueError(
                f"closed-loop observation history cannot provide positive delta indices: {sorted(positive)}"
            )
        lookback = max(
            (-min(indices, default=0) for indices in self._delta_indices.values()), default=0
        )
        self._observations: deque[dict[str, Any]] = deque(maxlen=lookback + 1)
        self._observations.extend(dict(observation) for _ in range(lookback + 1))

    def append(self, observation: Mapping[str, Any]) -> None:
        self._observations.append(dict(observation))

    def policy_observation(self) -> dict[str, Any]:
        """Return the latest observation with required temporal history stacked."""
        latest = dict(self._observations[-1])
        for modality, indices in self._delta_indices.items():
            for key in tuple(latest):
                if key.startswith(f"{modality}."):
                    latest[key] = np.stack(
                        [self._observations[index - 1][key] for index in indices], axis=0
                    )  # [T,...]
        return latest


class N2RolloutPolicy:
    """Load an N2 checkpoint and expose the shared vector-policy lifecycle."""

    def __init__(
        self,
        *,
        checkpoint: str,
        cache_root: str | None,
        options: Mapping[str, Any],
    ) -> None:
        unknown = sorted(
            set(options)
            - {
                "embodiment_tag",
                "denoising_steps",
                "device",
                "action_steps",
                "trt_engine_dir",
                "enable_trt",
                "trt_cache_dir",
                "rebuild_trt",
                "trt_rebuild_token",
                "use_cuda_graph",
                "codec_roundtrip",
            }
        )
        if unknown:
            raise ValueError(f"unsupported N2 policy options: {', '.join(unknown)}")
        embodiment_tag = str(options.get("embodiment_tag", "libero_sim"))
        denoising_steps_value = options.get("denoising_steps")
        denoising_steps = None if denoising_steps_value is None else int(denoising_steps_value)
        if denoising_steps is not None and denoising_steps <= 0:
            raise ValueError("N2 denoising_steps must be positive")
        self._action_steps = int(options.get("action_steps", 8))
        if self._action_steps <= 0:
            raise ValueError("N2 action_steps must be positive")
        trt_engine_dir_value = options.get("trt_engine_dir")
        trt_engine_dir = None if trt_engine_dir_value in (None, "") else str(trt_engine_dir_value)
        enable_trt = options.get("enable_trt", False)
        rebuild_trt = options.get("rebuild_trt", False)
        if not isinstance(enable_trt, bool):
            raise TypeError("N2 enable_trt must be a boolean")
        if not isinstance(rebuild_trt, bool):
            raise TypeError("N2 rebuild_trt must be a boolean")
        if trt_engine_dir is not None and enable_trt:
            raise ValueError("N2 trt_engine_dir and enable_trt are mutually exclusive")
        trt_cache_dir_value = options.get("trt_cache_dir")
        trt_cache_dir = None if trt_cache_dir_value in (None, "") else str(trt_cache_dir_value)
        if not enable_trt and rebuild_trt:
            raise ValueError("N2 rebuild_trt requires enable_trt")
        if not enable_trt and trt_cache_dir is not None:
            raise ValueError("N2 trt_cache_dir requires enable_trt")
        trt_rebuild_token_value = options.get("trt_rebuild_token")
        trt_rebuild_token = (
            None if trt_rebuild_token_value in (None, "") else str(trt_rebuild_token_value)
        )
        use_cuda_graph = options.get("use_cuda_graph", False)
        if not isinstance(use_cuda_graph, bool):
            raise TypeError("N2 use_cuda_graph must be a boolean")
        codec_roundtrip_value = options.get("codec_roundtrip")
        codec_roundtrip = (
            None if codec_roundtrip_value is None else str(codec_roundtrip_value).lower()
        )
        if codec_roundtrip is not None:
            from groot.core.policy.n2_codec import N2_CODEC_ROUNDTRIPS

            if codec_roundtrip not in N2_CODEC_ROUNDTRIPS:
                raise ValueError(
                    f"N2 codec_roundtrip must be one of {N2_CODEC_ROUNDTRIPS}, got {codec_roundtrip!r}"
                )
        self._policy = self._load_policy(
            checkpoint=checkpoint,
            cache_root=cache_root,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            device=str(options.get("device", "cuda:0")),
            trt_engine_dir=trt_engine_dir,
            enable_trt=enable_trt,
            trt_cache_dir=trt_cache_dir,
            rebuild_trt=rebuild_trt,
            trt_rebuild_token=trt_rebuild_token,
            use_cuda_graph=use_cuda_graph,
            codec_roundtrip=codec_roundtrip,
        )
        self._history: TemporalObservationHistory | None = None
        self._action_chunk: dict[str, np.ndarray] = {}
        self._action_index = 0

    @staticmethod
    def _load_policy(
        *,
        checkpoint: str,
        cache_root: str | None,
        embodiment_tag: str,
        denoising_steps: int | None,
        device: str,
        trt_engine_dir: str | None,
        enable_trt: bool,
        trt_cache_dir: str | None,
        rebuild_trt: bool,
        trt_rebuild_token: str | None,
        use_cuda_graph: bool,
        codec_roundtrip: str | None,
    ) -> Any:
        from groot.core.data import schema
        from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
        from groot.core.policy.n2_codec import N2_CODEC_ROUNDTRIPS, resolve_n2_codec_roundtrip
        from groot.core.policy.n2_policy import Gr00tN2Policy
        from groot.core.training.models.n2.utils.cosmos_model_deps import ensure_cosmos_model_deps
        from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig
        from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator

        model_path = OssOpenLoopEvaluator(
            OssOpenLoopEvalConfig(checkpoint=checkpoint, cache_root=cache_root)
        ).resolve_model_path()
        if codec_roundtrip is None:
            codec = resolve_n2_codec_roundtrip()
        elif codec_roundtrip not in N2_CODEC_ROUNDTRIPS:
            raise ValueError(
                f"N2 codec_roundtrip must be one of {N2_CODEC_ROUNDTRIPS}, got {codec_roundtrip!r}"
            )
        else:
            # An explicit policy option is serialized to Ray actors and must
            # win over any stale worker environment inherited by the cluster.
            codec = codec_roundtrip
        # Gr00tN2Policy keeps environment override compatibility for legacy
        # callers. A serialized closed-loop option is stronger: hide any stale
        # worker override while resolving a cold automatic TensorRT bundle and
        # constructing this explicitly configured policy. The builder copies
        # this environment into its policy-measurement subprocess.
        env_name = "GROOT_N2_CODEC_ROUNDTRIP"
        missing = object()
        previous: object = missing
        if codec_roundtrip is not None:
            previous = os.environ.pop(env_name, missing)
        try:
            if enable_trt:
                device, trt_engine_dir = _resolve_automatic_trt(
                    checkpoint=str(model_path),
                    embodiment=embodiment_tag,
                    denoising_steps=denoising_steps,
                    cache_root=trt_cache_dir,
                    rebuild=rebuild_trt,
                    rebuild_token=trt_rebuild_token,
                    device=device,
                    codec=codec,
                )
            ensure_cosmos_model_deps(str(model_path))
            try:
                tag = schema.EmbodimentTag[embodiment_tag]
            except KeyError:
                tag = schema.EmbodimentTag(embodiment_tag)
            policy_options: dict[str, Any] = {
                "trt_engine_dir": trt_engine_dir,
                "use_cuda_graph": use_cuda_graph,
            }
            if denoising_steps is not None:
                policy_options["num_inference_timesteps"] = denoising_steps
            policy = Gr00tN2Policy(
                embodiment_tag=tag,
                model_path=str(model_path),
                device=device,
                strict=True,
                codec_roundtrip=codec,
                **policy_options,
            )
        finally:
            if codec_roundtrip is not None and previous is not missing:
                os.environ[env_name] = str(previous)
        return Gr00tFlatPolicyWrapper(policy, strict=True)

    def on_env_init(self, env: Any) -> None:
        """Accept the shared lifecycle hook; N2 has no environment-owned initialization."""
        del env

    def on_env_reset(self, env_indices: np.ndarray) -> None:  # env_indices: [N]
        """Clear temporal policy state whenever an environment slot resets."""
        if env_indices.size:
            self._policy.reset()
            self._history = None
            self._action_chunk = {}
            self._action_index = 0

    def action(self, observations: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Infer action chunks and emit one batched simulator step at a time."""
        observation = {
            key: _unbatch(value) for key, value in observations.items()
        }  # each value: unbatched [...]
        if self._history is None:
            self._history = TemporalObservationHistory(observation, self._policy)
        else:
            self._history.append(observation)
        if not self._action_chunk or self._action_index >= self._chunk_length():
            action, _ = self._policy.get_action(self._history.policy_observation())
            self._action_chunk = {key: np.asarray(value) for key, value in action.items()}
            self._action_index = 0
        selected = {
            key: np.expand_dims(value[self._action_index], axis=0)
            for key, value in self._action_chunk.items()
        }  # each action: [B=1,A]
        self._action_index += 1
        return selected

    def _chunk_length(self) -> int:
        if not self._action_chunk:
            return 0
        horizon = min(value.shape[0] for value in self._action_chunk.values())
        return min(self._action_steps, horizon)


def _unbatch(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape[0] != 1:
            raise ValueError("portable N2 evaluation currently requires one environment per worker")
        return value[0]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value
