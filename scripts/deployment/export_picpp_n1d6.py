#!/usr/bin/env python3
"""Export GR00T N1.6 action-step assets for pi.cpp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch


SEED = 1234


def _rec_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=dtype)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _rec_to_dtype(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype) for item in value]
    return value


def _parse_observation(obs: dict[str, Any], modality_configs: dict[str, Any]) -> dict[str, Any]:
    parsed = {}
    for modality in ("video", "state", "language"):
        parsed[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            value = obs[parsed_key]
            parsed[modality][key] = [[value]] if isinstance(value, str) else value[None, :]
    return parsed


def _prepare_observation(policy: Gr00tPolicy, dataset: LeRobotEpisodeLoader) -> dict[str, Any]:
    traj = dataset[0]
    data_point = extract_step_data(
        traj,
        0,
        modality_configs=policy.get_modality_config(),
        embodiment_tag=policy.embodiment_tag,
    )

    observation = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.array(value)
    for key in policy.get_modality_config()["language"].modality_keys:
        observation[key] = data_point.text
    return _parse_observation(observation, policy.get_modality_config())


def _prepare_model_inputs(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    processed_inputs = []
    batch_size = observation["video"][list(observation["video"].keys())[0]].shape[0]
    for index in range(batch_size):
        obs = {
            "video": {key: value[index] for key, value in observation["video"].items()},
            "state": {key: value[index] for key, value in observation["state"].items()},
            "language": {key: value[index] for key, value in observation["language"].items()},
        }
        vla_step_data = VLAStepData(
            images=obs["video"],
            states=obs["state"],
            actions={},
            text=obs["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )
        processed_inputs.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]))

    collated = policy.collate_fn(processed_inputs)["inputs"]
    return _rec_to_dtype(collated, dtype=torch.bfloat16)


class GrootN1d6ActionStep(torch.nn.Module):
    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()
        self.action_head = policy.model.action_head

    def forward(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        action_features = self.action_head.action_encoder(actions, timestep, embodiment_id)
        if self.action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
            action_features = action_features + self.action_head.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = self.action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=backbone_features,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )
        decoded = self.action_head.action_decoder(model_output, embodiment_id)
        return decoded[:, -self.action_head.action_horizon :]


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GR00T N1.6 action-step assets for pi.cpp")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="./groot_n1d6_picpp")
    parser.add_argument("--embodiment_tag", type=EmbodimentTag, default=EmbodimentTag.GR1)
    parser.add_argument("--video_backend", default="torchcodec")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    output_dir = Path(args.output_dir)
    onnx_dir = output_dir / "onnx"
    io_dir = output_dir / "io"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )
    policy.model.eval()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )
    collated_inputs = _prepare_model_inputs(policy, _prepare_observation(policy, dataset))
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
        backbone_outputs = policy.model.backbone(backbone_inputs)
        features = policy.model.action_head._encode_features(backbone_outputs, action_inputs)

    batch_size = features.backbone_features.shape[0]
    dtype = features.backbone_features.dtype
    device = features.backbone_features.device
    actions = torch.randn(
        (batch_size, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
        dtype=dtype,
        device=device,
    )
    timestep = torch.zeros((batch_size,), dtype=torch.long, device=device)
    image_mask = backbone_outputs.image_mask
    backbone_attention_mask = backbone_outputs.backbone_attention_mask

    wrapper = GrootN1d6ActionStep(policy).eval()
    inputs = (
        features.backbone_features,
        features.state_features,
        action_inputs.embodiment_id,
        backbone_attention_mask,
        image_mask,
        actions,
        timestep,
    )
    input_names = [
        "backbone_features",
        "state_features",
        "embodiment_id",
        "backbone_attention_mask",
        "image_mask",
        "actions",
        "timestep",
    ]
    output_names = ["velocity"]
    onnx_path = onnx_dir / "action_step.onnx"
    with torch.inference_mode():
        velocity = wrapper(*inputs)
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=19,
            do_constant_folding=True,
            dynamo=False,
        )

    np.savez(
        io_dir / "action_step_inputs.npz",
        backbone_features=_array(features.backbone_features),
        state_features=_array(features.state_features),
        embodiment_id=action_inputs.embodiment_id.detach().cpu().numpy(),
        backbone_attention_mask=backbone_attention_mask.detach().cpu().numpy(),
        image_mask=image_mask.detach().cpu().numpy(),
        actions=_array(actions),
        timestep=timestep.detach().cpu().numpy(),
    )
    np.save(io_dir / "action_step_velocity.npy", _array(velocity))
    manifest = {
        "model": "groot_n1d6",
        "source_model": args.model_path,
        "embodiment_tag": args.embodiment_tag.value,
        "precision": "bf16",
        "seed": SEED,
        "onnx": {"action_step": "onnx/action_step.onnx"},
        "io": {
            "inputs": "io/action_step_inputs.npz",
            "velocity": "io/action_step_velocity.npy",
        },
        "inputs": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in zip(input_names, inputs, strict=True)
        },
        "outputs": {"velocity": {"shape": list(velocity.shape), "dtype": str(velocity.dtype)}},
        "action": {
            "horizon": int(policy.model.action_head.action_horizon),
            "dim": int(policy.model.action_head.action_dim),
            "denoise_steps": int(policy.model.action_head.num_inference_timesteps),
            "timestep_buckets": int(policy.model.action_head.num_timestep_buckets),
        },
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
