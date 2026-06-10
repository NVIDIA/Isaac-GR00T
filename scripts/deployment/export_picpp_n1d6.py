#!/usr/bin/env python3
"""Export GR00T N1.6 LIBERO assets for pi.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


SEED = 1234
EXPORT_DTYPE = torch.float16


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


def _prepare_observation(policy: Gr00tPolicy, dataset: Any) -> dict[str, Any]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

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
    return _rec_to_dtype(collated, dtype=EXPORT_DTYPE)


def _prepare_synthetic_observation(
    policy: Gr00tPolicy,
    task: str,
    embodiment_tag: EmbodimentTag,
) -> dict[str, Any]:
    stats = policy.processor.state_action_processor.statistics[embodiment_tag.value]
    observation: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
    for key in policy.get_modality_config()["video"].modality_keys:
        observation["video"][key] = np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)
    for key in policy.get_modality_config()["state"].modality_keys:
        dim = len(stats["state"][key]["min"])
        observation["state"][key] = np.zeros((1, 1, dim), dtype=np.float32)
    for key in policy.get_modality_config()["language"].modality_keys:
        observation["language"][key] = [[task]]
    return observation


LIBERO_V21_DIRS = (
    "aopolin-lv__libero_spatial_no_noops_lerobot_v21",
    "aopolin-lv__libero_object_no_noops_lerobot_v21",
    "aopolin-lv__libero_goal_no_noops_lerobot_v21",
    "aopolin-lv__libero_10_no_noops_lerobot_v21",
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)


class GrootN1d6Backbone(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values_0: torch.Tensor,
        pixel_values_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            BatchFeature(
                data={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": [pixel_values_0, pixel_values_1],
                }
            )
        )
        return outputs.backbone_features, outputs.backbone_attention_mask, outputs.image_mask


class GrootN1d6ActionLoop(torch.nn.Module):
    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()
        self.action_head = policy.model.action_head

    def forward(
        self,
        backbone_features: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        initial_actions: torch.Tensor,
    ) -> torch.Tensor:
        backbone_features = self.action_head.vlln(backbone_features)
        state_features = self.action_head.state_encoder(state, embodiment_id)
        actions = initial_actions
        dt = 1.0 / self.action_head.num_inference_timesteps
        for index in range(self.action_head.num_inference_timesteps):
            timestep = int(index / float(self.action_head.num_inference_timesteps) * self.action_head.num_timestep_buckets)
            timestep = torch.full((actions.shape[0],), timestep, dtype=torch.long, device=actions.device)
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
            actions = actions + dt * decoded[:, -self.action_head.action_horizon :]
        return actions


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def _normalized_task(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _formalized_task(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _lerobot_tasks(dataset_path: Path) -> list[str]:
    paths = []
    for dirname in LIBERO_V21_DIRS:
        path = dataset_path.parent / dirname
        if path.exists():
            paths.append(path)
    if not paths:
        paths.append(dataset_path)

    tasks = []
    seen = set()
    for path in paths:
        for line in (path / "meta" / "tasks.jsonl").read_text().splitlines():
            task = str(json.loads(line)["task"]).strip()
            key = _normalized_task(task)
            if key not in seen:
                tasks.append(task)
                seen.add(key)
    return tasks


def _tasks_from_file(path: Path) -> list[str]:
    tasks = []
    seen = set()
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("{"):
            task = str(json.loads(text)["task"]).strip()
        else:
            task = text
        key = _normalized_task(task)
        if key not in seen:
            tasks.append(task)
            seen.add(key)
    return tasks


def _inputs_for_task(policy: Gr00tPolicy, observation: dict[str, Any], task: str) -> dict[str, Any]:
    obs = {
        "video": observation["video"],
        "state": observation["state"],
        "language": dict(observation["language"]),
    }
    obs["language"][policy.language_key] = [[task]]
    return _prepare_model_inputs(policy, obs)


def _decode_actions(policy: Gr00tPolicy, actions: torch.Tensor, embodiment_tag: EmbodimentTag) -> np.ndarray:
    decoded = policy.processor.decode_action(_array(actions), embodiment_tag)
    keys = policy.modality_configs["action"].modality_keys
    return np.concatenate([decoded[key] for key in keys], axis=-1).astype(np.float32)[0]


def _pad_token_array(tensor: torch.Tensor, token_length: int) -> np.ndarray:
    values = tensor.detach().cpu().numpy()
    padded = np.zeros((values.shape[0], token_length), dtype=values.dtype)
    padded[:, : values.shape[1]] = values
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GR00T N1.6 LIBERO assets for pi.cpp")
    parser.add_argument("--model_path", required=True)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--dataset_path")
    input_source.add_argument("--tasks_path")
    parser.add_argument("--output_dir", default="./groot_n1d6_picpp")
    parser.add_argument("--embodiment_tag", type=EmbodimentTag, default=EmbodimentTag.LIBERO_PANDA)
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
    policy.model.to(dtype=EXPORT_DTYPE)
    policy.model.eval()
    if args.dataset_path is not None:
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        dataset = LeRobotEpisodeLoader(
            dataset_path=args.dataset_path,
            modality_configs=policy.get_modality_config(),
            video_backend=args.video_backend,
            video_backend_kwargs=None,
        )
        observation = _prepare_observation(policy, dataset)
        tasks = _lerobot_tasks(Path(args.dataset_path))
    else:
        tasks = _tasks_from_file(Path(args.tasks_path))
        observation = _prepare_synthetic_observation(policy, tasks[0], args.embodiment_tag)
    task_backbone_inputs = []
    max_token_length = 0
    for task in tasks:
        task_inputs = _inputs_for_task(policy, observation, task)
        with torch.inference_mode():
            task_inputs_backbone, _ = policy.model.prepare_input(task_inputs)
        token_length = int(task_inputs_backbone["input_ids"].shape[1])
        max_token_length = max(max_token_length, token_length)
        task_backbone_inputs.append((task, task_inputs_backbone))

    export_task, _ = max(task_backbone_inputs, key=lambda item: int(item[1]["input_ids"].shape[1]))
    collated_inputs = _inputs_for_task(policy, observation, export_task)
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
        pixel_values_0, pixel_values_1 = backbone_inputs["pixel_values"]
        backbone_wrapper = GrootN1d6Backbone(policy.model.backbone).eval()
        backbone_export_inputs = (
            backbone_inputs["input_ids"],
            backbone_inputs["attention_mask"],
            pixel_values_0,
            pixel_values_1,
        )
        backbone_outputs = backbone_wrapper(*backbone_export_inputs)

    backbone_input_names = ["input_ids", "attention_mask", "pixel_values_0", "pixel_values_1"]
    backbone_output_names = ["backbone_features", "backbone_attention_mask", "image_mask"]
    with torch.inference_mode():
        torch.onnx.export(
            backbone_wrapper,
            backbone_export_inputs,
            str(onnx_dir / "backbone.onnx"),
            input_names=backbone_input_names,
            output_names=backbone_output_names,
            opset_version=19,
            do_constant_folding=True,
            dynamo=False,
        )

    batch_size = backbone_outputs[0].shape[0]
    dtype = backbone_outputs[0].dtype
    device = backbone_outputs[0].device
    initial_actions = torch.randn(
        (batch_size, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
        dtype=dtype,
        device=device,
    )
    action_wrapper = GrootN1d6ActionLoop(policy).eval()
    action_inputs_tuple = (
        backbone_outputs[0],
        backbone_outputs[1],
        backbone_outputs[2],
        action_inputs.state,
        action_inputs.embodiment_id,
        initial_actions,
    )
    action_input_names = [
        "backbone_features",
        "backbone_attention_mask",
        "image_mask",
        "state",
        "embodiment_id",
        "initial_actions",
    ]
    with torch.inference_mode():
        normalized_actions = action_wrapper(*action_inputs_tuple)
        torch.onnx.export(
            action_wrapper,
            action_inputs_tuple,
            str(onnx_dir / "action_loop.onnx"),
            input_names=action_input_names,
            output_names=["normalized_actions"],
            opset_version=19,
            do_constant_folding=False,
            dynamo=False,
        )

    prompt_cache_dir = output_dir / "prompt_cache" / "libero"
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_cache_index = []
    for task, task_inputs in task_backbone_inputs:
        digest = hashlib.sha256(_normalized_task(task).encode("utf-8")).hexdigest()
        np.savez(
            prompt_cache_dir / f"{digest}.npz",
            input_ids=_pad_token_array(task_inputs["input_ids"], max_token_length),
            attention_mask=_pad_token_array(task_inputs["attention_mask"], max_token_length),
        )
        prompt_cache_index.append(
            {
                "task": task,
                "normalized_task": _normalized_task(task),
                "formalized_task": _formalized_task(task),
                "sha256": digest,
            }
        )

    np.savez(
        io_dir / "backbone_inputs.npz",
        input_ids=backbone_inputs["input_ids"].detach().cpu().numpy(),
        attention_mask=backbone_inputs["attention_mask"].detach().cpu().numpy(),
        pixel_values_0=_array(pixel_values_0),
        pixel_values_1=_array(pixel_values_1),
    )
    np.savez(
        io_dir / "backbone_outputs.npz",
        backbone_features=_array(backbone_outputs[0]),
        backbone_attention_mask=backbone_outputs[1].detach().cpu().numpy(),
        image_mask=backbone_outputs[2].detach().cpu().numpy(),
    )
    np.savez(
        io_dir / "action_loop_inputs.npz",
        backbone_features=_array(backbone_outputs[0]),
        backbone_attention_mask=backbone_outputs[1].detach().cpu().numpy(),
        image_mask=backbone_outputs[2].detach().cpu().numpy(),
        state=_array(action_inputs.state),
        embodiment_id=action_inputs.embodiment_id.detach().cpu().numpy(),
        initial_actions=_array(initial_actions),
    )
    np.save(io_dir / "normalized_actions.npy", _array(normalized_actions))
    np.save(io_dir / "actions.npy", _decode_actions(policy, normalized_actions, args.embodiment_tag))
    (output_dir / "prompt_cache_index.json").write_text(json.dumps(prompt_cache_index, indent=2) + "\n")
    statistics = policy.processor.state_action_processor.statistics[args.embodiment_tag.value]
    (output_dir / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")

    image_size = [int(value) for value in pixel_values_0.shape[-2:]]
    action_steps = len(policy.modality_configs["action"].delta_indices)
    manifest = {
        "model": "groot_n1d6",
        "source_model": args.model_path,
        "embodiment_tag": args.embodiment_tag.value,
        "precision": "fp16",
        "seed": SEED,
        "onnx": {
            "backbone": "onnx/backbone.onnx",
            "action_loop": "onnx/action_loop.onnx",
        },
        "io": {
            "backbone_inputs": "io/backbone_inputs.npz",
            "backbone_outputs": "io/backbone_outputs.npz",
            "action_loop_inputs": "io/action_loop_inputs.npz",
            "normalized_actions": "io/normalized_actions.npy",
            "actions": "io/actions.npy",
        },
        "assets": {
            "prompt_cache": "prompt_cache/libero",
            "prompt_cache_index": "prompt_cache_index.json",
            "dataset_statistics": "dataset_statistics.json",
        },
        "inputs": {
            "num_cameras": 2,
            "image_size": image_size,
            "state_dim": int(sum(len(statistics["state"][key]["min"]) for key in policy.modality_configs["state"].modality_keys)),
            "max_state_dim": int(action_inputs.state.shape[-1]),
            "token_length": int(backbone_inputs["input_ids"].shape[1]),
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "shortest_image_edge": int(policy.processor.shortest_image_edge),
            "crop_fraction": float(policy.processor.crop_fraction),
        },
        "engine_inputs": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in zip(backbone_input_names, backbone_export_inputs, strict=True)
        }
        | {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in zip(action_input_names, action_inputs_tuple, strict=True)
        },
        "outputs": {
            "normalized_actions": {"shape": list(normalized_actions.shape), "dtype": str(normalized_actions.dtype)}
        },
        "action": {
            "horizon": int(action_steps),
            "dim": int(sum(len(statistics["action"][key]["min"]) for key in policy.modality_configs["action"].modality_keys)),
            "latent_horizon": int(policy.model.action_head.action_horizon),
            "latent_dim": int(policy.model.action_head.action_dim),
            "denoise_steps": int(policy.model.action_head.num_inference_timesteps),
            "timestep_buckets": int(policy.model.action_head.num_timestep_buckets),
        },
        "features": {
            "images": list(policy.modality_configs["video"].modality_keys),
            "states": list(policy.modality_configs["state"].modality_keys),
            "actions": list(policy.modality_configs["action"].modality_keys),
        },
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
