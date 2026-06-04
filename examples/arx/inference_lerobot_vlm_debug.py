import argparse
from copy import deepcopy
import json
from pathlib import Path
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.server_client import PolicyClient


def _leaf_key(path_like_key: str) -> str:
    return path_like_key.split(".")[-1]


def _find_mapping_entry(mapping: dict, target_key: str) -> dict | None:
    if target_key in mapping:
        return mapping[target_key]
    if target_key.startswith("annotation."):
        stripped = target_key[len("annotation.") :]
        if stripped in mapping:
            return mapping[stripped]
    return None


def build_source_modality_configs(
    target_modality_configs: dict[str, ModalityConfig], modality_mapping: dict
) -> tuple[dict[str, ModalityConfig], dict[str, str], dict[str, str], dict[str, str]]:
    source_modality_configs = deepcopy(target_modality_configs)
    source_modality_configs.pop("action", None)

    state_map: dict[str, str] = {}
    video_map: dict[str, str] = {}
    language_map: dict[str, str] = {}

    for target_key in target_modality_configs["state"].modality_keys:
        entry = _find_mapping_entry(modality_mapping.get("state", {}), target_key)
        source_key = (
            _leaf_key(entry["original_key"]) if entry is not None and "original_key" in entry else target_key
        )
        state_map[target_key] = source_key

    for target_key in target_modality_configs["video"].modality_keys:
        entry = _find_mapping_entry(modality_mapping.get("video", {}), target_key)
        source_key = (
            _leaf_key(entry["original_key"]) if entry is not None and "original_key" in entry else target_key
        )
        video_map[target_key] = source_key

    for target_key in target_modality_configs["language"].modality_keys:
        annotation_mapping = modality_mapping.get("annotation", {})
        entry = _find_mapping_entry(annotation_mapping, target_key)
        if entry is not None and "original_key" in entry and "." in entry["original_key"]:
            source_key = _leaf_key(entry["original_key"])
        else:
            # Mappings like "task_index" refer to raw metadata before LeRobot conversion.
            # For dataset loading we keep the policy language key.
            source_key = target_key
        language_map[target_key] = source_key

    source_modality_configs["state"].modality_keys = [
        state_map[target_key] for target_key in target_modality_configs["state"].modality_keys
    ]
    source_modality_configs["video"].modality_keys = [
        video_map[target_key] for target_key in target_modality_configs["video"].modality_keys
    ]
    source_modality_configs["language"].modality_keys = [
        language_map[target_key] for target_key in target_modality_configs["language"].modality_keys
    ]
    return source_modality_configs, state_map, video_map, language_map


def parse_observation_for_policy(
    obs: dict[str, object], modality_configs: dict[str, object]
) -> dict[str, dict[str, object]]:
    parsed = {}
    for modality in ["video", "state", "language"]:
        parsed[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            value = obs[parsed_key]
            if isinstance(value, str):
                parsed[modality][key] = [[value]]
            else:
                parsed[modality][key] = value[None, :]
    return parsed


def build_policy_observation(
    data_point,
    target_modality_configs: dict[str, ModalityConfig],
    state_map: dict[str, str],
    video_map: dict[str, str],
    language_map: dict[str, str],
) -> tuple[dict[str, dict[str, object]], str]:
    obs = {}
    # data_point.text = "What is on the desk?"
    for target_key in target_modality_configs["state"].modality_keys:
        source_key = state_map[target_key]
        if source_key not in data_point.states:
            raise KeyError(f"Missing mapped state key '{source_key}' for target '{target_key}'")
        obs[f"state.{target_key}"] = data_point.states[source_key]
    for target_key in target_modality_configs["video"].modality_keys:
        source_key = video_map[target_key]
        if source_key not in data_point.images:
            raise KeyError(f"Missing mapped video key '{source_key}' for target '{target_key}'")
        obs[f"video.{target_key}"] = np.array(data_point.images[source_key])
    for language_key in target_modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text
    return parse_observation_for_policy(obs, target_modality_configs), data_point.text


def make_mosaic(images: dict[str, list[np.ndarray]], ordered_video_keys: list[str]) -> np.ndarray:
    frames = [np.asarray(images[key][0], dtype=np.uint8) for key in ordered_video_keys]
    max_h = max(frame.shape[0] for frame in frames)
    padded_frames = []
    for frame in frames:
        pad_h = max_h - frame.shape[0]
        if pad_h > 0:
            frame = np.pad(frame, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)
        padded_frames.append(frame)
    return np.concatenate(padded_frames, axis=1)


def render_debug_figure(
    mosaic: np.ndarray,
    video_keys: list[str],
    input_text: str,
    generated_text: str,
    save_path: Path,
    show_window: bool,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 10),
        gridspec_kw={"height_ratios": [7, 1.5, 2.0]},
    )
    image_ax, input_ax, output_ax = axes

    image_ax.imshow(mosaic)
    image_ax.axis("off")
    image_ax.set_title(" | ".join(video_keys))

    input_ax.axis("off")
    output_ax.axis("off")
    input_block = textwrap.fill(input_text, width=160)
    output_block = textwrap.fill(generated_text if generated_text else "<empty>", width=160)
    input_ax.text(0.0, 1.0, f"Input language:\n{input_block}", fontsize=11, va="top", ha="left")
    output_ax.text(0.0, 1.0, f"VLM output:\n{output_block}", fontsize=11, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    if show_window:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Load one LeRobot episode/frame, call GR00T server, and visualize VLM debug output."
    )
    parser.add_argument("--dataset-path", required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--episode-idx", type=int, required=True, help="Episode index")
    parser.add_argument("--frame-idx", type=int, required=True, help="Frame index within episode")
    parser.add_argument(
        "--embodiment-tag",
        default="oxe_droid_relative_eef_relative_joint",
        help="Embodiment tag",
    )
    parser.add_argument("--host", default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Client timeout in milliseconds")
    parser.add_argument("--video-backend", default="torchcodec", help="LeRobot video backend")
    parser.add_argument(
        "--modality-mapping-path",
        default="examples/arx/modality.json",
        help="Dataset-to-policy key mapping json (e.g. ARX modality.json)",
    )
    parser.add_argument(
        "--save-path",
        default="outputs/arx_vlm_debug.png",
        help="Path to save the stitched visualization image",
    )
    parser.add_argument("--show-window", action="store_true", help="Show matplotlib window")
    args = parser.parse_args()

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = PolicyClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.timeout_ms,
        strict=False,
    )
    if not policy.ping():
        raise RuntimeError(f"Cannot connect to policy server at {args.host}:{args.port}")

    modality_configs = policy.get_modality_config()
    with open(args.modality_mapping_path, "r") as f:
        modality_mapping = json.load(f)
    source_modality_configs, state_map, video_map, language_map = build_source_modality_configs(
        modality_configs, modality_mapping
    )

    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=source_modality_configs,
        video_backend=args.video_backend,
    )
    trajectory = loader[args.episode_idx]
    if args.frame_idx < 0 or args.frame_idx >= len(trajectory):
        raise IndexError(
            f"frame_idx={args.frame_idx} out of range for episode length={len(trajectory)}"
        )

    data_point = extract_step_data(
        trajectory,
        args.frame_idx,
        source_modality_configs,
        embodiment_tag,
        allow_padding=True,
    )
    observation, input_text = build_policy_observation(
        data_point,
        modality_configs,
        state_map=state_map,
        video_map=video_map,
        language_map=language_map,
    )

    options = {
        "debug_episode_idx": args.episode_idx,
        "debug_frame_idx": args.frame_idx,
    }
    ts = time.time()
    action, info = policy.get_action(observation, options=options)
    te = time.time()

    print(f"inference_time_sec: {te - ts:.3f}")
    print(f"action_keys: {list(action.keys())}")
    for action_key, action_val in action.items():
        print(f"{action_key}: {action_val.shape}")

    vlm_debug = info.get("vlm_debug", {})
    generated_text = vlm_debug.get("generated_text", "")
    print(f"vlm_debug: {vlm_debug}")

    video_keys = modality_configs["video"].modality_keys
    mapped_images = {
        target_key: data_point.images[source_key]
        for target_key, source_key in video_map.items()
        if source_key in data_point.images
    }
    mosaic = make_mosaic(mapped_images, video_keys)
    save_path = Path(args.save_path)
    render_debug_figure(
        mosaic=mosaic,
        video_keys=video_keys,
        input_text=input_text,
        generated_text=generated_text,
        save_path=save_path,
        show_window=args.show_window,
    )
    print(f"saved_visualization: {save_path}")


if __name__ == "__main__":
    main()
