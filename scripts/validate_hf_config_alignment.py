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

"""Validate HuggingFace config alignment against source-of-truth definitions.

Usage:
    # Internal consistency checks only (no HF download required):
    uv run python scripts/validate_hf_config_alignment.py

    # Full check with HF configs (requires auth + local dirs):
    uv run python scripts/validate_hf_config_alignment.py --hf-config-dir /tmp/hf_configs
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
INFO = "\033[94mℹ INFO\033[0m"
SKIP = "\033[90m⊘ SKIP\033[0m"

pass_count = 0
fail_count = 0
warn_count = 0
skip_count = 0


def check(condition, msg, *, warn_only=False, skip=False):
    global pass_count, fail_count, warn_count, skip_count
    if skip:
        skip_count += 1
        print(f"  {SKIP} {msg}")
        return True
    if condition:
        pass_count += 1
        print(f"  {PASS} {msg}")
        return True
    if warn_only:
        warn_count += 1
        print(f"  {WARN} {msg}")
        return True
    fail_count += 1
    print(f"  {FAIL} {msg}")
    return False


def info(msg):
    print(f"  {INFO} {msg}")


# ──────────────────────── Source-of-Truth Loaders ────────────────────────


def load_modality_configs():
    """Load MODALITY_CONFIGS from embodiment_configs.py as serializable dicts."""
    sys.path.insert(0, str(REPO_ROOT))
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.utils import to_json_serializable

    raw = to_json_serializable(MODALITY_CONFIGS)
    return raw


def load_model_config_defaults():
    """Load Gr00tN1d7Config defaults."""
    sys.path.insert(0, str(REPO_ROOT))
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    cfg = Gr00tN1d7Config()
    return cfg


def load_embodiment_tags():
    sys.path.insert(0, str(REPO_ROOT))
    from gr00t.data.embodiment_tags import POSTTRAIN_TAGS, PRETRAIN_TAGS, EmbodimentTag

    return EmbodimentTag, PRETRAIN_TAGS, POSTTRAIN_TAGS


def load_projector_index():
    sys.path.insert(0, str(REPO_ROOT))
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX

    return EMBODIMENT_TAG_TO_PROJECTOR_INDEX


# ──────────────────────── HF Model Definitions ────────────────────────

HF_MODELS = {
    "GR00T-N1.7-3B": {
        "hf_id": "nvidia/GR00T-N1.7-3B",
        "type": "base",
        "embodiment_tags": [
            "oxe_droid_relative_eef_relative_joint",
            "xdof_relative_eef_relative_joint",
            "xdof_relative_eef_relative_joint_subtask",
            "real_g1_relative_eef_relative_joints",
            "real_r1_pro_sharpa_relative_eef",
            "real_r1_pro_sharpa_relative_eef_human",
            "real_r1_pro_sharpa_relative_eef_maxinsights",
            "real_r1_pro_sharpa_relative_eef_mecka",
        ],
        "subdir": None,
    },
    "GR00T-N1.7-DROID": {
        "hf_id": "nvidia/GR00T-N1.7-DROID",
        "type": "finetuned",
        "embodiment_tags": ["oxe_droid_relative_eef_relative_joint"],
        "subdir": None,
    },
    "GR00T-N1.7-LIBERO": {
        "hf_id": "nvidia/GR00T-N1.7-LIBERO",
        "type": "finetuned",
        "embodiment_tags": ["libero_sim"],
        "subdir": "libero_10",
    },
    "SimplerEnv-Fractal": {
        "hf_id": "nvidia/GR00T-N1.7-SimplerEnv-Fractal",
        "type": "finetuned",
        "embodiment_tags": ["simpler_env_google"],
        "subdir": None,
    },
    "SimplerEnv-Bridge": {
        "hf_id": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "type": "finetuned",
        "embodiment_tags": ["simpler_env_widowx"],
        "subdir": None,
    },
}


# ──────────────────────── Dimension F & Internal Consistency ────────────────────────


def check_dim_f_internal_consistency():
    """Dimension F — Cross-file consistency (source-of-truth only)."""
    print("\n" + "=" * 70)
    print("DIMENSION F — Internal Source-of-Truth Consistency")
    print("=" * 70)

    modality_configs = load_modality_configs()
    model_cfg = load_model_config_defaults()
    EmbodimentTag, PRETRAIN_TAGS, POSTTRAIN_TAGS = load_embodiment_tags()
    projector_index = load_projector_index()

    # F3: action horizon ≤ model max
    print("\n[F3] Action horizon ≤ model max capacity")
    for tag, cfg in modality_configs.items():
        actual_horizon = len(cfg["action"]["delta_indices"])
        check(
            actual_horizon <= model_cfg.action_horizon,
            f"  {tag}: actual={actual_horizon} ≤ max={model_cfg.action_horizon}",
        )

    # F5: EMBODIMENT_TAG_TO_PROJECTOR_INDEX ↔ EmbodimentTag
    print("\n[F5] EMBODIMENT_TAG_TO_PROJECTOR_INDEX ↔ EmbodimentTag enum")
    for member in EmbodimentTag:
        if member.value in modality_configs:
            check(
                member.value in projector_index,
                f"  {member.value} in MODALITY_CONFIGS → has projector index: {projector_index.get(member.value, 'MISSING')}",
            )

    all_tag_values = {m.value for m in EmbodimentTag}
    for tag in projector_index:
        check(
            tag in all_tag_values,
            f"  projector index key '{tag}' → is valid EmbodimentTag value",
        )

    # F6: naming mismatch awareness
    print("\n[F6] Known naming mismatches (informational)")
    info(f"Model config: action_horizon={model_cfg.action_horizon}")
    info("Processor uses: max_action_horizon (same value, different key name)")
    info(f"Model config: use_albumentations_transforms={model_cfg.use_albumentations_transforms}")
    info("Processor uses: use_albumentations (same semantics, different key name)")


def check_dim_e_documentation():
    """Dimension E — README & Documentation Consistency."""
    print("\n" + "=" * 70)
    print("DIMENSION E — README & Documentation Consistency")
    print("=" * 70)

    EmbodimentTag, PRETRAIN_TAGS, POSTTRAIN_TAGS = load_embodiment_tags()
    modality_configs = load_modality_configs()

    # E1: Checkpoint table in README.md
    print("\n[E1] Checkpoint table in README.md")
    readme = (REPO_ROOT / "README.md").read_text()
    for model_name, model_info in HF_MODELS.items():
        check(
            model_info["hf_id"] in readme,
            f"  {model_info['hf_id']} found in README.md",
        )

    # E2: --embodiment-tag in example commands uses enum NAMES
    print("\n[E2] --embodiment-tag uses enum NAMES in example commands")
    example_readmes = {
        "DROID": REPO_ROOT / "examples/DROID/README.md",
        "LIBERO": REPO_ROOT / "examples/LIBERO/README.md",
        "SimplerEnv": REPO_ROOT / "examples/SimplerEnv/README.md",
    }
    tag_name_to_value = {m.name: m.value for m in EmbodimentTag}

    for name, path in example_readmes.items():
        if not path.exists():
            check(False, f"  {path} exists", skip=True)
            continue
        content = path.read_text()
        tags_in_commands = re.findall(r"--embodiment-tag\s+(\S+)", content)
        for tag in tags_in_commands:
            is_enum_name = tag in tag_name_to_value
            is_enum_value = tag in {m.value for m in EmbodimentTag}
            check(
                is_enum_name,
                f"  {name}: --embodiment-tag {tag} is valid enum NAME"
                + (" (used value instead of name)" if is_enum_value and not is_enum_name else ""),
            )

    # E4: DROID modality table
    print("\n[E4] DROID modality table matches MODALITY_CONFIGS")
    droid_readme = (REPO_ROOT / "examples/DROID/README.md").read_text()
    droid_cfg = modality_configs.get("oxe_droid_relative_eef_relative_joint", {})
    if droid_cfg:
        for vkey in droid_cfg["video"]["modality_keys"]:
            check(vkey in droid_readme, f"  Video key '{vkey}' mentioned in DROID README")
        for skey in droid_cfg["state"]["modality_keys"]:
            check(skey in droid_readme, f"  State key '{skey}' mentioned in DROID README")
        check(
            "17D" in droid_readme or "17d" in droid_readme.lower(),
            "  17D dimension mentioned in DROID README",
            warn_only=True,
        )

    # E5: Example modality.json files match MODALITY_CONFIGS
    print("\n[E5] Example modality.json ↔ MODALITY_CONFIGS key consistency")
    modality_json_map = {
        "simpler_env_google": REPO_ROOT / "examples/SimplerEnv/fractal_modality.json",
        "simpler_env_widowx": REPO_ROOT / "examples/SimplerEnv/bridge_modality.json",
        "libero_sim": REPO_ROOT / "examples/LIBERO/modality.json",
    }
    for tag, json_path in modality_json_map.items():
        if not json_path.exists():
            check(False, f"  {json_path} exists", skip=True)
            continue
        with open(json_path) as f:
            mj = json.load(f)
        code_cfg = modality_configs.get(tag, {})
        if not code_cfg:
            check(False, f"  {tag} in MODALITY_CONFIGS")
            continue

        mj_state_keys = list(mj.get("state", {}).keys())
        code_state_keys = code_cfg["state"]["modality_keys"]
        check(
            mj_state_keys == code_state_keys,
            f"  {tag} state keys: modality.json={mj_state_keys} vs code={code_state_keys}",
        )

        mj_action_keys = list(mj.get("action", {}).keys())
        code_action_keys = code_cfg["action"]["modality_keys"]
        check(
            mj_action_keys == code_action_keys,
            f"  {tag} action keys: modality.json={mj_action_keys} vs code={code_action_keys}",
        )

        mj_video_keys = list(mj.get("video", {}).keys())
        code_video_keys = code_cfg["video"]["modality_keys"]
        check(
            mj_video_keys == code_video_keys,
            f"  {tag} video keys: modality.json={mj_video_keys} vs code={code_video_keys}",
        )

    # E7: --action-horizon in commands
    print("\n[E7] --action-horizon in commands ≤ embodiment actual horizon")
    for name, path in example_readmes.items():
        if not path.exists():
            continue
        content = path.read_text()
        horizons = re.findall(r"--action-horizon\s+(\d+)", content)
        for h in horizons:
            info(f"  {name}: --action-horizon {h} found in commands")


def check_dim_f2_modality_json():
    """Dimension F2 — MODALITY_CONFIGS ↔ examples/*/modality.json."""
    print("\n" + "=" * 70)
    print("DIMENSION F2 — MODALITY_CONFIGS ↔ modality.json Structural Check")
    print("=" * 70)

    modality_configs = load_modality_configs()
    modality_json_files = {
        "simpler_env_google": REPO_ROOT / "examples/SimplerEnv/fractal_modality.json",
        "simpler_env_widowx": REPO_ROOT / "examples/SimplerEnv/bridge_modality.json",
        "libero_sim": REPO_ROOT / "examples/LIBERO/modality.json",
    }

    for tag, json_path in modality_json_files.items():
        print(f"\n  [{tag}]")
        if not json_path.exists():
            check(False, f"    {json_path.name} exists", skip=True)
            continue
        with open(json_path) as f:
            mj = json.load(f)

        code_cfg = modality_configs[tag]
        code_state_count = len(code_cfg["state"]["modality_keys"])
        mj_state_count = len(mj.get("state", {}))
        check(
            code_state_count == mj_state_count,
            f"    State key count: code={code_state_count} vs modality.json={mj_state_count}",
        )

        code_action_count = len(code_cfg["action"]["modality_keys"])
        mj_action_count = len(mj.get("action", {}))
        check(
            code_action_count == mj_action_count,
            f"    Action key count: code={code_action_count} vs modality.json={mj_action_count}",
        )


# ──────────────────────── Dimension J — Enum Serialization ────────────────────────


def check_dim_j_enum_serialization():
    """Dimension J — Verify enum serialization uses names not values."""
    print("\n" + "=" * 70)
    print("DIMENSION J — Enum Serialization Format (code-level)")
    print("=" * 70)

    modality_configs = load_modality_configs()
    valid_rep_names = {"RELATIVE", "ABSOLUTE"}
    valid_type_names = {"EEF", "NON_EEF"}
    valid_format_names = {"DEFAULT", "XYZ_ROT6D", "ROTATION_6D", "SCALAR"}

    for tag, cfg in modality_configs.items():
        action_configs = cfg.get("action", {}).get("action_configs")
        if not action_configs:
            continue
        print(f"\n  [{tag}]")
        for i, ac in enumerate(action_configs):
            rep = ac.get("rep")
            atype = ac.get("type")
            afmt = ac.get("format")
            check(
                rep in valid_rep_names,
                f"    action_configs[{i}].rep = '{rep}' (valid name: {rep in valid_rep_names})",
            )
            check(
                atype in valid_type_names,
                f"    action_configs[{i}].type = '{atype}' (valid name: {atype in valid_type_names})",
            )
            if afmt:
                check(
                    afmt in valid_format_names,
                    f"    action_configs[{i}].format = '{afmt}' (valid name: {afmt in valid_format_names})",
                )


# ──────────────────────── HF Config Checks (require downloads) ────────────────────────


def load_hf_json(base_dir, model_name, filename, subdir=None):
    model_dir = Path(base_dir) / model_name
    if subdir:
        model_dir = model_dir / subdir
    path = model_dir / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_dim_a_processor_config(hf_dir, model_name, model_def):
    """Dimension A — processor_config.json checks for a single model."""
    print(f"\n--- {model_name} ---")
    pc = load_hf_json(hf_dir, model_name, "processor_config.json", model_def.get("subdir"))
    if pc is None:
        check(False, "processor_config.json found", skip=True)
        return

    modality_configs = load_modality_configs()

    # A10: processor_class
    check(
        pc.get("processor_class") == "Gr00tN1d7Processor",
        f"[A10] processor_class = '{pc.get('processor_class')}' (expected 'Gr00tN1d7Processor')",
    )

    pk = pc.get("processor_kwargs", {})

    # A1: modality_configs top-level keys
    hf_modality_keys = list(pk.get("modality_configs", {}).keys())
    for expected_tag in model_def["embodiment_tags"]:
        check(
            expected_tag in hf_modality_keys,
            f"[A1] modality_configs contains '{expected_tag}'",
        )

    # Per-tag modality checks
    for tag in model_def["embodiment_tags"]:
        hf_mc = pk.get("modality_configs", {}).get(tag)
        code_mc = modality_configs.get(tag)
        if not hf_mc:
            check(False, f"[A1] {tag} present in HF modality_configs")
            continue
        if not code_mc:
            info(f"  {tag} not in code MODALITY_CONFIGS (pretrain-only tag, expected)")
            continue

        # A2: video.delta_indices
        check(
            hf_mc["video"]["delta_indices"] == code_mc["video"]["delta_indices"],
            f"[A2] {tag} video.delta_indices: HF={hf_mc['video']['delta_indices']} vs code={code_mc['video']['delta_indices']}",
        )

        # A3: video.modality_keys count
        hf_vkeys = hf_mc["video"]["modality_keys"]
        code_vkeys = code_mc["video"]["modality_keys"]
        check(
            len(hf_vkeys) == len(code_vkeys),
            f"[A3] {tag} video key count: HF={len(hf_vkeys)} ({hf_vkeys}) vs code={len(code_vkeys)} ({code_vkeys})",
        )

        # A4: state.delta_indices
        check(
            hf_mc["state"]["delta_indices"] == code_mc["state"]["delta_indices"],
            f"[A4] {tag} state.delta_indices: HF={hf_mc['state']['delta_indices']} vs code={code_mc['state']['delta_indices']}",
        )

        # A5: state.modality_keys
        check(
            hf_mc["state"]["modality_keys"] == code_mc["state"]["modality_keys"],
            f"[A5] {tag} state.modality_keys match",
        )

        # A6: action.delta_indices
        check(
            hf_mc["action"]["delta_indices"] == code_mc["action"]["delta_indices"],
            f"[A6] {tag} action.delta_indices: HF len={len(hf_mc['action']['delta_indices'])} vs code len={len(code_mc['action']['delta_indices'])}",
        )

        # A7: action.modality_keys
        check(
            hf_mc["action"]["modality_keys"] == code_mc["action"]["modality_keys"],
            f"[A7] {tag} action.modality_keys match",
        )

        # A8: action.action_configs
        hf_ac = hf_mc["action"].get("action_configs")
        code_ac = code_mc["action"].get("action_configs")
        if code_ac:
            check(
                hf_ac is not None,
                f"[A8] {tag} action.action_configs present in HF",
            )
            if hf_ac:
                check(
                    len(hf_ac) == len(code_ac),
                    f"[A8] {tag} action_configs count: HF={len(hf_ac)} vs code={len(code_ac)}",
                )
                for i, (h, c) in enumerate(zip(hf_ac, code_ac)):
                    for field in ("rep", "type", "format"):
                        check(
                            h.get(field) == c.get(field),
                            f"[A8] {tag} action_configs[{i}].{field}: HF={h.get(field)} vs code={c.get(field)}",
                        )

        # A9: language.modality_keys
        check(
            hf_mc["language"]["modality_keys"] == code_mc["language"]["modality_keys"],
            f"[A9] {tag} language.modality_keys match",
        )

    # A11-A31: scalar parameters
    scalar_checks = {
        "max_state_dim": ("A11", None),
        "max_action_dim": ("A12", None),
        "max_action_horizon": ("A13", None),
        "model_name": ("A14", "nvidia/Cosmos-Reason2-2B"),
        "model_type": ("A15", "qwen"),
        "use_percentiles": ("A16", None),
        "apply_sincos_state_encoding": ("A17", None),
        "use_relative_action": ("A18", None),
        "formalize_language": ("A19", True),
        "clip_outliers": ("A20", True),
        "use_mean_std": ("A21", False),
        "letter_box_transform": ("A22", None),
        "exclude_state": ("A23", None),
        "state_dropout_prob": ("A24", None),
        "image_crop_size": ("A25", None),
        "image_target_size": ("A26", None),
        "shortest_image_edge": ("A27", 256),
        "crop_fraction": ("A28", 0.95),
        "use_albumentations": ("A29", None),
        "random_rotation_angle": ("A30", None),
        "color_jitter_params": ("A31", None),
    }
    for field, (item_id, expected) in scalar_checks.items():
        actual = pk.get(field)
        if expected is not None:
            check(
                actual == expected,
                f"[{item_id}] {field}: HF={actual!r} (expected {expected!r})",
            )
        else:
            info(f"[{item_id}] {field} = {actual!r}")


def check_dim_b_config_json(hf_dir, model_name, model_def):
    """Dimension B — config.json checks for a single model."""
    print(f"\n--- {model_name} ---")
    cfg = load_hf_json(hf_dir, model_name, "config.json", model_def.get("subdir"))
    if cfg is None:
        check(False, "config.json found", skip=True)
        return

    model_cfg = load_model_config_defaults()

    b_checks = {
        "B1": ("model_type", "Gr00tN1d7"),
        "B2": ("max_state_dim", None),
        "B3": ("max_action_dim", None),
        "B4": ("action_horizon", model_cfg.action_horizon),
        "B5": ("backbone_embedding_dim", model_cfg.backbone_embedding_dim),
        "B6": ("hidden_size", model_cfg.hidden_size),
        "B7": ("input_embedding_dim", model_cfg.input_embedding_dim),
        "B11": ("num_inference_timesteps", model_cfg.num_inference_timesteps),
        "B12": ("max_num_embodiments", model_cfg.max_num_embodiments),
        "B13": ("model_name", "nvidia/Cosmos-Reason2-2B"),
        "B14": ("select_layer", model_cfg.select_layer),
        "B15": ("state_history_length", model_cfg.state_history_length),
        "B16": ("noise_beta_alpha", model_cfg.noise_beta_alpha),
        "B17": ("noise_beta_beta", model_cfg.noise_beta_beta),
        "B18": ("noise_s", model_cfg.noise_s),
        "B19": ("num_timestep_buckets", model_cfg.num_timestep_buckets),
        "B20": ("add_pos_embed", model_cfg.add_pos_embed),
        "B21": ("attn_dropout", model_cfg.attn_dropout),
        "B22": ("use_vlln", model_cfg.use_vlln),
        "B23": ("max_seq_len", model_cfg.max_seq_len),
        "B24": ("use_alternate_vl_dit", model_cfg.use_alternate_vl_dit),
        "B25": ("attend_text_every_n_blocks", model_cfg.attend_text_every_n_blocks),
        "B27": ("backbone_model_type", model_cfg.backbone_model_type),
        "B28": ("reproject_vision", model_cfg.reproject_vision),
        "B29": ("use_percentiles", model_cfg.use_percentiles),
        "B30": ("use_relative_action", model_cfg.use_relative_action),
    }

    for item_id, (field, expected) in b_checks.items():
        actual = cfg.get(field)
        if expected is not None:
            check(
                actual == expected,
                f"[{item_id}] {field}: HF={actual!r} (expected {expected!r})",
            )
        else:
            info(f"[{item_id}] {field} = {actual!r}")

    # B8-B10: diffusion_model_cfg nested
    diff_cfg = cfg.get("diffusion_model_cfg", {})
    check(
        diff_cfg.get("num_layers") == 16,
        f"[B8] diffusion_model_cfg.num_layers: {diff_cfg.get('num_layers')} (expected 16)",
    )
    check(
        diff_cfg.get("num_attention_heads") == 32,
        f"[B9] diffusion_model_cfg.num_attention_heads: {diff_cfg.get('num_attention_heads')} (expected 32)",
    )
    check(
        diff_cfg.get("attention_head_dim") == 48,
        f"[B10] diffusion_model_cfg.attention_head_dim: {diff_cfg.get('attention_head_dim')} (expected 48)",
    )

    # I4: No internal/legacy field names
    legacy_fields = ["vlm_model_path", "GrootN1d7"]
    for lf in legacy_fields:
        check(lf not in cfg, f"[I4] No legacy field '{lf}' in config.json")

    # B26 / I2: torch_dtype
    dtype_val = cfg.get("torch_dtype") or cfg.get("model_dtype")
    info(f"[B26/I2] torch_dtype/model_dtype = {dtype_val!r}")

    # I1: architectures
    archs = cfg.get("architectures")
    if archs is not None:
        check(
            "Gr00tN1d7" in archs,
            f"[I1] architectures contains 'Gr00tN1d7': {archs}",
        )
    else:
        info("[I1] 'architectures' field not present")


def check_dim_c_embodiment_id(hf_dir, model_name, model_def):
    """Dimension C — embodiment_id.json checks."""
    print(f"\n--- {model_name} ---")
    eid = load_hf_json(hf_dir, model_name, "embodiment_id.json", model_def.get("subdir"))
    if eid is None:
        check(False, "embodiment_id.json found", skip=True)
        return

    projector_index = load_projector_index()

    # C1: all entries match code
    for tag, idx in eid.items():
        code_idx = projector_index.get(tag)
        if code_idx is not None:
            check(
                idx == code_idx,
                f"[C1] {tag}: HF={idx} vs code={code_idx}",
            )
        else:
            check(
                False, f"[C1] {tag} not in code EMBODIMENT_TAG_TO_PROJECTOR_INDEX", warn_only=True
            )

    # C2: pretrain tags present (derived from source of truth)
    _, PRETRAIN_TAGS, _ = load_embodiment_tags()
    pretrain_tag_values = [t.value for t in PRETRAIN_TAGS]
    for pt in pretrain_tag_values:
        check(
            pt in eid,
            f"[C2] Pretrain tag '{pt}' present in embodiment_id.json",
        )


def check_dim_d_statistics(hf_dir, model_name, model_def):
    """Dimension D — statistics.json checks."""
    print(f"\n--- {model_name} ---")
    stats = load_hf_json(hf_dir, model_name, "statistics.json", model_def.get("subdir"))
    pc = load_hf_json(hf_dir, model_name, "processor_config.json", model_def.get("subdir"))
    if stats is None:
        check(False, "statistics.json found", skip=True)
        return

    pk = pc.get("processor_kwargs", {}) if pc else {}
    use_percentiles = pk.get("use_percentiles", True)

    for tag in model_def["embodiment_tags"]:
        tag_stats = stats.get(tag)
        check(tag_stats is not None, f"[D1] Top-level key '{tag}' in statistics.json")
        if not tag_stats:
            continue

        # D2: state/action sub-dicts
        check("state" in tag_stats, f"[D2] {tag} has 'state' sub-dict")
        check("action" in tag_stats, f"[D2] {tag} has 'action' sub-dict")

        # D3: modality key coverage
        hf_mc = pk.get("modality_configs", {}).get(tag, {})
        for modality in ("state", "action"):
            if modality not in tag_stats or modality not in hf_mc:
                continue
            expected_keys = hf_mc[modality].get("modality_keys", [])
            actual_keys = list(tag_stats[modality].keys())
            for ek in expected_keys:
                check(
                    ek in actual_keys,
                    f"[D3] {tag}/{modality}: key '{ek}' in statistics",
                )

        # D4: normalization fields
        for modality in ("state", "action"):
            if modality not in tag_stats:
                continue
            for key, key_stats in tag_stats[modality].items():
                check(
                    "min" in key_stats and "max" in key_stats,
                    f"[D4] {tag}/{modality}/{key}: has min/max",
                )
                if use_percentiles:
                    has_pct = "q01" in key_stats or "p01" in key_stats
                    check(
                        has_pct,
                        f"[D4] {tag}/{modality}/{key}: has percentile fields (use_percentiles={use_percentiles})",
                    )

        # D6: No NaN/Inf
        def check_finite(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_finite(v, f"{path}/{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if isinstance(v, (int, float)):
                        check(
                            math.isfinite(v),
                            f"[D6] {path}[{i}] = {v} is finite",
                        )
            elif isinstance(obj, (int, float)):
                check(math.isfinite(obj), f"[D6] {path} = {obj} is finite")

        check_finite(tag_stats, f"{tag}")


def check_dim_f1_cross_file(hf_dir, model_name, model_def):
    """Dimension F1 — config.json ↔ processor_config.json agreement."""
    print(f"\n--- {model_name} ---")
    cfg = load_hf_json(hf_dir, model_name, "config.json", model_def.get("subdir"))
    pc = load_hf_json(hf_dir, model_name, "processor_config.json", model_def.get("subdir"))
    if cfg is None or pc is None:
        check(False, "Both config.json and processor_config.json found", skip=True)
        return

    pk = pc.get("processor_kwargs", {})

    # F1: max_state_dim, max_action_dim, action_horizon
    check(
        cfg.get("max_state_dim") == pk.get("max_state_dim"),
        f"[F1] max_state_dim: config.json={cfg.get('max_state_dim')} vs processor={pk.get('max_state_dim')}",
    )
    check(
        cfg.get("max_action_dim") == pk.get("max_action_dim"),
        f"[F1] max_action_dim: config.json={cfg.get('max_action_dim')} vs processor={pk.get('max_action_dim')}",
    )
    check(
        cfg.get("action_horizon") == pk.get("max_action_horizon"),
        f"[F1] action_horizon={cfg.get('action_horizon')} vs max_action_horizon={pk.get('max_action_horizon')}",
    )

    # F7: use_percentiles, use_relative_action
    check(
        cfg.get("use_percentiles") == pk.get("use_percentiles"),
        f"[F7] use_percentiles: config.json={cfg.get('use_percentiles')} vs processor={pk.get('use_percentiles')}",
    )
    check(
        cfg.get("use_relative_action") == pk.get("use_relative_action"),
        f"[F7] use_relative_action: config.json={cfg.get('use_relative_action')} vs processor={pk.get('use_relative_action')}",
    )

    # B13 cross: model_name
    check(
        cfg.get("model_name") == pk.get("model_name"),
        f"[B13] model_name: config.json={cfg.get('model_name')} vs processor={pk.get('model_name')}",
    )


# ──────────────────────── Test Fixture Check ────────────────────────


def check_test_fixture():
    """Check the test fixture processor_config against source of truth."""
    print("\n" + "=" * 70)
    print("TEST FIXTURE — tests/fixtures/processor_config/ Check")
    print("=" * 70)

    fixture_dir = REPO_ROOT / "tests/fixtures/processor_config"
    pc_path = fixture_dir / "processor_config.json"
    eid_path = fixture_dir / "embodiment_id.json"
    stats_path = fixture_dir / "statistics.json"

    if not pc_path.exists():
        check(False, "Test fixture processor_config.json exists", skip=True)
        return

    with open(pc_path) as f:
        pc = json.load(f)
    if not eid_path.exists():
        check(False, "Test fixture embodiment_id.json exists", skip=True)
        return
    with open(eid_path) as f:
        eid = json.load(f)
    if not stats_path.exists():
        check(False, "Test fixture statistics.json exists", skip=True)
        return
    with open(stats_path) as f:
        stats = json.load(f)

    modality_configs = load_modality_configs()
    model_cfg = load_model_config_defaults()
    projector_index = load_projector_index()

    pk = pc.get("processor_kwargs", {})

    # processor_class
    check(
        pc.get("processor_class") == "Gr00tN1d7Processor",
        f"processor_class = '{pc.get('processor_class')}'",
    )

    # modality_configs: libero_sim
    hf_mc = pk.get("modality_configs", {}).get("libero_sim")
    code_mc = modality_configs.get("libero_sim")
    check(hf_mc is not None, "modality_configs contains 'libero_sim'")

    if hf_mc and code_mc:
        # video delta_indices
        check(
            hf_mc["video"]["delta_indices"] == code_mc["video"]["delta_indices"],
            f"video.delta_indices: fixture={hf_mc['video']['delta_indices']} vs code={code_mc['video']['delta_indices']}",
        )
        # video key count
        check(
            len(hf_mc["video"]["modality_keys"]) == len(code_mc["video"]["modality_keys"]),
            f"video key count: fixture={len(hf_mc['video']['modality_keys'])} vs code={len(code_mc['video']['modality_keys'])}",
        )
        # state keys
        check(
            hf_mc["state"]["modality_keys"] == code_mc["state"]["modality_keys"],
            "state.modality_keys match",
        )
        # action delta_indices
        check(
            hf_mc["action"]["delta_indices"] == code_mc["action"]["delta_indices"],
            f"action.delta_indices: fixture len={len(hf_mc['action']['delta_indices'])} vs code len={len(code_mc['action']['delta_indices'])}",
        )
        # action keys
        check(
            hf_mc["action"]["modality_keys"] == code_mc["action"]["modality_keys"],
            "action.modality_keys match",
        )
        # language keys
        check(
            hf_mc["language"]["modality_keys"] == code_mc["language"]["modality_keys"],
            "language.modality_keys match",
        )

    # Scalar params — notable mismatches to flag
    print("\n  Scalar Parameter Comparison (fixture vs model config defaults):")
    info(
        f"max_state_dim: fixture={pk.get('max_state_dim')} vs model_cfg default={model_cfg.max_state_dim}"
    )
    info(
        f"max_action_dim: fixture={pk.get('max_action_dim')} vs model_cfg default={model_cfg.max_action_dim}"
    )
    info(
        f"max_action_horizon: fixture={pk.get('max_action_horizon')} vs model_cfg.action_horizon={model_cfg.action_horizon}"
    )
    info(
        f"use_percentiles: fixture={pk.get('use_percentiles')} vs model_cfg={model_cfg.use_percentiles}"
    )
    info(
        f"apply_sincos_state_encoding: fixture={pk.get('apply_sincos_state_encoding')} vs model_cfg={model_cfg.apply_sincos_state_encoding}"
    )
    info(
        f"use_relative_action: fixture={pk.get('use_relative_action')} vs model_cfg={model_cfg.use_relative_action}"
    )

    # Check missing fields (new fields added to save_pretrained)
    expected_fields = [
        "letter_box_transform",
        "exclude_state",
        "state_dropout_prob",
        "use_mean_std",
    ]
    print("\n  New Fields Check (may be missing in older fixtures):")
    for field in expected_fields:
        present = field in pk
        check(present, f"Field '{field}' present in fixture processor_config", warn_only=True)

    # embodiment_id.json
    print("\n  Embodiment ID Check:")
    for tag, idx in eid.items():
        code_idx = projector_index.get(tag)
        check(
            code_idx is not None and idx == code_idx,
            f"  {tag}: fixture={idx} vs code={code_idx}",
        )

    # statistics.json structure
    print("\n  Statistics Structure Check:")
    for tag in pk.get("modality_configs", {}).keys():
        check(tag in stats, f"  statistics.json has key '{tag}'")
        if tag in stats:
            check("state" in stats[tag], f"  {tag}/state present")
            check("action" in stats[tag], f"  {tag}/action present")


# ──────────────────────── Main ────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate HF config alignment")
    parser.add_argument(
        "--hf-config-dir",
        type=str,
        default=None,
        help="Directory containing downloaded HF configs (subdirs per model)",
    )
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║  HuggingFace Config Alignment Validation                         ║")
    print("╚" + "═" * 68 + "╝")

    # Always run: internal consistency checks
    check_dim_f_internal_consistency()
    check_dim_e_documentation()
    check_dim_f2_modality_json()
    check_dim_j_enum_serialization()
    check_test_fixture()

    # HF config checks (if directory provided)
    if args.hf_config_dir:
        hf_dir = Path(args.hf_config_dir)
        if not hf_dir.exists():
            print(f"\n[ERROR] HF config directory not found: {hf_dir}")
            sys.exit(1)

        for model_name, model_def in HF_MODELS.items():
            print("\n" + "=" * 70)
            print(f"DIMENSION A — processor_config.json: {model_name}")
            print("=" * 70)
            check_dim_a_processor_config(hf_dir, model_name, model_def)

        for model_name, model_def in HF_MODELS.items():
            print("\n" + "=" * 70)
            print(f"DIMENSION B — config.json: {model_name}")
            print("=" * 70)
            check_dim_b_config_json(hf_dir, model_name, model_def)

        for model_name, model_def in HF_MODELS.items():
            print("\n" + "=" * 70)
            print(f"DIMENSION C — embodiment_id.json: {model_name}")
            print("=" * 70)
            check_dim_c_embodiment_id(hf_dir, model_name, model_def)

        for model_name, model_def in HF_MODELS.items():
            print("\n" + "=" * 70)
            print(f"DIMENSION D — statistics.json: {model_name}")
            print("=" * 70)
            check_dim_d_statistics(hf_dir, model_name, model_def)

        for model_name, model_def in HF_MODELS.items():
            print("\n" + "=" * 70)
            print(f"DIMENSION F1 — Cross-file: {model_name}")
            print("=" * 70)
            check_dim_f1_cross_file(hf_dir, model_name, model_def)
    else:
        print("\n" + "=" * 70)
        print("HF CONFIG CHECKS SKIPPED — No --hf-config-dir provided")
        print("To run full checks, download HF configs first:")
        print("  uv run huggingface-cli login")
        print("  # Then download configs for each model (see checklist)")
        print(
            "  uv run python scripts/validate_hf_config_alignment.py --hf-config-dir /tmp/hf_configs"
        )
        print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {PASS}: {pass_count}")
    print(f"  {FAIL}: {fail_count}")
    print(f"  {WARN}: {warn_count}")
    print(f"  {SKIP}: {skip_count}")
    total = pass_count + fail_count
    if total > 0:
        print(f"  Pass rate: {pass_count}/{total} ({100 * pass_count / total:.1f}%)")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
