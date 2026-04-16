# HuggingFace Model Config Alignment Checklist

> **Purpose:** Ensure that the configuration files shipped with each HuggingFace model checkpoint
> (`processor_config.json`, `config.json`, `embodiment_id.json`, `statistics.json`) are consistent
> with the source-of-truth definitions in this repository.
>
> **Applicable models:**
>
> | # | HF Model | Local Embodiment Tag(s) | Type |
> |---|----------|------------------------|------|
> | 1 | `nvidia/GR00T-N1.7-3B` | All pretrain tags | Base model |
> | 2 | `nvidia/GR00T-N1.7-DROID` | `oxe_droid_relative_eef_relative_joint` | Finetuned |
> | 3 | `nvidia/GR00T-N1.7-LIBERO` | `libero_sim` | Finetuned (nested dir `libero_10/`) |
> | 4 | `nvidia/GR00T-N1.7-SimplerEnv-Fractal` | `simpler_env_google` | Finetuned |
> | 5 | `nvidia/GR00T-N1.7-SimplerEnv-Bridge` | `simpler_env_widowx` | Finetuned |

---

## Prerequisites

| ID | Task | Details |
|----|------|---------|
| P1 | Pull latest branch | `git pull origin n17_redo` |
| P2 | Download HF config files locally | For each of the 5 models, download `processor_config.json`, `config.json`, `statistics.json`, `embodiment_id.json` (if present) to a local temp directory for comparison. Use `huggingface-cli download` or `hf download`. |

---

## Source-of-Truth Files in This Repo

| File | What It Defines |
|------|-----------------|
| `gr00t/configs/model/gr00t_n1d7.py` → `Gr00tN1d7Config` | Model architecture defaults (`max_state_dim`, `max_action_dim`, `action_horizon`, backbone, diffusion cfg, etc.) |
| `gr00t/configs/data/embodiment_configs.py` → `MODALITY_CONFIGS` | Per-embodiment modality layout (video/state/action/language keys, `delta_indices`, `ActionConfig`) |
| `gr00t/data/embodiment_tags.py` → `EmbodimentTag` | Canonical enum of embodiment tag names ↔ string values |
| `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` → `Gr00tN1d7Processor` | Processor logic, `save_pretrained()` serialization format, `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` |
| `examples/{LIBERO,SimplerEnv,DROID,SO100}/modality.json` | Dataset-level modality key mappings (state/action ranges, video original keys) |
| `README.md`, `examples/*/README.md` | User-facing commands, checkpoint table, embodiment tag references |

---

## Dimension A — `processor_config.json` Core Parameters

Each HF model's `processor_config.json` is produced by `Gr00tN1d7Processor.save_pretrained()`.
The following fields must be checked against local sources.

### A.1 Modality Configs

> **Local source of truth:** `gr00t/configs/data/embodiment_configs.py` → `MODALITY_CONFIGS`

| ID | Field | What to Check | Notes |
|----|-------|---------------|-------|
| A1 | `modality_configs` top-level key | Must match the `EmbodimentTag.value` string (e.g. `"simpler_env_google"`, `"libero_sim"`) | Cross-ref with `embodiment_tags.py` |
| A2 | `video.delta_indices` | Must match `MODALITY_CONFIGS[tag]["video"].delta_indices` | e.g. `[0]` for most posttrain, `[-15, 0]` for DROID |
| A3 | `video.modality_keys` | Should correspond to `MODALITY_CONFIGS[tag]["video"].modality_keys` | **Caveat:** HF may store dataset-remapped keys (e.g. `observation.images.rgb.head_256_256`) while code uses generic keys (e.g. `image`). This is expected for finetuned models where the processor was saved with dataset-specific keys. Verify the key count and ordering still match. |
| A4 | `state.delta_indices` | Must match `MODALITY_CONFIGS[tag]["state"].delta_indices` | Usually `[0]` |
| A5 | `state.modality_keys` | Must match `MODALITY_CONFIGS[tag]["state"].modality_keys` | e.g. `simpler_env_google`: `["x","y","z","rx","ry","rz","rw","gripper"]` |
| A6 | `action.delta_indices` | Must match `MODALITY_CONFIGS[tag]["action"].delta_indices` | Determines effective action horizon per embodiment (e.g. `range(8)` → horizon 8) |
| A7 | `action.modality_keys` | Must match `MODALITY_CONFIGS[tag]["action"].modality_keys` | e.g. `["x","y","z","roll","pitch","yaw","gripper"]` |
| A8 | `action.action_configs` | Each entry's `rep`, `type`, `format`, `state_key` must match the `ActionConfig` list in `MODALITY_CONFIGS` | Pay attention to RELATIVE vs ABSOLUTE, EEF vs NON_EEF |
| A9 | `language.modality_keys` | Must match `MODALITY_CONFIGS[tag]["language"].modality_keys` | e.g. `["annotation.human.action.task_description"]` |

### A.2 Model / Processor Scalar Parameters

> **Local source of truth:** `Gr00tN1d7Config` (model defaults) and `Gr00tN1d7Processor.__init__` (processor defaults)

| ID | Field in `processor_config.json` | Expected Value / Source | Notes |
|----|----------------------------------|------------------------|-------|
| A10 | `processor_class` | `"Gr00tN1d7Processor"` | Top-level field; required for `AutoProcessor.from_pretrained` to resolve the correct class |
| A11 | `max_state_dim` | `Gr00tN1d7Config.max_state_dim` (default `132`) | Finetuned models may override; processor `__init__` default is `29` (overridden at training time) |
| A12 | `max_action_dim` | `Gr00tN1d7Config.max_action_dim` (default `132`) | Same as above |
| A13 | `max_action_horizon` | `Gr00tN1d7Config.action_horizon` (default `40`) | Note naming difference: model config uses `action_horizon`, processor uses `max_action_horizon`. These represent the model's **max capacity**, not the per-embodiment actual horizon. |
| A14 | `model_name` | `"nvidia/Cosmos-Reason2-2B"` | VLM backbone identifier |
| A15 | `model_type` | `"qwen"` | Backbone architecture type (processor-level; distinct from `config.json`'s `model_type` which is `"Gr00tN1d7"`) |
| A16 | `use_percentiles` | `Gr00tN1d7Config.use_percentiles` (default `True`) | Controls normalization strategy; finetuned models may differ |
| A17 | `apply_sincos_state_encoding` | `Gr00tN1d7Config.apply_sincos_state_encoding` (default `False`) | |
| A18 | `use_relative_action` | `Gr00tN1d7Config.use_relative_action` (default `False`) | |
| A19 | `formalize_language` | `True` | Lowercases and strips punctuation from language instructions |
| A20 | `clip_outliers` | `True` (processor default) | |
| A21 | `use_mean_std` | `False` (processor default) | Alternative normalization mode |
| A22 | `letter_box_transform` | `False` (processor default) | Serialized by `save_pretrained` but easy to miss |
| A23 | `exclude_state` | `False` (processor default) | If `True`, state input is zeroed out during inference |
| A24 | `state_dropout_prob` | `0.0` for inference checkpoints | Training-time augmentation; should be 0 for published inference models |

### A.3 Image Processing Parameters

| ID | Field | Expected Value / Source | Notes |
|----|-------|------------------------|-------|
| A25 | `image_crop_size` | `Gr00tN1d7Config.image_crop_size` (default `(230, 230)`) or `None` | May be `null` in finetuned models that use `shortest_image_edge` + `crop_fraction` instead |
| A26 | `image_target_size` | `Gr00tN1d7Config.image_target_size` (default `(256, 256)`) or `None` | Same caveat as above |
| A27 | `shortest_image_edge` | Processor default `256` | Used when `image_crop_size`/`image_target_size` are `None` |
| A28 | `crop_fraction` | Processor default `0.95` | |
| A29 | `use_albumentations` | Processor default `False` | **Naming caveat:** processor uses `use_albumentations`, model config uses `use_albumentations_transforms` — same semantics, different key |
| A30 | `random_rotation_angle` | Usually `null` for inference | |
| A31 | `color_jitter_params` | Training augmentation; may be non-null in saved configs | Typical: `{"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}` |

---

## Dimension B — `config.json` (Model Architecture)

> **Local source of truth:** `gr00t/configs/model/gr00t_n1d7.py` → `Gr00tN1d7Config`
>
> This file is produced by `model.config.save_pretrained()` (via HF `PretrainedConfig`).
> Architecture parameters should **not** change between base and finetuned models.

| ID | Field | Expected Default | Notes |
|----|-------|-----------------|-------|
| B1 | `model_type` | `"Gr00tN1d7"` | Registered model type (distinct from processor's `model_type` = `"qwen"`) |
| B2 | `max_state_dim` | `132` | Must match `processor_config.json` |
| B3 | `max_action_dim` | `132` | Must match `processor_config.json` |
| B4 | `action_horizon` | `40` | Must match `processor_config.json`'s `max_action_horizon` |
| B5 | `backbone_embedding_dim` | `1536` | Should not change across checkpoints |
| B6 | `hidden_size` | `1024` | Should not change |
| B7 | `input_embedding_dim` | `1536` | Should not change |
| B8 | `diffusion_model_cfg.num_layers` | `16` | |
| B9 | `diffusion_model_cfg.num_attention_heads` | `32` | |
| B10 | `diffusion_model_cfg.attention_head_dim` | `48` | |
| B11 | `num_inference_timesteps` | `4` | Flow-matching denoising steps |
| B12 | `max_num_embodiments` | `32` | Projector count upper bound |
| B13 | `model_name` | `"nvidia/Cosmos-Reason2-2B"` | Must match processor's `model_name` |
| B14 | `select_layer` | `12` | VLM layer selection |
| B15 | `state_history_length` | `1` | |
| B16 | `noise_beta_alpha` | `1.5` | Flow-matching noise schedule; silent behavior change if wrong |
| B17 | `noise_beta_beta` | `1.0` | Flow-matching noise schedule |
| B18 | `noise_s` | `0.999` | Flow-matching noise schedule |
| B19 | `num_timestep_buckets` | `1000` | Timestep discretization for training/inference |
| B20 | `add_pos_embed` | `True` | Positional embedding in DiT; architecture-level |
| B21 | `attn_dropout` | `0.0` | Non-zero value would affect inference |
| B22 | `use_vlln` | `False` | VL-LN architecture switch; must match weights |
| B23 | `max_seq_len` | `2048` | Maximum sequence length |
| B24 | `use_alternate_vl_dit` | `False` | Alternative VL-DiT architecture switch |
| B25 | `attend_text_every_n_blocks` | `4` | Cross-attention frequency |
| B26 | `model_dtype` | `"torch.bfloat16"` | Affects loading behavior |
| B27 | `backbone_model_type` | `"qwen"` | Not to confuse with `model_type` (`"Gr00tN1d7"`) |
| B28 | `reproject_vision` | `False` | Controls vision reprojection layer |
| B29 | `use_percentiles` | `True` | Must match `processor_config.json` |
| B30 | `use_relative_action` | `False` | Must match `processor_config.json` |

---

## Dimension C — `embodiment_id.json`

> **Local source of truth:** `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` → `EMBODIMENT_TAG_TO_PROJECTOR_INDEX`

| ID | Check | Details |
|----|-------|---------|
| C1 | Tag → projector index mapping | Every entry in HF `embodiment_id.json` must match the hardcoded `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` dict |
| C2 | Pretrain tags present | Finetuned models should still contain all pretrain tag entries (code merges them in `__init__`) |
| C3 | Posttrain tag indices | `simpler_env_google: 0`, `simpler_env_widowx: 1`, `libero_sim: 2`, `unitree_g1_full_body_with_waist_height_nav_cmd: 25`, `new_embodiment: 10` |

**Reference mapping (from code):**

```
# Pretrain
oxe_droid_relative_eef_relative_joint: 24
xdof_relative_eef_relative_joint: 27
xdof_relative_eef_relative_joint_subtask: 27
real_g1_relative_eef_relative_joints: 25
real_r1_pro_sharpa_relative_eef: 26
real_r1_pro_sharpa_relative_eef_human: 26
real_r1_pro_sharpa_relative_eef_maxinsights: 26
real_r1_pro_sharpa_relative_eef_mecka: 26

# Posttrain
unitree_g1_full_body_with_waist_height_nav_cmd: 25
simpler_env_google: 0
simpler_env_widowx: 1
libero_sim: 2
new_embodiment: 10
```

---

## Dimension D — `statistics.json`

> **Note:** `statistics.json` contains per-embodiment normalization parameters computed from training data.
> These are dataset-specific and not expected to match code defaults — but structural consistency must hold.

| ID | Check | Details |
|----|-------|---------|
| D1 | Top-level key matches embodiment tag | The key in `statistics.json` must match the `modality_configs` key in `processor_config.json` |
| D2 | State/action sub-keys present | Each embodiment entry must contain `"state"` and `"action"` sub-dicts |
| D3 | Modality key coverage | State/action sub-keys must cover all `modality_keys` declared in the corresponding `modality_configs` |
| D4 | Normalization fields | Each key should have `min`, `max` arrays. If `use_percentiles=true`, should also have `p01`, `p99`. If `use_mean_std=true`, should have `mean`, `std`. |
| D5 | Dimension consistency | Array lengths in statistics should match the dimensionality implied by `modality.json` (e.g. `gripper` → dim 1, `eef_9d` → dim 9) |
| D6 | No NaN / Inf values | All numeric arrays in statistics must be finite; NaN/Inf indicates a data pipeline failure |

---

## Dimension E — README & Documentation Consistency

> **Scope:** `README.md` and `examples/*/README.md`

| ID | Check | File(s) | Details |
|----|-------|---------|---------|
| E1 | Checkpoint table | `README.md` L207-213 | Model names, Types (Base/Finetuned), Embodiment Tags must match `embodiment_tags.py` |
| E2 | `--embodiment-tag` in example commands | `examples/{DROID,LIBERO,SimplerEnv}/README.md` | Tags used in CLI commands must be valid `EmbodimentTag` enum **names** (e.g. `SIMPLER_ENV_GOOGLE`, not the value `simpler_env_google`) |
| E3 | `--model-path` in example commands | Same files | HF model IDs must be correct and reachable |
| E4 | DROID modality table | `examples/DROID/README.md` L11-16 | Video/State/Action keys and dimensions (17D) must match `MODALITY_CONFIGS["oxe_droid_relative_eef_relative_joint"]` |
| E5 | Example `modality.json` files | `examples/SimplerEnv/fractal_modality.json`, `bridge_modality.json`, `examples/LIBERO/modality.json` | State/action key ranges and video keys must correspond to `MODALITY_CONFIGS` |
| E6 | LIBERO nested directory structure | `README.md` download commands | The `hf download --include "libero_10/..."` paths must match what actually exists on HF |
| E7 | `--action-horizon` in commands | Various READMEs | Should be ≤ the effective `len(action.delta_indices)` for that embodiment |

---

## Dimension F — Cross-File Consistency

| ID | Check | Details |
|----|-------|---------|
| F1 | `config.json` ↔ `processor_config.json` | `max_state_dim`, `max_action_dim`, `action_horizon`/`max_action_horizon` must agree between the two files in the same checkpoint |
| F2 | `MODALITY_CONFIGS` ↔ `examples/*/modality.json` | Code-level embodiment config keys should correspond to dataset modality.json key ranges. e.g. if `MODALITY_CONFIGS["simpler_env_google"]["state"].modality_keys` has 8 keys, the modality.json `state` section should define 8 key ranges. |
| F3 | `action_horizon` semantics | `Gr00tN1d7Config.action_horizon` (default 40) is the **model's maximum capacity**. `len(MODALITY_CONFIGS[tag]["action"].delta_indices)` is the **per-embodiment actual horizon** (e.g. 8, 16, 40, 50). The actual horizon must be ≤ the model max. |
| F4 | `Gr00tN1d7Config` defaults vs `Gr00tN1d7Processor.__init__` defaults | Some parameters have different defaults in the two classes (e.g. `max_state_dim`: 132 in Config vs 29 in Processor). The **training pipeline** overrides the Processor defaults from the Config. HF checkpoints should reflect the training-time values, not the Processor `__init__` defaults. |
| F5 | `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` ↔ `EmbodimentTag` enum | Every `EmbodimentTag` value that appears in `MODALITY_CONFIGS` should have a corresponding entry in `EMBODIMENT_TAG_TO_PROJECTOR_INDEX`. |
| F6 | `use_albumentations` ↔ `use_albumentations_transforms` | Processor uses `use_albumentations`, model config uses `use_albumentations_transforms`. Values must agree semantically. |
| F7 | `use_percentiles` / `use_relative_action` cross-file | These fields exist in both `config.json` and `processor_config.json`; values must match within the same checkpoint. |

---

## Dimension G — Checkpoint File Layout & Completeness

> **Scope:** Verify the directory structure and file inventory of each published HF checkpoint.

| ID | Check | Details |
|----|-------|---------|
| G1 | Required files present | Each checkpoint must contain: `config.json`, `processor_config.json`, `statistics.json`, `embodiment_id.json`, weight files (`model.safetensors` or `model-*.safetensors` + `model.safetensors.index.json`) |
| G2 | No training-only artifacts | Checkpoints should NOT contain `trainer_state.json`, `training_args.bin`, `optimizer.pt`, or `rng_state*.pth` (unless intentionally published) |
| G3 | Processor file location | Config JSON files should be at **root** level (not under a `processor/` subdirectory). `Gr00tPolicy` falls back to `processor/` but canonical HF layout is flat. |
| G4 | LIBERO nested structure | `libero_10/` subdirectory must contain the full set of config + weight files |
| G5 | Safetensors index validity | `model.safetensors.index.json` (if present) must reference all shard files that actually exist |

---

## Dimension H — `extra_augmentation_config` Serialization Gap

> **Note:** `extra_augmentation_config` exists on both `Gr00tN1d7Config` and `Gr00tN1d7Processor.__init__`
> but is **NOT serialized** by `Gr00tN1d7Processor.save_pretrained()`. It IS saved in `config.json` (model side).

| ID | Check | Details |
|----|-------|---------|
| H1 | Presence in `config.json` | If the model was trained with non-null `extra_augmentation_config`, it should appear in `config.json` |
| H2 | Absence in `processor_config.json` | This field is **expected to be missing** from `processor_config.json` (known serialization gap) |
| H3 | Impact assessment | If any published checkpoint was trained with custom albumentations via this field, document the gap — the augmentation config is lost on processor save/load round-trip |

---

## Dimension I — HF Hub Metadata Fields

> **Scope:** Fields injected by `PretrainedConfig.to_dict()` into `config.json`

| ID | Field | Expected Value | Notes |
|----|-------|---------------|-------|
| I1 | `architectures` | `["Gr00tN1d7"]` | Required for `AutoModel` class resolution |
| I2 | `torch_dtype` | `"bfloat16"` | Affects model loading precision |
| I3 | `_name_or_path` | Should match HF repo ID | e.g. `"nvidia/GR00T-N1.7-3B"` |
| I4 | No internal/legacy field names | Must not contain `vlm_model_path`, `GrootN1d7`, or other pre-OSS field names | See `scripts/internal/patch_checkpoint.py` for the mapping |

---

## Dimension J — Enum Serialization Format

> **Scope:** `ActionConfig` fields in `modality_configs` within `processor_config.json`

| ID | Check | Details |
|----|-------|---------|
| J1 | `action_configs[].rep` | Must use **enum name** strings (e.g. `"RELATIVE"`, `"ABSOLUTE"`), not integer values. `ModalityConfig.__post_init__` uses `ActionRepresentation[name]` to reconstruct. |
| J2 | `action_configs[].type` | Must use enum name strings (e.g. `"EEF"`, `"NON_EEF"`) |
| J3 | `action_configs[].format` | Must use enum name strings (e.g. `"ROTATION_6D"`, `"SCALAR"`) if present |

---

## Known Caveats & Expected Differences

1. **`video.modality_keys` remapping:** Finetuned checkpoints may store dataset-specific video keys
   (e.g. `observation.images.rgb.head_256_256`) rather than the generic keys in `embodiment_configs.py`
   (e.g. `image`). This is expected — the processor is saved after the data pipeline resolves
   dataset-level key mappings. The key **count** and **ordering** should still match.

2. **`max_state_dim` / `max_action_dim` values:** The model config default is `132`, but finetuned
   models saved from older training runs may use `128` or other values. As long as `config.json` and
   `processor_config.json` within the same checkpoint agree, this is acceptable.

3. **`image_crop_size` / `image_target_size` may be `null`:** When the processor uses
   `shortest_image_edge` + `crop_fraction` for resizing (the albumentations path), the crop/target
   size fields are `null`. This is normal for N1.7 models.

4. **`color_jitter_params` in processor_config:** These are training-time augmentation parameters
   that get serialized into the saved config. They are **not used at inference time** (the processor
   uses `eval_image_transform`), but their presence is expected.

5. **Base model (`GR00T-N1.7-3B`) has multiple embodiment configs:** Its `processor_config.json`
   should contain `modality_configs` entries for **all pretrain tags**, not just one.

6. **`extra_augmentation_config` not in processor JSON:** This field is present in model config
   and processor `__init__` but is intentionally omitted from `save_pretrained()` serialization.
   This is a known gap — if a model was trained with custom albumentations augmentation, that
   config is only recoverable from `config.json` (model side), not from `processor_config.json`.

7. **Naming mismatches between model config and processor config:** Several fields share the same
   semantics but have different key names: `action_horizon` (model) vs `max_action_horizon` (processor),
   `use_albumentations_transforms` (model) vs `use_albumentations` (processor). Always compare the
   correct field name for each file.

---

## Per-Model Quick Reference

### nvidia/GR00T-N1.7-3B (Base)

- **Embodiment tags in processor_config:** All 8 pretrain tags
- **Expected action horizons:** `oxe_droid...` → 40, others vary
- **Expected `max_state_dim` / `max_action_dim`:** 132

### nvidia/GR00T-N1.7-DROID

- **Embodiment tag:** `oxe_droid_relative_eef_relative_joint`
- **Action horizon:** 40 (`range(40)`)
- **Video keys:** `exterior_image_1_left`, `wrist_image_left`
- **State/Action dim:** 17D (eef_9d=9 + gripper=1 + joint_position=7)
- **Action configs:** eef_9d=RELATIVE/EEF, gripper=ABSOLUTE/NON_EEF, joint=RELATIVE/NON_EEF

### nvidia/GR00T-N1.7-LIBERO

- **Embodiment tag:** `libero_sim`
- **Action horizon:** 16 (`range(16)`)
- **Video keys:** `image`, `wrist_image` (may be remapped to `observation.images.*`)
- **State/Action dim:** 7D (x,y,z,roll,pitch,yaw,gripper)
- **Action configs:** No `action_configs` in `MODALITY_CONFIGS` (defaults apply)
- **Directory structure:** Nested under `libero_10/` on HF

### nvidia/GR00T-N1.7-SimplerEnv-Fractal

- **Embodiment tag:** `simpler_env_google`
- **Action horizon:** 8 (`range(8)`)
- **Video keys:** `image`
- **State keys:** `x,y,z,rx,ry,rz,rw,gripper` (8D, quaternion orientation)
- **Action keys:** `x,y,z,roll,pitch,yaw,gripper` (7D, euler orientation)
- **Note:** State uses quaternion (`rx,ry,rz,rw`) but action uses euler (`roll,pitch,yaw`)

### nvidia/GR00T-N1.7-SimplerEnv-Bridge

- **Embodiment tag:** `simpler_env_widowx`
- **Action horizon:** 8 (`range(8)`)
- **Video keys:** `image_0`
- **State keys:** `x,y,z,roll,pitch,yaw,pad,gripper` (8D, includes padding)
- **Action keys:** `x,y,z,roll,pitch,yaw,gripper` (7D)
- **Note:** State has an extra `pad` key not present in action

---

## Execution Steps

### Automated Validation

```bash
# Run internal consistency checks (no HF auth required):
uv run python scripts/validate_hf_config_alignment.py

# Full check with HF configs (requires auth):
uv run huggingface-cli login
uv run huggingface-cli download nvidia/GR00T-N1.7-3B \
    --include "processor_config.json" "config.json" "statistics.json" "embodiment_id.json" \
    --local-dir /tmp/hf_configs/GR00T-N1.7-3B
uv run huggingface-cli download nvidia/GR00T-N1.7-DROID \
    --include "processor_config.json" "config.json" "statistics.json" "embodiment_id.json" \
    --local-dir /tmp/hf_configs/GR00T-N1.7-DROID
uv run huggingface-cli download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/processor_config.json" "libero_10/config.json" \
             "libero_10/statistics.json" "libero_10/embodiment_id.json" \
    --local-dir /tmp/hf_configs/GR00T-N1.7-LIBERO
uv run huggingface-cli download nvidia/GR00T-N1.7-SimplerEnv-Fractal \
    --include "processor_config.json" "config.json" "statistics.json" "embodiment_id.json" \
    --local-dir /tmp/hf_configs/SimplerEnv-Fractal
uv run huggingface-cli download nvidia/GR00T-N1.7-SimplerEnv-Bridge \
    --include "processor_config.json" "config.json" "statistics.json" "embodiment_id.json" \
    --local-dir /tmp/hf_configs/SimplerEnv-Bridge

# Run full validation:
uv run python scripts/validate_hf_config_alignment.py --hf-config-dir /tmp/hf_configs
```

---

## Validation Results (Internal Consistency)

> Run date: 2026-04-15 | Branch: `regen/droid-demo-data`
> HF model configs: **NOT CHECKED** (gated models, 401 auth required)

### Summary: 104 PASS / 2 FAIL / 4 WARN

### Findings

#### FAIL 1 — [F3] `unitree_g1` action horizon exceeds model max

- **Issue:** `MODALITY_CONFIGS["unitree_g1_full_body_with_waist_height_nav_cmd"]["action"].delta_indices`
  has length **50**, but `Gr00tN1d7Config.action_horizon` default is **40**.
- **Location:** `gr00t/configs/data/embodiment_configs.py` line 85 vs `gr00t/configs/model/gr00t_n1d7.py` line 75
- **Impact:** The per-embodiment actual horizon (50) exceeds the model's declared max capacity (40).
  Either the model config default needs to be ≥ 50, or the G1 config horizon needs to be reduced.
- **Note:** `unitree_g1` is not associated with any published HF checkpoint (posttrain tag only),
  so this does not affect the 5 published models. However, it is a source-of-truth inconsistency.

#### FAIL 2 — [C1] Test fixture `embodiment_id.json` has wrong projector index

- **Issue:** `tests/fixtures/processor_config/embodiment_id.json` maps `"libero_sim": 24`,
  but `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` in code maps `"libero_sim": 2`.
- **Location:** `tests/fixtures/processor_config/embodiment_id.json`
- **Impact:** Tests using this fixture will use the wrong projector index for LIBERO.
  This could silently produce incorrect inference results in tests.

#### WARN — Test fixture missing new `processor_config` fields

- `letter_box_transform`, `exclude_state`, `state_dropout_prob`, `use_mean_std` are **not present**
  in `tests/fixtures/processor_config/processor_config.json`.
- These fields were added to `save_pretrained()` after the fixture was created.
- **Impact:** Low — the processor `__init__` has sensible defaults for these. But the fixture
  should be updated for complete coverage.

#### WARN — Test fixture scalar params differ from model config defaults

These are informational; the fixture represents a specific training run, not the base model defaults:

| Field | Fixture | Model Config Default |
|-------|---------|---------------------|
| `max_state_dim` | 128 | 132 |
| `max_action_dim` | 128 | 132 |
| `max_action_horizon` | 50 | 40 |
| `use_percentiles` | false | true |
| `apply_sincos_state_encoding` | true | false |
| `use_relative_action` | true | false |

### Dimensions Fully Passing

| Dimension | Result | Details |
|-----------|--------|---------|
| E — README & Docs | **ALL PASS** | All 5 HF model IDs in README, all `--embodiment-tag` values use correct enum names, DROID modality table matches code, all modality.json files match MODALITY_CONFIGS |
| F2 — modality.json ↔ code | **ALL PASS** | State/action/video key counts and names match for all 3 example modality.json files |
| F5 — Projector index ↔ EmbodimentTag | **ALL PASS** | All tags in MODALITY_CONFIGS have projector indices; all projector index keys are valid enum values |
| J — Enum serialization | **ALL PASS** | All ActionConfig enums use correct name strings (RELATIVE, ABSOLUTE, EEF, NON_EEF, etc.) |

### Blocked Checks (require HF auth)

| Dimension | Scope | Status |
|-----------|-------|--------|
| A — processor_config.json | All 5 HF models | BLOCKED |
| B — config.json | All 5 HF models | BLOCKED |
| C — embodiment_id.json | All 5 HF models | BLOCKED |
| D — statistics.json | All 5 HF models | BLOCKED |
| F1 — Cross-file agreement | All 5 HF models | BLOCKED |
| G — File layout | All 5 HF models | BLOCKED |
| H — extra_augmentation_config | All 5 HF models | BLOCKED |
| I — HF metadata fields | All 5 HF models | BLOCKED |
