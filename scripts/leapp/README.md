# Export GR00T N2 with LEAPP

> **Warning: relative end-effector (EEF) exports are partial, not deployment-ready.**
> For relative EEF action groups, the exporter keeps denormalization and the
> checkpoint's clipping behavior but skips relative-to-absolute pose conversion.
> These outputs are relative `xyz + rot6d` predictions, **not absolute pose targets**,
> even if the checkpoint requests another absolute output format. Other action
> groups retain their normal decoding. The exporter prints a warning and writes
> `relative_eef_outputs.json` beside the pipeline YAML, recording affected outputs,
> reference states, rotation convention, and omitted postprocessing.
> The deployment application must implement and validate pose conversion before
> execution. Numerical export validation covers only this partial graph, not
> end-to-end EEF control. A full relative EEF checkpoint export has not yet been
> validated; other unsupported operations still cause export to fail.

This directory contains the LEAPP entry points for exporting GR00T policies. Use
`n2_export.py` to capture the production N2 flat-policy path from numeric camera
and state inputs to decoded action outputs as one ONNX model.

The exporter uses a synthetic observation only to establish the graph interface.
A dataset is not required.

## What the N2 export contains

The exported node includes:

- Image preprocessing, including center crop and resize.
- State normalization and packing.
- The N2 video/action model and denoising loop.
- Action decoding, slicing, and unnormalization.

The language instruction is a compile-time constant. Text tokenization and the
Cosmos text encoder run while exporting, but their approximately 16 GB of weights
are not stored in the ONNX model. Changing the instruction requires another
export.

Network transport, robot safety logic, and SONIC decoding are not included.

## Requirements

- Linux with an NVIDIA CUDA GPU. `n2_export.py` currently uses `cuda:0`.
- Python 3.12 for the current `groot-core` project.
- A GR00T environment capable of running N2 inference, including the
  `g1-inference-n2` dependencies and FlashAttention.
- LEAPP and ONNX installed in that same environment. GPU-enabled ONNX Runtime
  is additionally required for numerical replay validation.
- Approximately 45 GB for the checkpoint and frozen model dependencies, plus at
  least 5 GB for the generated ONNX artifacts.

## 1. Prepare the export environment

Start from the GR00T repository root. Enable the optional `leapp` extra in the
standard N2 inference environment; no separate LEAPP environment is needed:

```bash
uv sync --project groot/core --locked --extra g1-inference-n2 --extra leapp
```

`uv` manages `groot/core/.venv`. Dependencies, including the tested LEAPP version,
are declared in `groot/core/pyproject.toml` and resolved together in its lockfile;
there is no version to choose in the install command. The monorepo CI image
selects its own `leapp` extra from the root manifest instead.

Use `--extra g1-inference-n2 --extra leapp` on each `uv run` below so both sets
of dependencies remain installed. Verify the resulting interpreter:

```bash
uv run --project groot/core --locked --extra g1-inference-n2 --extra leapp \
  python -c "import flash_attn, leapp, onnx, torch; import onnxruntime as ort; print(leapp.__version__, torch.__version__); print(ort.get_available_providers())"
```

The locked extra installs CPU ONNX Runtime. Export still requires a CUDA GPU
for PyTorch, but export and schema validation do not need GPU ONNX Runtime.
The smoke command below explicitly skips numerical replay. For GPU replay, see
the optional setup in the validation section.

LEAPP documentation is available at
<https://nvidia-isaac.github.io/leapp/>.

## 2. Download a checkpoint

The exporter does not download the N2 checkpoint itself. Obtain a complete N2
checkpoint directory from the training artifact store or a machine that already
contains it. Preserve the whole directory rather than copying only the model
shards.

For example:

```bash
ASSET_ROOT="$HOME/checkpoints"
CHECKPOINT="$ASSET_ROOT/my_n2_run/checkpoint-5000"

mkdir -p "$CHECKPOINT"
rsync -ah --partial --info=progress2 \
  user@training-host:/path/to/my_n2_run/checkpoint-5000/ \
  "$CHECKPOINT/"
```

A complete checkpoint normally contains at least:

```text
checkpoint-5000/
├── config.json
├── processor_config.json
├── statistics.json
├── model.safetensors.index.json
└── model-*-of-*.safetensors
```

The checkpoint used for the validated smoke export occupies approximately 20 GB.

## 3. Download the frozen model dependencies

N2 checkpoints reference three large frozen dependencies that are not embedded
in the checkpoint:

```text
MODEL_DEPS/
├── cosmos25/
│   ├── Cosmos-Predict2.5-2B-diffusers/
│   │   └── transformer/
│   └── cosmos_reason1/
└── Wan2.1-I2V-14B-480P/
    └── Wan2.1_VAE.pth
```

Approximate sizes are:

- Cosmos text encoder and tokenizer: 16 GB.
- Cosmos diffusion transformer: 3.9 GB.
- Wan video VAE: 485 MB.

Copy these dependencies from the artifact store or a machine that has them, and
pass their common root as `--model-deps`. If the checkpoint already references
valid mounted paths, `--model-deps` may be omitted. On configured NVIDIA systems,
the GR00T dependency resolver can also fetch missing files from the internal
model-dependency bucket when credentials are available.

For example, when copying from another machine:

```bash
mkdir -p "$ASSET_ROOT/cosmos25" "$ASSET_ROOT/Wan2.1-I2V-14B-480P"

rsync -ah --partial --info=progress2 \
  user@model-host:/path/to/cosmos25/ \
  "$ASSET_ROOT/cosmos25/"

rsync -ah --partial --info=progress2 \
  user@model-host:/path/to/Wan2.1-I2V-14B-480P/ \
  "$ASSET_ROOT/Wan2.1-I2V-14B-480P/"
```

The dependency resolver may rewrite absolute model paths in the checkpoint's
JSON configuration files to point at the local cache. Keep the checkpoint
writable, or make a writable copy before exporting.

## 4. Run a one-step smoke export

From the GR00T repository root:

```bash
uv run --project groot/core --locked --extra g1-inference-n2 --extra leapp \
  python scripts/leapp/n2_export.py \
  --checkpoint "$CHECKPOINT" \
  --model-deps "$ASSET_ROOT" \
  --output-dir "$HOME/n2_leapp_export" \
  --instruction "Pick up the red object from the table." \
  --denoise-steps 1 \
  --video-denoise-steps 1 \
  --opset-version 22 \
  --skip-runtime-validation
```

Use a new or empty output directory. A successful export writes several
gigabytes and should not be mixed with artifacts from a different checkpoint or
configuration.

The one-step configuration is intended as the fastest integrity check. Export
again with the denoising-step configuration intended for deployment. The action
step count must be divisible by the video step count.

Exporter options:

| Option | Default | Description |
| --- | --- | --- |
| `--checkpoint PATH` | Required | N2 checkpoint directory. |
| `--model-deps PATH` | Checkpoint paths or configured resolver | Root containing `cosmos25/` and `Wan2.1-I2V-14B-480P/`. |
| `--output-dir PATH` | `./n2_leapp_export` | Directory for the LEAPP graph, ONNX model, external weights, and logs. |
| `--embodiment NAME` | `unitree_g1_whole_body_teleop_latent` | Embodiment configuration loaded from the checkpoint. |
| `--instruction TEXT` | `Pick up the red object from the table.` | Compile-time language instruction; it is not an ONNX input. |
| `--input-height INT` | `360` | Fixed exported camera height. |
| `--input-width INT` | `640` | Fixed exported camera width. |
| `--denoise-steps INT` | `1` | Number of action denoising steps. |
| `--video-denoise-steps INT` | `1` | Number of video denoising steps. |
| `--opset-version INT` | `22` | ONNX opset used by LEAPP. |
| `--joint-names PATH` | None | Optional JSON mapping flat state/action tensor names to ordered joint names. |
| `--skip-runtime-validation` | Off | Run ONNX schema validation but skip numerical ONNX Runtime replay. |
| `--validation-rtol FLOAT` | `0.02` | Relative tolerance for numerical replay. |
| `--validation-atol FLOAT` | `0.05` | Absolute tolerance for numerical replay. |

Both denoising step counts must be positive, and `--denoise-steps` must be
divisible by `--video-denoise-steps`. Camera size, embodiment, instruction,
attention pattern, and denoising configuration are fixed in the exported graph;
changing one requires another export.

The exporter rejects checkpoints configured with multiple stochastic attention
patterns because a deployment export must have one deterministic execution path.

### Optional joint-name metadata

Joint names are deployment metadata, not an export requirement. Without
`--joint-names`, LEAPP still exports and runs the model; robot-specific
`element_names` are simply omitted from the generated YAML. Conventional state
and action names ending in `_joints` still receive joint-position semantics;
ambiguous robot-specific tensors remain untyped rather than being mislabeled.

The [`joint_names/g1.json`](joint_names/g1.json) file is a template for the
validated G1 checkpoint. Pass it explicitly when exporting that checkpoint:

```bash
uv run --project groot/core --locked --extra g1-inference-n2 --extra leapp \
  python scripts/leapp/n2_export.py \
  --checkpoint "$CHECKPOINT" \
  --model-deps "$ASSET_ROOT" \
  --joint-names scripts/leapp/joint_names/g1.json \
  --skip-runtime-validation
```

For another robot, copy the template and replace its entries. Keys are exact
GR00T flat-observation or action names, and each value lists the corresponding
feature-axis joints in model order:

```json
{
  "state.arm": ["shoulder_joint", "elbow_joint"],
  "action.arm_joints": ["shoulder_joint", "elbow_joint"]
}
```

The exporter rejects unknown tensor keys and lists whose length does not match
the tensor's feature dimension. A URDF-wide joint list is not sufficient unless
it is first split and ordered to match each model tensor.

## 5. Inspect the result

The command writes:

```text
n2_leapp_export/
└── n2_get_actions/
    ├── log.txt
    ├── n2_get_actions.png
    ├── n2_get_actions.yaml
    ├── n2_get_actions.onnx
    └── n2_get_actions.onnx.data
```

Keep the ONNX protobuf and its `.onnx.data` file together. The validated G1
checkpoint produces approximately 3.8 GB of external ONNX data.

The generated YAML is the source of truth for input names, tensor dtypes, fixed
shapes, and outputs. Numeric input names are derived directly from GR00T's
flat-observation keys,
with dots replaced by underscores because Torch FX input placeholders must be
valid Python identifiers. For example, `video.ego_view` is exported as
`video_ego_view`; the original dotted keys are restored before `get_action`. The
language instruction is not an input because it is compiled into the export.

LEAPP semantic `kind` metadata follows the IsaacGr00t export convention:

- Camera images use `state/camera/image`.
- Joint-state tensors identified by `--joint-names` or a `_joints` suffix use
  `state/joint/position`.
- Projected gravity uses `state/vector3d`.
- Initial action diffusion noise currently has no semantic `kind`.
- Decoded joint outputs identified the same way use `target/joint/position`.
- Motion tokens have no semantic `kind`.

When `--joint-names` is provided, matching joint-state and joint-action tensors
include the supplied deployment names in `element_names`.

For the validated G1 checkpoint, the interface was:

```text
inputs:
  video_<modality>: uint8 [1, 360, 640, 3]
  state_<modality>: float32 tensors with checkpoint-defined widths
  initial_noise: float32 [1, 50, 128]

outputs:
  action.motion_token:       float32 [1, 40, 64]
  action.left_hand_joints:   float32 [1, 40, 7]
  action.right_hand_joints:  float32 [1, 40, 7]
```

Other checkpoints and embodiments can have different modality counts and action
outputs. Always inspect their generated YAML rather than assuming the example
interface.

## Validation and deployment notes

### Optional GPU numerical replay

The locked CI/user extra uses CPU ONNX Runtime. GPU replay is a separate local
environment adjustment, not part of the CI image: CPU and GPU distributions
share the `onnxruntime` import path and must not be installed over each other
implicitly during a sync. After the locked setup above, install the GPU build
matching the resolved CPU version:

```bash
ORT_VERSION=$(groot/core/.venv/bin/python -c 'from importlib.metadata import version; print(version("onnxruntime"))')
uv pip install --python groot/core/.venv/bin/python \
  --reinstall --no-deps "onnxruntime-gpu==$ORT_VERSION"
groot/core/.venv/bin/python -c 'import onnxruntime as ort; assert "CUDAExecutionProvider" in ort.get_available_providers(); print(ort.get_available_providers())'
```

This requires a compatible GPU wheel and CUDA/cuDNN libraries on the host.
Run the same export command with `uv run --project groot/core --no-sync` in
place of the locked `uv run` prefix, and omit `--skip-runtime-validation`.
Do not resync this adjusted environment before replay: a sync can remove the
GPU package or restore CPU package files. A listed CUDA provider is a prerequisite;
successful numerical replay is the actual runtime check.

### Lightweight tests

Run the lightweight CI tests with:

```bash
python -m pytest -q tests/scripts/leapp/test_n2_export.py
```

Like the N2 TensorRT package tests, these check numerical helpers, tensor
contracts, export orchestration, and failure cleanup using small CPU tensors and
mocked runtime calls. They do not load checkpoints or run LEAPP/ONNX inference.
Checkpoint export and numerical replay remain separate GPU checks using the
export command above. LEAPP replay compares against the patched capture; it is
not an independent original-policy parity test. That comparison still needs a
checkpoint-backed validation workflow with matching noise and deployment settings.

- `onnx.checker` schema validation always runs.
- Numerical ONNX Runtime replay runs only when the installed runtime exposes
  `CUDAExecutionProvider`. CPU ONNX Runtime cannot execute the N2 BF16
  convolution graph.
- The export is fixed shape. Changing camera resolution, embodiment, instruction,
  attention pattern, or denoising-step configuration requires another export.
- The exported graph is intended to reproduce model inference. ROS acquisition,
  message conversion, action limiting, safety checks, and command publication
  remain deployment-application responsibilities.
- Relative EEF pose conversion is excluded from the graph (see the warning above).
  Outputs retain their existing names, but affected outputs have no absolute-pose
  or joint-position semantic kind. Preserve `relative_eef_outputs.json` with the
  bundle: each chunk uses the last observed reference pose, not sequential deltas.
  Rotation repair/normalization and requested absolute-format conversion are also
  omitted; match GR00T's native pose-conversion behavior in the deployment adapter.
- The default G1 latent checkpoint is the currently validated configuration.
  Treat a new checkpoint or embodiment as a new export target and rerun validation.

The trace-time compatibility substitutions are isolated in
`n2_export_patch.py`. Resize, normalization, and attention use LEAPP's
`FunctionPatch` API, registered through `leapp.start(patching=...)` and restored by
`leapp.stop()`. These replacements run only on traced inputs. Class methods and
calls without traced arguments use a separate export-scoped context manager.
The substitutions express causal VAE padding as concatenation, calculate VAE RMS
reductions in FP32, and promote the initial action-noise matrix to a LEAPP input.
GR00T's core N2 inference files are not modified.
