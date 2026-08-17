# Isaac GR00T 2

## Start With an Officially Supported Agent

> [!WARNING]
> The agentic workflows in this repository are officially supported only with
> **Codex, Claude Code, and Cursor**. Other coding agents may be able to read the
> instructions, but are not officially supported or tested.

Start Codex, Claude Code, or Cursor in the package root, the directory
containing this README and `pyproject.toml`, then ask:

> Onboard me to Isaac GR00T 2 using the minimal end-to-end example. Set up the
> uv environment, prepare the example dataset, and verify training, open-loop
> evaluation, and the policy server.

The repository guidance and [`onboarding` skill](skills/onboarding/SKILL.md) give the agent
the supported workflow and success criteria. To run the same onboarding
manually, follow the
[minimal end-to-end example](docs/minimal_example.md).

This package exposes OSS-local post-training, open-loop evaluation, and policy
serving through `uv run --no-sync python -m groot.infra.oss` after installing
the training environment.

See the [agentic workflows and skills guide](skills/README.md) to choose among
the onboarding, AutoResearch, AutoResearch visualization, AutoDebug, and FAQ
workflows and start them with copy-ready prompts.

## Minimal End-to-End Example

The [minimal example](docs/minimal_example.md) uses a pinned, small public
LeRobot dataset to run one post-training step, a short open-loop evaluation,
and the policy server. It installs the locked environment directly with `uv`
and keeps data, caches, and outputs on explicit local paths.

TorchCodec requires FFmpeg shared libraries. On Ubuntu or Debian, install
FFmpeg before creating the uv environment:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
uv sync --frozen --extra training
```

When running as `root`, omit `sudo`.

Prepare the example dataset after completing the environment setup in the
walkthrough:

```bash
uv run --no-sync python examples/prepare_minimal_dataset.py \
  --output-dir "$HOME/gr00t2/data/gr00t_red_test"
```

Each workflow has a different input boundary:

| Workflow | Required inputs | Primary output |
| --- | --- | --- |
| Prepare dataset | Network access to the public dataset; writable dataset path | Local LeRobot dataset |
| Post-train | Local dataset, embodiment tag, accessible pretrain/model artifacts, GPU, writable output path | Local checkpoint |
| Open-loop eval | Local dataset, matching embodiment tag, local checkpoint or accessible HF URI, GPU | JSON evaluation report |
| Policy server | Matching embodiment tag, local checkpoint or accessible HF URI, GPU | TCP policy endpoint |
| Train and eval | All post-training and open-loop evaluation inputs | Checkpoint and JSON report |

See the detailed guide for the exact environment variables and command-line
arguments used for each input.

## Train

Run from the OSS package root with a local LeRobot dataset:

```bash
uv run --no-sync python -m groot.infra.oss train \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag>
```

For a smaller local smoke run:

```bash
uv run --no-sync python -m groot.infra.oss train \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag> \
  --num-gpus 8 \
  --batch-size 4 \
  --max-steps 2
```

The OSS CLI defaults to its public N2 training experiment, so no experiment
name is required. OSS training requires an explicit `--embodiment-tag` and
either `--dataset-path` or an intentional replacement of the complete
`data.datasets` list with `--set`.

## Evaluate

Run open-loop evaluation for an existing checkpoint:

```bash
uv run --no-sync python -m groot.infra.oss eval \
  --checkpoint <hf-uri-or-local-path> \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag>
```

Run post-training followed by evaluation:

```bash
uv run --no-sync python -m groot.infra.oss train-eval \
  --dataset-path /path/to/lerobot_dataset \
  --embodiment-tag <dataset-embodiment-tag>
```

Use `--dry-run` with any command to print the resolved command or evaluation
configuration without starting training or evaluation.

## Policy Server

Start a GR00T policy server for a local or Hugging Face checkpoint:

```bash
uv run --no-sync python -m groot.infra.oss policy-server \
  --checkpoint <hf-uri-or-local-path> \
  --host 0.0.0.0 \
  --port 5555
```

Use `--dry-run` to print the resolved core server command without starting the
server.

## Model Artifacts

By default the OSS training entrypoint loads GR00T 2 model dependencies from
Hugging Face:

- `hf://model/nvidia/Cosmos-Reason1-7B@375e24000b24baed78f4618d3dd779e47cd96323`
- `hf://model/nvidia/Cosmos-Predict2.5-2B@0d37c7498f54cee3c599d438d895a0a4a8608064/transformer`
- `hf://model/Wan-AI/Wan2.1-I2V-14B-480P@6b73f84e66371cdfe870c72acd6826e1d61cf279/Wan2.1_VAE.pth`
- `hf://model/nvidia/GR00T2Internal@4fe662abaf302ee46916881c447377da8c4bd15a/base_200k`

Pass `--pretrain-checkpoint` if you want to override the default pretrain
checkpoint with another `hf://model/...` URI or a local checkpoint path.
The default `base_200k` checkpoint and OSS training configuration both retain
all 32 embodiment-specific robot heads. The dataset's `--embodiment-tag`
selects the head used by each training example.

Set `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` when training
or evaluation needs to download a private Hugging Face checkpoint or model
dependency.

## Boundary

The OSS package includes reusable launcher and evaluation code:

- `groot.infra.launcher`
- `groot.infra.experiments.training.configs.n2_oss_pretrain`
- `groot.infra.experiments.training.configs._n2_base_*`
- `groot.infra.experiments.training.configs._n2_model_config`
- `groot.infra.eval`
- `groot.infra.oss`
- `groot.infra.job.data_credentials` (temporary symlink for hydrator credential loading)

TODO: remove the temporary `groot.infra.job.data_credentials` symlink once the
hydrator credential helper no longer lives under job infra.

It intentionally does not package the rest of `groot.infra.job`,
`groot.infra.ci`, or `groot.internal`, which contain internal GitLab, OSMO, and
cluster submission code.
