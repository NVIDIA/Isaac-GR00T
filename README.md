# Isaac-GR00T-N2

This package exposes OSS-local post-training, open-loop evaluation, and policy
serving through `python -m groot.infra.oss`.

## AutoDebug

AutoDebug reproduces a GR00T problem, tests focused explanations, writes a fix package
for a separate coding agent, and independently verifies the returned commit. Start with
[the AutoDebug guide](.agents/autodebug.md).

## Train

Run from the OSS package root with a local LeRobot dataset:

```bash
uv run python -m groot.infra.oss train n2_yam_simulation_stack_bowls \
  --dataset-path /path/to/lerobot_dataset
```

For a smaller local smoke run:

```bash
uv run python -m groot.infra.oss train n2_yam_simulation_stack_bowls \
  --dataset-path /path/to/lerobot_dataset \
  --num-gpus 8 \
  --batch-size 4 \
  --max-steps 2
```

The OSS CLI takes the one-launcher experiment name directly. The documented OSS
training path exposes only `n2_yam_simulation_stack_bowls`. OSS training
requires `--dataset-path` unless you intentionally override
`data.datasets.*.dataset_paths` with `--set`.

## Evaluate

Run open-loop evaluation for an existing checkpoint:

```bash
uv run python -m groot.infra.oss eval n2_yam_simulation_stack_bowls \
  --checkpoint <hf-uri-or-local-path> \
  --dataset-path /path/to/lerobot_dataset
```

Run post-training followed by evaluation:

```bash
uv run python -m groot.infra.oss train-eval n2_yam_simulation_stack_bowls \
  --dataset-path /path/to/lerobot_dataset
```

`train-eval` uses the same `--dataset-path` for training and evaluation. Run
`train` and `eval` separately when evaluating the new checkpoint on a different
held-out dataset.

Use `--dry-run` with any command to print the resolved command or evaluation
configuration without starting training or evaluation.

## Policy Server

Start a GR00T policy server for a local or Hugging Face checkpoint:

```bash
uv run python -m groot.infra.oss policy-server \
  --checkpoint <hf-uri-or-local-path> \
  --host 0.0.0.0 \
  --port 5555
```

Use `--dry-run` to print the effective server configuration without starting
the server.

## Model Artifacts

By default the OSS training entrypoint loads N2 model dependencies from Hugging
Face:

- `hf://model/nvidia/Cosmos-Reason1-7B@375e24000b24baed78f4618d3dd779e47cd96323`
- `hf://model/nvidia/Cosmos-Predict2.5-2B@0d37c7498f54cee3c599d438d895a0a4a8608064/transformer`
- `hf://model/Wan-AI/Wan2.1-I2V-14B-480P@6b73f84e66371cdfe870c72acd6826e1d61cf279/Wan2.1_VAE.pth`
- `hf://model/nvidia/GR00T2Internal@4fe662abaf302ee46916881c447377da8c4bd15a/base_200k`

Pass `--pretrain-checkpoint` if you want to override the default pretrain
checkpoint with another `hf://model/...` URI or a local checkpoint path.

Set `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` when training
or evaluation needs to download a private Hugging Face checkpoint or model
dependency.

## Boundary

The OSS package links reusable launcher and evaluation code only:

- `groot.infra.launcher`
- `groot.infra.experiments.training.configs.n2_yam_simulation_stack_bowls`
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
