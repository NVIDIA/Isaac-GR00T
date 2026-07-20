# Isaac-GR00T-N2

This package exposes OSS-local post-training and open-loop evaluation through
`python -m groot.infra.oss`.

## Train

Run from the OSS package root:

```bash
uv run python -m groot.infra.oss train n2_yam_simulation_stack_bowls
```

For a smaller local smoke run:

```bash
uv run python -m groot.infra.oss train n2_yam_simulation_stack_bowls \
  --num-gpus 8 \
  --batch-size 4 \
  --max-steps 2
```

The OSS CLI takes the one-launcher experiment name directly. The documented OSS
training path uses `n2_yam_simulation_stack_bowls`.

## Evaluate

Run open-loop evaluation for an existing checkpoint:

```bash
uv run python -m groot.infra.oss eval n2_yam_simulation_stack_bowls --checkpoint <hf-uri-or-local-path>
```

Run post-training followed by evaluation:

```bash
uv run python -m groot.infra.oss train-eval n2_yam_simulation_stack_bowls
```

Use `--dry-run` with any command to print the resolved command or evaluation
configuration without starting training or evaluation.

## Required Environment

Local training requires model dependency paths:

```bash
export COSMOS_MODEL_PATH=<local-cosmos-tokenizer-path>
export DIFFUSION_TRANSFORMER_PATH=<local-transformer-path>
export VAE_PATH=<local-vae-path>
```

Set `PRETRAIN_CHECKPOINT` or pass `--pretrain-checkpoint` if you want to
override the checkpoint baked into the experiment config.

Set `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` when evaluation
needs to download a private Hugging Face checkpoint or dataset.

## Boundary

The OSS package links reusable launcher and evaluation code only:

- `groot.infra.launcher`
- `groot.infra.experiments`
- `groot.infra.eval`
- `groot.infra.oss`
- `groot.infra.job.data_credentials` (temporary symlink for hydrator credential loading)

TODO: remove the temporary `groot.infra.job.data_credentials` symlink once the
hydrator credential helper no longer lives under job infra.

It intentionally does not package the rest of `groot.infra.job`,
`groot.infra.ci`, or `groot.internal`, which contain internal GitLab, OSMO, and
cluster submission code.
