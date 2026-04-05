# AGENTS.md — Agent Workflows for Isaac GR00T

This document describes end-to-end workflows that AI coding agents (Claude Code, Copilot, Cursor, etc.) can follow when working on this repo. Each workflow lists the steps and copy-pasteable commands.

## Workflow 1: Bug fix in core library

1. **Reproduce** — Run the failing test or script to confirm the bug.
2. **Locate** — Search `gr00t/` for the relevant module.
3. **Fix** — Edit the source.
4. **Lint** — `pre-commit run --all-files`
5. **Test** — `python -m pytest tests/ -m "not gpu" -v --timeout=300 -k <test_name>`
6. **Verify** — Re-run the originally failing test.

## Workflow 2: Add or update an embodiment example

1. **Check existing** — Review `examples/` for a similar embodiment.
2. **Create config** — Add a modality config under `examples/<name>/` following the pattern in `examples/SO100/`.
3. **Register tag** — If new, add to `gr00t/data/embodiment_tags.py`.
4. **Test** — `python -m pytest tests/examples/ -v -k <example_name>`
5. **Lint** — `pre-commit run --all-files`

## Workflow 3: Fine-tuning pipeline change

1. **Read** — `gr00t/experiment/launch_finetune.py`, `gr00t/experiment/trainer.py`, `gr00t/configs/finetune_config.py`
2. **Edit** — Make changes to the training loop or config.
3. **Quick test** — `python -m pytest tests/ -m "not gpu" -v -k finetune --timeout=300`
4. **GPU test** — `python -m pytest tests/ -m gpu -v -k finetune --timeout=300`

## Workflow 4: Deployment script update (Thor / Spark / Orin / dGPU)

1. **Read** — `scripts/deployment/<platform>/install_deps.sh` and `scripts/deployment/<platform>/Dockerfile`
2. **Edit** — Update dependencies or build steps.
3. **Validate lockfile** — If `pyproject.toml` changed: `cd scripts/deployment/<platform> && uv lock --locked`
4. **Lint** — `pre-commit run --all-files`

## Workflow 5: CI pipeline change

1. **Read** — `.gitlab-ci.yml` and the relevant `ci/*.gitlab-ci.yml` file.
2. **Edit** — Update job definitions.
3. **Validate** — Check syntax: the CI uses modular includes, so verify the `include:` paths are correct.
4. **Test locally** — Run the commands the CI job would run (see `ci/README.md` for details).

## Workflow 6: TensorRT / ONNX export change

1. **Read** — `scripts/deployment/export_onnx_n1d7.py`, `scripts/deployment/build_trt_pipeline.py`
2. **Edit** — Update export or build logic.
3. **Test** — `python -m pytest tests/examples/deployment/ -v --timeout=300`

## Common commands reference

```bash
# Environment setup
uv sync --all-extras

# Lint + format
pre-commit run --all-files
ruff check --fix gr00t/ tests/
ruff format gr00t/ tests/

# Tests
python -m pytest tests/ -m "not gpu" -v --timeout=300        # CPU tests
python -m pytest tests/ -m gpu -v --timeout=300               # GPU tests
python -m pytest tests/ -v -k <pattern> --timeout=300         # Filtered

# Build validation
uv lock --locked
uv build
python scripts/generate_attributions.py --check

# Fine-tuning (SO100 example)
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/cube_to_bowl_5 \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/SO100/modality_config.py \
  --output-dir output/so100_finetune

# Inference server
python gr00t/eval/run_gr00t_server.py \
  --model-path <checkpoint_path> \
  --embodiment-tag <tag> \
  --port 5555

# Deployment
python scripts/deployment/export_onnx_n1d7.py --model-path <path> --output-dir <dir>
python scripts/deployment/build_trt_pipeline.py --onnx-dir <dir> --output-dir <dir>
python scripts/deployment/benchmark_inference.py --model-path <path>
```
