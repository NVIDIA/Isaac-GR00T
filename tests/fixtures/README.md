# Test Fixtures

Small, static files committed to the repo so that CI can run tests without
external dependencies (checkpoints, datasets, network access, etc.).

## Guidelines

- Keep files small — configs, sample data, mock inputs, etc. are all fine.
- Do **not** commit large binaries (model weights, `.safetensors`, `.bin`).
- Organize by feature or test in descriptive subdirectories.
- Reference in tests via relative path: `Path(__file__).parent... / "fixtures" / "subdir"`.

## Current fixtures

| Directory | Used by | Description |
|-----------|---------|-------------|
| `processor_config/` | `tests/gr00t/model/test_variable_image_size.py` | Minimal `Gr00tN1d6Processor` config (libero_panda only) |
