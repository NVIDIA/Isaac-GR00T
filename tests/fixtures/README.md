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
| `processor_config/` | `tests/gr00t/model/test_gr00t_processor.py`<br>`tests/gr00t/model/test_variable_image_size.py`<br>`tests/gr00t/policy/test_gr00t_policy.py`<br>`tests/gr00t/data/state_action/test_state_action_processor.py` | Minimal `Gr00tN1d7Processor` config (libero_sim only) |
