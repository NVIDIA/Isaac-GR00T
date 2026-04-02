from __future__ import annotations

from test_support.readme import extract_code_blocks, find_block, run_readme_python_blocks
from test_support.runtime import get_root


REPO_ROOT = get_root()
REAL_WORLD_README = REPO_ROOT / "getting_started" / "real_world_deployment.md"


def test_quantitative_metrics() -> None:
    """Run all three quantitative diagnostic metrics from real_world_deployment.md."""
    blocks = extract_code_blocks(REAL_WORLD_README)

    intra_accel = find_block(blocks, "def metric_intra_accel", language="python")
    boundary_jump = find_block(blocks, "def metric_boundary_jump", language="python")
    momentum_shift = find_block(blocks, "def metric_momentum_shift", language="python")

    run_readme_python_blocks(
        [
            "import numpy as np",
            intra_accel,
            boundary_jump,
            momentum_shift,
            # exercise each function with compatible dummy data
            "chunks = np.random.randn(3, 5, 4)",
            "assert isinstance(metric_intra_accel(chunks), float)",
            "assert isinstance(metric_boundary_jump(chunks), float)",
            "assert isinstance(metric_momentum_shift(chunks, execute_steps=3), float)",
        ],
        readme_path=REAL_WORLD_README,
        repo_root=REPO_ROOT,
    )
