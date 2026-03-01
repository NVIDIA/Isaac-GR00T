import subprocess
import sys
from pathlib import Path


def test_validate_dataset_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "validate_dataset.py"
    demo = repo_root / "demo_data" / "cube_to_bowl_5"
    assert script.exists(), f"Script not found: {script}"
    assert demo.exists(), f"Demo dataset not found: {demo}"

    # Run the script and ensure it exits successfully and prints a success marker
    proc = subprocess.run([sys.executable, str(script), str(demo)], capture_output=True)
    out = proc.stdout.decode(errors="ignore") + proc.stderr.decode(errors="ignore")
    assert proc.returncode == 0, f"validate_dataset.py exited with {proc.returncode}\nOutput:\n{out}"
    assert "ALL VALIDATION CHECKS PASSED" in out
