from __future__ import annotations

import hashlib
from pathlib import Path


FROZEN_SHA256 = {
    "eval_toppra_bimanual.py": ("c655a6aaacb618c04f65aea07263761e4b00e703fca91cebed4ad3d45d9606e0"),
    "kion_client/README.md": ("eba9c38c715ae61a19e046c01e09940389616ffba91bf9730ffb4517be86d6e9"),
    "kion_client/__init__.py": ("259b654ab74babddc9e1bc2a9c722c9aa719a368471df9842b380623ca151b3a"),
    "kion_client/__main__.py": ("82f022b57cc0118fbe43c242c633f4b2a2b7fc6626506ce4d07ea7084ca687e0"),
    "kion_client/analyze_tracking.py": (
        "a2bae61a81634f5500e01ee832df9edd553dbffb7e129551eee0762847da2554"
    ),
    "kion_client/client.py": ("a39e6b37d2b427371617a6951706a26b429334eba1adc0c633c7fc1f08035dd8"),
    "kion_client/tracking.py": ("764e1dde2d2626b9b29204b2f088a4d3b22c8a48bee2c06fb0f6f826b4ef0b6c"),
}


def verify_frozen_baseline() -> dict[str, str]:
    """Raise if the rollout or existing Kion client differs from the captured baseline."""

    toppra_root = Path(__file__).resolve().parent.parent
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for relative_path, expected in FROZEN_SHA256.items():
        path = toppra_root / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[relative_path] = digest
        if digest != expected:
            mismatches.append(f"{relative_path}: expected {expected}, received {digest}")
    if mismatches:
        raise RuntimeError("Frozen TOPPRA baseline changed:\n" + "\n".join(mismatches))
    return actual
