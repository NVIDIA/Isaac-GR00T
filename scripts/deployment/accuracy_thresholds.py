"""Centralized accuracy thresholds for TRT deployment validation.

Single source of truth for all cosine similarity and L1 thresholds used
in benchmark_inference.py (compare modes) and export_onnx_n1d6.py (ViT
wrapper verification).  Import from here instead of hardcoding values.

See also: scripts/deployment/run_trt_pipeline.sh (acceptance criteria).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyThreshold:
    """Immutable pair of cosine-similarity minimum and optional L1 maximum."""

    cosine_min: float
    l1_max: float | None  # None = not checked


# ── BF16 thresholds ──────────────────────────────────────────────────────────

BF16_BACKBONE = AccuracyThreshold(cosine_min=0.999, l1_max=0.01)
BF16_ACTION_PRED = AccuracyThreshold(cosine_min=0.99, l1_max=0.05)
BF16_E2E_ACTION = AccuracyThreshold(cosine_min=0.99, l1_max=None)

# Per-component pass/warn/fail boundaries (detailed compare mode)
BF16_COMPONENT_PASS = 0.999  # cosine >= this => PASS
BF16_COMPONENT_WARN = 0.99  # cosine >= this => WARN, else FAIL

# ViT wrapper verification (export_onnx_n1d6.py)
VIT_WRAPPER_COSINE_MIN = 0.999

# ── FP8 thresholds (relaxed — more drift expected, especially in ViT) ────────

FP8_BACKBONE = AccuracyThreshold(cosine_min=0.99, l1_max=0.05)
FP8_ACTION_PRED = AccuracyThreshold(cosine_min=0.98, l1_max=0.1)
FP8_E2E_ACTION = AccuracyThreshold(cosine_min=0.98, l1_max=None)

FP8_COMPONENT_PASS = 0.99
FP8_COMPONENT_WARN = 0.98
