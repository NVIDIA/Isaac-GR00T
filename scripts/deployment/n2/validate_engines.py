#!/usr/bin/env python3
"""Validate a built N2 TensorRT bundle and emit its admitted manifest."""

from __future__ import annotations

from groot.deployment.n2_trt.validation import main

if __name__ == "__main__":
    raise SystemExit(main())
