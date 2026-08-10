#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Embedded HTML lines, validation errors, and CLI output are intentional here.
# ruff: noqa: E501, T201, TRY004
"""Render an AutoResearch state.json as a self-contained interactive HTML report."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import html
import json
import math
from pathlib import Path
import textwrap
from typing import Any

NODE_WIDTH = 264
NODE_HEIGHT = 170
BASELINE_WIDTH = 224
BASELINE_HEIGHT = 132
COLUMN_GAP = 96
ROW_GAP = 48
GRAPH_MARGIN_X = 56
GRAPH_MARGIN_Y = 76


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --ink: #172033;
    --muted: #667085;
    --line: #d8dee9;
    --panel: #ffffff;
    --canvas: #f4f7fb;
    --navy: #0b1730;
    --navy-soft: #15284d;
    --accent: #5b8cff;
    --gold: #e0a100;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--canvas);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
      sans-serif;
  }
  body.drawer-open { overflow: hidden; }
  .hero {
    color: white;
    background:
      radial-gradient(circle at 86% 10%, rgba(91, 140, 255, .32), transparent 28rem),
      linear-gradient(135deg, var(--navy), var(--navy-soft));
    padding: 42px clamp(24px, 5vw, 72px) 38px;
  }
  .hero-inner, main { max-width: 1680px; margin: 0 auto; }
  .eyebrow {
    color: #a8c2ff;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: .14em;
    text-transform: uppercase;
  }
  h1 {
    margin: 10px 0 8px;
    font-size: clamp(30px, 4vw, 50px);
    line-height: 1.08;
    letter-spacing: -.035em;
  }
  .hero-subtitle {
    margin: 0;
    color: #c9d6ed;
    font-size: 16px;
  }
  .hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border: 1px solid rgba(255, 255, 255, .18);
    border-radius: 999px;
    background: rgba(255, 255, 255, .08);
    color: #e9effa;
    padding: 7px 11px;
    font-size: 12px;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #7cc7ff;
  }
  .status-dot.complete { background: #5ee39a; }
  .status-dot.attention { background: #ffbd66; }
  main { padding: 28px clamp(18px, 4vw, 56px) 64px; }
  .metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 14px;
    margin-bottom: 28px;
  }
  .metric-card {
    min-height: 118px;
    border: 1px solid #e1e6ef;
    border-radius: 16px;
    background: var(--panel);
    padding: 18px 19px;
    box-shadow: 0 7px 22px rgba(18, 35, 68, .055);
  }
  .metric-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 750;
    letter-spacing: .075em;
    text-transform: uppercase;
  }
  .metric-value {
    margin-top: 10px;
    font-size: 26px;
    font-weight: 760;
    line-height: 1.05;
    letter-spacing: -.025em;
    overflow-wrap: anywhere;
  }
  .metric-note {
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.35;
  }
  .section-heading {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    margin: 4px 2px 14px;
  }
  .section-heading h2 {
    margin: 0;
    font-size: 22px;
    letter-spacing: -.02em;
  }
  .section-heading p {
    margin: 5px 0 0;
    color: var(--muted);
    font-size: 13px;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 9px 15px;
    color: var(--muted);
    font-size: 12px;
  }
  .legend-item { display: inline-flex; align-items: center; gap: 6px; }
  .swatch {
    width: 18px;
    height: 10px;
    border-radius: 3px;
    border: 1px solid rgba(23, 32, 51, .16);
  }
  .swatch.green { background: #65bd87; }
  .swatch.red { background: #e48680; }
  .swatch.gray { background: #edf1f6; }
  .line-sample {
    width: 23px;
    border-top: 2px solid #8691a5;
  }
  .line-sample.composition {
    border-color: #8b5cf6;
    border-top-style: dashed;
  }
  .graph-shell {
    overflow: hidden;
    border: 1px solid #dfe5ef;
    border-radius: 19px;
    background: white;
    box-shadow: 0 10px 32px rgba(18, 35, 68, .07);
  }
  .graph-scroll {
    overflow: auto;
    max-height: min(72vh, 980px);
    background:
      linear-gradient(#ffffff, #ffffff) padding-box,
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 23px,
        rgba(51, 65, 85, .026) 24px
      ),
      repeating-linear-gradient(
        90deg,
        transparent,
        transparent 23px,
        rgba(51, 65, 85, .026) 24px
      );
  }
  svg { display: block; margin: 0 auto; }
  .edge {
    fill: none;
    stroke: #9aa5b7;
    stroke-width: 2;
    opacity: .85;
  }
  .edge.composition {
    stroke: #8b5cf6;
    stroke-dasharray: 7 6;
  }
  .wave-guide {
    stroke: #e4e9f1;
    stroke-width: 1;
    stroke-dasharray: 3 7;
  }
  .wave-title {
    fill: #475467;
    font-size: 11px;
    font-weight: 800;
  }
  .wave-state {
    fill: #8a94a6;
    font-size: 9px;
    font-weight: 700;
  }
  .wave-title.active, .wave-state.active { fill: #2563eb; }
  .node-group { cursor: pointer; outline: none; }
  .node-card {
    stroke-width: 2;
    transition: filter .16s ease, stroke-width .16s ease;
    filter: drop-shadow(0 5px 7px rgba(16, 32, 61, .11));
  }
  .node-group:hover .node-card,
  .node-group:focus .node-card {
    stroke-width: 3.5;
    filter: drop-shadow(0 9px 12px rgba(16, 32, 61, .18));
  }
  .node-kicker {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .11em;
  }
  .node-title {
    font-size: 15px;
    font-weight: 760;
  }
  .node-copy { font-size: 12px; }
  .node-score {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.02em;
  }
  .node-delta { font-size: 12px; font-weight: 700; }
  .node-stage { font-size: 10px; font-weight: 650; }
  .best-ring {
    fill: none;
    stroke: var(--gold);
    stroke-width: 4;
  }
  .best-badge {
    fill: #fff4c7;
    stroke: #e0a100;
    stroke-width: 1;
  }
  .best-badge-text {
    fill: #795700;
    font-size: 9px;
    font-weight: 850;
    letter-spacing: .075em;
  }
  .drawer-overlay {
    position: fixed;
    inset: 0;
    z-index: 20;
    background: rgba(5, 14, 31, .42);
    opacity: 0;
    pointer-events: none;
    transition: opacity .18s ease;
  }
  .drawer {
    position: fixed;
    z-index: 21;
    top: 0;
    right: 0;
    width: min(590px, 94vw);
    height: 100vh;
    overflow-y: auto;
    background: white;
    box-shadow: -18px 0 50px rgba(6, 18, 40, .22);
    transform: translateX(102%);
    transition: transform .22s ease;
  }
  .drawer.open { transform: translateX(0); }
  .drawer-overlay.open { opacity: 1; pointer-events: auto; }
  .drawer-header {
    position: sticky;
    top: 0;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    border-bottom: 1px solid #e5e9f0;
    background: rgba(255, 255, 255, .96);
    backdrop-filter: blur(12px);
    padding: 19px 22px;
  }
  .drawer-title {
    margin: 0;
    font-size: 18px;
    line-height: 1.25;
  }
  .close-button {
    flex: 0 0 auto;
    width: 36px;
    height: 36px;
    border: 1px solid #d9e0ea;
    border-radius: 10px;
    background: #f8fafc;
    color: #344054;
    cursor: pointer;
    font-size: 21px;
  }
  .drawer-body { padding: 8px 22px 42px; }
  .detail-section {
    border-bottom: 1px solid #e9edf3;
    padding: 19px 0;
  }
  .detail-section:last-child { border-bottom: 0; }
  .detail-section h3 {
    margin: 0 0 10px;
    font-size: 12px;
    letter-spacing: .075em;
    text-transform: uppercase;
    color: #475467;
  }
  .detail-section p {
    margin: 0;
    line-height: 1.55;
    white-space: pre-wrap;
  }
  .detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }
  .detail-field {
    min-width: 0;
    border-radius: 10px;
    background: #f7f9fc;
    padding: 10px 11px;
  }
  .detail-field dt {
    color: var(--muted);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .06em;
    text-transform: uppercase;
  }
  .detail-field dd {
    margin: 5px 0 0;
    font-size: 13px;
    font-weight: 650;
    overflow-wrap: anywhere;
  }
  .detail-list {
    margin: 0;
    padding-left: 20px;
    line-height: 1.55;
  }
  .detail-list li + li { margin-top: 5px; }
  pre {
    margin: 0;
    overflow: auto;
    border-radius: 10px;
    background: #101828;
    color: #e5edfb;
    padding: 12px;
    font-size: 11px;
    line-height: 1.45;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .empty-note {
    padding: 58px 24px;
    color: var(--muted);
    text-align: center;
  }
  footer {
    margin-top: 18px;
    color: #7b8496;
    font-size: 11px;
    text-align: right;
  }
  @media (max-width: 760px) {
    .hero { padding-top: 30px; }
    .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .metric-card { min-height: 104px; padding: 14px; }
    .metric-value { font-size: 21px; }
    .section-heading { align-items: flex-start; flex-direction: column; }
    .legend { justify-content: flex-start; }
    .detail-grid { grid-template-columns: 1fr; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition: none !important; }
  }
</style>
</head>
<body>
<header class="hero">
  <div class="hero-inner">
    <div class="eyebrow">GR00T AutoResearch</div>
    <h1>__STUDY_ID__</h1>
    <p class="hero-subtitle">Interactive experiment lineage and results report</p>
    <div class="hero-meta">__HERO_META__</div>
  </div>
</header>
<main>
  <section class="metrics" aria-label="Study summary">__METRICS__</section>
  <div class="section-heading">
    <div>
      <h2>Experiment lineage</h2>
      <p>Click any node for the full hypothesis, code, stage, and artifact details.</p>
    </div>
    <div class="legend" aria-label="Diagram legend">
      <span class="legend-item"><span class="swatch green"></span>Improved</span>
      <span class="legend-item"><span class="swatch red"></span>Regressed</span>
      <span class="legend-item"><span class="swatch gray"></span>Unscored / neutral</span>
      <span class="legend-item"><span class="line-sample"></span>Parent</span>
      <span class="legend-item"><span class="line-sample composition"></span>Composed change</span>
    </div>
  </div>
  <section class="graph-shell">
    <div class="graph-scroll">__GRAPH__</div>
  </section>
  <footer>Generated from __SOURCE_PATH__ · Colors are normalized to the largest absolute parent-relative score change.</footer>
</main>
<div class="drawer-overlay" id="drawer-overlay"></div>
<aside class="drawer" id="drawer" aria-hidden="true" aria-label="Trial details">
  <div class="drawer-header">
    <h2 class="drawer-title" id="drawer-title">Trial details</h2>
    <button class="close-button" id="drawer-close" type="button" aria-label="Close details">×</button>
  </div>
  <div class="drawer-body" id="drawer-body"></div>
</aside>
__DETAIL_TEMPLATES__
<script>
(function () {
  "use strict";
  var drawer = document.getElementById("drawer");
  var overlay = document.getElementById("drawer-overlay");
  var title = document.getElementById("drawer-title");
  var body = document.getElementById("drawer-body");
  var closeButton = document.getElementById("drawer-close");
  var lastTrigger = null;

  function openDrawer(trigger) {
    var key = trigger.getAttribute("data-detail-id");
    var template = document.getElementById("detail-" + key);
    if (!template) return;
    lastTrigger = trigger;
    title.textContent = trigger.getAttribute("data-detail-title") || "Trial details";
    body.replaceChildren(template.content.cloneNode(true));
    drawer.classList.add("open");
    overlay.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    document.body.classList.add("drawer-open");
    closeButton.focus();
  }

  function closeDrawer() {
    drawer.classList.remove("open");
    overlay.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    document.body.classList.remove("drawer-open");
    if (lastTrigger) lastTrigger.focus();
  }

  document.querySelectorAll("[data-detail-id]").forEach(function (trigger) {
    trigger.addEventListener("click", function () { openDrawer(trigger); });
    trigger.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openDrawer(trigger);
      }
    });
  });
  closeButton.addEventListener("click", closeDrawer);
  overlay.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && drawer.classList.contains("open")) closeDrawer();
  });
})();
</script>
</body>
</html>
"""


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Render AutoResearch state.json as interactive HTML."
    )
    parser.add_argument("state_json", type=Path, help="Path to AutoResearch state.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: state.visualization.html beside state.json)",
    )
    return parser


def load_state(path: Path) -> dict[str, Any]:
    """Load and minimally validate an AutoResearch state file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"State file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"State file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("AutoResearch state root must be a JSON object")
    for key in ("study_id", "baseline_score", "best_score", "direction", "trials"):
        if key not in data:
            raise ValueError(f"AutoResearch state is missing required key: {key}")
    if not isinstance(data["study_id"], str) or not data["study_id"].strip():
        raise ValueError("state.study_id must be a non-empty string")
    _as_number(data["baseline_score"], "state.baseline_score")
    _as_number(data["best_score"], "state.best_score")
    if not isinstance(data["trials"], list):
        raise ValueError("state.trials must be a list")
    return data


def build_report(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build graph layout, score semantics, and summary data."""
    study_id = str(state["study_id"])
    baseline_score = _as_number(state["baseline_score"], "state.baseline_score")
    best_score = _as_number(state["best_score"], "state.best_score")
    raw_trials = state["trials"]
    if not isinstance(raw_trials, Sequence) or isinstance(raw_trials, str | bytes):
        raise ValueError("state.trials must be a list")

    direction = state["direction"]
    if direction not in ("minimize", "maximize"):
        raise ValueError("state.direction must be 'minimize' or 'maximize'")

    trials: list[dict[str, Any]] = []
    trials_by_id: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(raw_trials, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"state.trials[{position - 1}] must be an object")
        trial_id = raw.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError(f"state.trials[{position - 1}].trial_id must be a string")
        if trial_id in trials_by_id:
            raise ValueError(f"Duplicate trial ID: {trial_id}")
        trial = {
            "raw": dict(raw),
            "trial_id": trial_id,
            "index": raw.get("index", position),
            "wave_index": raw.get("wave_index", 0),
            "parent_trial_id": raw.get("parent_trial_id"),
            "composition_source_trial_id": raw.get("composition_source_trial_id"),
            "score": _optional_number(raw.get("score"), f"{trial_id}.score"),
            "parent_score": _optional_number(raw.get("parent_score"), f"{trial_id}.parent_score"),
        }
        trials.append(trial)
        trials_by_id[trial_id] = trial

    for trial in trials:
        parent_id = trial["parent_trial_id"]
        if parent_id is not None and parent_id not in trials_by_id:
            raise ValueError(f"Trial {trial['trial_id']} references unknown parent {parent_id}")
        source_id = trial["composition_source_trial_id"]
        if source_id is not None and source_id not in trials_by_id:
            raise ValueError(
                f"Trial {trial['trial_id']} references unknown composition source {source_id}"
            )

    depth_cache: dict[str, int] = {}

    def depth(trial: Mapping[str, Any], visiting: set[str]) -> int:
        trial_id = str(trial["trial_id"])
        if trial_id in depth_cache:
            return depth_cache[trial_id]
        if trial_id in visiting:
            raise ValueError(f"Cycle detected in trial parents at {trial_id}")
        parent_id = trial["parent_trial_id"]
        value = (
            1
            if parent_id is None
            else depth(trials_by_id[str(parent_id)], {*visiting, trial_id}) + 1
        )
        depth_cache[trial_id] = value
        return value

    signed_changes: list[float] = []
    for trial in trials:
        raw = trial["raw"]
        score = trial["score"]
        reference = trial["parent_score"]
        if reference is None:
            parent_id = trial["parent_trial_id"]
            reference = (
                trials_by_id[parent_id]["score"] if parent_id is not None else baseline_score
            )
        trial["reference_score"] = reference
        trial["depth"] = depth(trial, set())
        change = _signed_improvement(
            score=score,
            reference=reference,
            direction=direction,
            improved_flag=raw.get("improved_over_parent"),
        )
        trial["improvement"] = change
        if change is not None and change != 0:
            signed_changes.append(change)

    max_abs_change = max((abs(value) for value in signed_changes), default=0.0)
    for trial in trials:
        raw = trial["raw"]
        improvement = trial["improvement"]
        status = _trial_result_status(raw, improvement)
        intensity = (
            0.0
            if improvement is None or improvement == 0 or max_abs_change == 0
            else max(0.28, math.sqrt(abs(improvement) / max_abs_change))
        )
        fill, stroke, text_color = _node_colors(status, intensity)
        trial.update(
            {
                "status": status,
                "intensity": intensity,
                "fill": fill,
                "stroke": stroke,
                "text_color": text_color,
                "summary_text": _trial_summary(raw),
                "score_text": _format_score(trial["score"]),
                "change_text": _format_change(improvement),
                "detail_key": f"trial-{trial['index']}",
            }
        )

    legacy_depths = [trial["depth"] for trial in trials if trial["wave_index"] == 0]
    legacy_column_count = max(legacy_depths, default=0)
    has_waves = any(trial["wave_index"] > 0 for trial in trials)
    columns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trial in sorted(trials, key=lambda item: (_sort_int(item["index"]), item["trial_id"])):
        wave_index = trial["wave_index"]
        column = int(trial["depth"]) if wave_index == 0 else legacy_column_count + int(wave_index)
        trial["layout_column"] = column
        columns[column].append(trial)

    max_rows = max((len(items) for items in columns.values()), default=1)
    graph_height = max(
        430,
        GRAPH_MARGIN_Y * 2 + max_rows * NODE_HEIGHT + max(0, max_rows - 1) * ROW_GAP,
    )
    max_depth = max(columns, default=0)
    graph_width = GRAPH_MARGIN_X * 2 + BASELINE_WIDTH + max_depth * (NODE_WIDTH + COLUMN_GAP) + 18
    baseline_position = {
        "x": GRAPH_MARGIN_X,
        "y": (graph_height - BASELINE_HEIGHT) / 2,
    }
    positions: dict[str, dict[str, float]] = {}
    column_metadata: list[dict[str, Any]] = []
    active_wave = state.get("active_wave_index")
    for column, items in columns.items():
        column_height = len(items) * NODE_HEIGHT + max(0, len(items) - 1) * ROW_GAP
        start_y = (graph_height - column_height) / 2
        x = GRAPH_MARGIN_X + BASELINE_WIDTH + COLUMN_GAP
        x += (column - 1) * (NODE_WIDTH + COLUMN_GAP)
        for row, trial in enumerate(items):
            position = {"x": x, "y": start_y + row * (NODE_HEIGHT + ROW_GAP)}
            positions[trial["trial_id"]] = position
            trial["position"] = position
        wave_indexes = {item["wave_index"] for item in items}
        if has_waves and len(wave_indexes) == 1:
            wave_index = next(iter(wave_indexes))
            if wave_index > 0:
                column_metadata.append(
                    {
                        "x": x,
                        "label": f"WAVE {wave_index}",
                        "state": "ACTIVE" if wave_index == active_wave else "SETTLED",
                        "active": wave_index == active_wave,
                        "trial_count": len(items),
                    }
                )

    counts = defaultdict(int)
    for trial in trials:
        counts[trial["status"]] += 1
    scored_count = sum(trial["score"] is not None for trial in trials)
    failed_count = sum(
        _stage_status(trial["raw"], "train") == "failed"
        or _stage_status(trial["raw"], "evaluation") == "failed"
        for trial in trials
    )
    running_count = sum(
        _stage_status(trial["raw"], "train") == "running"
        or _stage_status(trial["raw"], "evaluation") == "running"
        for trial in trials
    )

    if running_count or active_wave is not None:
        study_status = "Running"
        status_class = ""
    elif not trials:
        study_status = "Initialized"
        status_class = ""
    elif failed_count:
        study_status = "Complete with failures"
        status_class = "attention"
    else:
        study_status = "Complete"
        status_class = "complete"

    best_improvement = _signed_improvement(
        score=best_score,
        reference=baseline_score,
        direction=direction,
        improved_flag=None,
    )
    return {
        "study_id": study_id,
        "base_commit": state.get("base_commit") or "",
        "baseline_score": baseline_score,
        "best_score": best_score,
        "best_trial_id": state.get("best_trial_id"),
        "eval_metric": state.get("eval_metric") or "Not recorded",
        "direction": direction,
        "min_delta": state.get("min_delta"),
        "active_wave_index": active_wave,
        "pending_change_trial_ids": list(state.get("pending_change_trial_ids") or []),
        "trials": trials,
        "trials_by_id": trials_by_id,
        "positions": positions,
        "baseline_position": baseline_position,
        "column_metadata": column_metadata,
        "graph_width": graph_width,
        "graph_height": graph_height,
        "max_abs_change": max_abs_change,
        "scored_count": scored_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "study_status": study_status,
        "status_class": status_class,
        "best_improvement": best_improvement,
        "result_counts": dict(counts),
    }


def render_html(state: Mapping[str, Any], source_path: Path) -> str:
    """Render one validated state as a complete HTML document."""
    report = build_report(state)
    graph = _render_graph(report)
    details = [_baseline_detail_template(report)]
    details.extend(_trial_detail_template(trial) for trial in report["trials"])
    direction_note = {
        "minimize": "Lower is better",
        "maximize": "Higher is better",
        None: "Direction not recorded",
    }[report["direction"]]
    best_trial = report["best_trial_id"] or "Baseline"
    base_commit = str(report["base_commit"])
    short_commit = base_commit[:12] if base_commit else "Not recorded"
    hero_meta = "".join(
        [
            _pill(report["study_status"], report["status_class"]),
            _plain_pill(f"Objective: {report['eval_metric']}"),
            _plain_pill(direction_note),
            _plain_pill(f"Best: {best_trial}"),
            _plain_pill(f"Base: {short_commit}"),
        ]
    )
    metrics = "".join(
        [
            _metric("Baseline", _format_score(report["baseline_score"]), "Starting score"),
            _metric(
                "Best score",
                _format_score(report["best_score"]),
                best_trial,
            ),
            _metric(
                "Baseline → best",
                _format_change(report["best_improvement"]),
                direction_note,
            ),
            _metric(
                "Trials",
                str(len(report["trials"])),
                f"{report['scored_count']} scored · {report['failed_count']} failed",
            ),
            _metric(
                "Current wave",
                (
                    str(report["active_wave_index"])
                    if report["active_wave_index"] is not None
                    else "—"
                ),
                f"{len(report['pending_change_trial_ids'])} queued changes",
            ),
            _metric(
                "Minimum delta",
                _format_score(report["min_delta"]),
                "required improvement",
            ),
        ]
    )
    page_title = f"{report['study_id']} · AutoResearch"
    output = HTML_TEMPLATE
    replacements = {
        "__PAGE_TITLE__": html.escape(page_title),
        "__STUDY_ID__": html.escape(report["study_id"]),
        "__HERO_META__": hero_meta,
        "__METRICS__": metrics,
        "__GRAPH__": graph,
        "__SOURCE_PATH__": html.escape(str(source_path)),
        "__DETAIL_TEMPLATES__": "".join(details),
    }
    for token, value in replacements.items():
        output = output.replace(token, value)
    return output


def _render_graph(report: Mapping[str, Any]) -> str:
    """Render the study lineage as an SVG."""
    width = report["graph_width"]
    height = report["graph_height"]
    trials = report["trials"]
    baseline = report["baseline_position"]
    if not trials:
        return '<div class="empty-note">No trials have been created yet.</div>'

    parts = [
        (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-label="AutoResearch trial lineage">'
        ),
        """<defs>
          <marker id="arrow-parent" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa5b7"></path>
          </marker>
          <marker id="arrow-composition" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b5cf6"></path>
          </marker>
        </defs>""",
    ]
    positions = report["positions"]
    parts.extend(_wave_headers(report))
    for trial in trials:
        parent_id = trial["parent_trial_id"]
        source = (
            {
                "x": baseline["x"],
                "y": baseline["y"],
                "width": BASELINE_WIDTH,
                "height": BASELINE_HEIGHT,
            }
            if parent_id is None
            else {
                **positions[parent_id],
                "width": NODE_WIDTH,
                "height": NODE_HEIGHT,
            }
        )
        target = {**positions[trial["trial_id"]], "width": NODE_WIDTH, "height": NODE_HEIGHT}
        parts.append(_edge(source, target, composition=False))
    for trial in trials:
        source_id = trial["composition_source_trial_id"]
        if source_id is None:
            continue
        source = {**positions[source_id], "width": NODE_WIDTH, "height": NODE_HEIGHT}
        target = {
            **positions[trial["trial_id"]],
            "width": NODE_WIDTH,
            "height": NODE_HEIGHT,
        }
        parts.append(_edge(source, target, composition=True))

    parts.append(_baseline_node(report))
    for trial in trials:
        parts.append(_trial_node(trial, report["best_trial_id"]))
    parts.append("</svg>")
    return "".join(parts)


def _wave_headers(report: Mapping[str, Any]) -> list[str]:
    """Render labels and guides for wave-based parallel trial columns."""
    parts: list[str] = []
    height = report["graph_height"]
    for column in report["column_metadata"]:
        center_x = column["x"] + NODE_WIDTH / 2
        active_class = " active" if column["active"] else ""
        state = f"{column['state']} · {column['trial_count']} TRIALS"
        parts.extend(
            [
                f'<line class="wave-guide" x1="{center_x}" y1="58" '
                f'x2="{center_x}" y2="{height - 28}"></line>',
                f'<text class="wave-title{active_class}" x="{center_x}" y="28" '
                f'text-anchor="middle">{html.escape(str(column["label"]))}</text>',
                f'<text class="wave-state{active_class}" x="{center_x}" y="45" '
                f'text-anchor="middle">{html.escape(state)}</text>',
            ]
        )
    return parts


def _edge(
    source: Mapping[str, float],
    target: Mapping[str, float],
    *,
    composition: bool,
) -> str:
    """Render a curved directed edge between two nodes."""
    x1 = source["x"] + source["width"]
    y1 = source["y"] + source["height"] / 2
    x2 = target["x"] - 7
    y2 = target["y"] + target["height"] / 2
    bend = max(44, (x2 - x1) * 0.46)
    class_name = "edge composition" if composition else "edge"
    marker = "arrow-composition" if composition else "arrow-parent"
    path = f"M {x1:.1f} {y1:.1f} C {x1 + bend:.1f} {y1:.1f}, {x2 - bend:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}"
    return (
        f'<path class="{class_name}" d="{path}" marker-end="url(#{marker})">'
        f"<title>{'Composed change' if composition else 'Parent lineage'}</title></path>"
    )


def _baseline_node(report: Mapping[str, Any]) -> str:
    """Render the baseline root node."""
    x = report["baseline_position"]["x"]
    y = report["baseline_position"]["y"]
    return f"""
    <g class="node-group" tabindex="0" role="button"
       data-detail-id="baseline" data-detail-title="Baseline">
      <rect class="node-card" x="{x}" y="{y}" width="{BASELINE_WIDTH}"
            height="{BASELINE_HEIGHT}" rx="16" fill="#edf3ff" stroke="#5b8cff"></rect>
      <text x="{x + 18}" y="{y + 27}" class="node-kicker" fill="#4168b5">STUDY ROOT</text>
      <text x="{x + 18}" y="{y + 55}" class="node-title" fill="#172033">Baseline</text>
      <text x="{x + 18}" y="{y + 88}" class="node-score" fill="#172033">{html.escape(_format_score(report["baseline_score"]))}</text>
      <text x="{x + 18}" y="{y + 111}" class="node-stage" fill="#596579">Click for objective and revision</text>
    </g>
    """


def _trial_node(trial: Mapping[str, Any], best_trial_id: str | None) -> str:
    """Render one clickable trial node."""
    raw = trial["raw"]
    x = trial["position"]["x"]
    y = trial["position"]["y"]
    color = trial["text_color"]
    title_lines = _wrap_text(trial["summary_text"], width=28, limit=2)
    title_spans = "".join(
        f'<tspan x="{x + 18}" dy="{0 if index == 0 else 17}">{html.escape(line)}</tspan>'
        for index, line in enumerate(title_lines)
    )
    wave = raw.get("wave_index", 0)
    idea_type = str(raw.get("idea_type") or "experiment").replace("_", " ")
    train_status = _stage_status(raw, "train")
    eval_status = _stage_status(raw, "evaluation")
    best = trial["trial_id"] == best_trial_id
    best_markup = (
        f'<rect class="best-ring" x="{x - 5}" y="{y - 5}" width="{NODE_WIDTH + 10}" '
        f'height="{NODE_HEIGHT + 10}" rx="20"></rect>'
        f'<rect class="best-badge" x="{x + NODE_WIDTH - 74}" y="{y + 12}" '
        'width="60" height="20" rx="10"></rect>'
        f'<text class="best-badge-text" x="{x + NODE_WIDTH - 44}" y="{y + 26}" '
        'text-anchor="middle">BEST</text>'
        if best
        else ""
    )
    kicker = f"TRIAL {trial['index']} · WAVE {wave} · {idea_type}"
    return f"""
    <g class="node-group" tabindex="0" role="button"
       data-detail-id="{html.escape(str(trial["detail_key"]))}"
       data-detail-title="{html.escape(str(trial["trial_id"]))}"
       data-result="{trial["status"]}" data-improvement="{trial["improvement"]}">
      {best_markup}
      <rect class="node-card" x="{x}" y="{y}" width="{NODE_WIDTH}" height="{NODE_HEIGHT}"
            rx="16" fill="{trial["fill"]}" stroke="{trial["stroke"]}"></rect>
      <text x="{x + 18}" y="{y + 25}" class="node-kicker" fill="{color}">{html.escape(kicker.upper())}</text>
      <text x="{x + 18}" y="{y + 52}" class="node-title" fill="{color}">{title_spans}</text>
      <text x="{x + 18}" y="{y + 105}" class="node-score" fill="{color}">{html.escape(trial["score_text"])}</text>
      <text x="{x + 18}" y="{y + 128}" class="node-delta" fill="{color}">{html.escape(trial["change_text"])}</text>
      <text x="{x + 18}" y="{y + 153}" class="node-stage" fill="{color}">TRAIN {html.escape(train_status.upper())} · EVAL {html.escape(eval_status.upper())}</text>
    </g>
    """


def _baseline_detail_template(report: Mapping[str, Any]) -> str:
    """Render the baseline drawer template."""
    direction = report["direction"] or "not recorded"
    fields = [
        ("Score", _format_score(report["baseline_score"])),
        ("Metric", report["eval_metric"]),
        ("Direction", direction),
        ("Minimum delta", _format_score(report["min_delta"])),
        ("Base commit", report["base_commit"] or "Not recorded"),
    ]
    return _detail_template(
        "baseline",
        _detail_section("Baseline", "The frozen reference used for every root trial.")
        + _detail_fields(fields),
    )


def _trial_detail_template(trial: Mapping[str, Any]) -> str:
    """Render the drawer template for one trial."""
    raw = trial["raw"]
    parent = trial["parent_trial_id"] or "Baseline"
    fields = [
        ("Trial ID", trial["trial_id"]),
        ("Wave", raw.get("wave_index", 0)),
        ("Idea type", raw.get("idea_type") or "Not recorded"),
        ("Parent", parent),
        ("Parent score", _format_score(trial["reference_score"])),
        ("Score", _format_score(trial["score"])),
        ("Change", trial["change_text"]),
        ("New global best", _format_bool(raw.get("is_new_best"))),
        ("Passed parent gate", _format_bool(raw.get("improved_over_parent"))),
        ("Published", _format_bool(raw.get("code_published"))),
    ]
    sections = [_detail_fields(fields)]
    for heading, value in (
        ("Hypothesis", raw.get("hypothesis")),
        ("Result summary", raw.get("result_summary")),
        ("Next guidance", raw.get("next_guidance")),
    ):
        if value:
            sections.append(_detail_section(heading, str(value)))

    composition_source = raw.get("composition_source_trial_id")
    if composition_source:
        sections.append(_detail_section("Composition", f"Source trial: {composition_source}"))

    for heading, values in (("Changed files", raw.get("changed_files")),):
        if isinstance(values, Sequence) and not isinstance(values, str | bytes) and values:
            sections.append(_detail_list(heading, [str(value) for value in values]))

    code_fields = [
        ("Branch", raw.get("code_branch") or "Not recorded"),
        ("Revision", raw.get("code_revision") or "Not recorded"),
        ("Parent revision", raw.get("parent_revision") or "Not recorded"),
    ]
    sections.append(_detail_section("Code", _definition_list(code_fields), raw_html=True))
    sections.append(_stage_detail("Training", raw.get("train")))
    sections.append(_stage_detail("Evaluation", raw.get("evaluation")))
    return _detail_template(str(trial["detail_key"]), "".join(sections))


def _stage_detail(name: str, stage: Any) -> str:
    """Render one train/evaluation stage in the detail drawer."""
    stage = stage if isinstance(stage, Mapping) else {}
    fields = [
        ("Status", stage.get("status", "pending")),
        ("Attempts", stage.get("attempts", 0)),
    ]
    body = _definition_list(fields)
    error = stage.get("error")
    if error:
        body += f"<h4>Error</h4><pre>{html.escape(str(error))}</pre>"
    checkpoint = stage.get("checkpoint")
    if checkpoint:
        body += "<h4>Checkpoint</h4>" + _definition_list([("Path", str(checkpoint))])
    return _detail_section(name, body, raw_html=True)


def _detail_template(key: str, content: str) -> str:
    """Wrap drawer content in a dormant HTML template."""
    return f'<template id="detail-{html.escape(key)}">{content}</template>'


def _detail_section(heading: str, value: str, *, raw_html: bool = False) -> str:
    """Render a named detail section."""
    content = value if raw_html else f"<p>{html.escape(value)}</p>"
    return f'<section class="detail-section"><h3>{html.escape(heading)}</h3>{content}</section>'


def _detail_fields(fields: Sequence[tuple[str, Any]]) -> str:
    """Render the primary trial field grid."""
    return (
        '<section class="detail-section"><div class="detail-grid">'
        + "".join(
            '<div class="detail-field">'
            f"<dt>{html.escape(str(label))}</dt>"
            f"<dd>{html.escape(str(value))}</dd>"
            "</div>"
            for label, value in fields
        )
        + "</div></section>"
    )


def _definition_list(fields: Sequence[tuple[str, Any]]) -> str:
    """Render compact key/value details."""
    return (
        '<dl class="detail-grid">'
        + "".join(
            '<div class="detail-field">'
            f"<dt>{html.escape(str(label))}</dt>"
            f"<dd>{html.escape(str(value))}</dd>"
            "</div>"
            for label, value in fields
        )
        + "</dl>"
    )


def _detail_list(heading: str, values: Sequence[str]) -> str:
    """Render a list section."""
    items = "".join(f"<li>{html.escape(value)}</li>" for value in values)
    return (
        f'<section class="detail-section"><h3>{html.escape(heading)}</h3>'
        f'<ul class="detail-list">{items}</ul></section>'
    )


def _metric(label: str, value: str, note: str) -> str:
    """Render a top-level metric card."""
    return (
        '<article class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        "</article>"
    )


def _pill(label: str, status_class: str) -> str:
    """Render the study status pill."""
    return (
        '<span class="pill">'
        f'<span class="status-dot {html.escape(status_class)}"></span>'
        f"{html.escape(label)}</span>"
    )


def _plain_pill(label: str) -> str:
    """Render a metadata pill."""
    return f'<span class="pill">{html.escape(label)}</span>'


def _trial_summary(raw: Mapping[str, Any]) -> str:
    """Choose concise node text from a trial."""
    for key in ("hypothesis", "result_summary", "idea_type"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Experiment"


def _wrap_text(value: str, *, width: int, limit: int) -> list[str]:
    """Wrap and truncate text for an SVG node."""
    lines = textwrap.wrap(value, width=width, break_long_words=False) or ["Experiment"]
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    return lines


def _signed_improvement(
    *,
    score: float | None,
    reference: float | None,
    direction: str | None,
    improved_flag: Any,
) -> float | None:
    """Return parent-relative improvement where positive always means better."""
    if score is None or reference is None:
        return None
    raw_delta = score - reference
    if direction == "maximize":
        return raw_delta
    if direction == "minimize":
        return -raw_delta
    if raw_delta == 0:
        return 0.0
    if improved_flag is True:
        return abs(raw_delta)
    if improved_flag is False:
        return -abs(raw_delta)
    return None


def _trial_result_status(raw: Mapping[str, Any], improvement: float | None) -> str:
    """Classify a trial without conflating failure and regression."""
    if improvement is not None:
        if improvement > 0:
            return "improved"
        if improvement < 0:
            return "regressed"
        return "unchanged"
    train_status = _stage_status(raw, "train")
    eval_status = _stage_status(raw, "evaluation")
    statuses = {train_status, eval_status}
    if "failed" in statuses:
        return "failed"
    if "running" in statuses:
        return "running"
    if statuses == {"skipped"}:
        return "skipped"
    if "succeeded" in statuses:
        return "unscored"
    return "pending"


def _node_colors(status: str, intensity: float) -> tuple[str, str, str]:
    """Return accessible fill, stroke, and text colors for one node."""
    if status == "improved":
        lightness = 94 - 47 * intensity
        text = "#ffffff" if lightness < 64 else "#113322"
        return f"hsl(145 43% {lightness:.1f}%)", "#16834a", text
    if status == "regressed":
        lightness = 95 - 45 * intensity
        text = "#ffffff" if lightness < 65 else "#4a1717"
        return f"hsl(5 63% {lightness:.1f}%)", "#d14343", text
    if status == "failed":
        return "#fff0f2", "#e11d48", "#4d1621"
    if status == "running":
        return "#eaf3ff", "#3b82f6", "#15355f"
    if status == "skipped":
        return "#f1f3f6", "#98a2b3", "#4b5565"
    return "#f7f9fc", "#aeb8c7", "#344054"


def _stage_status(raw: Mapping[str, Any], key: str) -> str:
    """Read a train or evaluation stage status."""
    stage = raw.get(key)
    if not isinstance(stage, Mapping):
        return "pending"
    value = stage.get("status", "pending")
    return str(value) if value is not None else "pending"


def _format_score(value: Any) -> str:
    """Format a score compactly without hiding useful precision."""
    number = _optional_number(value, "score")
    if number is None:
        return "—"
    if number == 0:
        return "0"
    absolute = abs(number)
    if absolute >= 10000 or absolute < 0.0001:
        return f"{number:.4g}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _format_change(value: float | None) -> str:
    """Format a signed improvement with explicit semantics."""
    if value is None:
        return "Not scored"
    if value > 0:
        return f"Improved +{_format_score(abs(value))}"
    if value < 0:
        return f"Regressed −{_format_score(abs(value))}"
    return "Unchanged"


def _format_bool(value: Any) -> str:
    """Format nullable boolean state."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Not decided"


def _sort_int(value: Any) -> int:
    """Return a stable integer sort key."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_number(value: Any, label: str) -> float:
    """Validate and convert a finite JSON number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _optional_number(value: Any, label: str) -> float | None:
    """Validate an optional finite JSON number."""
    if value is None:
        return None
    return _as_number(value, label)


def main() -> int:
    """Run the renderer CLI."""
    args = _parser().parse_args()
    try:
        state = load_state(args.state_json)
        output = args.output or args.state_json.with_name(
            f"{args.state_json.stem}.visualization.html"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_html(state, args.state_json.resolve())
        output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
