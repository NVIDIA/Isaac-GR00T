# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for the published insight projection and AutoDebug lookup."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from skills.autodebug.infra import AutoDebugIncident
from skills.insights.infra.insight_projection import (
    FINDING_ID_PREFIX,
    InsightProjection,
    locate_insights_root,
    search_insights,
)

REPORT = """# GR00T bug report

## Problem

Relative EEF actions are inaccurate during open-loop evaluation.

## Expected behavior

Predicted relative EEF actions match the held-out target.

## Observed behavior

Relative EEF arm error is higher than absolute-action arm error.

## Reproduction

Run the matched evaluator.

## Acceptance check

The matched evaluator reaches the declared arm-error threshold.
"""


def _write_projection(repository: Path) -> Path:
    root = repository / ".insights"
    nodes = root / "nodes"
    nodes.mkdir(parents=True)
    index = {
        "schema_version": "groot_insights_projection.v0",
        "generated_at": "2026-07-22T23:49:33+00:00",
        "topics": ["general", "playbook"],
        "findings": ["finding:doc:relative", "finding:doc:unpaired"],
        "chunks": ["doc:relative"],
    }
    general = {
        "topic": "general",
        "findings": [],
        "chunks": [
            {
                "chunk_id": "doc:relative",
                "title": "Relative EEF evaluation",
                "text": "Compare relative EEF normalization in open-loop evaluation.",
                "concepts": [],
            }
        ],
    }
    playbook = {
        "topic": "playbook",
        "findings": [
            {
                "id": "finding:doc:relative",
                "title": "Check relative EEF normalization",
                "body": "Use a matched open-loop comparison before changing training.",
                "concepts": ["playbook", "evaluation"],
            },
            {
                "id": "finding:doc:unpaired",
                "title": "Inspect camera timing",
                "body": "Camera timing can affect deployment.",
                "concepts": ["playbook"],
            },
        ],
        "chunks": [],
    }
    (root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (nodes / "general.json").write_text(json.dumps(general, indent=2), encoding="utf-8")
    (nodes / "playbook.json").write_text(json.dumps(playbook, indent=2), encoding="utf-8")
    (root / "graph.html").write_text(
        "Relative EEF fake viewer-only duplicate that must not be indexed.",
        encoding="utf-8",
    )
    (root / "vis-network.min.js").write_text(
        "Relative EEF fake library-only duplicate that must not be indexed.",
        encoding="utf-8",
    )
    return root


def _init_repository(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Insights Test",
            "-c",
            "user.email=insights@example.invalid",
            "commit",
            "-m",
            "Projection fixture",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_projection_is_discovered_from_nested_oss_and_pairs_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    workspace = repository / "oss"
    workspace.mkdir(parents=True)
    root = _write_projection(repository)

    assert locate_insights_root(workspace) == root
    result = search_insights("relative EEF open-loop", start=workspace)

    assert result["available"] is True
    assert result["projection"]["finding_count"] == 2
    assert result["projection"]["chunk_count"] == 1
    assert result["record_ids"][:2] == [
        "finding:doc:relative",
        "doc:relative",
    ]
    assert result["results"][0]["finding"]["topic"] == "playbook"
    assert result["results"][0]["chunk"]["topic"] == "general"
    assert all("fake" not in str(item) for item in result["results"])


def test_projection_rejects_an_index_node_mismatch(tmp_path: Path) -> None:
    root = _write_projection(tmp_path)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["chunks"].append("doc:missing")
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="index chunks do not match"):
        InsightProjection.load(root)


def test_a_record_without_concepts_is_rejected_rather_than_silently_empty(
    tmp_path: Path,
) -> None:
    # This field was read with a default, so when the producer renamed it the records
    # still validated and search quietly lost every concept term. Absence must be loud.
    root = _write_projection(tmp_path)
    nodes_path = root / "nodes" / "playbook.json"
    bucket = json.loads(nodes_path.read_text(encoding="utf-8"))
    del bucket["findings"][0]["concepts"]
    nodes_path.write_text(json.dumps(bucket), encoding="utf-8")

    with pytest.raises(ValueError, match="requires concepts"):
        InsightProjection.load(root)


def test_the_projection_shipped_in_this_repository_loads() -> None:
    # Every other test here builds its own fixture, so the corpus we actually ship was
    # covered by nothing: a reader change could reject all 217 findings and the suite
    # would stay green. Load the real thing.
    root = locate_insights_root(Path(__file__).resolve().parents[3])

    projection = InsightProjection.load(root)

    assert projection.findings, "the shipped projection has no findings"
    assert all(finding_id.startswith(FINDING_ID_PREFIX) for finding_id in projection.findings)
    # Findings pair with their evidence on the id remainder. If the producer ever stops
    # deriving the two ids from each other, grounding degrades silently, so pin it.
    paired = [
        finding_id
        for finding_id in projection.findings
        if finding_id.removeprefix(FINDING_ID_PREFIX) in projection.chunks
    ]
    assert paired, "no shipped finding pairs with its evidence chunk"


def test_missing_projection_is_a_nonfatal_grounding_gap(tmp_path: Path) -> None:
    result = search_insights("relative EEF", start=tmp_path)

    assert result["available"] is False
    assert result["record_ids"] == []
    assert result["results"] == []
    assert result["error"].startswith("FileNotFoundError:")


def test_shipped_search_wrapper_uses_the_projection_contract(tmp_path: Path) -> None:
    root = _write_projection(tmp_path)
    oss_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(oss_root / "skills" / "insights" / "scripts" / "search.py"),
            "--query",
            "relative EEF open-loop",
            "--insights-root",
            str(root),
        ],
        cwd=oss_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["available"] is True
    assert payload["record_ids"][0] == "finding:doc:relative"


def test_new_autodebug_incident_records_the_projection_and_ids(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    workspace = repository / "oss"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text("diagnosis workspace\n", encoding="utf-8")
    _write_projection(repository)
    _init_repository(repository)
    report = tmp_path / "report.md"
    report.write_text(REPORT, encoding="utf-8")

    incident = AutoDebugIncident.init_from_report(
        report,
        workspace=workspace,
        run_root=tmp_path / "runs",
    )

    lookup = incident.runs[0]
    assert lookup["kind"] == "insights"
    assert lookup["available"] is True
    assert lookup["record_ids"][:2] == ["finding:doc:relative", "doc:relative"]
    assert lookup["projection_sha256"]
    assert Path(lookup["log"]).is_file()
    persisted = AutoDebugIncident.init_from_path(incident.path)
    assert persisted.runs[0]["projection_sha256"] == lookup["projection_sha256"]


def test_autodebug_cli_can_repeat_a_narrower_insight_lookup(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    workspace = repository / "oss"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text("diagnosis workspace\n", encoding="utf-8")
    root = _write_projection(repository)
    _init_repository(repository)
    report = tmp_path / "report.md"
    report.write_text(REPORT, encoding="utf-8")
    incident = AutoDebugIncident.init_from_report(
        report,
        workspace=workspace,
        run_root=tmp_path / "runs",
    )
    oss_root = Path(__file__).resolve().parents[3]

    result = subprocess.run(
        [
            sys.executable,
            str(oss_root / "skills" / "autodebug" / "scripts" / "autodebug.py"),
            "insights",
            "--incident",
            str(incident.path),
            "--query",
            "relative EEF normalization",
            "--insights-root",
            str(root),
            "--limit",
            "1",
        ],
        cwd=oss_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["available"] is True
    assert payload["incident"] == str(incident.path)
    assert payload["record_ids"] == ["finding:doc:relative", "doc:relative"]
    persisted = AutoDebugIncident.init_from_path(incident.path)
    assert [run["kind"] for run in persisted.runs] == ["insights", "insights"]
