# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Verified precedents used to prioritize AutoDebug checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class SolvedCase:
    id: str
    title: str
    source: str
    scope: tuple[str, ...]
    tags: tuple[str, ...]
    match_terms: tuple[str, ...]
    symptom: str
    first_test: str
    cause: str
    fix: str
    compute: str
    verification: str
    status: str = "verified"
    version: int = 1

    @classmethod
    def init_from_path(cls, path: Path) -> SolvedCase:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Solved case must be one JSON object: {path}")
        required_strings = (
            "id",
            "status",
            "title",
            "source",
            "symptom",
            "first_test",
            "cause",
            "fix",
            "compute",
            "verification",
        )
        missing = [
            key
            for key in required_strings
            if not isinstance(data.get(key), str) or not data[key].strip()
        ]
        if missing:
            raise ValueError(f"Solved case {path} has missing fields: {', '.join(missing)}")
        if data.get("version") != 1:
            raise ValueError(f"Solved case {path} must use version 1")
        if data["status"] != "verified":
            raise ValueError(f"Runtime solved case {path} must have status 'verified'")
        sequence_fields = ("scope", "tags", "match_terms")
        for key in sequence_fields:
            value = data.get(key)
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                raise ValueError(f"Solved case {path} needs a non-empty string list: {key}")
        return cls(
            id=data["id"],
            title=data["title"],
            source=data["source"],
            scope=tuple(data["scope"]),
            tags=tuple(data["tags"]),
            match_terms=tuple(data["match_terms"]),
            symptom=data["symptom"],
            first_test=data["first_test"],
            cause=data["cause"],
            fix=data["fix"],
            compute=data["compute"],
            verification=data["verification"],
            status=data["status"],
            version=data["version"],
        )

    @classmethod
    def get_all(cls, directory: Path) -> list[SolvedCase]:
        if not directory.exists():
            return []
        return [cls.init_from_path(path) for path in sorted(directory.glob("*.json"))]

    def match_score(self, report: str) -> int:
        normalized = re.sub(r"[^a-z0-9]+", " ", report.lower())
        phrases = (*self.match_terms, *self.tags)
        score = 0
        for phrase in phrases:
            normalized_phrase = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
            if re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized):
                score += 1
        return score

    def version_affinity(self, report: str) -> int:
        """Prefer the same GR00T generation without excluding older precedents."""
        version_pattern = re.compile(r"\bn\d+(?:\.\d+)*\b", flags=re.IGNORECASE)
        report_versions = {value.lower() for value in version_pattern.findall(report)}
        case_versions = {value.lower() for value in version_pattern.findall(" ".join(self.scope))}
        return int(bool(report_versions & case_versions))

    def to_reference_text(self) -> str:
        return (
            f"## {self.id}: {self.title}\n\n"
            f"- Source: {self.source}\n"
            f"- Scope: {', '.join(self.scope)}\n"
            f"- Symptom: {self.symptom}\n"
            f"- Best first test: {self.first_test}\n"
            f"- Confirmed cause: {self.cause}\n"
            f"- Verified fix: {self.fix}\n"
            f"- Compute: {self.compute}\n"
            f"- Verification: {self.verification}\n"
        )
