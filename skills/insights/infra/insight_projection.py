# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Validate and deterministically search the published GR00T insight projection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

SCHEMA_VERSION = "groot_insights_projection.v0"
# Published ids are the producer's graph ids. The prefix separates a finding from a
# chunk, and the remainder is what relates the two.
FINDING_ID_PREFIX = "finding:"
_TOPIC_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*")
_STOP_WORDS = {
    "acceptance",
    "behavior",
    "check",
    "environment",
    "expected",
    "observed",
    "problem",
    "reproduction",
    "run",
    "the",
    "this",
    "with",
}


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid insight JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Insight JSON must contain an object: {path}")  # noqa: TRY004
    return value, raw


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Insight field {field_name!r} must be a list of strings")
    return list(value)


def locate_insights_root(start: Path, explicit_root: Path | None = None) -> Path:
    """Find ``.insights`` from either a repository root or a nested OSS directory."""
    if explicit_root is not None:
        candidate = explicit_root.expanduser().resolve()
        if (candidate / "index.json").is_file():
            return candidate
        if (candidate / ".insights" / "index.json").is_file():
            return candidate / ".insights"
        raise FileNotFoundError(f"No .insights/index.json under {candidate}")

    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / ".insights"
        if (candidate / "index.json").is_file():
            return candidate
    raise FileNotFoundError(f"No .insights/index.json found from {current} or its parents")


@dataclass(frozen=True)
class InsightProjection:
    root: Path
    schema_version: str
    generated_at: str
    topics: tuple[str, ...]
    findings: dict[str, dict[str, Any]]
    chunks: dict[str, dict[str, Any]]
    sha256: str

    @classmethod
    def load(cls, root: Path) -> InsightProjection:
        root = root.resolve()
        index, index_raw = _read_json(root / "index.json")
        schema_version = index.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported insight schema {schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        generated_at = index.get("generated_at")
        if not isinstance(generated_at, str) or not generated_at.strip():
            raise ValueError("Insight index requires a non-empty generated_at")
        topics = _string_list(index.get("topics"), "topics")
        if len(set(topics)) != len(topics):
            raise ValueError("Insight topics must be unique")
        if any(_TOPIC_PATTERN.fullmatch(topic) is None for topic in topics):
            raise ValueError("Insight topic names may contain only letters, numbers, ., _, and -")

        findings: dict[str, dict[str, Any]] = {}
        chunks: dict[str, dict[str, Any]] = {}
        digest = hashlib.sha256()
        digest.update(b"index.json\0")
        digest.update(index_raw)
        for topic in topics:
            path = root / "nodes" / f"{topic}.json"
            bucket, raw = _read_json(path)
            digest.update(f"\0nodes/{topic}.json\0".encode())
            digest.update(raw)
            if bucket.get("topic") != topic:
                raise ValueError(f"Insight bucket topic mismatch in {path}")
            bucket_findings = bucket.get("findings")
            bucket_chunks = bucket.get("chunks")
            if not isinstance(bucket_findings, list) or not isinstance(bucket_chunks, list):
                raise ValueError(  # noqa: TRY004
                    f"Insight bucket requires findings and chunks lists: {path}"
                )

            for record in bucket_findings:
                normalized = cls._validate_record(record, "finding", topic, path)
                record_id = normalized["id"]
                if record_id in findings:
                    raise ValueError(f"Duplicate insight finding id: {record_id}")
                findings[record_id] = normalized
            for record in bucket_chunks:
                normalized = cls._validate_record(record, "chunk", topic, path)
                record_id = normalized["id"]
                if record_id in chunks:
                    raise ValueError(f"Duplicate insight chunk id: {record_id}")
                chunks[record_id] = normalized

        index_findings = set(_string_list(index.get("findings"), "findings"))
        index_chunks = set(_string_list(index.get("chunks"), "chunks"))
        if index_findings != set(findings):
            raise ValueError("Insight index findings do not match nodes/*.json")
        if index_chunks != set(chunks):
            raise ValueError("Insight index chunks do not match nodes/*.json")

        return cls(
            root=root,
            schema_version=schema_version,
            generated_at=generated_at,
            topics=tuple(topics),
            findings=findings,
            chunks=chunks,
            sha256=digest.hexdigest(),
        )

    @staticmethod
    def _validate_record(
        value: object,
        kind: str,
        topic: str,
        path: Path,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"Insight {kind} must be an object in {path}")  # noqa: TRY004
        id_field = "id" if kind == "finding" else "chunk_id"
        text_field = "body" if kind == "finding" else "text"
        record_id = value.get(id_field)
        title = value.get("title")
        text = value.get(text_field)
        if "concepts" not in value:
            raise ValueError(f"Insight {kind} requires concepts in {path}")
        concepts = _string_list(value["concepts"], f"{kind}.concepts")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Insight {kind} requires {id_field} in {path}")
        if kind == "finding" and not record_id.startswith(FINDING_ID_PREFIX):
            raise ValueError(
                f"Insight finding id must start with {FINDING_ID_PREFIX!r}: {record_id}"
            )
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(  # noqa: TRY004
                f"Insight {kind} requires string title and {text_field}: {record_id}"
            )
        return {
            "id": record_id,
            "kind": kind,
            "topic": topic,
            "title": title,
            "text": text,
            "concepts": concepts,
        }

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        query = " ".join(query.split())
        if not query:
            raise ValueError("Insight query cannot be empty")
        if limit < 1 or limit > 20:
            raise ValueError("Insight result limit must be between 1 and 20")
        query_tokens = [
            token
            for token in dict.fromkeys(_TOKEN_PATTERN.findall(query.casefold()))
            if token not in _STOP_WORDS
        ]
        if not query_tokens:
            query_tokens = list(dict.fromkeys(_TOKEN_PATTERN.findall(query.casefold())))

        items: dict[str, dict[str, dict[str, Any]]] = {}
        for record in self.findings.values():
            key = record["id"].removeprefix(FINDING_ID_PREFIX)
            items.setdefault(key, {})["finding"] = record
        for record in self.chunks.values():
            items.setdefault(record["id"], {})["chunk"] = record

        ranked: list[dict[str, Any]] = []
        query_phrase = query.casefold()
        for key, item in items.items():
            title_text = " ".join(
                record["title"] for record in item.values() if record.get("title")
            ).casefold()
            body_text = " ".join(
                record["text"] for record in item.values() if record.get("text")
            ).casefold()
            metadata_text = " ".join(
                [
                    *(record["topic"] for record in item.values()),
                    *(concept for record in item.values() for concept in record["concepts"]),
                ]
            ).casefold()
            title_tokens = set(_TOKEN_PATTERN.findall(title_text))
            body_tokens = set(_TOKEN_PATTERN.findall(body_text))
            metadata_tokens = set(_TOKEN_PATTERN.findall(metadata_text))
            matched_terms = [
                token
                for token in query_tokens
                if token in title_tokens or token in body_tokens or token in metadata_tokens
            ]
            if not matched_terms:
                continue
            score = sum(
                8 if token in title_tokens else 3 if token in body_tokens else 2
                for token in matched_terms
            )
            combined_text = f"{title_text} {body_text}"
            if len(query_phrase) <= 240 and query_phrase in combined_text:
                score += 30
            if len(matched_terms) == len(query_tokens):
                score += 10
            if "finding" in item and "chunk" in item:
                score += 2
            ranked.append(
                {
                    "key": key,
                    "score": score,
                    "matched_terms": matched_terms,
                    "finding": item.get("finding"),
                    "chunk": item.get("chunk"),
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["key"]))
        results = ranked[:limit]
        record_ids = [
            record["id"]
            for item in results
            for record in (item.get("finding"), item.get("chunk"))
            if record is not None
        ]
        return {
            "available": True,
            "query": query,
            "projection": {
                "root": str(self.root),
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "sha256": self.sha256,
                "topics": list(self.topics),
                "finding_count": len(self.findings),
                "chunk_count": len(self.chunks),
            },
            "record_ids": record_ids,
            "results": results,
        }


def search_insights(
    query: str,
    *,
    start: Path,
    insights_root: Path | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search a projection, returning a non-fatal grounding gap when unavailable."""
    try:
        root = locate_insights_root(start, insights_root)
        return InsightProjection.load(root).search(query, limit)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {
            "available": False,
            "query": " ".join(query.split()),
            "projection": None,
            "record_ids": [],
            "results": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
