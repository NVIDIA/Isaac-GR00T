# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Guard: exported comments and docstrings must not point at internal VCS objects.

``oss/`` ships outside the monorepo, so a comment that says "see
``<handle>/<topic>``" or "from MR !1234" resolves to nothing for its reader
while still disclosing an internal branch, author handle, or review thread.
Both leaks this guard was written for cited a private branch as the place to go
find a fuller implementation.

Only comments and docstrings are scanned. Runtime string literals are excluded
because metric keys and default paths (``wbc/total_demo_steps``,
``tmp/groot_outputs``) share the slash shape without being VCS references.
"""

from __future__ import annotations

import ast
import io
import itertools
import os
from pathlib import Path
import re
import tokenize

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "groot"

# Internal review threads and tracker items. The exported tree has no merge
# requests, so this vocabulary is always a finding.
_TRACKER_REFERENCE = re.compile(
    r"(?<![\w!])!\d{3,}\b|\b(?:GEAR|PAI)-\d+\b|\bnvbugs?\b|\bmerge requests?\b|\bMRs?\b"
)

# Words that mean the surrounding text is talking about version control rather
# than about a control-flow branch or a dataset merge.
_VCS_CUE = re.compile(
    r"\bbranch(?:es|'s)?\b|\bcherry-pick\w*\b|\bbackport\w*\b|\brebase\w*\b|\blanded in\b",
    re.IGNORECASE,
)

_SLASH_TOKEN = re.compile(r"\b([a-z][a-z0-9_.-]{1,20})/([a-z0-9][a-z0-9_.-]{3,60})\b")

# Branch topics join at least two real words (`fix/rmsnorm-dtype`). Prose
# alternations (`warn/kill`, `video-only/no-action`) and metric prefixes
# (`val/loss_`) fail that shape, which is what tells them apart.
_BRANCH_TOPIC = re.compile(r"[a-z0-9]{2,}_[a-z0-9]{2,}|-.{8,}")

# A path is a file, not a ref.
_FILE_SUFFIX = re.compile(r"\.[a-z0-9]{1,6}$")

# A branch cited as the place to find code is written as an identifier, in
# backticks. That quoting is the signal when no VCS noun is nearby -- the
# docstring leak this guard was written for never used the word "branch".
_BACKTICKED = re.compile(r"`")

# Slash-separated names that legitimately appear as identifiers in prose.
_ALLOWED_PREFIXES = frozenset(
    {
        "codex",  # autoresearch names its own trial branches this way
        "git",  # config keys such as git/base_branch
        "meta",  # dataset layout, e.g. meta/info.json
        "refs",
    }
)


def _python_sources(root: Path) -> list[Path]:
    """Every ``.py`` file under ``root``, descending through symlinked directories.

    ``oss/groot`` is assembled from directory symlinks into the monorepo
    (``oss/groot/core/policy -> ../../../groot/core/policy``). ``Path.rglob``
    does not descend into those before Python 3.13, which silently halves the
    scanned tree, so the walk has to opt in explicitly.
    """
    return [
        Path(dirpath) / name
        for dirpath, _dirs, files in os.walk(root, followlinks=True)
        for name in files
        if name.endswith(".py")
    ]


def _comment_runs(source: str):
    """Yield ``(lineno, text)`` for each run of consecutive ``#`` comment lines."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    pending: list[str] = []
    start = 0
    for token in tokens:
        if token.type == tokenize.COMMENT:
            if not pending:
                start = token.start[0]
            pending.append(token.string)
        elif pending and token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT):
            yield start, "\n".join(pending)
            pending = []
    if pending:
        yield start, "\n".join(pending)


def _is_string_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _docstrings(source: str):
    """Yield ``(lineno, text)`` for every module, class, and function docstring."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, documented) and (text := ast.get_docstring(node, clean=False)):
            yield getattr(node, "lineno", 1), text
    # Attribute docstrings -- a bare string after an assignment -- are how the
    # embodiment tag enum documents each member, and ast.get_docstring skips them.
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for previous, statement in itertools.pairwise(body):
            if isinstance(previous, ast.Assign | ast.AnnAssign) and _is_string_expr(statement):
                yield statement.lineno, statement.value.value


def _findings(text: str) -> list[str]:
    hits = _TRACKER_REFERENCE.findall(text)
    near_vcs_noun = bool(_VCS_CUE.search(text))
    for match in _SLASH_TOKEN.finditer(text):
        prefix, topic = match.group(1), match.group(2)
        if prefix in _ALLOWED_PREFIXES or _FILE_SUFFIX.search(topic):
            continue
        if not _BRANCH_TOPIC.search(topic):
            continue
        if near_vcs_noun or _BACKTICKED.search(text[max(0, match.start() - 2) : match.start()]):
            hits.append(match.group(0))
    return hits


# Without this the repo-wide scan below could silently degrade into a test that
# passes because it detects nothing.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see `seungjael/dex_tactile_aug3_p2_override_fix`, which carries the full entry", True),
        ("resolver from MR !2897", True),
        ("tracked as GEAR-75", True),
        ("the lint job only runs on files changed in an MR", True),
        ("Keep this an alias so that branch's richer entry is a clean superset", False),
        # Prose alternations and config keys sharing a slash with VCS vocabulary.
        ("the model still skips the action branch for video-only/no-action batches", False),
        ("cannot resolve git/base_branch in workspace", False),
        ("create the trial branch codex/autoresearch-yam-0001", False),
        # "branch" as control flow must not turn an ordinary path into a finding.
        ("never reaches 0 and this branch is always hit", False),
    ],
)
def test_findings_detects_internal_references(text, expected):
    assert bool(_findings(text)) is expected


def test_no_internal_vcs_references_in_comments():
    offenders = []
    for path in _python_sources(PACKAGE_ROOT):
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, text in (*_comment_runs(source), *_docstrings(source)):
            rel = path.relative_to(PACKAGE_ROOT.parent)
            offenders += [f"{rel}:{lineno}: {hit}" for hit in _findings(text)]

    assert not offenders, (
        "Comments/docstrings reference internal branches, merge requests, or "
        "trackers that do not exist in the exported tree. State the constraint "
        "the code needs instead of pointing at the internal object:\n  "
        + "\n  ".join(sorted(offenders))
    )
