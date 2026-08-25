# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Report whether the vendored schema still matches the one xagents validates with.

``wire_contract`` checks this repository's code against the schema sitting beside it.
That leaves one gap it cannot close by itself: the copy can fall behind. This closes it
by fetching the original and comparing, and it is deliberately a separate thing to run —
``wire_contract`` is an offline test with no excuse to be flaky, and this one needs the
network and a token to read a private project.

Run it:

    python -m skills.groot_trajectory.check_schema_freshness

Exit codes are distinct on purpose, because "they differ" and "I could not look" are
different states and a caller that treats them alike is back to a guard that passes when
it has learned nothing:

    0  the copy matches the original
    1  the copy has fallen behind, with the differing fields listed
    2  the original could not be read, with the reason

Both sides are compared as parsed JSON rather than bytes, so reformatting the file is not
reported as drift while any change to what it declares is.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from skills.groot_trajectory.wire_contract import SCHEMA_PATH

# The project the contract belongs to, and the path within it. `main` is the ref worth
# tracking: a producer must agree with what xagents ships, not with an open branch. Until
# agentics/xagents!396 merges there is no such file on main and this reports code 2, which
# is the truthful answer -- the contract is not shipped yet, so the copy is of a proposal.
# Pass --ref to compare against that branch in the meantime.
# The GitLab host is not hardcoded: this module ships in the public OSS tree.
XAGENTS_PROJECT = "agentics/xagents"
XAGENTS_SCHEMA_PATH = "xagents/trajectory/schemas/groot_agent_trajectory.v0.schema.json"
HOST_ENV = "XAGENTS_GITLAB_URL"
DEFAULT_REF = "main"
TOKEN_VARIABLES = ("XAGENTS_READ_TOKEN", "GITLAB_TOKEN", "CI_JOB_TOKEN")

MATCHES = 0
DRIFTED = 1
UNREADABLE = 2


def _token() -> str | None:
    """Return the first configured read token."""
    for name in TOKEN_VARIABLES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def raw_file_url(host: str, project: str, path: str, ref: str) -> str:
    """Build the GitLab raw-file API URL for one path at one ref."""
    return (
        f"{host}/api/v4/projects/{urllib.parse.quote(project, safe='')}"
        f"/repository/files/{urllib.parse.quote(path, safe='')}/raw"
        f"?ref={urllib.parse.quote(ref, safe='')}"
    )


def fetch_upstream(url: str, token: str | None, timeout: float = 30.0) -> dict[str, Any]:
    """Read the upstream schema, raising RuntimeError with why on any failure."""
    request = urllib.request.Request(url)
    if token:
        request.add_header("PRIVATE-TOKEN", token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        hint = " (is a read token set?)" if exc.code in (401, 403, 404) else ""
        raise RuntimeError(f"HTTP {exc.code} reading {url}{hint}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not reach {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{url} did not return JSON: {exc}") from exc


def _contract_body(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Drop location metadata so a public $id is not reported as contract drift."""
    body = json.loads(json.dumps(schema))
    if isinstance(body, dict):
        body.pop("$id", None)
    return body


def differences(vendored: Mapping[str, Any], upstream: Mapping[str, Any]) -> list[str]:
    """Describe how the two schemas disagree, in terms of what they declare."""
    if _contract_body(vendored) == _contract_body(upstream):
        return []
    problems = []
    for label, path in (("record", ()), ("step", ("$defs", "step"))):
        ours = _at(vendored, path)
        theirs = _at(upstream, path)
        ours_fields = set(ours.get("properties", {}))
        theirs_fields = set(theirs.get("properties", {}))
        for name in sorted(theirs_fields - ours_fields):
            problems.append(f"{label}: upstream declares {name!r} and the copy does not")
        for name in sorted(ours_fields - theirs_fields):
            problems.append(f"{label}: the copy declares {name!r} and upstream does not")
        ours_required = set(ours.get("required", ()))
        theirs_required = set(theirs.get("required", ()))
        for name in sorted(theirs_required - ours_required):
            problems.append(f"{label}: upstream requires {name!r} and the copy does not")
        for name in sorted(ours_required - theirs_required):
            problems.append(f"{label}: the copy requires {name!r} and upstream does not")
    # The two are unequal, so say so even when the field sets happen to agree: a changed
    # enum or description is still a copy that is no longer the contract.
    return problems or ["the two differ somewhere other than their field sets"]


def _at(schema: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, Any]:
    """Walk a nested schema path, returning an empty mapping when it is absent."""
    node: Any = schema
    for key in path:
        node = node.get(key, {}) if isinstance(node, Mapping) else {}
    return node if isinstance(node, Mapping) else {}


def main(argv: list[str] | None = None) -> int:
    """Compare the vendored schema with the upstream one and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get(HOST_ENV, ""),
        help=f"GitLab origin URL, or set {HOST_ENV}",
    )
    parser.add_argument("--project", default=XAGENTS_PROJECT)
    parser.add_argument("--path", default=XAGENTS_SCHEMA_PATH)
    parser.add_argument("--ref", default=DEFAULT_REF, help="the ref to compare against")
    args = parser.parse_args(argv)

    host = args.host.strip()
    if not host:
        print(  # noqa: T201
            f"could not read the upstream schema: set --host or {HOST_ENV}",
            file=sys.stderr,
        )
        return UNREADABLE

    url = raw_file_url(host, args.project, args.path, args.ref)
    try:
        upstream = fetch_upstream(url, _token())
    except RuntimeError as exc:
        print(f"could not read the upstream schema: {exc}", file=sys.stderr)  # noqa: T201
        return UNREADABLE

    vendored = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    problems = differences(vendored, upstream)
    if not problems:
        print(f"the vendored schema matches {args.project}@{args.ref}")  # noqa: T201
        return MATCHES
    print(  # noqa: T201
        f"the vendored schema has fallen behind {args.project}@{args.ref}:",
        file=sys.stderr,
    )
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)  # noqa: T201
    print(  # noqa: T201
        f"refresh {SCHEMA_PATH} and rerun the wire-contract tests.",
        file=sys.stderr,
    )
    return DRIFTED


if __name__ == "__main__":
    raise SystemExit(main())
