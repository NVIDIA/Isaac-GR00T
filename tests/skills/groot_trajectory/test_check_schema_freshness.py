# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests that the freshness check reports drift, and says so when it cannot look."""

from __future__ import annotations

import json

from skills.groot_trajectory import check_schema_freshness as freshness
from skills.groot_trajectory.wire_contract import SCHEMA_PATH


def _vendored() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_the_copy_matches_itself():
    assert freshness.differences(_vendored(), _vendored()) == []


def test_a_different_schema_id_is_not_contract_drift():
    upstream = _vendored()
    upstream["$id"] = "https://example.invalid/schema.json"
    assert freshness.differences(_vendored(), upstream) == []


def test_a_field_added_upstream_is_reported_as_the_copy_falling_behind():
    upstream = _vendored()
    upstream["$defs"]["step"]["properties"]["cost_usd"] = {"type": "number"}
    upstream["$defs"]["step"]["required"].append("cost_usd")

    assert freshness.differences(_vendored(), upstream) == [
        "step: upstream declares 'cost_usd' and the copy does not",
        "step: upstream requires 'cost_usd' and the copy does not",
    ]


def test_a_field_dropped_upstream_is_reported_the_other_way_round():
    # This is the step_id removal seen from the copy's side: upstream deleted a field and
    # a producer trusting the stale copy would keep emitting it.
    upstream = _vendored()
    del upstream["properties"]["task"]
    upstream["required"].remove("task")

    assert freshness.differences(_vendored(), upstream) == [
        "record: the copy declares 'task' and upstream does not",
        "record: the copy requires 'task' and upstream does not",
    ]


def test_a_change_that_is_not_about_field_sets_is_still_drift():
    # A narrowed enum leaves both field sets identical and still makes the copy wrong.
    upstream = _vendored()
    upstream["$defs"]["step"]["properties"]["status"]["enum"] = ["ok"]

    assert freshness.differences(_vendored(), upstream) == [
        "the two differ somewhere other than their field sets"
    ]


def test_the_url_names_the_file_and_the_ref():
    url = freshness.raw_file_url("https://host", "group/proj", "a/b.json", "main")

    assert url == (
        "https://host/api/v4/projects/group%2Fproj/repository/files/a%2Fb.json/raw?ref=main"
    )


def test_missing_host_is_not_reported_as_agreement(monkeypatch, capsys) -> None:
    monkeypatch.delenv(freshness.HOST_ENV, raising=False)
    assert freshness.main([]) == freshness.UNREADABLE
    assert "could not read the upstream schema" in capsys.readouterr().err


def test_being_unable_to_look_is_not_reported_as_agreement(monkeypatch, capsys):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("HTTP 404 reading ... (is a read token set?)")

    monkeypatch.setattr(freshness, "fetch_upstream", refuse)

    assert freshness.main(["--host", "https://example.invalid"]) == freshness.UNREADABLE
    assert "could not read the upstream schema" in capsys.readouterr().err
