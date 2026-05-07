# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the HF local-first cache probe in :mod:`gr00t`.

Regression coverage for CI Job 308778931, where the previous probe
(``snapshot_download(local_files_only=True)``) returned 71/71 false misses
across a 1.5h job because ``transformers.from_pretrained`` does not always
populate the ``refs/main`` snapshot metadata that ``snapshot_download``
requires.  The new strategy uses ``from_pretrained(local_files_only=True)``
itself as the cache probe — exactly the same logic the downloader uses —
so a hit/miss is consistent with what the underlying loader sees.
"""

from __future__ import annotations

from gr00t import _hf_local_first_call
import pytest


class _FakeKlass:
    """Sentinel class used as the first argument to a from_pretrained-shaped fn."""


def _make_orig(side_effect_by_local_files_only=None, default_return="ok"):
    """Build a fake ``orig_func`` recording each call.

    ``side_effect_by_local_files_only`` maps the value of the
    ``local_files_only`` kwarg (True/False/None) to either a return value or
    an Exception instance to raise.  Anything not in the map returns
    ``default_return``.
    """
    side_effect_by_local_files_only = side_effect_by_local_files_only or {}
    calls = []

    def fake(klass, name, *args, **kwargs):
        calls.append({"klass": klass, "name": name, "args": args, "kwargs": dict(kwargs)})
        local_only = kwargs.get("local_files_only", None)
        outcome = side_effect_by_local_files_only.get(local_only, default_return)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    fake.calls = calls
    return fake


class TestLocalFilesystemPath:
    """When ``pretrained_model_name_or_path`` is an existing directory, no
    cache probing is needed — pass through to ``orig_func`` unchanged."""

    def test_calls_orig_once_with_unchanged_kwargs(self, tmp_path):
        orig = _make_orig()
        result = _hf_local_first_call(orig, _FakeKlass, str(tmp_path), trust_remote_code=True)
        assert result == "ok"
        assert len(orig.calls) == 1
        call = orig.calls[0]
        assert call["name"] == str(tmp_path)
        assert call["kwargs"] == {"trust_remote_code": True}


class TestExplicitOfflineRequest:
    """If the caller already passed ``local_files_only=True``, honor it: do
    not retry on failure (caller has explicitly forbidden network)."""

    def test_passes_through_on_success(self):
        orig = _make_orig(default_return="cached")
        result = _hf_local_first_call(
            orig, _FakeKlass, "nvidia/Cosmos-Reason2-2B", local_files_only=True
        )
        assert result == "cached"
        assert len(orig.calls) == 1
        assert orig.calls[0]["kwargs"]["local_files_only"] is True

    def test_propagates_failure_without_retry(self):
        orig = _make_orig(
            side_effect_by_local_files_only={True: OSError("not in cache")},
        )
        with pytest.raises(OSError):
            _hf_local_first_call(
                orig, _FakeKlass, "nvidia/Cosmos-Reason2-2B", local_files_only=True
            )
        assert len(orig.calls) == 1, "must not retry when caller demanded offline"


class TestRepoIdCacheHit:
    """The common warm-cache case: probing with ``local_files_only=True``
    succeeds, so we never go to the network."""

    def test_calls_orig_once_with_local_files_only_true(self, capsys):
        orig = _make_orig(side_effect_by_local_files_only={True: "from-cache"})
        result = _hf_local_first_call(orig, _FakeKlass, "nvidia/Cosmos-Reason2-2B")
        assert result == "from-cache"
        assert len(orig.calls) == 1
        assert orig.calls[0]["kwargs"]["local_files_only"] is True
        out = capsys.readouterr().out
        assert "[groot/hf] cache hit:" in out
        assert "[groot/hf] cache miss" not in out

    def test_preserves_other_kwargs_on_probe(self):
        orig = _make_orig(side_effect_by_local_files_only={True: "from-cache"})
        _hf_local_first_call(orig, _FakeKlass, "repo/x", trust_remote_code=True, revision="abc")
        kwargs = orig.calls[0]["kwargs"]
        assert kwargs["trust_remote_code"] is True
        assert kwargs["revision"] == "abc"
        assert kwargs["local_files_only"] is True


class TestRepoIdCacheMissFallthrough:
    """Cold-cache case: probe raises, fall through to a normal call that the
    HF Hub will service via download."""

    def test_falls_through_to_normal_call_on_probe_failure(self, capsys):
        orig = _make_orig(
            side_effect_by_local_files_only={
                True: OSError("LocalEntryNotFoundError"),
                None: "downloaded",
            }
        )
        result = _hf_local_first_call(orig, _FakeKlass, "nvidia/Cosmos-Reason2-2B")
        assert result == "downloaded"
        assert len(orig.calls) == 2, (
            "expected one probe call and one download call, got "
            f"{[c['kwargs'] for c in orig.calls]}"
        )
        assert orig.calls[0]["kwargs"]["local_files_only"] is True
        assert "local_files_only" not in orig.calls[1]["kwargs"]
        out = capsys.readouterr().out
        assert "[groot/hf] cache miss (will download):" in out
        assert "[groot/hf] cache hit" not in out

    def test_propagates_failure_from_download_call(self):
        boom = RuntimeError("network down")
        orig = _make_orig(
            side_effect_by_local_files_only={
                True: OSError("not cached"),
                None: boom,
            }
        )
        with pytest.raises(RuntimeError, match="network down"):
            _hf_local_first_call(orig, _FakeKlass, "repo/x")
