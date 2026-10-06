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

"""CPU-only regression tests for the safe-YAML contract on
:class:`gr00t.configs.base_config.Config`: the save side must emit
plain-dict YAML (no ``!!python/`` tags) and the load sides
(``Config.load`` / ``Config.from_pretrained``) must refuse to
instantiate any Python object from a maliciously crafted config."""

from __future__ import annotations

import pytest


def _import_config():
    """Lazy import — ``Config`` pulls heavy deps that the CPU-only test
    venv may lack at collection time. Skip if unimportable."""
    try:
        from gr00t.configs.base_config import Config, get_default_config
    except Exception as e:  # pragma: no cover - skip path
        pytest.skip(f"gr00t.configs.base_config not importable in this env: {e}")
    return Config, get_default_config


# ---------------------------------------------------------------------------
# Security: the !!python/object tag must never instantiate
# ---------------------------------------------------------------------------


def test_save_does_not_emit_python_object_tags(tmp_path):
    """Save() output must be plain-dict YAML — no ``!!python/`` tag
    anywhere, so the file cannot reopen the construct-from-disk path."""
    Config, get_default_config = _import_config()
    cfg = get_default_config()

    out = tmp_path / "config.yaml"
    cfg.save(out)
    text = out.read_text()

    assert "!!python/" not in text, (
        "Config.save() must not emit !!python/ tags — those re-open the RCE "
        "path the safe loader is meant to close. Found python tags in:\n"
        f"{text[:500]}"
    )


def test_load_rejects_malicious_python_object_tag(tmp_path, monkeypatch):
    """``!!python/object/apply:os.system`` must surface a ``ValueError``
    with a migration hint and execute no side effect — proven by the
    untouched canary file."""
    Config, _ = _import_config()
    canary = tmp_path / "canary.txt"
    assert not canary.exists()

    # PyYAML RCE template equivalent to ``os.system(f"touch {canary}")``.
    malicious = tmp_path / "malicious.yaml"
    malicious.write_text(f"!!python/object/apply:os.system\n- 'touch {canary}'\n")

    with pytest.raises(ValueError) as excinfo:
        Config().load(malicious)

    assert not canary.exists(), (
        "Loading the malicious YAML must not have executed the os.system "
        "payload — that would mean the safe loader is not in effect."
    )
    msg = str(excinfo.value)
    assert "rejected unsafe legacy config YAML" in msg or "python/object" in msg.lower(), (
        "Migration error should clearly signal that the legacy unsafe YAML "
        f"format was rejected. Got: {msg!r}"
    )


def test_from_pretrained_rejects_malicious_python_object_tag(tmp_path):
    """The class-level loader shares the same safe-loader contract — pin
    the RCE rejection here so a future refactor cannot diverge the two
    code paths."""
    Config, _ = _import_config()
    canary = tmp_path / "canary_classmethod.txt"
    malicious = tmp_path / "malicious.yaml"
    malicious.write_text(f"!!python/object/apply:os.system\n- 'touch {canary}'\n")
    with pytest.raises(ValueError):
        Config.from_pretrained(malicious)
    assert not canary.exists()


def test_load_rejects_non_mapping_top_level(tmp_path):
    """``Config.save()`` always emits a top-level mapping; a YAML that
    decodes to e.g. a list must fail loud rather than silently produce
    an inert Config."""
    Config, _ = _import_config()
    odd = tmp_path / "odd.yaml"
    odd.write_text("- a\n- b\n- c\n")
    with pytest.raises(ValueError, match="Expected a YAML mapping"):
        Config().load(odd)


# ---------------------------------------------------------------------------
# Round-trip — save → load reconstructs equivalent dict-form fields
# ---------------------------------------------------------------------------


def test_safe_dump_yaml_can_be_loaded_back(tmp_path):
    """``save()`` output must be safe-loadable and reconstruct the
    top-level dataclasses that :meth:`Config.load_dict` knows how to
    rebuild (``model`` / ``data`` / ``training``). Nested dict-typed
    fields (e.g. ``data.modality_configs``) come back as raw dicts
    under ``yaml.safe_load`` — that's the deliberate type-erasure cost
    of dropping ``!!python/object``; recovering nested dataclass types
    would belong to a separate load_dict enhancement."""
    Config, get_default_config = _import_config()
    from dataclasses import asdict

    cfg = get_default_config()
    out = tmp_path / "config.yaml"
    cfg.save(out)

    # Round-trip succeeds: no RepresenterError on enums (save side),
    # no ConstructorError on safe_load (load side).
    loaded = Config().load(out)

    assert type(loaded.training) is type(cfg.training)
    assert type(loaded.data) is type(cfg.data)
    assert asdict(loaded.training) == asdict(cfg.training)
    assert loaded.data.datasets == cfg.data.datasets


def test_save_serialises_enum_action_configs(tmp_path):
    """``ActionConfig`` carries ``ActionRepresentation`` / ``ActionType`` /
    ``ActionFormat`` enum fields and is reachable from the default
    ``MODALITY_CONFIGS``. ``yaml.safe_dump`` refuses raw ``Enum``
    instances, so ``Config.save()`` must lower them to ``.value``
    strings via ``_build_safe_tree`` before dumping — otherwise the
    pre-fix production save path (``experiment.py``) regresses to a
    ``RepresenterError`` at training start."""
    Config, get_default_config = _import_config()
    cfg = get_default_config()
    out = tmp_path / "config.yaml"
    cfg.save(out)
    text = out.read_text()

    assert "!!python" not in text
    # No enum repr leaks (e.g. ``ActionRepresentation.RELATIVE``); only
    # the bare ``.value`` strings should appear.
    assert "ActionRepresentation" not in text
    assert "ActionFormat" not in text
    assert "ActionType" not in text
    # At least one known enum value from the default MODALITY_CONFIGS
    # made it to disk as a plain string.
    assert "relative" in text


def test_save_creates_parent_directory(tmp_path):
    """Nested target directories must still be created — pin
    ``mkdir(parents=True)`` so the safe-yaml refactor cannot drop it."""
    Config, get_default_config = _import_config()
    cfg = get_default_config()
    nested = tmp_path / "a" / "b" / "c" / "config.yaml"
    cfg.save(nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# Belt-and-suspenders: canary file must never appear during teardown.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_stray_canary(tmp_path):
    """Teardown safety net: if any security test accidentally executed
    its YAML payload, the canary file would exist under ``tmp_path``.
    Assert per-test rather than per-process (a stray ``/tmp/canary.txt``
    from an unrelated CI workload would otherwise cause spurious
    failures with a misleading regression message)."""
    yield
    for canary_name in ("canary.txt", "canary_classmethod.txt"):
        assert not (tmp_path / canary_name).exists(), (
            f"{canary_name} must not exist — its presence proves a malicious "
            "YAML payload executed and the RCE fix regressed."
        )
