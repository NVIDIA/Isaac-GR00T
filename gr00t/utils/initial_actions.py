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

"""Safe (pickle-free) save/load for the per-dataset initial-actions cache.

Why this isn't just ``np.savez(path, the_nested_list)``:
    The cache is a nested ``list[dict[trajectory, dict[action_key, ndarray]]]``.
    numpy can only store plain arrays, so handing it the nested object forces
    it to *pickle* the whole structure — and reading a pickle runs whatever
    code the file tells it to. An ``initial_actions.npz`` can come from an
    untrusted place (a HuggingFace dataset bundle, shared NFS), so that load
    would be an arbitrary-code-execution hole.

How we avoid pickle:
    We flatten the nested structure to one plain array per leaf, keyed by a
    string ``"{dataset_idx}::{trajectory}::{action_key}"``. For example::

        [{"traj_0": {"action.arm": arr}}]       # nested input
        ->  {"0::traj_0::action.arm": arr,       # what's stored on disk
             "__schema__": <json: format / version / num_datasets>}

    Every stored value is now a plain numeric array, so saving needs no pickle
    and loading uses ``np.load(..., allow_pickle=False)`` (numpy's safe
    default). The ``__schema__`` marker lets the loader recognise our format
    and reject anything else. ``::`` is reserved as the field separator, so a
    trajectory / action-key string containing it is rejected at save time
    (otherwise the loader might split a key in the wrong place).
"""

import json
from pathlib import Path

import numpy as np


INITIAL_ACTIONS_FILENAME = "initial_actions.npz"

_KEY_SEP = "::"
_SCHEMA_KEY = "__schema__"
_FORMAT_VERSION = 1


def _encode_key(dataset_idx: int, trajectory: str, action_key: str) -> str:
    if _KEY_SEP in trajectory or _KEY_SEP in action_key:
        raise ValueError(
            f"trajectory / action_key must not contain {_KEY_SEP!r}; got "
            f"trajectory={trajectory!r}, action_key={action_key!r}. The "
            f"keyed-npz format reserves {_KEY_SEP!r} as a structural separator."
        )
    return f"{dataset_idx}{_KEY_SEP}{trajectory}{_KEY_SEP}{action_key}"


def _decode_key(key: str) -> tuple[int, str, str]:
    parts = key.split(_KEY_SEP, 2)
    if len(parts) != 3:
        raise ValueError(
            f"Malformed initial-actions key {key!r}: expected three "
            f"{_KEY_SEP!r}-separated fields (dataset_idx, trajectory, action_key)."
        )
    dataset_idx_str, trajectory, action_key = parts
    return int(dataset_idx_str), trajectory, action_key


def save_initial_actions(
    initial_actions: list[dict[str, dict[str, np.ndarray]]],
    initial_actions_path: str | Path,
) -> None:
    """Save the cache as a flat, pickle-free npz (see module docstring).

    Flattens the nested ``list[dict[traj, dict[action_key, ndarray]]]`` to one
    array per leaf under ``"{dataset_idx}::{trajectory}::{action_key}"`` keys,
    plus a ``__schema__`` marker. Every stored value is a plain array, so the
    matching loader can stay on the safe ``allow_pickle=False`` path.

    Raises ``ValueError`` if a trajectory or action-key string contains the
    reserved ``::`` separator.
    """
    flat: dict[str, np.ndarray] = {}
    for dataset_idx, dataset_actions in enumerate(initial_actions):
        for trajectory, action_dict in dataset_actions.items():
            for action_key, array in action_dict.items():
                encoded = _encode_key(dataset_idx, trajectory, action_key)
                flat[encoded] = np.asarray(array)
    schema = {
        "format": "gr00t.initial_actions",
        "version": _FORMAT_VERSION,
        "key_sep": _KEY_SEP,
        "num_datasets": len(initial_actions),
    }
    flat[_SCHEMA_KEY] = np.frombuffer(json.dumps(schema).encode("utf-8"), dtype=np.uint8)
    np.savez(str(initial_actions_path), **flat)


def load_initial_actions(
    initial_actions_path: str | Path,
) -> list[dict[str, dict[str, np.ndarray]]]:
    """Load the cache without ever enabling pickle, rebuilding the nested
    ``list[dict[traj, dict[action_key, ndarray]]]`` from the flat keys.

    Old files written before this fix were pickle-encoded (``np.savez(path,
    the_nested_list)``). Rather than silently re-enabling pickle to read them
    — which would re-open the code-execution hole — they are rejected with a
    clear "re-generate the cache" error. Files missing or mismatching the
    ``__schema__`` marker are rejected for the same reason.
    """
    # Narrow the np.load() call so its ValueError can't shadow our own
    # schema / decode validators below.
    try:
        npz_ctx = np.load(str(initial_actions_path), allow_pickle=False)
    except ValueError as e:
        # Match broadly: numpy's exact wording for the allow_pickle=False
        # rejection has shifted across releases ("Object arrays cannot be
        # loaded when allow_pickle=False" / "pickle"). Catch any of them.
        if "pickle" in str(e).lower():
            raise ValueError(
                f"{initial_actions_path}: rejected unsafe pickle-encoded "
                "initial-actions file. The pre-2026-05 save format embedded "
                "Python objects via numpy's pickle path, which is an arbitrary-"
                "code-execution gadget when the file comes from an external "
                "dataset / shared NFS. Re-generate the cache via the current "
                "save_initial_actions()."
            ) from e
        raise

    with npz_ctx as npz:
        keys = list(npz.files)
        if _SCHEMA_KEY not in keys:
            raise ValueError(
                f"{initial_actions_path}: missing {_SCHEMA_KEY!r} marker — "
                "this file appears to be in the pre-2026-05 pickle-based "
                "format (or an unrelated npz). Re-save it via "
                "save_initial_actions() to migrate to the safe keyed-npz "
                "format."
            )
        schema = json.loads(bytes(npz[_SCHEMA_KEY]).decode("utf-8"))
        if schema.get("format") != "gr00t.initial_actions":
            raise ValueError(
                f"{initial_actions_path}: unrecognised schema {schema!r}; "
                "expected format='gr00t.initial_actions'."
            )
        if schema.get("version") != _FORMAT_VERSION:
            raise ValueError(
                f"{initial_actions_path}: unsupported version "
                f"{schema.get('version')!r} (this build expects "
                f"{_FORMAT_VERSION})."
            )
        num_datasets = int(schema.get("num_datasets", 0))
        grouped: list[dict[str, dict[str, np.ndarray]]] = [{} for _ in range(num_datasets)]
        for key in keys:
            if key == _SCHEMA_KEY:
                continue
            dataset_idx, trajectory, action_key = _decode_key(key)
            if dataset_idx < 0 or dataset_idx >= num_datasets:
                raise ValueError(
                    f"{initial_actions_path}: key {key!r} references "
                    f"dataset_idx={dataset_idx} outside [0, {num_datasets})."
                )
            grouped[dataset_idx].setdefault(trajectory, {})[action_key] = np.asarray(npz[key])
    return grouped
