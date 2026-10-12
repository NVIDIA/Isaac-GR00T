# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for locked AutoResearch training-config fields."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from skills.autoresearch.train_config_lock import (
    DEFAULT_LOCKED_FIELDS,
    MISSING,
    TrainConfigLock,
)

CONFIG_PATH = "skills/autoresearch/scripts/train_config.yaml"


def _baseline() -> dict:
    return {
        "data": {
            "datasets": [
                {
                    "dataset_paths": ["/data/yam/train"],
                    "mix_ratio": 1.0,
                    "embodiment_tag": "yam",
                }
            ]
        },
        "training": {
            "start_from_checkpoint": "/checkpoints/base",
            "max_steps": 1000,
            "per_device_batch_size": 8,
            "num_gpus": 4,
            "learning_rate": 5.0e-5,
        },
        "model": {
            "tokenizer_path": "/checkpoints/text-encoder",
            "text_encoder_cfg": {"text_encoder_pretrained_path": "/checkpoints/text-encoder"},
        },
    }


def _edited(**training) -> dict:
    config = copy.deepcopy(_baseline())
    config["training"].update(training)
    return config


def _with_text_encoder(path: str) -> dict:
    config = copy.deepcopy(_baseline())
    config["model"]["text_encoder_cfg"]["text_encoder_pretrained_path"] = path
    return config


def _without_training_field(name: str) -> dict:
    config = copy.deepcopy(_baseline())
    del config["training"][name]
    return config


def _with_train_paths(*paths: str) -> dict:
    config = copy.deepcopy(_baseline())
    config["data"]["datasets"][0]["dataset_paths"] = list(paths)
    return config


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(_edited(learning_rate=1.0e-4), id="recipe-only-edit"),
        # Halving the micro-batch while doubling accumulation preserves the global batch.
        pytest.param(
            _edited(per_device_batch_size=4, gradient_accumulation_steps=2),
            id="micro-batch-with-accumulation",
        ),
    ],
)
def test_recipe_edits_that_keep_the_budget_are_accepted(candidate: dict) -> None:
    lock = TrainConfigLock(path=CONFIG_PATH, eval_dataset_paths=("/data/yam/eval",))

    assert lock.violations(baseline=_baseline(), candidate=candidate) == []


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        pytest.param(
            _with_train_paths("/data/yam/train", "/data/yam/extra"),
            "locked field data.datasets changed",
            id="extra-training-data",
        ),
        pytest.param(
            _edited(max_steps=5000), "locked field training.max_steps changed", id="max-steps"
        ),
        pytest.param(
            _edited(start_from_checkpoint="/checkpoints/other"),
            "locked field training.start_from_checkpoint changed",
            id="base-model",
        ),
        # Resume loads another checkpoint's weights and trainer state after the base model.
        pytest.param(
            _edited(resume_from_checkpoint="/runs/trial-1/training"),
            "locked field training.resume_from_checkpoint changed from <missing>",
            id="resume-checkpoint",
        ),
        # Tokenizer and processor files come from this path, not from checkpoint weights.
        pytest.param(
            _with_text_encoder("/checkpoints/other-vlm"),
            "locked field model.text_encoder_cfg.text_encoder_pretrained_path changed",
            id="text-encoder",
        ),
        # Removing a locked field falls back to a trainer default, which is still a change.
        pytest.param(
            _without_training_field("max_steps"),
            "locked field training.max_steps changed from 1000 to <missing>",
            id="removed-locked-field",
        ),
        pytest.param(
            _edited(per_device_batch_size=16),
            "effective global batch changed from 32 to 64",
            id="global-batch",
        ),
        pytest.param(MISSING, "locked training config was deleted", id="deleted-config"),
    ],
)
def test_budget_data_and_base_model_changes_are_rejected(candidate, expected: str) -> None:
    lock = TrainConfigLock(path=CONFIG_PATH)

    violations = lock.violations(baseline=_baseline(), candidate=candidate)

    assert len(violations) == 1
    assert violations[0].startswith(f"{CONFIG_PATH}: ")
    assert expected in violations[0]


@pytest.mark.parametrize(
    ("train_path", "eval_path"),
    [
        pytest.param("/data/yam/eval", "/data/yam/eval/", id="same-directory"),
        pytest.param("/data/yam", "/data/yam/eval", id="train-contains-eval"),
        pytest.param("/data/yam/eval/episode_0", "/data/yam/eval", id="eval-contains-train"),
        pytest.param("hf://datasets/org/yam/eval", "hf://datasets/org/yam", id="uri"),
    ],
)
def test_training_on_evaluation_data_is_rejected(train_path: str, eval_path: str) -> None:
    # No field locks, so only the held-out overlap check can reject the candidate.
    lock = TrainConfigLock(path=CONFIG_PATH, fields=())

    violations = lock.violations(
        baseline=_baseline(),
        candidate=_with_train_paths("/data/yam/train", train_path),
        extra_eval_dataset_paths=(eval_path,),
    )

    assert violations == [
        f"{CONFIG_PATH}: training dataset {train_path} overlaps evaluation dataset {eval_path}"
    ]


def test_symlinked_evaluation_data_is_rejected(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    alias = tmp_path / "train_alias"
    alias.symlink_to(eval_dir)
    lock = TrainConfigLock(path=CONFIG_PATH, fields=(), eval_dataset_paths=(str(eval_dir),))

    violations = lock.violations(baseline=_baseline(), candidate=_with_train_paths(str(alias)))

    assert len(violations) == 1
    assert "overlaps evaluation dataset" in violations[0]


def test_sibling_dataset_with_shared_prefix_is_not_an_overlap() -> None:
    lock = TrainConfigLock(path=CONFIG_PATH, eval_dataset_paths=("/data/yam/eval",))

    candidate = _with_train_paths("/data/yam/evaluation_extra")

    assert lock.violations(baseline=candidate, candidate=candidate) == []


def test_from_mapping_defaults_and_validation(tmp_path: Path) -> None:
    source = tmp_path / "autoresearch.yaml"

    lock = TrainConfigLock.from_mapping({"path": CONFIG_PATH}, source)

    assert lock.fields == DEFAULT_LOCKED_FIELDS
    assert lock.eval_dataset_paths == ()
    with pytest.raises(ValueError, match="relative to the workspace"):
        TrainConfigLock.from_mapping({"path": "/abs/train_config.yaml"}, source)
    with pytest.raises(TypeError, match="fields must be a list"):
        TrainConfigLock.from_mapping({"path": CONFIG_PATH, "fields": "data.datasets"}, source)
