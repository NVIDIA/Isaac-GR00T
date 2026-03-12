from __future__ import annotations

import pathlib
import shutil

import pytest
from tests.examples.utils import build_shared_runtime_env, run_subprocess_step


ROOT = pathlib.Path(__file__).resolve().parents[3]

TRAINING_STEPS = 2

README = ROOT / "examples/SO100/README.md"

DATASET_REPO_ID = "izuluaga/finish_sandwich"
DATASET_ROOT = ROOT / "examples/SO100/finish_sandwich_lerobot"
DATASET_PATH = DATASET_ROOT / "izuluaga/finish_sandwich"
MODALITY_SRC = ROOT / "examples/SO100/modality.json"
MODALITY_DST = DATASET_PATH / "meta/modality.json"
MODEL_CHECKPOINT = pathlib.Path(f"/tmp/so100_finetune/checkpoint-{TRAINING_STEPS}")


def _cleanup_dataset_path() -> None:
    """Remove the dataset directory created by the SO100 workflow."""
    try:
        if DATASET_ROOT.is_symlink():
            DATASET_ROOT.unlink()
        elif DATASET_ROOT.exists():
            shutil.rmtree(DATASET_ROOT)
    except OSError as exc:
        print(f"[so100] cleanup_warning path={DATASET_PATH} error={exc}", flush=True)


def _normalize_ws(text: str) -> str:
    """Normalize all whitespace to single spaces for robust snippet matching."""
    # Ignore shell line-continuation backslashes in README code blocks.
    return " ".join(text.replace("\\\n", " ").replace("\\", " ").split())


@pytest.mark.gpu
@pytest.mark.timeout(400)
def test_so100_readme_workflow_executes_via_subprocess() -> None:
    """Run the README's dataset handling, finetuning, and open-loop eval commands."""

    env = build_shared_runtime_env(
        "so100",
        extra_env={
            # Prevent git-lfs smudge failures for non-runtime artifacts when uv
            # resolves git dependencies (e.g. huggingface/lerobot).
            "GIT_LFS_SKIP_SMUDGE": "1",
        },
    )
    print(f"[so100] uv_env={env.get('UV_PROJECT_ENVIRONMENT', '<unset>')}", flush=True)
    try:
        # Step 1: Convert dataset (README: Handling the dataset)
        run_subprocess_step(
            [
                "uv",
                "run",
                "--project",
                "scripts/lerobot_conversion",
                "python",
                "scripts/lerobot_conversion/convert_v3_to_v2.py",
                "--repo-id",
                DATASET_REPO_ID,
                "--root",
                str(DATASET_ROOT),
            ],
            step="convert_v3_to_v2",
            cwd=ROOT,
            env=env,
            log_prefix="so100",
            failure_prefix="SO100 README step failed",
            output_tail_chars=8000,
        )

        # Step 2: Copy modality.json into dataset meta directory.
        MODALITY_DST.parent.mkdir(parents=True, exist_ok=True)
        run_subprocess_step(
            ["cp", str(MODALITY_SRC), str(MODALITY_DST)],
            step="copy_modality_json",
            cwd=ROOT,
            env=env,
            log_prefix="so100",
            failure_prefix="SO100 README step failed",
            output_tail_chars=8000,
        )
        assert MODALITY_DST.is_file(), f"Expected modality file after copy: {MODALITY_DST}"

        # Step 3: Finetune model (README: Finetuning).
        run_subprocess_step(
            [
                "uv",
                "run",
                "bash",
                "examples/SO100/finetune_so100.sh",
            ],
            step="finetune_so100",
            cwd=ROOT,
            env={
                **env,
                "SAVE_STEPS": str(TRAINING_STEPS),
                "MAX_STEPS": str(TRAINING_STEPS),
                "USE_WANDB": "0",
                "DATALOADER_NUM_WORKERS": "0",
                "GLOBAL_BATCH_SIZE": "2",
                "SHARD_SIZE": "64",
                "NUM_SHARDS_PER_EPOCH": "1",
                "EPISODE_SAMPLING_RATE": "0.02",
            },
            log_prefix="so100",
            failure_prefix="SO100 README step failed",
            output_tail_chars=8000,
        )
        assert MODEL_CHECKPOINT.exists(), (
            f"Expected model checkpoint after finetune: {MODEL_CHECKPOINT}"
        )

        # Step 4: Open-loop evaluation.
        run_subprocess_step(
            [
                "uv",
                "run",
                "python",
                "gr00t/eval/open_loop_eval.py",
                "--dataset-path",
                str(DATASET_PATH),
                "--embodiment-tag",
                "NEW_EMBODIMENT",
                "--model-path",
                str(MODEL_CHECKPOINT),
                "--traj-ids",
                "0",
                "--action-horizon",
                "16",
                "--steps",
                "5",
            ],
            step="open_loop_eval",
            cwd=ROOT,
            env=env,
            log_prefix="so100",
            failure_prefix="SO100 README step failed",
            output_tail_chars=8000,
        )
    finally:
        _cleanup_dataset_path()


def test_so100_readme_commands_have_not_drifted() -> None:
    """Ensure the README still documents the command flow this test validates."""
    readme_text = README.read_text(encoding="utf-8")
    normalized_readme = _normalize_ws(readme_text)
    expected_snippets = [
        (
            "dataset_conversion",
            "uv run --project scripts/lerobot_conversion python "
            "scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id "
            "izuluaga/finish_sandwich --root examples/SO100/finish_sandwich_lerobot",
        ),
        (
            "copy_modality",
            "cp examples/SO100/modality.json "
            "examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich/meta/modality.json",
        ),
        (
            "finetune",
            "uv run bash examples/SO100/finetune_so100.sh",
        ),
        (
            "open_loop_eval",
            "uv run python gr00t/eval/open_loop_eval.py --dataset-path "
            "examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich/ "
            "--embodiment-tag NEW_EMBODIMENT --model-path "
            "/tmp/so100_finetune/checkpoint-10000 --traj-ids 0 --action-horizon 16 --steps 400",
        ),
    ]

    missing = [
        name
        for name, snippet in expected_snippets
        if _normalize_ws(snippet) not in normalized_readme
    ]
    assert not missing, (
        "SO100 README command drift detected for sections: "
        f"{', '.join(missing)}. Update tests or README to keep them aligned."
    )
