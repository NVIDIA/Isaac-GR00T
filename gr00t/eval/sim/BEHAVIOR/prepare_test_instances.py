from pathlib import Path

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    dir_path = Path(__file__).parent

    test_instances_path = dir_path / "test_instances"
    test_instances_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="behavior-1k/2025-challenge-hidden-instances",
        repo_type="dataset",
        local_dir=test_instances_path,
    )
