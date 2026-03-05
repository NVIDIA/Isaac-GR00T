import pytest


@pytest.mark.gpu
def test_torchcodec_importable() -> None:
    """Smoke test that torchcodec is importable in the CI environment."""
    import torchcodec

    assert torchcodec is not None
