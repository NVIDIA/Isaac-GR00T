from __future__ import annotations

import pathlib

import numpy as np
import pytest
from test_support.readme import extract_code_blocks, find_block, run_readme_python_blocks
from test_support.runtime import get_root
import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


REPO_ROOT = get_root()
POLICY_README = REPO_ROOT / "getting_started" / "policy.md"


# ---------------------------------------------------------------------------
# TinyGr00t — minimal HuggingFace-compatible model registered once per session
# ---------------------------------------------------------------------------


class TinyGr00tConfig(PretrainedConfig):
    model_type = "TinyGr00t"

    def __init__(self, action_horizon: int = 16, max_action_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.action_horizon = action_horizon
        self.max_action_dim = max_action_dim


class TinyGr00tModel(PreTrainedModel):
    """Minimal stand-in for Gr00tN1d6: accepts any kwargs, returns zero action_pred."""

    config_class = TinyGr00tConfig

    def __init__(self, config: TinyGr00tConfig):
        super().__init__(config)
        self._dummy = torch.nn.Linear(1, 1)

    def get_action(self, **kwargs) -> dict:
        batch_size = 1
        for v in kwargs.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                batch_size = v.shape[0]
                break
        return {
            "action_pred": torch.zeros(
                batch_size,
                self.config.action_horizon,
                self.config.max_action_dim,
                device=self._dummy.weight.device,
                dtype=self._dummy.weight.dtype,
            )
        }


AutoConfig.register("TinyGr00t", TinyGr00tConfig)
AutoModel.register(TinyGr00tConfig, TinyGr00tModel)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Save a TinyGr00t checkpoint to a temp directory and return its path."""
    ckpt_dir = tmp_path_factory.mktemp("tiny_gr00t")
    model = TinyGr00tModel(TinyGr00tConfig(action_horizon=16, max_action_dim=128))
    model.save_pretrained(ckpt_dir)
    return ckpt_dir


# ---------------------------------------------------------------------------
# MockProcessor — processor stand-in for policy.md inference tests
# ---------------------------------------------------------------------------

_POLICY_MD_ACTION_HORIZON = 16
_POLICY_MD_ACTION_DIM = 7


def _make_policy_md_modality_configs() -> dict:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ModalityConfig

    tag = EmbodimentTag.NEW_EMBODIMENT.value
    return {
        tag: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["wrist_cam"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
            "action": ModalityConfig(
                delta_indices=list(range(_POLICY_MD_ACTION_HORIZON)),
                modality_keys=["joints"],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        }
    }


class MockProcessor:
    def eval(self) -> None:
        pass

    def get_modality_configs(self) -> dict:
        return _make_policy_md_modality_configs()

    def __call__(self, messages) -> dict:
        return {}

    @property
    def collator(self):
        def _collate(inputs: list) -> dict:
            return {}

        return _collate

    def decode_action(self, action_array, embodiment_tag, batched_states) -> dict:
        return {
            "joints": action_array[:, :_POLICY_MD_ACTION_HORIZON, :_POLICY_MD_ACTION_DIM].astype(
                np.float32
            )
        }


@pytest.fixture
def mock_processor() -> MockProcessor:
    """Return a MockProcessor instance for policy.md tests."""
    return MockProcessor()


# ---------------------------------------------------------------------------
# Tiny model smoke tests
# ---------------------------------------------------------------------------


def test_automodel_loads_tiny_checkpoint(tiny_checkpoint: pathlib.Path) -> None:
    """AutoModel.from_pretrained round-trips the tiny checkpoint."""
    model = AutoModel.from_pretrained(tiny_checkpoint)
    assert isinstance(model, TinyGr00tModel)


def test_get_action_returns_correct_shape(tiny_checkpoint: pathlib.Path) -> None:
    """get_action returns action_pred with shape (B, action_horizon, max_action_dim)."""
    model = AutoModel.from_pretrained(tiny_checkpoint)
    result = model.get_action()
    assert "action_pred" in result
    assert result["action_pred"].shape == (1, 16, 128)


def test_get_action_respects_batch_size(tiny_checkpoint: pathlib.Path) -> None:
    """get_action infers batch size from the first tensor kwarg."""
    model = AutoModel.from_pretrained(tiny_checkpoint)
    result = model.get_action(state=torch.zeros(4, 10))
    assert result["action_pred"].shape[0] == 4


# ---------------------------------------------------------------------------
# Policy.md integration test — extracts blocks from the README directly
# ---------------------------------------------------------------------------


def test_policy_md_steps(
    monkeypatch: pytest.MonkeyPatch,
    tiny_checkpoint: pathlib.Path,
    mock_processor: MockProcessor,
) -> None:
    """Run every named step from policy.md using extracted README code blocks."""
    import gr00t.policy.gr00t_policy as _policy_module

    loaded_model = AutoModel.from_pretrained(tiny_checkpoint)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **_):
            return loaded_model

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path):
            return mock_processor

    monkeypatch.setattr(_policy_module, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(_policy_module, "AutoProcessor", _FakeAutoProcessor)

    blocks = extract_code_blocks(POLICY_README)

    # --- Loading the Policy ---
    loading_code = (
        find_block(blocks, "Gr00tPolicy(", language="python")
        .code.replace('"/path/to/your/checkpoint"', f'r"{tiny_checkpoint}"')
        .replace('"cuda:0"', '"cpu"')
    )

    # --- Querying Modality Configurations ---
    modality_code = find_block(blocks, "policy.get_modality_config()", language="python").code

    # Injected preamble: resolve undefined dimension variables used by the
    # batched-inference block.
    dims_preamble = (
        "import numpy as np\n"
        "T_video = video_horizon\n"
        "T_state = state_horizon\n"
        "H, W = 224, 224\n"
        "D_state = 7\n"
    )

    # --- Batched Inference ---
    batched_inference_code = find_block(blocks, "wrist_cam", language="python").code.replace(
        "batch_size = 4", "batch_size = 1"
    )

    # --- Running Inference — action access ---
    inference_code = find_block(blocks, "arm_action", language="python").code.replace(
        '"action_name"', "action_keys[0]"
    )

    # --- Resetting the Policy ---
    reset_code = find_block(blocks, "policy.reset()", language="python").code

    run_readme_python_blocks(
        [
            loading_code,
            modality_code,
            dims_preamble,
            batched_inference_code,
            inference_code,
            reset_code,
        ],
        readme_path=POLICY_README,
        repo_root=REPO_ROOT,
    )
