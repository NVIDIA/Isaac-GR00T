from unittest.mock import MagicMock, patch

import pytest
from gr00t.model.gr00t_n1 import GR00T_N1, select_attn_impl
from gr00t.utils.attn_impl import _flash2_available
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat import (
    Eagle2ChatConfig,
)
from gr00t.model.backbone.eagle_backbone import EagleBackbone

@pytest.fixture(autouse=True)
def patch_metadata_and_horizons(monkeypatch, tmp_path):
    monkeypatch.setattr(Gr00tPolicy, "_load_metadata", lambda self, cfg: None)
    monkeypatch.setattr(Gr00tPolicy, "_load_horizons", lambda self: None)
    fake_cfg = tmp_path / "experiment_cfg"
    fake_cfg.mkdir()
    (fake_cfg / "metadata.json").write_text("{}")
    return tmp_path

@pytest.fixture
def common_args():
    return {
        "model_path": "fake/model",
        "embodiment_tag": "gr1",
        "modality_config": {"video": None, "state": None},
        "modality_transform": MagicMock(),
    }

@pytest.mark.parametrize("capability,expected", [
    ((8, 0), True),   # Ampere
    ((8, 6), True),   # Ada
    ((9, 0), True),   # Hopper
    ((7, 5), False),  # Turing and below
])
def test_flash2_availability_by_compute(capability, expected):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=capability), \
         patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch.dict("sys.modules", {"flash_attn": MagicMock()}):
        assert _flash2_available() is expected


@pytest.mark.parametrize("cuda_available,cc,flash_installed,user_choice,expected", [
    # auto on flash2-compatible GPU → flash_attention_2
    (True, (8, 0), True, "auto", "flash_attention_2"),
    # auto on older GPU → sdpa
    (True, (7, 0), False, "auto", "sdpa"),
    # auto on CPU → eager
    (False, None, False, "auto", "eager"),
    # explicit flash two on unsupported → fallback eager
    (True, (7, 0), False, "flash_attention_2", "eager"),
    # explicit sdpa on CPU → fallback eager
    (False, None, False, "sdpa", "eager"),
    # explicit eager always
    (False, None, False, "eager", "eager"),
])
def test_select_attn_impl_all_branches(cuda_available, cc, flash_installed, user_choice, expected):
    spec_val = MagicMock() if flash_installed else None
    module_dict = {"flash_attn": MagicMock()} if flash_installed else {}
    with patch("torch.cuda.is_available", return_value=cuda_available), \
         patch("importlib.util.find_spec", return_value=spec_val), \
         patch.dict("sys.modules", module_dict), \
         patch("torch.cuda.get_device_capability", return_value=cc if cc is not None else (0, 0)):
        result = select_attn_impl(user_choice)
        assert result == expected


# -------------------------------------------------------------------
# Tests for GR00T_N1.from_pretrained injection logic
# -------------------------------------------------------------------

@patch("gr00t.model.gr00t_n1.snapshot_download", lambda p, **kw: p)
@patch("gr00t.model.gr00t_n1.PreTrainedModel.from_pretrained")
def test_from_pretrained_attn_impl_injection_cpu(mock_super_from_pretrained):
    with patch("torch.cuda.is_available", return_value=False), \
         patch("importlib.util.find_spec", side_effect=ImportError):
        mock_model = MagicMock()
        mock_model.config = MagicMock(backbone_cfg={})
        mock_super_from_pretrained.return_value = mock_model

        _ = GR00T_N1.from_pretrained("fake_path", attn_implementation="auto")

        assert mock_model.config.backbone_cfg["attn_implementation"] == "eager"
        mock_super_from_pretrained.assert_called_once_with(
            "fake_path",
            local_model_path="fake_path",
        )


@patch("gr00t.model.gr00t_n1.snapshot_download", lambda p, **kw: p)
@patch("gr00t.model.gr00t_n1.PreTrainedModel.from_pretrained")
def test_from_pretrained_attn_impl_injection_gpu_ampere(mock_super_from_pretrained):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(8, 0)), \
         patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch.dict("sys.modules", {"flash_attn": MagicMock()}):
        mock_model = MagicMock()
        mock_model.config = MagicMock(backbone_cfg={})
        mock_super_from_pretrained.return_value = mock_model

        _ = GR00T_N1.from_pretrained("fake_path", attn_implementation="auto")

        assert mock_model.config.backbone_cfg["attn_implementation"] == "flash_attention_2"
        mock_super_from_pretrained.assert_called_once_with(
            "fake_path",
            local_model_path="fake_path",
        )


@patch("gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat.SiglipVisionConfig")
@patch("gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat.LlamaConfig")
def test_config_attn_propagation(mock_llama, mock_siglip):
    vision = {"model_type": "siglip_vision_model"}
    llm = {"architectures": ["LlamaForCausalLM"], "vocab_size": 123}

    Eagle2ChatConfig(
        vision_config=vision.copy(), llm_config=llm.copy(), attn_implementation="flash_attention_2"
    )
    mock_siglip.assert_called_with(
        model_type="siglip_vision_model", _attn_implementation="flash_attention_2"
    )
    mock_llama.assert_called_with(
        architectures=["LlamaForCausalLM"], vocab_size=123, attn_implementation="flash_attention_2"
    )

    mock_siglip.reset_mock()
    mock_llama.reset_mock()
    Eagle2ChatConfig(vision_config=vision.copy(), llm_config=llm.copy())
    mock_siglip.assert_called_with(model_type="siglip_vision_model")
    mock_llama.assert_called_with(architectures=["LlamaForCausalLM"], vocab_size=123)


@patch("gr00t.model.backbone.eagle_backbone.AutoConfig")
@patch("gr00t.model.backbone.eagle_backbone.AutoModel")
def test_backbone_attn_impl_forward(mock_auto_model, mock_auto_config):
    fake_cfg = mock_auto_config.from_pretrained.return_value
    fake_model = mock_auto_model.from_config.return_value

    # **Add these lines** to satisfy the internal assertion
    fake_model.template = "test_template"
    fake_model.num_image_token = 5

    processor_cfg = {
        "model_path": "fake/processor/path",
        "max_input_tiles": 1,
        "model_spec": {"template": "test_template", "num_image_token": 5},
    }

    # Test explicit 'flash_attention_2'
    EagleBackbone(
        processor_cfg=processor_cfg,
        use_local_eagle_hg_model=False,
        model_name="m",
        attn_implementation="flash_attention_2",
    )
    mock_auto_config.from_pretrained.assert_called_with(
        "m", trust_remote_code=True, attn_implementation="flash_attention_2"
    )
    mock_auto_model.from_config.assert_called_with(
        fake_cfg, trust_remote_code=True, attn_implementation="flash_attention_2"
    )

    # Reset mocks for default case
    mock_auto_config.from_pretrained.reset_mock()
    mock_auto_model.from_config.reset_mock()

    # Test default 'eager'
    fake_model.template = "test_template"  # re‑set for the new instance
    fake_model.num_image_token = 5

    EagleBackbone(processor_cfg=processor_cfg, use_local_eagle_hg_model=False, model_name="m")
    mock_auto_config.from_pretrained.assert_called_with(
        "m", trust_remote_code=True, attn_implementation="eager"
    )
    mock_auto_model.from_config.assert_called_with(
        fake_cfg, trust_remote_code=True, attn_implementation="eager"
    )

# -------------------------------------------------------------------
# Smoke test for loading the LoRA checkpoint with Gr00tPolicy
# -------------------------------------------------------------------

@patch.object(Gr00tPolicy, "_load_model", autospec=True)
@patch("gr00t.model.policy.EmbodimentTag")
@patch("gr00t.model.policy.GR00T_N1.from_pretrained")
@patch("gr00t.model.policy.snapshot_download")
def test_policy_loads_lora_checkpoint(
    mock_snapshot_download,
    mock_from_pretrained,
    mock_EmbodimentTag,
    mock_load_model,
    tmp_path,
    patch_metadata_and_horizons
):
    fake_repo = tmp_path / "fake-lora"
    fake_repo.mkdir()
    (fake_repo / "experiment_cfg").mkdir()
    (fake_repo / "experiment_cfg" / "metadata.json").write_text("{}")

    mock_snapshot_download.return_value = str(fake_repo)
    def fake_load(self, model_path):
        object.__setattr__(self, "_model", MagicMock())
    mock_load_model.side_effect = fake_load
    policy = Gr00tPolicy(
        model_path="fake-lora",
        embodiment_tag="test",
        modality_config={},
        modality_transform=MagicMock(),
        device="cpu",
        attn_implementation="eager"
    )
    mock_snapshot_download.assert_called_once_with("fake-lora", repo_type="model")
    mock_load_model.assert_called_once()
    assert hasattr(policy, "_model"), "Policy did not load the internal model"
