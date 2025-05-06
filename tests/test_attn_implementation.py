from unittest.mock import MagicMock, patch

import pytest

from gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat import (
    Eagle2ChatConfig,
)
from gr00t.model.backbone.eagle_backbone import EagleBackbone
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.policy import Gr00tPolicy


@pytest.fixture(autouse=True)
def patch_metadata_and_horizons(monkeypatch):
    monkeypatch.setattr(Gr00tPolicy, "_load_metadata", lambda self, cfg: None)
    monkeypatch.setattr(Gr00tPolicy, "_load_horizons", lambda self: None)


@pytest.fixture
def common_args():
    return {
        "model_path": "fake/model",
        "embodiment_tag": "gr1",
        "modality_config": {"video": None, "state": None},
        "modality_transform": MagicMock(),
    }


@patch("gr00t.model.gr00t_n1.snapshot_download", lambda p, **kw: p)
@patch("gr00t.model.gr00t_n1.PreTrainedModel.from_pretrained")
@patch("gr00t.model.gr00t_n1.AutoConfig.from_pretrained")
def test_from_pretrained_attn_impl_injection_cpu(mock_auto_config, mock_super_from_pretrained):
    # Simulate CPU environment
    with patch("torch.cuda.is_available", return_value=False):
        with patch("importlib.util.find_spec", side_effect=ImportError):
            # Setup mock config with backbone_cfg dict
            mock_config = MagicMock()
            mock_config.backbone_cfg = {}
            mock_auto_config.return_value = mock_config

            _ = GR00T_N1.from_pretrained("fake_path", attn_implementation="auto")

            # Assert final attn impl injected as 'eager' on CPU
            assert mock_config.backbone_cfg["attn_implementation"] == "eager"
            mock_super_from_pretrained.assert_called_once_with(
                "fake_path",
                config=mock_config,
                local_model_path="fake_path",
            )


@patch("gr00t.model.gr00t_n1.snapshot_download", lambda p, **kw: p)
@patch("gr00t.model.gr00t_n1.PreTrainedModel.from_pretrained")
@patch("gr00t.model.gr00t_n1.AutoConfig.from_pretrained")
def test_from_pretrained_attn_impl_injection_gpu(mock_auto_config, mock_super_from_pretrained):
    # Simulate GPU environment and flash_attn available
    with patch("torch.cuda.is_available", return_value=True):
        with patch("importlib.util.find_spec", return_value=True):
            mock_config = MagicMock()
            mock_config.backbone_cfg = {}
            mock_auto_config.return_value = mock_config

            _ = GR00T_N1.from_pretrained("fake_path", attn_implementation="auto")

            # Assert final attn impl injected as 'flash_attention_2' on GPU
            assert mock_config.backbone_cfg["attn_implementation"] == "flash_attention_2"
            mock_super_from_pretrained.assert_called_once_with(
                "fake_path",
                config=mock_config,
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
