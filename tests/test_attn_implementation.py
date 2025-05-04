import pytest
from unittest.mock import patch, MagicMock

from gr00t.model.backbone.eagle_backbone import EagleBackbone
from gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat import Eagle2ChatConfig

@patch('gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat.SiglipVisionConfig')
@patch('gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat.LlamaConfig')
def test_config_attn_propagation(mock_llama, mock_siglip):
    vision = {"model_type":"siglip_vision_model"}
    llm    = {"architectures":["LlamaForCausalLM"], "vocab_size":123}

    # With override
    Eagle2ChatConfig(vision_config=vision.copy(), llm_config=llm.copy(),
                     attn_implementation="flash_attention_2")
    mock_siglip.assert_called_with(model_type='siglip_vision_model', _attn_implementation='flash_attention_2')
    mock_llama.assert_called_with(architectures=['LlamaForCausalLM'], vocab_size=123, attn_implementation='flash_attention_2')

    # Without override
    mock_siglip.reset_mock(); mock_llama.reset_mock()
    Eagle2ChatConfig(vision_config=vision.copy(), llm_config=llm.copy())
    mock_siglip.assert_called_with(model_type='siglip_vision_model')
    mock_llama.assert_called_with(architectures=['LlamaForCausalLM'], vocab_size=123)





@patch('gr00t.model.backbone.eagle_backbone.AutoConfig')
@patch('gr00t.model.backbone.eagle_backbone.AutoModel')
def test_backbone_attn_impl_forward(mock_auto_model, mock_auto_config):
    fake_cfg = mock_auto_config.from_pretrained.return_value
    fake_model = mock_auto_model.from_config.return_value

    # **Add these lines** to satisfy the internal assertion
    fake_model.template = "test_template"
    fake_model.num_image_token = 5

    processor_cfg = {
        "model_path": "fake/processor/path",
        "max_input_tiles": 1,
        "model_spec": {"template": "test_template", "num_image_token": 5}
    }

    # Test explicit 'flash_attention_2'
    EagleBackbone(
        processor_cfg=processor_cfg,
        use_local_eagle_hg_model=False,
        model_name="m",
        attn_implementation="flash_attention_2"
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
    fake_model.template = "test_template"   # re‑set for the new instance
    fake_model.num_image_token = 5

    EagleBackbone(
        processor_cfg=processor_cfg,
        use_local_eagle_hg_model=False,
        model_name="m"
    )
    mock_auto_config.from_pretrained.assert_called_with(
        "m", trust_remote_code=True, attn_implementation="eager"
    )
    mock_auto_model.from_config.assert_called_with(
        fake_cfg, trust_remote_code=True, attn_implementation="eager"
    )
