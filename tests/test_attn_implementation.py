import pytest
from unittest.mock import patch, MagicMock

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
