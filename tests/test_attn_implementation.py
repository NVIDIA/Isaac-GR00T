import pytest
from unittest.mock import patch, MagicMock, ANY

from gr00t.model.backbone.eagle_backbone import EagleBackbone
from gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat import Eagle2ChatConfig
from gr00t.model.policy import Gr00tPolicy




@pytest.fixture(autouse=True)
def patch_metadata_and_horizons(monkeypatch):
    monkeypatch.setattr(Gr00tPolicy, '_load_metadata', lambda self, cfg: None)
    monkeypatch.setattr(Gr00tPolicy, '_load_horizons',  lambda self: None)

@pytest.fixture
def common_args():
    return {
        "model_path": "fake/model",
        "embodiment_tag": "gr1",
        "modality_config": {"video": None, "state": None},
        "modality_transform": MagicMock(),
    }

@patch('gr00t.model.policy.snapshot_download', lambda p, **kw: p)
@patch('gr00t.model.policy.GR00T_N1')
@patch('torch.cuda.is_available', return_value=True)
def test_policy_explicit_flash(mock_cuda, mock_gr00t_n1, common_args):
    Gr00tPolicy(**common_args, attn_implementation="flash_attention_2", device="cuda")
    mock_gr00t_n1.from_pretrained.assert_called_once_with(
        "fake/model", torch_dtype=ANY, attn_implementation="flash_attention_2"
    )



@patch('gr00t.model.policy.snapshot_download', lambda p, **kw: p)
@patch('gr00t.model.policy.GR00T_N1')
@patch('torch.cuda.is_available', return_value=True)
def test_policy_auto_on_cuda(mock_cuda, mock_gr00t_n1, common_args):
    Gr00tPolicy(**common_args, attn_implementation="auto", device="cuda")
    mock_gr00t_n1.from_pretrained.assert_called_once_with(
        "fake/model",
        torch_dtype=ANY,
        attn_implementation=None
    )

@patch('gr00t.model.policy.snapshot_download', lambda p, **kw: p)
@patch('gr00t.model.policy.GR00T_N1')
@patch('torch.cuda.is_available', return_value=False)
def test_policy_auto_on_cpu(mock_cuda, mock_gr00t_n1, common_args):
    Gr00tPolicy(**common_args, attn_implementation="auto", device="cpu")
    mock_gr00t_n1.from_pretrained.assert_called_once_with(
        "fake/model",
        torch_dtype=ANY,
        attn_implementation="eager"
    )

@patch('gr00t.model.policy.snapshot_download', lambda p, **kw: p)
@patch('gr00t.model.policy.GR00T_N1')
@patch('torch.cuda.is_available', return_value=False)
def test_policy_explicit_eager(mock_cuda, mock_gr00t_n1, common_args):
    Gr00tPolicy(**common_args, attn_implementation="eager", device="cpu")
    mock_gr00t_n1.from_pretrained.assert_called_once_with(
        "fake/model",
        torch_dtype=ANY,
        attn_implementation="eager"
    )

@patch('gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat.SiglipVisionConfig')
@patch('gr00t.model.backbone.eagle2_hg_model.configuration_eagle_chat.LlamaConfig')
def test_config_attn_propagation(mock_llama, mock_siglip):
    vision = {"model_type":"siglip_vision_model"}
    llm    = {"architectures":["LlamaForCausalLM"], "vocab_size":123}


    Eagle2ChatConfig(vision_config=vision.copy(), llm_config=llm.copy(),
                     attn_implementation="flash_attention_2")
    mock_siglip.assert_called_with(model_type='siglip_vision_model', _attn_implementation='flash_attention_2')
    mock_llama.assert_called_with(architectures=['LlamaForCausalLM'], vocab_size=123, attn_implementation='flash_attention_2')


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






