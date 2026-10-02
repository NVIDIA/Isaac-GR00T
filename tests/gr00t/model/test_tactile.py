"""CPU checks for optional tactile input using the existing GR00T classes."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

from gr00t.configs.base_config import Config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.types import VLAStepData
from gr00t.experiment.launch_finetune import load_modality_config
from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as modeling, processing_gr00t_n1d7 as processing
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.model.modules.tactile_encoders import build_tactile_encoder
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoModel, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
import tyro


JSON_PATH = Path(__file__).resolve().parents[3] / "gr00t/configs/data/sh5.json"
JOINTS = {"left_arm": 7, "right_arm": 7, "left_hand": 20, "right_hand": 20}
TOUCH = {"tactile_left": 45, "tactile_right": 45}


def modality_config(tactile=False):
    modalities, options = load_modality_config(
        str(JSON_PATH), embodiment_tag="new_embodiment", use_tactile=tactile
    )
    modalities["action"].delta_indices = list(range(4))
    return modalities, options


def small_config(use_tactile=False, **kwargs):
    _, tactile_options = modality_config(use_tactile)
    values = dict(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        action_horizon=4,
        num_inference_timesteps=2,
        max_seq_len=16,
        use_alternate_vl_dit=False,
        use_flash_attention=False,
        state_dropout_prob=0.0,
        diffusion_model_cfg=dict(
            positional_embeddings=None,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            norm_type="ada_norm",
            dropout=0.0,
            final_dropout=False,
            output_dim=32,
            interleave_self_attention=True,
        ),
    )
    values.update(tactile_options)
    values.update(kwargs)
    return Gr00tN1d7Config(**values)


def inputs(config):
    state = torch.zeros(2, config.state_history_length, config.max_state_dim)
    state[..., :54] = torch.randn(2, config.state_history_length, 54)
    mask = torch.zeros(2, config.action_horizon, config.max_action_dim)
    mask[..., :54] = 1
    data = BatchFeature(
        data=dict(
            state=state,
            action=torch.randn_like(mask),
            action_mask=mask,
            embodiment_id=torch.full((2,), 10, dtype=torch.long),
        )
    )
    if config.use_tactile:
        data["tactile"] = torch.randn(2, config.state_history_length, 2, 5, 9)
    return data


def backbone_output():
    return BatchFeature(
        data=dict(
            backbone_features=torch.randn(2, 4, 32),
            backbone_attention_mask=torch.ones(2, 4, dtype=torch.bool),
            image_mask=torch.ones(2, 4, dtype=torch.bool),
        )
    )


class TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(1, 32)

    def prepare_input(self, batch):
        return {"state": batch["state"]}

    def forward(self, batch):
        features = self.projection(batch["state"].mean(dim=-1, keepdim=True))
        features = features[:, :1].expand(-1, 4, -1)
        return BatchFeature(
            data=dict(
                backbone_features=features,
                backbone_attention_mask=torch.ones(features.shape[:2], dtype=torch.bool),
                image_mask=torch.ones(features.shape[:2], dtype=torch.bool),
            )
        )


class TinyCollator:
    def __init__(self, **kwargs):
        pass

    def __call__(self, rows):
        return {"inputs": {k: torch.as_tensor(np.stack([r[k] for r in rows])) for k in rows[0]}}


@pytest.fixture
def no_vlm(monkeypatch):
    monkeypatch.setattr(modeling, "get_backbone_cls", lambda config: TinyBackbone)
    monkeypatch.setattr(processing, "Gr00tN1d7DataCollator", TinyCollator)
    monkeypatch.setattr(processing, "build_processor", lambda *a, **k: MagicMock())
    monkeypatch.setattr(processing.Gr00tN1d7Processor, "data_collator_class", TinyCollator)
    monkeypatch.setattr(processing.Gr00tN1d7Processor, "_get_vlm_inputs", lambda *a, **k: {})


def statistics(tactile):
    def stats(n):
        return {
            "min": [-1.0] * n,
            "max": [1.0] * n,
            "q01": [-1.0] * n,
            "q99": [1.0] * n,
            "mean": [0.0] * n,
            "std": [1.0] * n,
        }

    return {
        "new_embodiment": {
            "state": {k: stats(n) for k, n in {**JOINTS, **(TOUCH if tactile else {})}.items()},
            "action": {k: stats(n) for k, n in JOINTS.items()},
        }
    }


def make_processor(tactile):
    modalities, options = modality_config(tactile)
    options.pop("tactile_encoder", None)
    return processing.Gr00tN1d7Processor(
        modality_configs={"new_embodiment": modalities},
        statistics=statistics(tactile),
        max_state_dim=132,
        max_action_dim=132,
        max_action_horizon=4,
        image_crop_size=(8, 8),
        image_target_size=(8, 8),
        state_dropout_prob=0.0,
        **options,
    )


def sample(tactile):
    states = {k: np.full((1, n), 0.2, dtype=np.float32) for k, n in JOINTS.items()}
    if tactile:
        states.update({k: np.full((1, n), 0.4, dtype=np.float32) for k, n in TOUCH.items()})
    return VLAStepData(
        images={},
        states=states,
        actions={k: np.full((4, n), 0.3, dtype=np.float32) for k, n in JOINTS.items()},
        text="hold the objects",
    )


@pytest.mark.parametrize("tactile", [False, True])
@pytest.mark.parametrize("history", [1, 2])
@pytest.mark.parametrize("alternate", [False, True])
def test_forward_backward_inference(tactile, history, alternate):
    config = small_config(tactile, state_history_length=history, use_alternate_vl_dit=alternate)
    head = Gr00tN1d7ActionHead(config)
    data = inputs(config)
    output = head(backbone_output(), data)
    assert torch.isfinite(output["loss"])
    assert output["action_loss"][..., 54:].count_nonzero() == 0
    output["loss"].backward()
    if tactile:
        assert head.tactile_encoder.finger[0].weight.grad.abs().sum() > 0
    else:
        assert not any("tactile" in k for k in head.state_dict())
        torch.testing.assert_close(
            head._encode_state(data),
            head.state_encoder(data.state.reshape(2, 1, -1), data.embodiment_id),
        )
    assert data.state.shape == (2, history, 132)
    head.eval()
    del data["action"]
    prediction = head.get_action(backbone_output(), data)["action_pred"]
    assert prediction.shape == (2, 4, 132)
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("name", ["finger_mlp", "finger_transformer"])
def test_encoder_gradients(name):
    encoder = build_tactile_encoder(name, 16)
    x = torch.randn(2, 3, 2, 5, 9, requires_grad=True)
    y = encoder(x)
    assert y.shape == (2, 3, 16)
    y.square().mean().backward()
    assert x.grad.abs().sum() > 0


def test_required_input_and_freezing():
    config = small_config(True, tune_projector=False)
    head = Gr00tN1d7ActionHead(config)
    assert not any(p.requires_grad for p in head.tactile_encoder.parameters())
    data = inputs(config)
    del data["tactile"]
    with pytest.raises(ValueError, match="tactile must have shape"):
        head(backbone_output(), data)
    with pytest.raises(ValueError, match="without tactile"):
        Gr00tN1d7ActionHead(small_config())(backbone_output(), inputs(small_config(True)))


@pytest.mark.parametrize("tactile", [False, True])
def test_processor_model_and_policy_roundtrip(no_vlm, tmp_path, tactile, load_hf_model_weights):
    with load_hf_model_weights():
        proc = make_processor(tactile)
        proc.save_pretrained(tmp_path / "processor")
        restored = AutoProcessor.from_pretrained(tmp_path / "processor")
        assert type(restored) is processing.Gr00tN1d7Processor
        before = proc([{"content": sample(tactile)}])
        after = restored([{"content": sample(tactile)}])
        assert before["state"].shape == (1, 132)
        assert before["state"][..., 54:].count_nonzero() == 0
        assert before["action_mask"][..., 54:].count_nonzero() == 0
        if tactile:
            assert before["tactile"].shape == (1, 2, 5, 9)
        for key in before:
            torch.testing.assert_close(torch.as_tensor(before[key]), torch.as_tensor(after[key]))
        model = Gr00tN1d7(small_config(tactile)).eval()
        model(**restored.collator([after, after]))["loss"].backward()
        model.save_pretrained(tmp_path)
        loaded = AutoModel.from_pretrained(tmp_path).eval()
        assert type(loaded) is Gr00tN1d7
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, loaded.state_dict()[key], rtol=0, atol=0)
        data = inputs(model.config)
        del data["action"]
        torch.manual_seed(33)
        expected = model.get_action(dict(data))["action_pred"]
        torch.manual_seed(33)
        torch.testing.assert_close(
            loaded.get_action(dict(data))["action_pred"], expected, rtol=0, atol=0
        )
        policy = Gr00tPolicy("new_embodiment", str(tmp_path), device="cpu", use_tactile=tactile)
        row = sample(tactile)
        observation = {
            "state": {k: v[None] for k, v in row.states.items()},
            "video": {
                k: np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)
                for k in modality_config()[0]["video"].modality_keys
            },
            "language": {"annotation.human.task_description": [["hold the objects"]]},
        }
        actions, _ = policy.get_action(observation)
        assert sum(v.shape[-1] for v in actions.values()) == 54
        assert all(v.shape[:2] == (1, 4) and np.isfinite(v).all() for v in actions.values())
        with pytest.raises(ValueError, match="use_tactile must match"):
            Gr00tPolicy("new_embodiment", str(tmp_path), device="cpu", use_tactile=not tactile)


@pytest.mark.parametrize("encoder", ["finger_mlp", "finger_transformer"])
def test_base_transfer_preserves_every_existing_weight(
    no_vlm, tmp_path, load_hf_model_weights, encoder
):
    with load_hf_model_weights():
        base = Gr00tN1d7(small_config())
        base.save_pretrained(tmp_path / "base")
        cfg = Config(model=small_config(True, tactile_encoder=encoder))
        cfg.training.start_from_checkpoint = str(tmp_path / "base")
        loaded = Gr00tN1d7Pipeline(cfg, tmp_path)._create_model()
        assert loaded.config.use_tactile and loaded.config.max_action_dim == 132
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
        assert torch.isfinite(loaded(inputs(loaded.config))["loss"])
        loaded.save_pretrained(tmp_path / "touch")
        cfg.training.start_from_checkpoint = str(tmp_path / "touch")
        reloaded = Gr00tN1d7Pipeline(cfg, tmp_path)._create_model()
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
        cfg.model.tactile_embed_dim = 16
        with pytest.raises(ValueError, match="tactile_embed_dim differs"):
            Gr00tN1d7Pipeline(cfg, tmp_path)._create_model()


def test_missing_tactile_weights_rejected(no_vlm, tmp_path, load_hf_model_weights):
    from safetensors.torch import load_file, save_file

    with load_hf_model_weights():
        Gr00tN1d7(small_config(True)).save_pretrained(tmp_path)
        make_processor(True).save_pretrained(tmp_path / "processor")
        file = tmp_path / "model.safetensors"
        weights = load_file(file)
        del weights["action_head.tactile_encoder.finger.0.weight"]
        save_file(weights, file, metadata={"format": "pt"})
        cfg = Config(model=small_config(True))
        cfg.training.start_from_checkpoint = str(tmp_path)
        with pytest.raises(RuntimeError, match="weight mismatch"):
            Gr00tN1d7Pipeline(cfg, tmp_path)._create_model()
        with pytest.raises(RuntimeError, match="weight mismatch"):
            Gr00tPolicy("new_embodiment", str(tmp_path), device="cpu", use_tactile=True)


def test_json_selection_leaves_registry_unchanged():
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    original = deepcopy(MODALITY_CONFIGS)
    joints, _ = modality_config()
    tactile, options = modality_config(True)
    assert joints["state"].modality_keys == list(JOINTS)
    assert tactile["state"].modality_keys == [*JOINTS, *TOUCH]
    assert options["tactile_embed_dim"] == 32
    assert MODALITY_CONFIGS == original


def test_cli_and_config_save(tmp_path):
    from gr00t.eval.open_loop_eval import ArgsConfig
    from gr00t.eval.run_gr00t_server import ServerConfig

    args = tyro.cli(
        FinetuneConfig,
        args=[
            "--base-model-path",
            "base",
            "--dataset-path",
            "data",
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--use_tactile",
            "--tactile_encoder",
            "finger_transformer",
            "--tactile_embed_dim",
            "16",
        ],
    )
    assert args.use_tactile and args.tactile_embed_dim == 16
    for cls in (ArgsConfig, ServerConfig):
        assert tyro.cli(cls, args=["--model-path", "model", "--use_tactile"]).use_tactile
    config = Config(model=small_config(True))
    config.save(tmp_path / "config.yaml")
    restored = Config.from_pretrained(tmp_path / "config.yaml")
    assert type(restored.model) is Gr00tN1d7Config
    assert restored.model.use_tactile
    assert restored.model.tactile_state_keys == list(TOUCH)


def test_processor_rejects_bad_dimensions_and_drops_touch(no_vlm):
    proc = make_processor(True)
    row = sample(True)
    row.states["tactile_left"] = np.zeros((1, 44), dtype=np.float32)
    # Inspect the split directly: normalization also rejects a wrong feature width.
    with pytest.raises(ValueError, match="dimensions do not match"):
        proc._pack_state(row.states, [*JOINTS, *TOUCH])
    proc.tactile_embed_dim = 100
    with pytest.raises(ValueError, match="exceeds max_state_dim"):
        proc([{"content": sample(True)}])
    proc.tactile_embed_dim = 32
    proc.state_dropout_prob = 1.0
    result = proc([{"content": sample(True)}])
    assert result["state"].count_nonzero() == result["tactile"].count_nonzero() == 0


def test_external_factory(monkeypatch):
    import sys
    from types import ModuleType

    external = ModuleType("test_tactile_external")
    external.factory = lambda embed_dim, input_shape: nn.Linear(input_shape[-1], embed_dim)
    monkeypatch.setitem(sys.modules, external.__name__, external)
    assert isinstance(build_tactile_encoder("test_tactile_external:factory", 8), nn.Module)
    with pytest.raises(ValueError, match="Unknown tactile encoder"):
        build_tactile_encoder("unknown_encoder", 8)


@pytest.mark.parametrize("from_base", [False, True])
def test_pipeline_processor_construction(no_vlm, tmp_path, monkeypatch, from_base):
    from gr00t.model.gr00t_n1d7 import setup

    cfg = Config(model=small_config(True))
    cfg.data.modality_configs = {"new_embodiment": modality_config(True)[0]}
    if from_base:
        make_processor(False).save_pretrained(tmp_path / "base_processor")
        cfg.training.start_from_checkpoint = str(tmp_path / "base_processor")
    train = MagicMock()
    train.get_dataset_statistics.return_value = statistics(True)
    factory = MagicMock()

    def build(*, processor):
        processor.set_statistics(
            statistics(True), override=cfg.data.override_pretraining_statistics
        )
        return train, None

    factory.return_value.build.side_effect = build
    monkeypatch.setattr(setup, "DatasetFactory", factory)
    pipeline = Gr00tN1d7Pipeline(cfg, tmp_path)
    pipeline.model = MagicMock(config=cfg.model)
    pipeline._create_dataset(tmp_path)
    data = pipeline.processor([{"content": sample(True)}])
    assert data["state"].shape == (1, 132)
    assert data["tactile"].shape == (1, 2, 5, 9)
    assert data["action_mask"].sum() == 4 * 54


def test_batched_observation_matches_step_processing(no_vlm, monkeypatch):
    from gr00t.data.embodiment_tags import EmbodimentTag

    proc = make_processor(True)
    proc.eval()
    monkeypatch.setattr(
        proc, "_apply_vlm_processing", lambda *a: {"vlm_content": {"text": "hold", "images": []}}
    )
    proc.processor.return_value = {}
    row = sample(True)
    obs = {f"state.{k}": np.stack([v, v]) for k, v in row.states.items()}
    obs.update(
        {
            f"video.{k}": np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)
            for k in modality_config()[0]["video"].modality_keys
        }
    )
    obs["annotation.human.task_description"] = [row.text, row.text]
    batched = proc.process_observation(obs, EmbodimentTag.NEW_EMBODIMENT)
    single = proc([{"content": row}])
    for k in ("state", "tactile"):
        torch.testing.assert_close(batched[k], torch.stack([single[k], single[k]]))


@pytest.mark.parametrize("width", [29, 132])
@pytest.mark.parametrize("history", [1, 2])
def test_legacy_config_and_mixed_embodiment_ids(
    no_vlm, tmp_path, load_hf_model_weights, width, history
):
    import json

    with load_hf_model_weights():
        config = small_config(
            max_state_dim=width, max_action_dim=width, state_history_length=history
        )
        original = Gr00tN1d7(config).eval()
        original.save_pretrained(tmp_path)
        path = tmp_path / "config.json"
        saved = json.loads(path.read_text())
        saved = {
            k: v for k, v in saved.items() if k != "use_tactile" and not k.startswith("tactile_")
        }
        path.write_text(json.dumps(saved))
        restored = AutoModel.from_pretrained(tmp_path).eval()
        assert restored.config.use_tactile is False
        assert restored.action_head.tactile_encoder is None
        ids = torch.tensor([0, 1, 10, 24, 26, 31])
        state = torch.randn(len(ids), history, width)
        batch = BatchFeature(data={"state": state, "embodiment_id": ids})
        expected = original.action_head.state_encoder(state.reshape(len(ids), 1, -1), ids)
        actual = restored.action_head._encode_state(batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
