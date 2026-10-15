from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from groot.core.data.state_action import state_action_processor
from groot.core.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from groot.core.training.models.n2.components import sequence_packing, transformer_cosmos
from groot.core.training.models.n2.components.action_modules import ActionDecoder
from groot.core.training.models.n2.components.wan_video_vae import CausalConv3d, RMS_norm

# Like the N2 TRT unit tests, use small tensors and mocked export/runtime seams.
# These tests do not load a checkpoint, trace LEAPP graphs, or execute ONNX.


class _TensorSemantics:
    def __init__(self, *, name, ref, kind=None, element_names=None):
        self.name = name
        self.ref = ref
        self.kind = kind
        self.element_names = element_names


@dataclass(frozen=True)
class _FunctionPatch:
    module: ModuleType
    function_name: str
    replacement: object


_leapp_stub = ModuleType("leapp")
_leapp_stub.InputKindEnum = SimpleNamespace(JOINT_POSITION="state/joint/position")
_leapp_stub.OutputKindEnum = SimpleNamespace(JOINT_POSITION="target/joint/position")
_leapp_stub.TensorSemantics = _TensorSemantics
_leapp_stub.FunctionPatch = _FunctionPatch
_leapp_stub.annotate = SimpleNamespace()
# Keep transitive native imports cached; restoring the whole module map can
# import ONNX's C++ bindings twice and abort the process.
with pytest.MonkeyPatch.context() as _leapp_import_patch:
    _leapp_import_patch.setitem(sys.modules, "leapp", _leapp_stub)
    from scripts.leapp import n2_export as exporter
    from scripts.leapp.n2_export import (
        _element_names,
        _input_kind,
        _output_element_names,
        _output_kind,
        load_joint_names,
        make_dummy_observation,
        run_annotated_get_action,
    )
    from scripts.leapp.n2_export_patch import (
        annotate_action_noise_for_leapp,
        apply_eval_image_transform_for_leapp,
        causal_conv3d_for_leapp,
        cv2_resize_for_leapp,
        dense_attention_for_leapp,
        dense_create_block_mask_for_leapp,
        n2_function_patches,
        normalize_values_meanstd_for_leapp,
        normalize_values_minmax_for_leapp,
        partial_eef_actions_for_leapp,
        patch_n2_for_leapp,
        relative_joint_action_to_absolute_for_leapp,
        rms_norm_for_leapp,
        zero_pad2d_for_leapp,
    )


def test_load_flat_policy_uses_localized_checkpoint_view(tmp_path, monkeypatch):
    source, view = tmp_path / "source", tmp_path / "view"
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    monkeypatch.setattr(exporter, "ensure_cosmos_model_deps", lambda *a, **kw: str(view))
    policy = SimpleNamespace(model=SimpleNamespace())
    loaded = []

    def load_policy(embodiment, checkpoint, **kwargs):
        loaded.append(checkpoint)
        return policy

    monkeypatch.setattr(exporter, "Gr00tN2Policy", load_policy)
    monkeypatch.setattr(exporter, "Gr00tFlatPolicyWrapper", lambda policy, **kw: policy)
    result = exporter.load_flat_policy(
        source,
        torch.device("cpu"),
        model_deps=None,
        embodiment="xdof_relative_eef_relative_joint",
        num_action_steps=4,
        num_video_steps=1,
    )
    assert result is policy
    assert loaded == [str(view)]


@pytest.mark.parametrize(
    ("mean", "steps", "expected"),
    [
        ([1.0, 2.0], 3, [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
        ([1.0, 2.0], 1, [[1.0, 2.0]]),
        ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 3, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 2, [[1.0, 2.0], [3.0, 4.0]]),
        ([[1.0, 2.0], [3.0, 4.0]], 3, None),
    ],
)
def test_dummy_observation_preserves_state_history(mean, steps, expected) -> None:
    mean = np.asarray(mean, dtype=np.float32)
    configs = {
        "video": ModalityConfig([0], ["camera"]),
        "state": ModalityConfig(list(range(1 - steps, 1)), ["arm"]),
    }
    processor = SimpleNamespace(norm_params={"test": {"state": {"arm": {"mean": mean}}}})
    policy = SimpleNamespace(
        get_modality_config=lambda: configs,
        processor=SimpleNamespace(state_action_processor=processor),
        embodiment_tag=SimpleNamespace(value="test"),
        language_key="annotation.task",
    )
    kwargs = {"instruction": "test", "input_height": 32, "input_width": 32}
    if expected is None:
        with pytest.raises(ValueError, match="state.arm.*3 timesteps"):
            make_dummy_observation(SimpleNamespace(policy=policy), **kwargs)
        return
    observation = make_dummy_observation(SimpleNamespace(policy=policy), **kwargs)
    assert observation["video.camera"].shape == (1, 32, 32, 3)
    assert observation["video.camera"].dtype == np.uint8
    actual = observation["state.arm"]
    assert actual.shape == (steps, 2)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, mean)


def test_function_patch_contract_uses_module_targets_without_installing_them() -> None:
    expected = (
        (cv2, "resize", cv2_resize_for_leapp),
        (transformer_cosmos, "flex_attention_op", dense_attention_for_leapp),
        (state_action_processor, "normalize_values_minmax", normalize_values_minmax_for_leapp),
        (state_action_processor, "normalize_values_meanstd", normalize_values_meanstd_for_leapp),
    )
    originals = [getattr(module, name) for module, name, _ in expected]
    definitions = n2_function_patches()
    assert [(item.module, item.function_name, item.replacement) for item in definitions] == list(
        expected
    )
    assert [getattr(module, name) for module, name, _ in expected] == originals


@pytest.fixture
def mocked_export(monkeypatch, tmp_path, mixed_eef_processor):
    events = []
    model = SimpleNamespace(
        action_decoder=ActionDecoder(hidden_dim=8, action_dim=4),
        config=SimpleNamespace(action_cfg={"action_horizon": 5, "max_action_dim": 7}),
    )
    policy = SimpleNamespace(
        policy=SimpleNamespace(
            model=model,
            processor=SimpleNamespace(state_action_processor=mixed_eef_processor),
            embodiment_tag=SimpleNamespace(value="test"),
        )
    )
    original_forward = ActionDecoder.forward
    original_randn = torch.randn
    observation = {"state.test": np.zeros((1, 2), dtype=np.float32)}
    monkeypatch.setattr(exporter, "load_flat_policy", lambda *args, **kwargs: policy)
    monkeypatch.setattr(exporter, "make_dummy_observation", lambda *args, **kwargs: observation)

    def start(**kwargs):
        events.append(("start", kwargs))
        (Path(kwargs["save_path"]) / kwargs["name"]).mkdir(parents=True)

    def capture(flat_policy, inputs, annotate, **kwargs):
        assert flat_policy is policy
        assert inputs is observation
        assert torch.randn is not original_randn
        torch.randn(1, 5, 7)  # Exercise the real action-noise capture context.
        events.append(("capture", kwargs))

    def compile_graph(**kwargs):
        assert ActionDecoder.forward is original_forward
        assert torch.randn is original_randn
        assert "unapply_action" not in vars(mixed_eef_processor)
        events.append(("compile", kwargs))

    monkeypatch.setattr(
        exporter,
        "leapp",
        SimpleNamespace(
            start=start,
            stop=lambda: events.append(("stop", {})),
            compile_graph=compile_graph,
        ),
    )
    monkeypatch.setattr(
        exporter, "annotate", SimpleNamespace(input_tensors=lambda name, tensor: tensor.ref)
    )
    monkeypatch.setattr(exporter, "run_annotated_get_action", capture)
    monkeypatch.setattr(
        exporter,
        "onnx",
        SimpleNamespace(
            checker=SimpleNamespace(
                check_model=lambda path: events.append(("schema", path)),
            )
        ),
    )
    monkeypatch.setattr(
        exporter, "ort", SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"])
    )
    kwargs = {
        "checkpoint": tmp_path / "checkpoint",
        "output_dir": tmp_path / "export",
        "model_deps": None,
        "embodiment": "test",
        "instruction": "test",
        "input_height": 32,
        "input_width": 32,
        "num_action_steps": 4,
        "num_video_steps": 2,
        "opset_version": 22,
        "validate_runtime": True,
        "validation_rtol": 0.02,
        "validation_atol": 0.05,
    }
    return events, kwargs


@pytest.mark.parametrize("partial_eef", [False, True])
@pytest.mark.parametrize(
    ("cuda_available", "requested"), [(True, True), (True, False), (False, True)]
)
def test_export_lifecycle_validation_and_partial_metadata(
    mocked_export,
    mixed_eef_processor,
    monkeypatch,
    partial_eef,
    cuda_available,
    requested,
):
    events, kwargs = mocked_export
    mixed_eef_processor.use_relative_action = partial_eef
    monkeypatch.setattr(
        exporter.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider"] if cuda_available else ["CPUExecutionProvider"],
    )
    kwargs["validate_runtime"] = requested
    path = exporter.export_n2(**kwargs)
    assert path == kwargs["output_dir"] / "n2_get_actions/n2_get_actions.yaml"
    assert [event for event, _ in events] == ["start", "capture", "stop", "compile", "schema"]
    assert events[0][1]["patching"] == n2_function_patches()
    assert events[3][1] == {
        "visualize": True,
        "validate": cuda_available and requested,
        "strict": True,
        "rtol": 0.02,
        "atol": 0.05,
    }
    assert events[4][1] == str(path.with_suffix(".onnx"))
    sidecar = path.with_name("relative_eef_outputs.json")
    if partial_eef:
        metadata = json.loads(sidecar.read_text())
        assert metadata["end_to_end"] is False
        assert metadata["outputs"] == events[1][1]["relative_eef_outputs"]
        assert metadata["outputs"]["action.eef"]["reference_state"] == "state.wrist"
    else:
        assert not sidecar.exists()


def test_capture_failure_stops_tracing_and_restores_patches(
    mocked_export, mixed_eef_processor, monkeypatch
):
    events, kwargs = mocked_export
    original_forward = ActionDecoder.forward
    original_randn = torch.randn
    original_inference_mode = torch.inference_mode

    def fail(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(exporter, "run_annotated_get_action", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        exporter.export_n2(**kwargs)
    assert [event for event, _ in events] == ["start", "stop"]
    assert ActionDecoder.forward is original_forward
    assert torch.randn is original_randn
    assert torch.inference_mode is original_inference_mode
    assert "unapply_action" not in vars(mixed_eef_processor)
    assert not (kwargs["output_dir"] / "n2_get_actions/relative_eef_outputs.json").exists()


def test_native_export_mask_matches_bidirectional_cond_table() -> None:
    ids = torch.tensor([[1, 0, 2, 3]])
    with patch.object(
        sequence_packing,
        "create_block_mask",
        dense_create_block_mask_for_leapp,
    ):
        mask, pad = sequence_packing._build_attention_mask(ids, ids.device, block_size=4)

    assert pad == 0
    assert mask.shape == (1, 1, 4, 4)
    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, False],
            [True, False, True, True],
        ]
    )
    torch.testing.assert_close(mask[0, 0], expected)


@pytest.mark.parametrize("flatten_heads", [False, True])
def test_dense_attention_matches_masked_attention(flatten_heads) -> None:
    q = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) / 24
    k, v = q + 0.5, q - 0.5
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) / 2
    scores = scores.masked_fill(~mask[..., :3, :3], float("-inf"))
    expected = (scores.softmax(dim=-1) @ v.transpose(1, 2)).transpose(1, 2)
    if flatten_heads:
        expected = expected.flatten(2)
    actual = dense_attention_for_leapp(
        q,
        k,
        v,
        attn_mask=mask,
        pad=1,
        flatten_heads=flatten_heads,
        compile_attention=False,
    )
    torch.testing.assert_close(actual, expected)


def test_export_patch_context_installs_and_restores_substitutions() -> None:
    decoder = ActionDecoder(hidden_dim=8, action_dim=4).to(torch.bfloat16)
    model = type("Model", (), {"action_decoder": decoder})()
    original_resize = cv2.resize
    original_create_block_mask = sequence_packing.create_block_mask
    original_flex_attention = transformer_cosmos.flex_attention_op
    original_minmax = state_action_processor.normalize_values_minmax
    original_meanstd = state_action_processor.normalize_values_meanstd
    original_decoder_forward = ActionDecoder.forward
    original_inference_mode = torch.inference_mode

    with patch.dict(
        os.environ,
        {"ENABLE_TENSORRT": "sentinel"},
    ):
        with patch_n2_for_leapp(model):
            assert os.environ["ENABLE_TENSORRT"] == "true"
            assert cv2.resize is original_resize
            assert sequence_packing.create_block_mask is dense_create_block_mask_for_leapp
            assert transformer_cosmos.flex_attention_op is original_flex_attention
            assert state_action_processor.normalize_values_minmax is original_minmax
            assert state_action_processor.normalize_values_meanstd is original_meanstd
            with torch.inference_mode():
                export_tensor = torch.ones(1)
            assert not export_tensor.is_inference()
            output = decoder(
                torch.ones(1, 2, 8, dtype=torch.float32),
                torch.zeros(1, dtype=torch.long),
            )
            assert output.dtype == torch.bfloat16
            with pytest.raises(
                NotImplementedError,
                match="Unexpected relative end-effector pose conversion",
            ):
                state_action_processor.StateActionProcessor._convert_to_absolute_action(
                    object(),
                    np.zeros((1, 9), dtype=np.float32),
                    np.zeros(9, dtype=np.float32),
                    ActionType.EEF,
                    ActionFormat.XYZ_ROT6D,
                )

        assert os.environ["ENABLE_TENSORRT"] == "sentinel"

    assert cv2.resize is original_resize
    assert sequence_packing.create_block_mask is original_create_block_mask
    assert transformer_cosmos.flex_attention_op is original_flex_attention
    assert state_action_processor.normalize_values_minmax is original_minmax
    assert state_action_processor.normalize_values_meanstd is original_meanstd
    assert ActionDecoder.forward is original_decoder_forward
    assert torch.inference_mode is original_inference_mode
    with torch.inference_mode():
        production_tensor = torch.ones(1)
    assert production_tensor.is_inference()


def test_traceable_normalization_preserves_degenerate_dimensions() -> None:
    values = np.array([[2.0, 7.0, 5.0]], dtype=np.float32)
    minmax = normalize_values_minmax_for_leapp(
        values,
        {
            "min": np.array([0.0, 7.0, 1.0], dtype=np.float32),
            "max": np.array([4.0, 7.0, 9.0], dtype=np.float32),
        },
    )
    meanstd = normalize_values_meanstd_for_leapp(
        values,
        {
            "mean": np.array([1.0, 3.0, 5.0], dtype=np.float32),
            "std": np.array([1.0, 0.0, 2.0], dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(minmax, np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(meanstd, np.array([[1.0, 7.0, 0.0]], dtype=np.float32))


def test_torch_eval_resize_matches_opencv_within_one_pixel_level() -> None:
    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, (2, 73, 119, 3), dtype=np.uint8)
    crop_fraction = 0.95
    crop_height = int(images.shape[-3] * crop_fraction)
    crop_width = int(images.shape[-2] * crop_fraction)
    y_min = (images.shape[-3] - crop_height) // 2
    x_min = (images.shape[-2] - crop_width) // 2
    expected = np.stack(
        [
            cv2.resize(
                image[y_min : y_min + crop_height, x_min : x_min + crop_width],
                (32, 24),
                interpolation=cv2.INTER_LINEAR,
            )
            for image in images
        ]
    )

    actual = np.stack(
        [
            cv2_resize_for_leapp(
                image[y_min : y_min + crop_height, x_min : x_min + crop_width],
                (32, 24),
                interpolation=cv2.INTER_LINEAR,
            )
            for image in images
        ]
    )

    assert actual.dtype == np.float32
    difference = np.abs(expected.astype(np.float32) - actual)
    assert difference.max() <= 1


def test_eval_image_transform_keeps_pixels_float32() -> None:
    image = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(4, 5, 3)

    transformed, replay = apply_eval_image_transform_for_leapp(
        lambda *, image: {"image": image},
        [image],
    )

    assert replay is None
    assert transformed[0].dtype == torch.float32
    np.testing.assert_array_equal(transformed[0].permute(1, 2, 0).numpy(), image)


@pytest.mark.parametrize("partial_eef", [False, True])
def test_annotations_surround_the_original_flat_policy_call(partial_eef) -> None:
    action_name = "action.test_joints" if partial_eef else "action.test"
    width = 9 if partial_eef else 2
    fields = ["x", "y", "z", *[f"rot6d_{i}" for i in range(6)]]
    relative_eef = (
        {action_name: {"reference_state": "state.test", "element_names": fields}}
        if partial_eef
        else {}
    )

    class RecordingPolicy:
        def get_action(self, observation):
            self.observation = observation
            return {action_name: observation["state.test"] + 1}, {}

    class PassthroughAnnotate:
        def input_tensors(self, node_name, tensors):
            self.node_name = node_name
            self.input_semantics = {tensor.name: tensor for tensor in tensors}
            self.inputs = {tensor.name: tensor.ref for tensor in tensors}
            return tuple(self.inputs.values())

        def output_tensors(self, node_name, tensors, **kwargs):
            self.output_semantics = {tensor.name: tensor for tensor in tensors}
            self.outputs = {tensor.name: tensor.ref for tensor in tensors}
            self.output_kwargs = kwargs

    policy = RecordingPolicy()
    annotate = PassthroughAnnotate()
    observation = {
        "video.test": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "state.test": np.zeros((1, width), dtype=np.float32),
        "annotation.task": "pick up the object",
    }

    actions = run_annotated_get_action(
        policy, observation, annotate, relative_eef_outputs=relative_eef
    )

    assert policy.observation["annotation.task"] == "pick up the object"
    assert annotate.input_semantics["video_test"].ref.dtype == np.uint8
    assert policy.observation["video.test"].dtype == np.float32
    assert set(annotate.inputs) == {"video_test", "state_test"}
    assert set(policy.observation) == set(observation)
    np.testing.assert_array_equal(actions[action_name], np.ones((1, 1, width), dtype=np.float32))
    assert set(annotate.outputs) == {action_name}
    np.testing.assert_array_equal(annotate.outputs[action_name], actions[action_name])
    assert annotate.input_semantics["video_test"].kind == "state/camera/image"
    assert annotate.input_semantics["state_test"].kind is None
    assert annotate.input_semantics["state_test"].element_names is None
    assert annotate.output_semantics[action_name].element_names == (fields if partial_eef else None)
    assert annotate.output_semantics[action_name].kind is None
    assert annotate.output_kwargs["export_with"] == "onnx"
    if partial_eef:
        for name in (action_name, "state.test"):
            with pytest.raises(ValueError, match="cannot label EEF pose tensors"):
                run_annotated_get_action(
                    policy,
                    observation,
                    annotate,
                    relative_eef_outputs=relative_eef,
                    joint_names={name: fields},
                )
        observation["state.test"] = np.zeros((1, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="9 features"):
            run_annotated_get_action(
                policy, observation, annotate, relative_eef_outputs=relative_eef
            )


def test_joint_names_are_optional_and_validated_against_feature_width() -> None:
    value = np.zeros((1, 2), dtype=np.float32)

    assert load_joint_names(None) == {}
    assert _element_names("state.arm", value, {}) is None
    assert _element_names("state.arm", value, {"state.arm": ["a", "b"]}) == ["a", "b"]

    with pytest.raises(ValueError, match="feature dimension is 2"):
        _element_names("state.arm", value, {"state.arm": ["a"]})


def test_g1_joint_name_template_matches_hand_tensor_order() -> None:
    template = load_joint_names(Path("scripts/leapp/joint_names/g1.json"))

    for prefix in ("state", "action"):
        for side in ("left", "right"):
            suffix = "_joints" if prefix == "action" else ""
            assert template[f"{prefix}.{side}_hand{suffix}"] == [
                f"{side}_hand_index_0_joint",
                f"{side}_hand_index_1_joint",
                f"{side}_hand_middle_0_joint",
                f"{side}_hand_middle_1_joint",
                f"{side}_hand_thumb_0_joint",
                f"{side}_hand_thumb_1_joint",
                f"{side}_hand_thumb_2_joint",
            ]


def test_input_and_output_semantics_cover_known_modalities() -> None:
    joint_kind = object()
    joint_names = {"action.arm": ["shoulder", "elbow"]}
    value = np.zeros((1, 4, 2), dtype=np.float32)

    assert _input_kind("video.ego_view", joint_kind, {}) == "state/camera/image"
    assert _input_kind("state.projected_gravity", joint_kind, {}) == "state/vector3d"
    assert _input_kind("state.arm_joints", joint_kind, {}) is joint_kind
    assert _input_kind("state.left_wrist_eef", joint_kind, {}) is None

    assert _output_kind("action.arm", joint_kind, joint_names) is joint_kind
    assert _output_element_names("action.arm", value, joint_names) == ["shoulder", "elbow"]
    assert _output_kind("action.hand_joints", joint_kind, {}) is joint_kind
    assert _output_element_names("action.hand_joints", value, {}) is None

    assert _output_kind("action.navigate_command", joint_kind, {}) == "velocity_command"
    assert _output_element_names("action.navigate_command", value, {}) == [
        "linear_x",
        "linear_y",
        "angular_z",
    ]
    assert _output_kind("action.base_height_command", joint_kind, {}) == "target/base/height"
    assert _output_element_names("action.base_height_command", value, {}) == ["height"]


def test_action_noise_is_promoted_to_semantic_input() -> None:
    class RecordingAnnotate:
        def input_tensors(self, node_name, semantics):
            self.node_name = node_name
            self.semantics = semantics
            return semantics.ref

    model = SimpleNamespace(
        config=SimpleNamespace(action_cfg={"action_horizon": 5, "max_action_dim": 7})
    )
    annotate = RecordingAnnotate()
    original_randn = torch.randn

    with annotate_action_noise_for_leapp(model, annotate, "policy", original_randn):
        unmatched = torch.randn(1, 2, 3)
        action_noise = torch.randn(2, 5, 7, dtype=torch.bfloat16)

    assert unmatched.shape == (1, 2, 3)
    assert action_noise.dtype == torch.bfloat16
    assert annotate.node_name == "policy"
    assert annotate.semantics.name == "initial_noise"
    assert annotate.semantics.kind is None
    assert annotate.semantics.ref.dtype == torch.float32
    torch.testing.assert_close(action_noise.float(), annotate.semantics.ref)


def test_relative_joint_actions_are_anchored_to_current_position() -> None:
    delta = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    current_position = torch.tensor([1.0, 2.0])

    absolute = relative_joint_action_to_absolute_for_leapp(delta, current_position)

    torch.testing.assert_close(
        absolute,
        torch.tensor([[1.1, 1.8], [1.3, 2.4]]),
    )


@pytest.fixture
def mixed_eef_processor():
    configs = [
        ActionConfig(
            ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROTVEC, "wrist"
        ),
        ActionConfig(ActionRepresentation.RELATIVE, ActionType.NON_EEF, ActionFormat.DEFAULT),
        ActionConfig(ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT),
    ]
    widths = {"eef": 9, "joints": 2, "gripper": 1}
    absolute_stats = {
        key: {
            "min": [-20.0] * width,
            "max": [20.0] * width,
            "mean": [0.0] * width,
            "std": [1.0] * width,
        }
        for key, width in widths.items()
    }
    relative_stats = {
        key: {
            "min": [-2.0] * width,
            "max": [2.0] * width,
            "mean": [0.0] * width,
            "std": [1.0] * width,
        }
        for key, width in widths.items()
        if key != "gripper"
    }
    return state_action_processor.StateActionProcessor(
        {"test": {"action": ModalityConfig([0, 1, 2], list(widths), action_configs=configs)}},
        statistics={"test": {"action": absolute_stats, "relative_action": relative_stats}},
        use_relative_action=True,
    )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("skip_clip", [False, True])
def test_partial_eef_decode_preserves_scaling_and_other_actions(
    mixed_eef_processor, batched, skip_clip
):
    processor = mixed_eef_processor
    processor.skip_inference_clip = skip_clip
    shape = (2, 3) if batched else (3,)
    actions = {
        key: np.full((*shape, width), 2.0, dtype=np.float32)
        for key, width in {"eef": 9, "joints": 2, "gripper": 1}.items()
    }
    # The EEF reference is intentionally absent: partial decoding must not use it.
    states = {"joints": np.full((2, 1, 2) if batched else (1, 2), 10.0, dtype=np.float32)}
    original_configs = processor.modality_configs
    original_unapply = processor.unapply_action
    with partial_eef_actions_for_leapp(processor, "test") as outputs:
        decoded = processor.unapply_action(actions, "test", states)
        assert processor.modality_configs is original_configs
        assert processor.use_relative_action
        assert set(outputs) == {"action.eef"}
        assert outputs["action.eef"]["reference_state"] == "state.wrist"
        assert outputs["action.eef"]["format"] == "xyz+rot6d"
        assert outputs["action.eef"]["requested_absolute_format"] == "xyz+rotvec"
    scale = 2 if skip_clip else 1
    np.testing.assert_allclose(decoded["eef"], 2 * scale)
    np.testing.assert_allclose(decoded["joints"], 10 + 2 * scale)
    np.testing.assert_allclose(decoded["gripper"], 20 * scale)
    assert processor.unapply_action == original_unapply
    assert processor.modality_configs is original_configs


def test_partial_eef_context_restores_after_failure_and_is_noop_when_disabled(mixed_eef_processor):
    processor = mixed_eef_processor
    original_configs = processor.modality_configs
    original_unapply = processor.unapply_action
    with pytest.raises(KeyError), partial_eef_actions_for_leapp(processor, "test"):
        processor.unapply_action({}, "test")
    assert processor.modality_configs is original_configs
    assert processor.unapply_action == original_unapply
    processor.use_relative_action = False
    with partial_eef_actions_for_leapp(processor, "test") as outputs:
        assert outputs == {}
        assert processor.unapply_action == original_unapply


@pytest.mark.parametrize("use_cache", [False, True])
def test_causal_conv3d_export_rewrite_matches_original(use_cache: bool) -> None:
    torch.manual_seed(7)
    module = CausalConv3d(3, 4, 3, padding=1)
    x = torch.randn(2, 3, 1 if use_cache else 2, 8, 10)
    cache = torch.randn(2, 3, 1, 8, 10) if use_cache else None

    expected = module(x, cache)
    actual = causal_conv3d_for_leapp(module, x, cache)

    torch.testing.assert_close(actual, expected)


def test_zero_pad2d_export_rewrite_matches_original() -> None:
    module = torch.nn.ZeroPad2d((0, 1, 0, 1))
    x = torch.randn(2, 3, 8, 10)

    expected = module(x)
    actual = zero_pad2d_for_leapp(module, x)

    torch.testing.assert_close(actual, expected)


def test_rms_norm_export_rewrite_matches_original_in_float32() -> None:
    module = RMS_norm(3)
    x = torch.randn(2, 3, 4, 5)

    expected = module(x)
    actual = rms_norm_for_leapp(module, x)

    torch.testing.assert_close(actual, expected)
