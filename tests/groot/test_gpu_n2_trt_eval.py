# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Functional OSS N2 TensorRT open-loop GPU smoke test."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from groot.infra import oss

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "OSS_PACKAGE_ROOT" not in os.environ,
        reason="Run this OSS-only GPU test through oss/tests.",
    ),
]

_EMBODIMENT = "xdof_relative_eef_relative_joint"
_CAMERAS = ("video.top", "video.left", "video.right")
_ACTION_HORIZON = 80
_VIDEO_INDICES = (-20, -15, -10, -5, 0, 5, 10, 15, 20)
_VAE_LATENT_SHAPE = (48, 2, 14, 20)
_TOTAL_LATENT_FRAMES = (len(_VIDEO_INDICES) - 1) // 4 + 1
_VIDEO_TOKENS = (
    len(_CAMERAS)
    * (_VAE_LATENT_SHAPE[-2] // 2)
    * (_VAE_LATENT_SHAPE[-1] // 2)
    * _TOTAL_LATENT_FRAMES
)


def test_oss_open_loop_eval_executes_real_tensorrt_and_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Enter through the OSS CLI and execute target-built TRT engines in both modes."""
    import torch
    import transformers

    from groot.core.training.models.n2 import model as n2_model
    from groot.core.training.models.n2.components import sequence_packing
    from groot.infra.eval.metric_summary import MetricSummary
    from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator

    assert torch.cuda.is_available(), "OSS TensorRT smoke requires CUDA"
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    engine_dir = _build_engine_bundle(checkpoint, tmp_path / "engines")
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    monkeypatch.setenv("GROOT_N2_CODEC_ROUNDTRIP", "none")
    monkeypatch.setenv("GROOT_N2_TRT_OFFLOAD_TEXT_ENCODER", "0")
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *_a, **_k: _make_model())
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *_a, **_k: _Processor(),
    )
    # PredictGraphRunner temporarily rebinds these module globals. Register them
    # with monkeypatch so pytest restores the process after graph capture.
    monkeypatch.setattr(n2_model, "BetaFlowMatchEulerScheduler", _Scheduler)
    monkeypatch.setattr(
        sequence_packing,
        "_build_attention_mask",
        sequence_packing._build_attention_mask,
    )

    observations: list[dict[str, Any]] = []

    def evaluate_fixture(
        _self: OssOpenLoopEvaluator,
        policy: Any,
        _dataset_path: Path,
    ) -> MetricSummary:
        images = torch.zeros(
            (1, 3, len(_CAMERAS) * 5, 224, 320),
            device="cuda",
            dtype=torch.bfloat16,
        )
        images.flatten()[0] = 7
        prompt = torch.zeros((1, 1, 1), device="cuda", dtype=torch.bfloat16)
        first = policy._run_predict(
            {
                "images": images,
                "action_seed": torch.full(
                    (1, _ACTION_HORIZON, 1),
                    3,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
            },
            prompt,
        )
        first_action = first["action_pred"].clone()
        second = policy._run_predict(
            {
                "images": images,
                "action_seed": torch.full(
                    (1, _ACTION_HORIZON, 1),
                    5,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
            },
            prompt,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(first_action, torch.full_like(first_action, 3))
        torch.testing.assert_close(second["action_pred"], torch.full_like(second["action_pred"], 5))
        assert second["video_latents"].flatten()[0].item() == 7
        observations.append(
            {
                "acceleration": dict(policy.inference_acceleration_state),
                "lifecycle": dict(
                    policy.benchmark_runtime_contract.get("cuda_graph_lifecycle", {})
                ),
                "loaded_engines": sorted(
                    policy.benchmark_runtime_contract["runtime_artifacts"]["loaded_engines"]
                ),
            }
        )
        graph_runner = getattr(policy, "_graph_runner", None)
        if graph_runner is not None:
            graph_runner.close()
        error = (second["action_pred"].float() - 5).abs()
        return MetricSummary(
            per_trajectory=[
                {"traj_id": 0, "mse": float((error * error).mean()), "mae": float(error.mean())}
            ],
            average={"mse": float((error * error).mean()), "mae": float(error.mean())},
        )

    monkeypatch.setattr(OssOpenLoopEvaluator, "evaluate_policy", evaluate_fixture)

    for use_cuda_graph in (False, True):
        output_json = tmp_path / f"trt-graph-{use_cuda_graph}.json"
        argv = [
            "eval",
            "--embodiment-tag",
            _EMBODIMENT,
            "--checkpoint",
            str(checkpoint),
            "--dataset-path",
            str(dataset_path),
            "--traj-ids",
            "0",
            "--steps",
            "1",
            "--action-horizon",
            str(_ACTION_HORIZON),
            "--denoising-steps",
            "4",
            "--trt-engine-dir",
            str(engine_dir),
            "--output-json",
            str(output_json),
            "--local-only",
        ]
        if use_cuda_graph:
            argv.append("--use-cuda-graph")

        assert oss.main(argv) == 0
        report = json.loads(output_json.read_text(encoding="utf-8"))
        assert report["results"]["local"]["average"] == {"mae": 0.0, "mse": 0.0}

    assert [record["acceleration"]["backend"] for record in observations] == [
        "trt",
        "trt_graph",
    ]
    assert all(record["acceleration"]["trt_engines"] for record in observations)
    assert all(
        record["loaded_engines"]
        == ["transformer_action_cached", "transformer_joint_capture", "vae_encode"]
        for record in observations
    )
    assert observations[1]["lifecycle"] == {
        "schema_version": "n2-cuda-graph-lifecycle-v1",
        "capture_count": 1,
        "recapture_count": 0,
        "replay_count": 2,
    }


class _Scheduler:
    def __init__(self, config: Any = None):
        self.config = config or {}

    def set_timesteps(self, steps: int, *, device: Any = None, **_kwargs: Any) -> None:
        import torch

        self.timesteps = torch.arange(steps, device=device)


def _make_model():
    # CPU CI collects this module without the optional training dependencies.
    from torch import nn

    class _Model(nn.Module):
        def __init__(self) -> None:
            import torch

            super().__init__()
            # Match production's module-owned parameters for selective TRT placement.
            self.state_encoder = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            self.state_encoder.requires_grad_(False)
            self.transformer = nn.Identity()
            self.vae = nn.Identity()
            self.text_encoder = nn.Identity()
            self.action_scheduler = _Scheduler()
            self.config = SimpleNamespace(
                model_type="GrootN2",
                model_size="2B",
                vae_type="wan22",
                action_cfg={
                    "action_horizon": _ACTION_HORIZON,
                    "max_state_dim": 128,
                    "max_action_dim": 128,
                },
                token_merge_kernel=None,
                token_merge_layers=None,
            )
            self.max_n_cameras = len(_CAMERAS)
            self.tiled = False

        @property
        def device(self):
            return self.state_encoder.weight.device

        def _resolve_attention_mask_pattern(self):
            from groot.core.training.models.n2.components.sequence_packing import (
                AttentionMaskPattern,
            )

            return AttentionMaskPattern.BIDIRECTIONAL_VIDEO_ACTION_FULL

        def _make_video_scheduler(self, _config: Any) -> _Scheduler:
            return _Scheduler()

        def predict(
            self, inputs: dict[str, Any], *, prompt_embs: Any, **_kwargs: Any
        ) -> dict[str, Any]:
            import torch

            from groot.core.training.models.n2.components.kv_cache import CondKVCache
            from groot.core.training.models.n2.components.sequence_packing import MODALITY_ACTION

            data = inputs.get("inputs", inputs)
            video_latents = self.encode_video(data["images"])
            video_tokens = video_latents.flatten()[:_VIDEO_TOKENS].reshape(1, _VIDEO_TOKENS, 1)
            state_tokens = torch.zeros((1, 1, 1), device=self.device, dtype=torch.bfloat16)
            action_tokens = data["action_seed"]

            def time_like(tokens):
                return torch.zeros_like(tokens)

            def rope(length: int):
                value = torch.zeros((length, 1, 1), device=self.device, dtype=torch.bfloat16)
                return value, value

            joint = self.packer.pack(
                video_tokens,
                time_like(video_tokens),
                rope(_VIDEO_TOKENS),
                video_cond_token_mask=torch.ones(
                    (1, _VIDEO_TOKENS), device=self.device, dtype=torch.bool
                ),
                action_tokens=torch.zeros_like(action_tokens),
                action_timesteps=time_like(action_tokens),
                action_rope=rope(_ACTION_HORIZON),
                state_tokens=state_tokens,
                state_timesteps=time_like(state_tokens),
                state_rope=rope(1),
            )
            cache = CondKVCache()
            self._call_transformer(joint, prompt_embs, kv_cache=cache)
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            action = replace(
                joint,
                tokens=action_tokens,
                timesteps=time_like(action_tokens),
                rope_freqs=rope(_ACTION_HORIZON),
                modality_ids=torch.full(
                    (1, _ACTION_HORIZON),
                    MODALITY_ACTION,
                    device=self.device,
                    dtype=torch.long,
                ),
                video_token_indices=empty,
                state_token_indices=empty,
                action_token_indices=torch.arange(_ACTION_HORIZON, device=self.device),
                non_video_token_indices=torch.arange(_ACTION_HORIZON, device=self.device),
                has_video_cond_tokens=False,
            )
            action_pred = self._call_transformer(action, prompt_embs, kv_cache=cache)
            return {
                "action_pred": action_pred,
                "video_latents": video_latents,
                "prompt_embs": prompt_embs,
            }

    return _Model()


class _Processor:
    def __init__(self) -> None:
        self.collator = lambda batch: batch

    def eval(self):
        return self

    def get_modality_configs(self):
        from groot.core.data.types import ModalityConfig

        return {
            _EMBODIMENT: {
                "video": ModalityConfig(
                    delta_indices=list(_VIDEO_INDICES), modality_keys=list(_CAMERAS)
                ),
                "state": ModalityConfig(delta_indices=[0], modality_keys=["state.joint"]),
                "action": ModalityConfig(
                    delta_indices=list(range(_ACTION_HORIZON)), modality_keys=["action.joint"]
                ),
                "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
            }
        }


def _write_checkpoint(path: Path) -> Path:
    from safetensors.torch import save_file
    import torch

    path.mkdir()
    vae_weights = path / "vae.pth"
    vae_weights.write_bytes(b"fixture-vae-weights")
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "GrootN2",
                "model_size": "2B",
                "vae_type": "wan22",
                "model_dtype": "bfloat16",
                "max_n_cameras": len(_CAMERAS),
                "vae_cfg": {"vae_pretrained_path": str(vae_weights)},
                "action_cfg": {
                    "action_horizon": _ACTION_HORIZON,
                    "max_state_dim": 128,
                    "max_action_dim": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "image_target_size": [224, 320],
                    "modality_configs": {
                        _EMBODIMENT: {
                            "video": {
                                "delta_indices": list(_VIDEO_INDICES),
                                "modality_keys": list(_CAMERAS),
                            },
                            "action": {
                                "delta_indices": list(range(_ACTION_HORIZON)),
                                "modality_keys": ["action.joint"],
                            },
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "statistics.json").write_text("{}\n", encoding="utf-8")
    save_file({"fixture": torch.zeros(1)}, path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"fixture": "model.safetensors"}}),
        encoding="utf-8",
    )
    return path


def _write_onnx_bundle(root: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    root.mkdir()

    def value(name: str, shape: list[int]):
        return helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)

    def transformer(name: str, length: int, *, cached: bool) -> None:
        inputs = [
            value("latents", [1, length, 1]),
            value("timesteps", [1, length, 1]),
            value("rope_cos", [length, 1, 1]),
            value("rope_sin", [length, 1, 1]),
            value("context", [1, 1, 1]),
        ]
        nodes = [helper.make_node("Add", ["latents", "timesteps"], ["sum0"])]
        current = "sum0"
        for index, source in enumerate(("rope_cos", "rope_sin", "context"), start=1):
            scalar = f"scalar{index}"
            output = f"sum{index}"
            nodes.append(helper.make_node("ReduceMean", [source], [scalar], keepdims=1))
            nodes.append(helper.make_node("Add", [current, scalar], [output]))
            current = output
        if cached:
            inputs.extend(
                [value("kv_cache_key", [1, length, 1]), value("kv_cache_value", [1, length, 1])]
            )
            nodes.extend(
                [
                    helper.make_node("Add", [current, "kv_cache_key"], ["with_key"]),
                    helper.make_node("Add", ["with_key", "kv_cache_value"], ["output1"]),
                ]
            )
            outputs = [value("output1", [1, length, 1])]
        else:
            starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
            ends = helper.make_tensor("ends", TensorProto.INT64, [1], [_ACTION_HORIZON])
            axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
            steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
            nodes.extend(
                [
                    helper.make_node("Identity", [current], ["output"]),
                    helper.make_node(
                        "Slice", [current, "starts", "ends", "axes", "steps"], ["output2"]
                    ),
                    helper.make_node("Neg", ["output2"], ["output3"]),
                ]
            )
            outputs = [
                value("output", [1, length, 1]),
                value("output2", [1, _ACTION_HORIZON, 1]),
                value("output3", [1, _ACTION_HORIZON, 1]),
            ]
            initializers = [starts, ends, axes, steps]
        graph = helper.make_graph(
            nodes,
            name,
            inputs,
            outputs,
            initializer=[] if cached else initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10
        onnx.checker.check_model(model)
        onnx.save(model, root / name)

    transformer("n2_transformer_joint.onnx", _VIDEO_TOKENS + 1 + _ACTION_HORIZON, cached=False)
    transformer("n2_transformer_action_cache.onnx", _ACTION_HORIZON, cached=True)

    videos = value("videos", [len(_CAMERAS), 3, 5, 224, 320])
    vae_output_shape = [len(_CAMERAS), *_VAE_LATENT_SHAPE]
    latents = value("latents", vae_output_shape)
    starts = helper.make_tensor("vae_starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("vae_ends", TensorProto.INT64, [1], [math.prod(vae_output_shape)])
    axes = helper.make_tensor("vae_axes", TensorProto.INT64, [1], [1])
    steps = helper.make_tensor("vae_steps", TensorProto.INT64, [1], [1])
    shape = helper.make_tensor("vae_shape", TensorProto.INT64, [5], vae_output_shape)
    vae_graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["videos"], ["flat"], axis=0),
            helper.make_node(
                "Slice",
                ["flat", "vae_starts", "vae_ends", "vae_axes", "vae_steps"],
                ["slice"],
            ),
            helper.make_node("Reshape", ["slice", "vae_shape"], ["latents"]),
        ],
        "n2_vae_encoder.onnx",
        [videos],
        [latents],
        initializer=[starts, ends, axes, steps, shape],
    )
    vae_model = helper.make_model(vae_graph, opset_imports=[helper.make_opsetid("", 17)])
    vae_model.ir_version = 10
    onnx.checker.check_model(vae_model)
    onnx.save(vae_model, root / "n2_vae_encoder.onnx")


def _build_engine_bundle(checkpoint: Path, root: Path) -> Path:
    from groot.deployment.n2_trt.contract import N2TensorRTContract
    from groot.deployment.n2_trt.manifest import (
        BUILD_FILENAME,
        MANIFEST_FILENAME,
        VALIDATION_FILENAME,
        VALIDATION_GATE_PROFILES,
        VALIDATION_POLICY,
        VALIDATION_POLICY_SHA256,
        VALIDATION_PROFILE_THRESHOLDS,
        VALIDATION_SAMPLE_PATHS,
        VALIDATION_SCHEMA,
        ArtifactRecord,
        BuildSettings,
        EngineManifest,
        RuntimeIdentity,
        SourceRevision,
        canonical_sha256,
        checkpoint_file_hashes,
        sha256_file,
    )
    from groot.deployment.n2_trt.validation import _artifact_records

    _write_onnx_bundle(root)
    sample_paths = {role: root / path for role, path in VALIDATION_SAMPLE_PATHS.items()}
    for role, path in sample_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}\n".encode())
    sample_digests = {role: sha256_file(path) for role, path in sample_paths.items()}
    (root / "n2_transformer_export.json").write_text(
        json.dumps(
            {
                "validation_sample_sha256": {
                    role: digest for role, digest in sample_digests.items() if role != "vae_eager"
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "n2_vae_export_manifest.json").write_text(
        json.dumps({"validation_sample_sha256": {"vae_eager": sample_digests["vae_eager"]}}),
        encoding="utf-8",
    )
    package_root = Path(os.environ.get("OSS_PACKAGE_ROOT", Path.cwd()))
    try:
        source_revision = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        source_revision = "oss-gpu-fixture"
    subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "groot.deployment.n2_trt.tools.build_engines",
            "--artifact-root",
            str(root),
            "--builder-optimization-level",
            "0",
            "--source-revision",
            source_revision,
        ],
        check=True,
    )
    build_path = root / BUILD_FILENAME
    build = json.loads(build_path.read_text(encoding="utf-8"))
    sources, engines = _artifact_records(root, build)
    contract = N2TensorRTContract.from_checkpoint(
        checkpoint,
        embodiment=_EMBODIMENT,
        video_denoising_steps=2,
        action_denoising_steps=4,
        joint_denoise=True,
        conditional_kv_cache=True,
    )
    runtime = RuntimeIdentity.capture()
    checkpoint_files = checkpoint_file_hashes(checkpoint)
    contract_sha256 = canonical_sha256(contract.to_dict())
    vae_weights_sha256 = sha256_file(checkpoint / "vae.pth")
    # Admission receipts are immutable production artifacts. This synthetic receipt is
    # fixture metadata only; the assertions above independently execute every engine.
    validation_payload = {
        "schema": VALIDATION_SCHEMA,
        "validation_policy": VALIDATION_POLICY,
        "validation_policy_sha256": VALIDATION_POLICY_SHA256,
        "passed": True,
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": canonical_sha256(checkpoint_files),
        "vae_weights_path": str(checkpoint / "vae.pth"),
        "vae_weights_sha256": vae_weights_sha256,
        "artifacts": {record.role: record.sha256 for record in (*sources, *engines)},
        "oracles": build["validation_samples"],
        "gates": {
            name: {
                "passed": True,
                "profile": profile,
                "thresholds": VALIDATION_PROFILE_THRESHOLDS[profile],
                "gating_metrics": VALIDATION_POLICY["gating_metrics"][profile],
            }
            for name, profile in VALIDATION_GATE_PROFILES.items()
        },
        "validator_runtime": asdict(runtime),
    }
    validation_path = root / VALIDATION_FILENAME
    validation_path.write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = ArtifactRecord.from_file(
        validation_path,
        role="engine_validation",
        root=root,
        kind="validation",
        numeric_format="json",
    )
    build_record = ArtifactRecord.from_file(
        build_path,
        role="engine_build",
        root=root,
        kind="build",
        numeric_format="json",
    )
    manifest = EngineManifest.create(
        contract=contract,
        runtime=runtime,
        build_settings=BuildSettings(
            strongly_typed=True,
            builder_optimization_level=0,
            transformer_precision="bf16",
            vae_precision="bf16",
        ),
        tool_sources=tuple(
            SourceRevision(component=component, revision="oss-gpu-fixture")
            for component in ("exporter", "builder", "validator")
        ),
        sources=sources,
        engines=engines,
        build=build_record,
        validation=validation,
        vae_weights_path=checkpoint / "vae.pth",
        vae_weights_sha256=vae_weights_sha256,
    )
    manifest.write(root / MANIFEST_FILENAME)
    return root
