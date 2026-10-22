#!/usr/bin/env python3
# ruff: noqa: T201
"""Decoded-action gate of a TensorRT bundle against the eager policy on recorded observations.

Both legs construct the serving policy for the same checkpoint and embodiment, denoise every
request from the same fixed seed, and run the same recorded observations with the same
instruction. The gate compares the normalized action prediction (the model output the processor
decodes) of the TensorRT leg with the eager leg:

* ``relative_rmse`` (pooled over observations, valid action channels) must not exceed
  ``--max-relative-rmse``;
* it must also stay below ``--max-spread-fraction`` of the eager leg's own seed-to-seed spread
  (the same observations denoised from ``seed + 1``), so the engines' numerical drift is small
  against the policy's sampling noise.

``--trt-cuda-graph`` serves the TensorRT leg through the whole-predict CUDA graph, as the server
does with ``--use-cuda-graph``. ``--prompt-sequence`` serves several prompts in order on one
TensorRT policy instance (e.g. ``0,1,0``: a prompt change and back). Every step must pass against
the eager leg on its own prompt, and a repeated prompt must reproduce its first step within the
same tolerance. For a ``text_keys="real"`` bundle every engine call must also read its own
request's real prompt length (``text_seq_len``), under CUDA graph replay included, so no prompt's
length or mask outlives its request. Under the graph the policy must end healthy with one capture.

Decoded actions are compared per key as diagnostics. The receipt is written to ``--out``; the exit
status is non-zero when the gate fails.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from groot.core.policy.benchmark_contract import action_equivalence
from groot.deployment.n2_trt.contract import TEXT_SEQ_LEN_INPUT

GATE_SCHEMA = "groot_n2_trt_policy_action_gate_v1"
DEFAULT_MAX_RELATIVE_RMSE = 0.05
DEFAULT_MAX_SPREAD_FRACTION = 0.25


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embodiment", required=True, help="embodiment tag (any case)")
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument(
        "--trt-checkpoint",
        default=None,
        help="checkpoint the TensorRT leg loads (default: --checkpoint); a control bundle "
        "built from another view of the same weights is gated against the eager --checkpoint",
    )
    parser.add_argument(
        "--observations",
        required=True,
        help="directory of obs_*.npz files with video.* and state.* arrays (client layout)",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        action="append",
        help="task prompt; repeat it to name the prompts of --prompt-sequence",
    )
    parser.add_argument(
        "--prompt-sequence",
        default="0",
        help="comma-separated --instruction indices the TensorRT leg serves in this order on "
        "one policy instance, every observation per step (e.g. 0,1,0)",
    )
    parser.add_argument(
        "--trt-cuda-graph",
        action="store_true",
        help="serve the TensorRT leg through the whole-predict CUDA graph (captured on its "
        "first request)",
    )
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument("--num-action-steps", type=int, default=2)
    parser.add_argument("--codec", choices=("h264", "mp4v", "none"), default="none")
    parser.add_argument(
        "--bare-engines",
        action="store_true",
        help="serve the engines of the build receipt under --engine-dir without a manifest "
        "(a candidate that was never admitted)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-relative-rmse", type=float, default=DEFAULT_MAX_RELATIVE_RMSE)
    parser.add_argument("--max-spread-fraction", type=float, default=DEFAULT_MAX_SPREAD_FRACTION)
    parser.add_argument("--save-actions", default=None, help="directory for <leg>.npz action dumps")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _prompt_sequence(spec: str, instructions: list[str]) -> list[int]:
    steps = [int(index) for index in spec.split(",")]
    if not all(0 <= index < len(instructions) for index in steps):
        raise ValueError(
            f"--prompt-sequence {spec!r} must index the {len(instructions)} --instruction values"
        )
    return steps


def _embodiment_tag(name: str) -> Any:
    from groot.core.data import schema

    try:
        return schema.EmbodimentTag[name.upper()]
    except KeyError:
        return schema.EmbodimentTag(name.lower())


def _language_keys(checkpoint: Path, embodiment: str) -> list[str]:
    processor = json.loads((checkpoint / "processor_config.json").read_text())
    config = processor["processor_kwargs"]["modality_configs"][embodiment]
    return list(config["language"]["modality_keys"])


def _observations(directory: Path) -> list[dict]:
    files = sorted(directory.glob("obs_*.npz"))
    if not files:
        raise FileNotFoundError(f"no obs_*.npz under {directory}")
    return [
        {k: v for k, v in dict(np.load(path)).items() if k.startswith(("video.", "state."))}
        for path in files
    ]


def _watch_engine_text_seq_len(model: Any) -> dict[str, Any]:
    """The ``text_seq_len`` tensor each transformer engine was last bound to, by engine mode.

    The runner binds the tensor's device address, and a CUDA graph replays the bindings of its
    capture, so reading the tensor after a request is the value that request's engines read.
    """
    bound: dict[str, Any] = {}
    for mode, runner in model.n2_trt_runtime.transformer_runners.items():
        original = runner._execute  # noqa: SLF001 - the gate observes the engine bindings

        def execute(tensors: dict[str, Any], _mode: str = mode, _execute: Any = original):
            _execute(tensors)
            bound[_mode] = tensors[TEXT_SEQ_LEN_INPUT]

        runner._execute = execute  # noqa: SLF001
    return bound


def _run_leg(
    *,
    leg: str,
    args: argparse.Namespace,
    observations: list[dict],
    language_keys: list[str],
    runs: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Normalized predictions and decoded actions of each (instruction, seed) run, in order.

    Every run serves every observation on the same policy instance.
    """
    import torch

    from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
    from groot.core.policy.n2_policy import Gr00tN2Policy

    os.environ["GROOT_N2_FIXED_SEED"] = "1"
    os.environ["GROOT_N2_NUM_VIDEO_STEPS"] = str(args.num_video_steps)
    os.environ["GROOT_N2_CODEC_ROUNDTRIP"] = args.codec
    checkpoint = args.checkpoint
    if leg == "trt" and args.trt_checkpoint:
        checkpoint = args.trt_checkpoint
    trt_kwargs: dict[str, Any] = {}
    if leg == "trt" and args.bare_engines:
        from groot.deployment.n2_trt.benchmark_adapter import _bare_engine_runtime_factory

        trt_kwargs["trt_runtime_factory"] = _bare_engine_runtime_factory(
            Path(args.engine_dir), codec=args.codec
        )
    cuda_graph = leg == "trt" and args.trt_cuda_graph
    policy = Gr00tN2Policy(
        _embodiment_tag(args.embodiment),
        checkpoint,
        num_inference_timesteps=args.num_action_steps,
        device="cuda",
        strict=True,
        seed=runs[0][1],
        trt_engine_dir=args.engine_dir if leg == "trt" else None,
        use_cuda_graph=cuda_graph,
        codec_roundtrip=args.codec,
        **trt_kwargs,
    )
    if (getattr(policy.model, "n2_trt_runtime", None) is not None) != (leg == "trt"):
        raise RuntimeError(f"{leg} leg built the wrong backend")
    real_text = getattr(policy.model, "text_keys", "all") == "real"
    engine_text_seq_len = (
        _watch_engine_text_seq_len(policy.model) if leg == "trt" and real_text else {}
    )
    pad_token_id = getattr(policy.model, "text_pad_token_id", None)
    predictions: list[np.ndarray] = []
    request_text_seq_len: list[list[int]] = []
    original_run_predict = policy._run_predict  # noqa: SLF001 - the policy's predict seam

    def recording_run_predict(collated_inputs: Any, prompt_embs: Any) -> Any:
        # The policy's model-computation seam: model.predict, or the CUDA graph replay (whose
        # warmup and capture call model.predict for no request of their own).
        if real_text:
            payload = collated_inputs if isinstance(collated_inputs, dict) else collated_inputs.data
            text = payload.get("inputs", payload)["text"]
            request_text_seq_len.append((text != pad_token_id).sum(1).tolist())
        result = original_run_predict(collated_inputs, prompt_embs)
        predictions.append(result["action_pred"].detach().float().cpu().numpy())
        return result

    policy._run_predict = recording_run_predict  # noqa: SLF001
    wrapper = Gr00tFlatPolicyWrapper(policy, strict=True)
    results: list[dict[str, Any]] = []
    try:
        for instruction, seed in runs:
            policy.seed = seed
            predictions.clear()
            request_text_seq_len.clear()
            decoded: dict[str, list[np.ndarray]] = {}
            latencies = []
            engine_reads: list[dict[str, list[int]]] = []
            for recorded in observations:
                observation = {**recorded, **{key: [instruction] for key in language_keys}}
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    action, _ = wrapper.get_action(observation)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - started) * 1000.0)
                engine_reads.append(
                    {mode: tensor.tolist() for mode, tensor in engine_text_seq_len.items()}
                )
                for key, value in action.items():
                    decoded.setdefault(key, []).append(np.asarray(value, dtype=np.float64))
            if len(predictions) != len(observations):
                raise RuntimeError(
                    f"{leg} leg recorded {len(predictions)} predictions for "
                    f"{len(observations)} observations"
                )
            text_seq_len = None
            if real_text:
                modes = sorted({mode for reads in engine_reads for mode in reads})
                text_seq_len = {
                    "request": sorted({n for lengths in request_text_seq_len for n in lengths}),
                    "engines": {
                        mode: sorted({n for reads in engine_reads for n in reads.get(mode, [])})
                        for mode in modes
                    },
                    # Every engine read its own request's length (vacuous on the eager leg).
                    "matches": all(
                        reads.get(mode) == lengths
                        for reads, lengths in zip(engine_reads, request_text_seq_len, strict=True)
                        for mode in modes
                    ),
                }
            results.append(
                {
                    "instruction": instruction,
                    "seed": seed,
                    "normalized": np.stack(predictions),
                    "decoded": {key: np.stack(values) for key, values in decoded.items()},
                    "latency_ms": latencies,
                    "text_seq_len": text_seq_len,
                }
            )
            print(
                f"[policy-action-gate] {leg} seed {seed}: {len(observations)} observations, "
                f"text_seq_len {text_seq_len}",
                flush=True,
            )
        if cuda_graph:
            results[-1]["cuda_graph_readiness"] = policy.server_readiness()
    finally:
        del wrapper, policy
        gc.collect()
        torch.cuda.empty_cache()
    return results


def _relative(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        return {"finite": False, "relative_rmse": float("inf"), "cosine": float("nan")}
    difference = actual - expected
    denominator = float(np.sqrt(np.mean(expected**2))) + 1e-12
    cosine = float(
        np.dot(actual.ravel(), expected.ravel())
        / (np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel()) + 1e-12)
    )
    return {
        "finite": True,
        "relative_rmse": float(np.sqrt(np.mean(difference**2)) / denominator),
        "max_abs": float(np.abs(difference).max()),
        "cosine": cosine,
    }


def _within(drift: dict[str, Any], spread: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        drift["finite"]
        and drift["relative_rmse"] <= args.max_relative_rmse
        and drift["relative_rmse"] / max(spread["relative_rmse"], 1e-12) <= args.max_spread_fraction
    )


def _pooled(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "normalized": np.concatenate([run["normalized"] for run in runs]),
        "decoded": {
            key: np.concatenate([run["decoded"][key] for run in runs]) for key in runs[0]["decoded"]
        },
    }


def main() -> int:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    embodiment = args.embodiment.lower()
    instructions = args.instruction
    sequence = _prompt_sequence(args.prompt_sequence, instructions)
    prompts = list(dict.fromkeys(sequence))
    observations = _observations(Path(args.observations))
    language_keys = _language_keys(checkpoint, embodiment)
    # The eager reference of every prompt, and its seed-to-seed spread.
    eager_keys = [(p, seed) for p in prompts for seed in (args.seed, args.seed + 1)]
    eager_runs = _run_leg(
        leg="eager",
        args=args,
        observations=observations,
        language_keys=language_keys,
        runs=[(instructions[p], seed) for p, seed in eager_keys],
    )
    eager = dict(zip(eager_keys, eager_runs, strict=True))
    trt = _run_leg(
        leg="trt",
        args=args,
        observations=observations,
        language_keys=language_keys,
        runs=[(instructions[p], args.seed) for p in sequence],
    )

    action_channels = sum(value.shape[-1] for value in eager_runs[0]["decoded"].values())
    valid = slice(0, action_channels)
    steps = []
    first_step: dict[int, int] = {}
    for index, (prompt, candidate) in enumerate(zip(sequence, trt, strict=True)):
        reference = eager[(prompt, args.seed)]
        normalized = _relative(
            candidate["normalized"][..., valid], reference["normalized"][..., valid]
        )
        spread = _relative(
            eager[(prompt, args.seed + 1)]["normalized"][..., valid],
            reference["normalized"][..., valid],
        )
        step = {
            "prompt": prompt,
            "instruction": instructions[prompt],
            "normalized_trt_vs_eager": normalized,
            "normalized_eager_seed_spread": spread,
            "spread_fraction": normalized["relative_rmse"] / max(spread["relative_rmse"], 1e-12),
            "decoded_trt_vs_eager": action_equivalence(reference["decoded"], candidate["decoded"]),
            "text_seq_len": candidate["text_seq_len"],
            "latency_ms": candidate["latency_ms"],
            "repeat_of_step": first_step.get(prompt),
            "passed": _within(normalized, spread, args),
        }
        if candidate["text_seq_len"] is not None:
            step["passed"] = step["passed"] and candidate["text_seq_len"]["matches"]
        if prompt in first_step:
            # A repeated prompt reproduces its first step: nothing of the prompts in between
            # (their length, mask or conditioning) outlives their requests.
            first = trt[first_step[prompt]]["normalized"]
            repeat = _relative(candidate["normalized"][..., valid], first[..., valid])
            repeat["bitwise_equal"] = bool(np.array_equal(candidate["normalized"], first))
            step["normalized_trt_vs_first_step"] = repeat
            step["passed"] = step["passed"] and _within(repeat, spread, args)
        first_step.setdefault(prompt, index)
        steps.append(step)

    reference = _pooled([eager[(p, args.seed)] for p in sequence])
    seed_spread = _pooled([eager[(p, args.seed + 1)] for p in sequence])
    candidate = _pooled(trt)
    normalized = _relative(candidate["normalized"][..., valid], reference["normalized"][..., valid])
    spread = _relative(seed_spread["normalized"][..., valid], reference["normalized"][..., valid])
    spread_fraction = normalized["relative_rmse"] / max(spread["relative_rmse"], 1e-12)
    decoded = action_equivalence(reference["decoded"], candidate["decoded"])
    decoded_spread = action_equivalence(reference["decoded"], seed_spread["decoded"])
    readiness = trt[-1].get("cuda_graph_readiness")
    graph_ok = readiness is None or (
        readiness.get("healthy") is True
        and readiness.get("ready") is True
        and readiness["cuda_graph_lifecycle"]["capture_count"] == 1
        and readiness["cuda_graph_lifecycle"]["recapture_count"] == 0
    )
    passed = _within(normalized, spread, args) and all(s["passed"] for s in steps) and graph_ok
    receipt = {
        "schema": GATE_SCHEMA,
        "checkpoint": str(checkpoint),
        "trt_checkpoint": str(args.trt_checkpoint or checkpoint),
        "embodiment": embodiment,
        "engine_dir": str(args.engine_dir),
        "bare_engines": bool(args.bare_engines),
        "trt_cuda_graph": bool(args.trt_cuda_graph),
        "observations": len(observations),
        "instruction": instructions[sequence[0]],
        "instructions": instructions,
        "prompt_sequence": sequence,
        "seed": args.seed,
        "schedule": f"{args.num_video_steps}v{args.num_action_steps}a",
        "codec": args.codec,
        "action_channels": action_channels,
        "thresholds": {
            "max_relative_rmse": args.max_relative_rmse,
            "max_spread_fraction": args.max_spread_fraction,
        },
        "normalized_trt_vs_eager": normalized,
        "normalized_eager_seed_spread": spread,
        "spread_fraction": spread_fraction,
        "decoded_trt_vs_eager": decoded,
        "decoded_eager_seed_spread": decoded_spread,
        "steps": steps,
        "cuda_graph_readiness": readiness,
        "latency_ms": {
            "eager": [ms for p in prompts for ms in eager[(p, args.seed)]["latency_ms"]],
            "trt": [ms for run in trt for ms in run["latency_ms"]],
        },
        "passed": bool(passed),
    }
    if args.save_actions:
        directory = Path(args.save_actions)
        directory.mkdir(parents=True, exist_ok=True)
        dumps = {f"eager_prompt{p}_seed{seed}": run for (p, seed), run in eager.items()}
        for index, (p, run) in enumerate(zip(sequence, trt, strict=True)):
            dumps[f"trt_step{index}_prompt{p}_seed{args.seed}"] = run
        for name, run in dumps.items():
            np.savez(
                directory / f"{name}.npz",
                normalized=run["normalized"],
                **{f"decoded|{key}": value for key, value in run["decoded"].items()},
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("POLICY_ACTION_GATE", json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
