#!/usr/bin/env python3
# ruff: noqa: T201
"""Serve one N2 TensorRT engine set that has no admitted bundle manifest, for a research evaluation.

The policy is constructed exactly as the measurement harness's bare-engine legs construct it
(``benchmark_adapter.build_benchmark_policy(bare_engines=True)``): the engines named by the build
receipt ``n2_trt_build.json`` beside them, no manifest, never admitted. The production launchers
are untouched; this is the research-only serving entry point and it says so on every start.
Serving quantized engines keeps its explicit opt-in (``GROOT_N2_TRT_ALLOW_QUANTIZED=1``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from groot.core.policy import base
from groot.deployment.n2_trt.manifest import BUILD_FILENAME, MANIFEST_FILENAME


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="the checkpoint the engines were exported from"
    )
    parser.add_argument("--embodiment", required=True, help="EmbodimentTag value (or member name)")
    parser.add_argument(
        "--engine-dir",
        required=True,
        type=Path,
        help=f"directory with the engines and {BUILD_FILENAME}, without {MANIFEST_FILENAME}",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-action-steps", type=int, default=4)
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="serving codec roundtrip; must equal the receipt's build codec",
    )
    parser.add_argument(
        "--no-cuda-graph", action="store_true", help="serve without the predict CUDA graph"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="keep requests here (observation arrays, returned chunk, wall time)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="with --log-dir, keep every Nth request (one request is ~9 MB of video; 50+ for an evaluation)",
    )
    return parser.parse_args(argv)


def resolve_embodiment(spec: str, tags: Any) -> Any:
    """Accept the tag's value (``unitree_g1_...``) or its member name (``UNITREE_G1_...``)."""
    try:
        return tags(spec)
    except ValueError:
        return tags[spec]


def check_engine_dir(engine_dir: Path) -> Path:
    """Bare engines only: a manifest means an admitted bundle, no receipt means nothing built."""
    engine_dir = engine_dir.expanduser().resolve()
    if (engine_dir / MANIFEST_FILENAME).is_file():
        raise SystemExit(
            f"{engine_dir} carries {MANIFEST_FILENAME}: an admitted bundle is served by "
            "run_gr00t_server --trt-engine-dir"
        )
    if not (engine_dir / BUILD_FILENAME).is_file():
        raise SystemExit(
            f"{engine_dir} has no {BUILD_FILENAME}: build the engines first "
            "(python -P -m groot.deployment.n2_trt.tools.build_engines --artifact-root <dir>)"
        )
    return engine_dir


def _to_np(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return None


def _flat_items(observation: dict[str, Any], prefix: str = ""):
    for key, value in observation.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flat_items(value, name + ".")
        else:
            yield name, value


class RequestLogger(base.PolicyWrapper):
    """Keep every Nth request: the observation as received, the returned chunk, the wall time.

    Validation stays with the wrapped policy (its own ``get_action`` runs its checks), so
    the wrapper is not strict itself.
    """

    def __init__(self, policy: Any, save_dir: Path, every: int = 1):
        super().__init__(policy, strict=False)
        self._dir = save_dir
        self._every = max(1, every)
        self._n = 0

    def _get_action(self, observation, options=None):
        t0 = time.perf_counter()
        chunk, extra = self.policy.get_action(observation, options)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        index = self._n
        self._n += 1
        if index % self._every:
            return chunk, extra
        step = self._dir / f"req_{index:06d}"
        step.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "request": index,
            "timestamp": time.time(),
            "wall_ms": wall_ms,
            "observation": {},
        }
        for key, value in _flat_items(observation):
            arr = _to_np(value)
            if arr is not None:
                np.save(step / f"obs__{key.replace('/', '_')}.npy", arr)
                meta["observation"][key] = f"{tuple(arr.shape)} {arr.dtype}"
            else:
                meta["observation"][key] = (
                    value if isinstance(value, str | int | float | bool) else repr(value)[:200]
                )
        for key, value in chunk.items():
            arr = _to_np(value)
            if arr is not None:
                np.save(step / f"chunk__{key}.npy", arr)
        meta["extra"] = {
            k: (float(v) if isinstance(v, int | float) else repr(v)[:100])
            for k, v in (extra or {}).items()
        }
        (step / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return chunk, extra

    def check_observation(self, observation):
        self.policy.check_observation(observation)

    def check_action(self, action):
        self.policy.check_action(action)

    def get_modality_config(self):
        return self.policy.get_modality_config()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if os.environ.get("GROOT_N2_TRT_ALLOW_QUANTIZED") != "1":
        raise SystemExit(
            "set GROOT_N2_TRT_ALLOW_QUANTIZED=1: serving quantized engines is an explicit opt-in"
        )
    engine_dir = check_engine_dir(args.engine_dir)

    from groot.core.data.schema import EmbodimentTag
    from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
    from groot.core.policy.server_client import PolicyServer
    from groot.deployment.n2_trt.benchmark_adapter import build_benchmark_policy

    tag = resolve_embodiment(args.embodiment, EmbodimentTag)
    policy = build_benchmark_policy(
        backend="trt" if args.no_cuda_graph else "trt_graph",
        checkpoint=args.checkpoint,
        embodiment_tag=tag,
        codec=args.codec,
        device=args.device,
        engine_dir=str(engine_dir),
        num_video_steps=args.num_video_steps,
        num_action_steps=args.num_action_steps,
        bare_engines=True,
    )
    contract = dict(getattr(policy, "benchmark_runtime_contract", None) or {})
    identity = {
        "research_only": (
            "bare engines through the measurement seam: no bundle manifest, no admission, "
            "no golden record"
        ),
        "engine_dir": str(engine_dir),
        "checkpoint": args.checkpoint,
        "embodiment": tag.value,
        "cuda_graph": not args.no_cuda_graph,
        "num_action_steps": args.num_action_steps,
        "num_video_steps": args.num_video_steps,
        "codec": args.codec,
        "runtime_contract": contract,
    }
    engines = (contract.get("bare_engines") or {}).get("engines", {})
    print(
        f"[serve_bare_engines] research-only serving of {engine_dir} "
        f"(precision {contract.get('precision')!r}, engines {json.dumps(engines, default=str)})",
        flush=True,
    )

    served: Any = policy
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        (args.log_dir / "identity.json").write_text(json.dumps(identity, indent=2, default=str))
        served = RequestLogger(policy, args.log_dir, args.log_every)
        print(
            f"[serve_bare_engines] keeping every {args.log_every} request(s) under {args.log_dir}",
            flush=True,
        )
    served = Gr00tFlatPolicyWrapper(served, strict=True)
    server = PolicyServer(policy=served, host=args.host, port=args.port)
    print(f"[serve_bare_engines] listening on {args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        print("[serve_bare_engines] shutting down", flush=True)
    finally:
        try:
            served.reset()
        except Exception as exc:  # noqa: BLE001
            print(f"[serve_bare_engines] reset failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
