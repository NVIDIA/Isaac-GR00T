# SPDX-License-Identifier: Apache-2.0
"""GR00T policy server for the Unitree G1 (bimanual, Dex3 hands, upper body).

A single process that loads the finetuned ``new_embodiment`` GR00T N1.7
checkpoint in-memory *and* speaks the TRI ``policy_interfaces`` gRPC contract, so
anzu's ``RosRobotPolicyRunner`` (the policy client, on the G1's Jetson Orin) can
drive it unchanged over ethernet. Runs on the workstation dGPU.

The server invokes three duck-typed methods on the policy object
(``get_policy_metadata``, ``reset_batch``, ``step_batch``); ``step`` / ``reset``
satisfy the abstract :class:`Policy` base and delegate to the batch methods.

GR00T predicts an action *chunk* (up to 16 steps); the gRPC contract returns one
action per ``step``. We therefore keep a per-client open-loop buffer: re-query
the model every ``open_loop_steps`` steps and index through the chunk in
between (mirroring vla_foundry's ``InferenceDiffusionPolicy``).

``--model-path`` accepts either a local checkpoint directory or an
``s3://bucket/prefix`` URI; an S3 URI is downloaded to a local cache directory
before loading (see :func:`resolve_model_path`).

Launch::

    python gr00t_g1_policy_server.py --model-path /path/to/finetuned_gr00t \\
        --device cuda --open-loop-steps 8 --server-uri 0.0.0.0:50051

    # or load straight from S3
    python gr00t_g1_policy_server.py --model-path s3://my-bucket/checkpoints/gr00t \\
        --device cuda --open-loop-steps 8 --server-uri 0.0.0.0:50051
"""

import argparse
import os
import uuid
from pathlib import Path

from gr00t.eval.real_robot.G1Dex3.conversions import (
    G1DexConversionConfig,
    gr00t_action_to_poses_and_grippers,
    multiarm_obs_to_gr00t,
)
from policy_interfaces.grpc_interface.policy_server import LbmPolicyServerConfig, run_policy_server
from policy_interfaces.robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from policy_interfaces.robot_gym.policy import Policy, PolicyMetadata


DEFAULT_SERVER_URI = "0.0.0.0:50051"


def resolve_model_path(model_path: str, cache_dir: str | None = None) -> str:
    """Return a local checkpoint directory, downloading from S3 if needed.

    Local paths pass through unchanged. For an ``s3://bucket/prefix`` URI every
    object under the prefix is downloaded (preserving its relative layout) into a
    local cache directory, and that directory is returned so the rest of the
    loader (``AutoModel`` / ``AutoProcessor.from_pretrained``) sees an ordinary
    local checkpoint.

    Args:
        model_path: A local path or an ``s3://bucket/prefix`` URI.
        cache_dir: Where to place the download. Defaults to the ``GR00T_S3_CACHE_DIR``
            env var, else ``~/.cache/gr00t/s3/<bucket>/<prefix>``.

    Files already present locally with a matching size are skipped, so restarts
    against the same S3 checkpoint don't re-download.
    """
    if not model_path.startswith("s3://"):
        return model_path

    import boto3

    bucket, _, prefix = model_path[len("s3://") :].partition("/")
    prefix = prefix.strip("/")
    if not bucket or not prefix:
        raise ValueError(
            f"Malformed S3 model path {model_path!r}; expected s3://bucket/prefix"
        )

    cache_dir = cache_dir or os.environ.get("GR00T_S3_CACHE_DIR")
    dest_root = (
        Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gr00t" / "s3"
    ) / bucket / prefix

    print(f"[gr00t_g1_policy] downloading {model_path} -> {dest_root}", flush=True)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0
    skipped = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # S3 "directory" placeholder
                continue
            rel = key[len(prefix) + 1 :]
            local_file = dest_root / rel
            if local_file.exists() and local_file.stat().st_size == obj["Size"]:
                skipped += 1
                continue
            local_file.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_file))
            downloaded += 1

    if downloaded == 0 and skipped == 0:
        raise FileNotFoundError(f"No objects found under {model_path!r}")
    print(
        f"[gr00t_g1_policy] S3 checkpoint ready ({downloaded} downloaded, "
        f"{skipped} cached) at {dest_root}",
        flush=True,
    )
    return str(dest_root)


class GrootG1Policy(Policy):
    """Serves the finetuned G1-Dex3 GR00T checkpoint over the TRI contract."""

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "new_embodiment",
        device: int | str = "cuda",
        open_loop_steps: int = 8,
        conversion_config: G1DexConversionConfig | None = None,
        s3_cache_dir: str | None = None,
    ):
        if open_loop_steps < 1:
            raise ValueError(f"open_loop_steps must be >= 1, got {open_loop_steps}")

        # Belt-and-suspenders: register the new_embodiment modality config. The
        # checkpoint's own processor already carries it, so this is a no-op in
        # the normal case but guards code paths that consult the global registry.
        try:
            import examples.G1Dex3.g1_dex3_modality_config  # noqa: F401
        except Exception:  # pragma: no cover - registration is optional
            pass

        # Imported lazily so the module imports cheaply (e.g. for unit tests)
        # without pulling in torch / the model stack.
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        # Keep the caller-supplied path (possibly an s3:// URI) for metadata, but
        # load from a resolved local directory.
        self.model_path = model_path
        local_model_path = resolve_model_path(model_path, cache_dir=s3_cache_dir)
        self.embodiment_tag = embodiment_tag
        self.open_loop_steps = open_loop_steps
        self.cfg = conversion_config or G1DexConversionConfig()
        self.policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag, model_path=local_model_path, device=device
        )

        # Per-client open-loop state: uuid -> {"chunk", "idx", "horizon"}.
        self._clients: dict[uuid.UUID, dict] = {}
        # Used when the (non-batch) step/reset interface is called directly.
        self._internal_uuid = uuid.uuid4()

    def get_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="GrootG1Policy",
            skill_type="LanguageConditionedManipulation",
            checkpoint_path=self.model_path,
            is_language_conditioned=True,
            git_repo="Isaac-GR00T",
            git_sha="Undefined",
            runtime_information={
                "embodiment_tag": self.embodiment_tag,
                "open_loop_steps": str(self.open_loop_steps),
            },
        )

    def reset(self, *, seed: int | None = None, options=None) -> None:
        self.reset_batch({self._internal_uuid: seed})

    def reset_batch(self, seeds: dict[uuid.UUID, int | None], options=None) -> None:
        for client_id in seeds:
            self._clients.pop(client_id, None)
        # Gr00tPolicy is stateless across calls, but reset() keeps parity.
        self.policy.reset()

    def step(self, observation: MultiarmObservation) -> PosesAndGrippers:
        return self.step_batch({self._internal_uuid: observation})[self._internal_uuid]

    def step_batch(
        self, observations: dict[uuid.UUID, MultiarmObservation]
    ) -> dict[uuid.UUID, PosesAndGrippers]:
        actions: dict[uuid.UUID, PosesAndGrippers] = {}
        for client_id, obs in observations.items():
            state = self._clients.get(client_id)
            needs_refill = (
                state is None
                or state["idx"] >= self.open_loop_steps
                or state["idx"] >= state["horizon"]
            )
            if needs_refill:
                gr00t_obs = multiarm_obs_to_gr00t(obs, self.cfg)
                chunk, _info = self.policy.get_action(gr00t_obs)
                horizon = next(iter(chunk.values())).shape[1]
                state = {"chunk": chunk, "idx": 0, "horizon": horizon}
                self._clients[client_id] = state

            actions[client_id] = gr00t_action_to_poses_and_grippers(
                state["chunk"], state["idx"], self.cfg
            )
            state["idx"] += 1
        return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local checkpoint directory or an s3://bucket/prefix URI "
        "(downloaded to a local cache before loading).",
    )
    parser.add_argument(
        "--s3-cache-dir",
        default=None,
        help="Directory for S3 checkpoint downloads (default: $GR00T_S3_CACHE_DIR "
        "or ~/.cache/gr00t/s3). Ignored for local --model-path.",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="new_embodiment",
        help="Embodiment tag the checkpoint was finetuned under.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device (e.g. cuda, cuda:0).")
    parser.add_argument(
        "--open-loop-steps",
        type=int,
        default=8,
        help="Steps replayed from an action chunk before re-querying the model.",
    )
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    args = parser.parse_args()

    # Bind on all interfaces by default so the Orin can reach us over ethernet.
    if args.server_uri == LbmPolicyServerConfig.server_uri:
        args.server_uri = DEFAULT_SERVER_URI

    policy = GrootG1Policy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        open_loop_steps=args.open_loop_steps,
        s3_cache_dir=args.s3_cache_dir,
    )
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
