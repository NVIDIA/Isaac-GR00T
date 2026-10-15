# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""Turn-key engine build from a golden package: pull, verify, build, inspect, admit, serve.

The operator chooses the package (by checkpoint) and stands on the platform; everything else
is decided by the package's record: which engines to build, at which build settings, and the
equivalence bounds the fresh engines must reproduce against the package's canonical outputs.
An uncertified platform (no admission row for its runtime envelope) is refused unless the
build is a certification candidate (``--certify``), which mints a candidate receipt that
``--promote`` turns servable once the platform's row has landed in the record.

    python -P -m groot.deployment.n2_trt.tools.build_quantized --golden <package_id> \\
        --checkpoint <dir> --embodiment <tag> --output-dir <dir> [--quantized-onnx-path <dir>] \\
        [--certify] [--dry-run] [--mirror-key <key>]
    python -P -m groot.deployment.n2_trt.tools.build_quantized --promote --golden <package_id> \\
        --output-dir <root> --checkpoint <dir> --embodiment <tag>

The artifact root must lie outside any source checkout: the admission mints a manifest whose
tool revisions must come from clean sources, and a checkout with build outputs inside it is
dirty.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from groot.deployment.n2_trt.golden import (
    GOLDEN_SIDECAR_FILENAME,
    PACKAGE_CONTROL_FILES,
    GoldenRecord,
    GoldenRecordError,
    admitted_served_modes,
    checkpoint_identity,
    checkpoint_identity_matches,
    envelope_provenance_matches,
    load_record,
    package_sha256,
    read_golden_sidecar,
    runtime_envelope_identity,
    served_mode_name,
    sha256_file,
    verify_package_files,
    write_golden_sidecar,
)
from groot.deployment.n2_trt.manifest import (
    ADMISSION_GOLDEN,
    BUILD_FILENAME,
    GOLDEN_ADMISSION_SCHEMA,
    MANIFEST_FILENAME,
    VALIDATION_FILENAME,
    EngineManifest,
    RuntimeIdentity,
    canonical_sha256,
    checkpoint_file_hashes,
)
from groot.deployment.n2_trt.tools.inspect_engines import (
    EngineInspectionError,
    require_current_sidecar,
)

SHA256SUMS_FILENAME = "SHA256SUMS"
# The gear CLI's recursive glob skips the key's own root entries, so those are pulled by exact key.
# The pull must never sync the environment it runs in: a bare `uv run` re-syncs the project's
# default extras and once downgraded torch under a running build, so the pull uses this
# environment's own `gear` console script and falls back to the frozen, no-sync uv form.
GEAR_DOWNLOAD = ("uv", "run", "--frozen", "--no-sync", "gear", "data", "download")
SERVE_ENV_ALWAYS = ("GROOT_N2_TRT_ALLOW_QUANTIZED=1", "OMP_NUM_THREADS=1")
# The files a promotion rewrites: the sidecar (first), then the receipt and the manifest the
# validator writes in that order. They are restored together when the validation fails.
PROMOTION_FILES = (GOLDEN_SIDECAR_FILENAME, VALIDATION_FILENAME, MANIFEST_FILENAME)


def gear_download_command() -> tuple[str, ...]:
    """The gear download command of THIS environment: its console script, else the frozen uv form."""
    gear = Path(sys.executable).with_name("gear")
    if gear.is_file():
        return (str(gear), "data", "download")
    return GEAR_DOWNLOAD


class BuildQuantizedError(RuntimeError):
    """A refused turn-key build or promotion (the message names the reason)."""


@dataclass(frozen=True)
class BuildTools:
    """The effectful seams of a build: injected as fakes by the tests, real by default."""

    pull: Callable[[str, Path, GoldenRecord], None]
    probe_gpu: Callable[[], RuntimeIdentity]
    busy_processes: Callable[[], list[str]]
    build: Callable[[Path, int], None]
    inspect: Callable[[Path], Path]
    validate: Callable[..., dict[str, Any]]
    emit: Callable[[str], None] = print


def default_tools() -> BuildTools:
    from groot.deployment.n2_trt import validation
    from groot.deployment.n2_trt.tools import build_engines, inspect_engines

    def build(root: Path, builder_optimization_level: int) -> None:
        argv = sys.argv
        sys.argv = [
            "build_engines",
            "--artifact-root",
            str(root),
            "--builder-optimization-level",
            str(builder_optimization_level),
        ]
        try:
            code = build_engines.main()
        finally:
            sys.argv = argv
        if code:
            raise BuildQuantizedError(f"engine build failed with exit code {code}")

    return BuildTools(
        pull=pull_package,
        probe_gpu=RuntimeIdentity.capture,
        busy_processes=busy_gpu_processes,
        build=build,
        inspect=inspect_engines.write_sidecar,
        validate=validation.validate_bundle,
    )


def pull_package(mirror_key: str, destination: Path, record: GoldenRecord) -> None:
    """Pull a package from the gear mirror: the tree by recursive glob, the root files by key.

    Every root object is fetched INTO the destination directory: the gear CLI appends '/' to a
    non-force destination and syncs into it, so a ``<dir>/<name>`` destination would nest the
    object as ``<dir>/<name>/<name>`` and fail the SHA256SUMS verification.
    """
    destination.mkdir(parents=True, exist_ok=True)
    command = gear_download_command()
    subprocess.run([*command, f"{mirror_key}/**/*", f"{destination}/"], check=True)
    root_files = sorted(
        {name for name in record.package["files"] if "/" not in name} | set(PACKAGE_CONTROL_FILES)
    )
    for name in root_files:
        subprocess.run([*command, f"{mirror_key}/{name}", f"{destination}/"], check=True)


def busy_gpu_processes() -> list[str]:
    """Compute processes holding the GPU (an idle GPU is a precondition of a certification build)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuildQuantizedError(f"cannot query the GPU's compute processes: {error}") from error
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def verify_sha256sums(package_dir: Path, record: GoldenRecord) -> dict[str, Any]:
    """Every listed file present with its digest, every record file listed, no foreign entry."""
    listing = package_dir / SHA256SUMS_FILENAME
    if not listing.is_file():
        raise BuildQuantizedError(f"{package_dir} has no {SHA256SUMS_FILENAME}")
    listed: dict[str, str] = {}
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise BuildQuantizedError(f"malformed {SHA256SUMS_FILENAME} line: {line!r}")
        digest, name = parts[0], parts[1].lstrip("*")
        listed[name] = digest
    failures = [
        f"{name}: {'missing' if not (package_dir / name).is_file() else 'digest mismatch'}"
        for name, digest in listed.items()
        if not (package_dir / name).is_file() or sha256_file(package_dir / name) != digest
    ]
    unlisted = sorted(set(record.package["files"]) - set(listed))
    if unlisted:
        failures.append(f"record files not listed: {unlisted[:8]}")
    if failures:
        raise BuildQuantizedError(
            f"{SHA256SUMS_FILENAME} verification failed: " + "; ".join(failures)
        )
    content = verify_package_files(package_dir, record.package, allow_extra=False)
    if not content["passed"]:
        raise BuildQuantizedError(
            "package content does not match the record: " + "; ".join(content["failures"][:8])
        )
    return {"listed": len(listed), "checked": content["checked"]}


def check_checkpoint(checkpoint: Path, record: GoldenRecord) -> dict[str, Any]:
    identity = checkpoint_identity(checkpoint)
    differences = checkpoint_identity_matches(record.package["checkpoint"], identity)
    if differences:
        raise BuildQuantizedError(
            f"checkpoint {checkpoint} is not the package's checkpoint: " + "; ".join(differences)
        )
    return identity


def stage_export_half(package_dir: Path, output_dir: Path, record: GoldenRecord) -> int:
    """Copy the record's files into a fresh build root (never over engines or a build receipt)."""
    if output_dir.exists() and (
        (output_dir / BUILD_FILENAME).exists() or any(output_dir.glob("*.engine"))
    ):
        raise BuildQuantizedError(
            f"{output_dir} already carries a build receipt or engines; use a fresh output dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative in record.package["files"]:
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_dir / relative, target)
    return len(record.package["files"])


def platform_row(
    record: GoldenRecord, identity: RuntimeIdentity, *, certify: bool
) -> dict[str, Any] | None:
    envelope = runtime_envelope_identity(identity)
    row = record.admission_row_for(envelope)
    if row is None and not certify:
        raise BuildQuantizedError(
            f"platform {envelope} has no admission row in {record.package_id!r}: this GPU class is "
            "not certified for the package; pass --certify to build a certification candidate"
        )
    return row


def serve_command(
    record: GoldenRecord,
    output_dir: Path,
    row: dict[str, Any] | None,
    *,
    checkpoint: Path,
    embodiment: str,
    identity: dict[str, Any] | None = None,
) -> list[str]:
    """The exact serving invocation (the documented server form) and this platform's certified numbers."""
    env = [SERVE_ENV_ALWAYS[0]]
    if record.certification.level == "amplitude":
        env.append("GROOT_N2_TRT_ALLOW_AMPLITUDE_ONLY=1")
    admitted = None if row is None else admitted_served_modes(row)
    bundle_default = served_mode_name(bool(record.package["build_settings"].get("cond_kv_freeze")))
    if admitted is not None and bundle_default not in admitted:
        # The bundle's default served mode is not admitted here: select the admitted one.
        env.append(f"GROOT_N2_COND_KV_FREEZE={'1' if admitted[0] == 'cond_on' else '0'}")
    env.append(f"GROOT_N2_CODEC_ROUNDTRIP={record.package['build_settings']['codec']}")
    env.append(SERVE_ENV_ALWAYS[1])
    lines = [
        "serve with:",
        "  " + " ".join(env) + " \\",
        "    python -P -m groot.core.policy.run_gr00t_server \\",
        f"      --model-path {checkpoint} --embodiment-tag {embodiment} \\",
        f"      --trt-engine-dir {output_dir} --use-cuda-graph",
        f"  certification level: {record.certification.level}"
        + (
            " (serving acknowledges the missing temporal row)"
            if record.certification.level == "amplitude"
            else ""
        ),
    ]
    if row is None:
        lines.append(
            "  candidate: no certified row for this runtime envelope (not servable until promoted)"
        )
    else:
        lines.append(
            f"  certified on {row['date']} for envelope {json.dumps(row['envelope'], sort_keys=True)}:"
        )
        if identity is not None:
            same_stack = envelope_provenance_matches(row["envelope"], identity)
            lines.append(
                "  certified speed measured on this software stack: "
                + ("yes" if same_stack else "no")
                + f" (row: TensorRT {row['envelope'].get('tensorrt')}, driver "
                + f"{row['envelope'].get('driver_major')}; this host: TensorRT "
                + f"{identity.get('tensorrt')}, driver {identity.get('driver')})"
            )
        for mode, numbers in sorted((row.get("certified") or {}).items()):
            lines.append(f"    {mode}: {json.dumps(numbers, sort_keys=True)}")
        refused = sorted(set(row.get("certified") or {}) - set(admitted or ()))
        if refused:
            lines.append(
                f"  admitted served modes: {', '.join(admitted or ())}; {', '.join(refused)} failed "
                "its gates on this platform and is refused unless "
                "GROOT_N2_TRT_ALLOW_UNADMITTED_MODE=1 acknowledges it"
            )
    return lines


def _engine_files(root: Path) -> list[Path]:
    build = json.loads((root / BUILD_FILENAME).read_text())
    return [root / record["engine"] for record in build.get("artifacts") or []]


def run_build(args: argparse.Namespace, tools: BuildTools) -> dict[str, Any]:
    record = load_record(args.golden)
    digest = package_sha256(record)
    mirror_key = args.mirror_key or record.package["mirror_key"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    settings = record.package["build_settings"]
    plan = {
        "package_id": record.package_id,
        "package_sha256": digest,
        "mirror_key": mirror_key,
        "files": len(record.package["files"]),
        "roles": sorted(record.roles),
        "build_settings": settings,
        "certification_level": record.certification.level,
        "mode": "certify" if args.certify else "serve",
        "output_dir": str(output_dir),
        "source": (
            str(args.quantized_onnx_path) if args.quantized_onnx_path else f"pull from {mirror_key}"
        ),
    }
    tools.emit("[build-quantized] plan: " + json.dumps(plan, sort_keys=True))
    if args.dry_run:
        return {"plan": plan, "dry_run": True}
    if args.quantized_onnx_path:
        package_dir = Path(args.quantized_onnx_path).expanduser().resolve()
    else:
        package_dir = output_dir.parent / f"{record.package_id}.package"
        tools.pull(mirror_key, package_dir, record)
    sums = verify_sha256sums(package_dir, record)
    tools.emit(f"[build-quantized] package verified: {sums['checked']} files match the record")
    check_checkpoint(Path(args.checkpoint).expanduser().resolve(), record)
    identity = tools.probe_gpu()
    busy = tools.busy_processes()
    if busy:
        raise BuildQuantizedError(f"the GPU is not idle: {busy}")
    envelope = runtime_envelope_identity(identity)
    row = platform_row(record, identity, certify=args.certify)
    staged = stage_export_half(package_dir, output_dir, record)
    tools.emit(f"[build-quantized] staged {staged} package files into {output_dir}")
    tools.build(output_dir, int(settings["builder_optimization_level"]))
    for engine in _engine_files(output_dir):
        tools.inspect(engine)
        try:
            require_current_sidecar(engine)
        except EngineInspectionError as error:
            raise BuildQuantizedError(
                f"census sidecar missing or stale before admission: {error}"
            ) from error
    write_golden_sidecar(
        output_dir,
        package_id=record.package_id,
        package_sha256=digest,
        mode="serve" if row is not None else "certify",
        platform_row_sha256=None if row is None else row["row_sha256"],
    )
    receipt = tools.validate(
        artifact_root=output_dir,
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        embodiment=args.embodiment,
    )
    for line in serve_command(
        record,
        output_dir,
        row,
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        embodiment=args.embodiment,
        identity=envelope,
    ):
        tools.emit(line)
    return {"plan": plan, "row": row, "receipt": receipt, "dry_run": False}


def _promotion_completed(
    root: Path, manifest: EngineManifest, record: GoldenRecord, row: dict[str, Any]
) -> bool:
    """Whether ``root`` carries a completed promotion under ``row``, bound by digest.

    The manifest must name the golden route, this package and the sidecar on disk; the receipt
    it minted must be the file on disk (its digest) and a passed serving receipt under this row.
    A serve-mode sidecar alone proves nothing: it is written before the validation runs.
    """
    if manifest.admission != ADMISSION_GOLDEN or manifest.package_id != record.package_id:
        return False
    if manifest.golden_sidecar_sha256 != sha256_file(root / GOLDEN_SIDECAR_FILENAME):
        return False
    receipt_path = root / manifest.validation.relative_path
    if not receipt_path.is_file() or sha256_file(receipt_path) != manifest.validation.sha256:
        return False
    receipt = json.loads(receipt_path.read_text())
    golden = receipt.get("golden") if isinstance(receipt, dict) else None
    return (
        isinstance(golden, dict)
        and receipt.get("schema") == GOLDEN_ADMISSION_SCHEMA
        and receipt.get("passed") is True
        and golden.get("mode") == "serve"
        and golden.get("platform_row_sha256") == row["row_sha256"]
    )


def _snapshot_promotion_files(root: Path) -> dict[str, bytes | None]:
    return {
        name: (root / name).read_bytes() if (root / name).is_file() else None
        for name in PROMOTION_FILES
    }


def _restore_promotion_files(root: Path, snapshot: dict[str, bytes | None]) -> None:
    for name, content in snapshot.items():
        if content is None:
            (root / name).unlink(missing_ok=True)
        else:
            (root / name).write_bytes(content)


def run_promote(args: argparse.Namespace, tools: BuildTools) -> dict[str, Any]:
    """Turn a certify-mode root servable once its platform row has landed in the record.

    Promotion is a transaction over the sidecar, the receipt and the manifest: a failed
    validation restores all three byte-for-byte and the root stays a candidate. A root in serve
    mode is skipped only with a completed promotion under the current row; anything else (a
    stranded sidecar, a replaced receipt, a row the record no longer carries) validates again.
    """
    record = load_record(args.golden)
    digest = package_sha256(record)
    root = Path(args.output_dir).expanduser().resolve()
    sidecar = read_golden_sidecar(root)
    if sidecar["package_id"] != record.package_id or sidecar["package_sha256"] != digest:
        raise BuildQuantizedError(f"{root} was not built from golden package {record.package_id!r}")
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise BuildQuantizedError(f"{root} carries no candidate manifest; build it first")
    manifest = EngineManifest.read(manifest_path)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if manifest.checkpoint_sha256 != canonical_sha256(checkpoint_file_hashes(checkpoint)):
        raise BuildQuantizedError("the candidate was built for another checkpoint")
    if manifest.contract.get("embodiment") != args.embodiment:
        raise BuildQuantizedError(
            f"the candidate was built for embodiment {manifest.contract.get('embodiment')!r}, "
            f"not {args.embodiment!r}"
        )
    envelope = runtime_envelope_identity(tools.probe_gpu())
    row = record.admission_row_for(envelope)
    if row is None:
        raise BuildQuantizedError(
            f"{record.package_id!r} has no admission row for this platform yet; nothing to promote"
        )
    if sidecar["mode"] == "serve":
        if sidecar["platform_row_sha256"] == row["row_sha256"]:
            if _promotion_completed(root, manifest, record, row):
                tools.emit("[build-quantized] already promoted under this platform row")
                return {"row": row, "promoted": False}
            tools.emit("[build-quantized] the promotion under this platform row did not complete")
        elif any(
            other["row_sha256"] == sidecar["platform_row_sha256"] for other in record.admission_rows
        ):
            raise BuildQuantizedError(
                "the root is already in serve mode under another platform row"
            )
        else:
            tools.emit("[build-quantized] the sidecar's platform row is no longer in the record")
    snapshot = _snapshot_promotion_files(root)
    write_golden_sidecar(
        root,
        package_id=record.package_id,
        package_sha256=digest,
        mode="serve",
        platform_row_sha256=row["row_sha256"],
    )
    try:
        receipt = tools.validate(
            artifact_root=root, checkpoint=checkpoint, embodiment=args.embodiment
        )
    except BaseException as error:
        _restore_promotion_files(root, snapshot)
        if isinstance(error, Exception):
            raise BuildQuantizedError(
                f"promotion rolled back; the root stays a candidate: {error}"
            ) from error
        raise
    for line in serve_command(
        record, root, row, checkpoint=checkpoint, embodiment=args.embodiment, identity=envelope
    ):
        tools.emit(line)
    return {"row": row, "receipt": receipt, "promoted": True}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--golden", required=True, help="golden package id (its record is repository data)"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="fresh build root outside any checkout"
    )
    parser.add_argument(
        "--quantized-onnx-path", type=Path, help="an already pulled package directory"
    )
    parser.add_argument("--mirror-key", help="override the record's mirror key")
    parser.add_argument(
        "--certify",
        action="store_true",
        help="build a certification candidate on an uncertified platform",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="turn a certify-mode root servable once its row landed",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the resolved plan and stop before the pull"
    )
    args = parser.parse_args(argv)
    if args.promote and (args.certify or args.dry_run or args.quantized_onnx_path):
        parser.error("--promote takes only --golden, --checkpoint, --embodiment and --output-dir")
    return args


def main(argv: Sequence[str] | None = None, tools: BuildTools | None = None) -> int:
    args = parse_args(argv)
    tools = tools or default_tools()
    try:
        if args.promote:
            run_promote(args, tools)
        else:
            run_build(args, tools)
    except (BuildQuantizedError, GoldenRecordError) as error:
        tools.emit(f"[build-quantized] refused: {error}")
        return 2
    return 0


__all__ = [
    "GOLDEN_SIDECAR_FILENAME",
    "BuildQuantizedError",
    "BuildTools",
    "check_checkpoint",
    "default_tools",
    "main",
    "parse_args",
    "platform_row",
    "pull_package",
    "run_build",
    "run_promote",
    "serve_command",
    "stage_export_half",
    "verify_sha256sums",
]


if __name__ == "__main__":
    raise SystemExit(main())
