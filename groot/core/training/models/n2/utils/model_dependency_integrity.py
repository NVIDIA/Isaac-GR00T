# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Local validation for downloaded and pre-existing N2 model dependencies.

Legacy directories receive structural checks, not a checksum guarantee. In
particular, an opaque legacy pickle or tokenizer binary can only be checked for
presence/nonzero size without deserializing it. A receipt records the bytes of a
size-verified download; it detects subsequent corruption, but is not a producer
checksum. Receipt verification hashes every payload unless the caller supplies a
cache of successful validations whose receipt, inventory and file identities are
unchanged. Stat-based reuse cannot detect silent bitrot without metadata changes.
No network access or model deserialization occurs.
"""

from collections.abc import Collection, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import tempfile
from typing import Any
import zipfile

DEPENDENCY_RECEIPT_FILENAME = ".model-dependency.json"


def _nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty dependency file: {path}")


def _json_object(path: Path) -> dict[str, Any]:
    _nonempty_file(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _relative_file(root: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"Invalid dependency relative filename: {name!r}")
    return root / name


def _weights_file(path: Path) -> None:
    _nonempty_file(path)
    if path.suffix == ".safetensors":
        from safetensors import SafetensorError, safe_open

        try:
            # Checks the header, tensor offsets and total file length without
            # loading tensors or allocating a model-sized buffer.
            with safe_open(str(path), framework="numpy") as weights:
                if not weights.keys():
                    raise ValueError(f"No tensors in {path}")
        except SafetensorError as error:
            raise ValueError(f"Invalid safetensors weights {path}: {error}") from error
    else:
        with path.open("rb") as stream:
            is_zip = stream.read(2) == b"PK"
        if is_zip:
            try:
                # Modern torch.save uses ZIP. Catch truncation without unsafe
                # pickle loading; opaque pre-ZIP saves remain legacy-compatible.
                with zipfile.ZipFile(path) as archive:
                    if not archive.infolist():
                        raise ValueError(f"Empty weights archive: {path}")
            except zipfile.BadZipFile as error:
                raise ValueError(f"Invalid weights archive {path}: {error}") from error


def _model_directory(root: Path) -> None:
    _json_object(root / "config.json")
    for stem in (
        "model.safetensors",
        "pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
    ):
        index = root / f"{stem}.index.json"
        if index.exists():
            weight_map = _json_object(index).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Missing weight_map in {index}")
            shards = {_relative_file(root, name) for name in weight_map.values()}
            for shard in shards:
                _weights_file(shard)
            return
        weights = root / stem
        if weights.exists():
            _weights_file(weights)
            return
    raise ValueError(f"Missing model weights or weight index in {root}")


def _tokenizer_directory(root: Path) -> None:
    if (root / "tokenizer.json").exists():
        _json_object(root / "tokenizer.json")
        return
    for filename in ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "vocab.txt"):
        if (root / filename).exists():
            _nonempty_file(root / filename)
            return
    if (root / "vocab.json").exists():
        _json_object(root / "vocab.json")
        _nonempty_file(root / "merges.txt")
        return
    raise ValueError(f"Missing tokenizer vocabulary in {root}")


def _text_encoder_processor(root: Path) -> None:
    if _json_object(root / "config.json").get("model_type") != "qwen2_5_vl":
        return
    # Qwen's AutoProcessor also loads image-processing metadata. Modern HF
    # saves can nest it; legacy saves keep it in preprocessor_config.json.
    processor = root / "processor_config.json"
    if processor.exists():
        image_processor = _json_object(processor).get("image_processor")
        if isinstance(image_processor, dict) and image_processor:
            return
    legacy = root / "preprocessor_config.json"
    if legacy.exists() and _json_object(legacy):
        return
    raise ValueError(f"Missing or empty Qwen image-processor configuration in {root}")


def _receipt_files(path: Path, names: Collection[str]) -> dict[str, Path]:
    if path.is_file():
        if set(names) != {"."}:
            raise ValueError("Single-file dependency receipt must contain only '.'")
        return {".": path}
    if not path.is_dir():
        raise ValueError(f"Missing model dependency: {path}")
    files = {name: _relative_file(path, name) for name in names}
    if DEPENDENCY_RECEIPT_FILENAME in files:
        raise ValueError("Dependency inventory includes its reserved receipt filename")
    actual = {
        entry.relative_to(path).as_posix()
        for entry in path.rglob("*")
        if entry.is_file() and entry != path / DEPENDENCY_RECEIPT_FILENAME
    }
    if actual != set(files):
        raise ValueError(
            f"Dependency inventory mismatch in {path}: "
            f"missing={sorted(set(files) - actual)}, unexpected={sorted(actual - set(files))}"
        )
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        # Keep large model shards bounded in memory on supported Python 3.10.
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dependency_receipt(path: str | Path, expected_sizes: Mapping[str, int]) -> dict[str, Any]:
    """Capture a complete download using sizes from its remote inventory.

    Directory keys are relative POSIX file paths. A single file uses the key
    ``'.'`` so its receipt survives the staging-to-final rename. Callers own
    structural role validation and atomic receipt/payload publication.
    """
    if not expected_sizes:
        raise ValueError("Dependency inventory must not be empty")
    files = _receipt_files(Path(path), expected_sizes)
    result = {}
    for name, file in files.items():
        expected_size = expected_sizes[name]
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"Invalid expected dependency size for {name}: {expected_size!r}")
        if not file.is_file() or file.stat().st_size != expected_size:
            raise ValueError(f"Dependency size mismatch for {file}; expected {expected_size} bytes")
        result[name] = {"size": expected_size, "sha256": _sha256(file)}
    return {"version": 1, "files": result}


def _verification_fingerprint(receipt: Mapping[str, Any], files: Mapping[str, Path]) -> str:
    identities = {}
    for name, file in files.items():
        resolved = file.resolve()
        stat = resolved.stat()
        identities[name] = (
            str(resolved),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    # Device/inode values belong to a host's filesystem namespace. A shared
    # lock root must not transfer this stat-based certificate to another host.
    return hashlib.sha256(
        json.dumps([1, socket.gethostname(), receipt, identities], sort_keys=True).encode()
    ).hexdigest()


def _remember_verification(path: Path, fingerprint: str) -> None:
    """Atomically record a successful check; an unavailable cache only costs I/O."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(fingerprint + "\n")
        os.replace(temporary, path)
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _validate_receipt(
    path: Path, receipt: Mapping[str, Any], verification_cache_dir: Path | None = None
) -> None:
    if not isinstance(receipt, Mapping):
        raise TypeError("Model dependency receipt must be an object")
    if type(receipt.get("version")) is not int or receipt["version"] != 1:
        raise ValueError("Unsupported model dependency receipt version")
    records = receipt.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("Model dependency receipt has no file inventory")
    files = _receipt_files(path, records)
    for name, file in files.items():
        record = records[name]
        if (
            not isinstance(record, dict)
            or type(record.get("size")) is not int
            or record["size"] < 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in record["sha256"])
        ):
            raise ValueError(f"Invalid model dependency receipt entry for {name}")
        if not file.is_file() or file.stat().st_size != record["size"]:
            raise ValueError(f"Dependency size mismatch for {file}")

    cached = None
    if verification_cache_dir is not None:
        key = hashlib.sha256(
            json.dumps([socket.gethostname(), str(path.absolute())]).encode()
        ).hexdigest()
        cached = verification_cache_dir / f"{key}.sha256"
        fingerprint = _verification_fingerprint(receipt, files)
        try:
            if cached.read_text() == fingerprint + "\n":
                return
        except (OSError, UnicodeError):
            pass

    for name, file in files.items():
        record = records[name]
        if _sha256(file) != record["sha256"]:
            raise ValueError(f"Dependency SHA256 mismatch for {file}")

    if cached is not None:
        # Do not certify a file replaced/modified while hashing, or a changed
        # directory inventory. The resolver holds its cache lock throughout.
        if _verification_fingerprint(receipt, _receipt_files(path, records)) != fingerprint:
            raise ValueError(f"Dependency changed during checksum validation: {path}")
        _remember_verification(cached, fingerprint)


def dependency_invalid_reason(
    path: str | Path,
    roles: Collection[str],
    *,
    receipt: Mapping[str, Any] | None = None,
    verification_cache_dir: Path | None = None,
) -> str | None:
    """Return a local validation failure, or ``None`` for a usable candidate.

    Roles aggregate all consumers of a dependency path. A tokenizer-only folder
    does not need weights; a folder shared by tokenizer and text encoder needs
    both. A supplied receipt verifies every byte unless a successful verification
    cache matches its current file identities. The resolver supplies a writable
    cache beside its lock; omitting it forces a full hash. Without a receipt,
    structural validity does not establish byte-level integrity.
    """
    path = Path(path)
    try:
        if not roles or not set(roles) <= {"model", "text_encoder", "tokenizer"}:
            raise ValueError(f"Unknown or empty model dependency roles: {roles}")
        if path.is_dir():
            if set(roles) & {"model", "text_encoder"}:
                _model_directory(path)
            if "text_encoder" in roles:
                _text_encoder_processor(path)
            if "tokenizer" in roles:
                _tokenizer_directory(path)
        elif "tokenizer" in roles or "text_encoder" in roles:
            raise ValueError(f"Expected a dependency directory for {sorted(roles)}: {path}")
        else:
            _weights_file(path)
        if receipt is not None:
            _validate_receipt(path, receipt, verification_cache_dir)
    except (OSError, TypeError, ValueError, UnicodeError) as error:
        return str(error)
    return None
