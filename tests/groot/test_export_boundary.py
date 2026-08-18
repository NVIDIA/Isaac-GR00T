from __future__ import annotations

from collections.abc import Iterator, Mapping
import os
from pathlib import Path
import re

OSS_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_EXPORT_EXCLUDED_PATHS = frozenset(
    {
        Path(".gitlab-ci.yml"),
        Path("ci"),
        Path("tests/groot/infra/ci"),
        Path("tests/groot/infra/job"),
    }
)
SKIPPED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
MAX_TEXT_FILE_BYTES = 5 * 1024 * 1024
MOUNT_ROOT = "/" + "mnt"
S3_SCHEME = "s3" + "://"
# Exact bucket names approved for documentation, tests, and examples. Concrete
# deployment buckets must be supplied through configuration rather than added
# here.
S3_BUCKET_ALLOWLIST = frozenset(
    {
        "bucket",
        "example-bucket",
        "explicit-bucket",
        "my-bucket",
        "new-bucket",
        "some-bucket",
        "your-bucket",
    }
)

CONTENT_RULES = {
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "confidentiality-marker": re.compile(
        r"\b(?:NVIDIA\s+CONFIDENTIAL|PROPRIETARY\s+AND\s+CONFIDENTIAL|DO\s+NOT\s+DISTRIBUTE)\b",
        re.IGNORECASE,
    ),
    "embedded-url-credentials": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
    "github-token": re.compile(r"\b(?:gh[opurs]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "gitlab-token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    "internal-gitlab-host": re.compile(r"\bgitlab-master[.]nvidia[.]com\b", re.IGNORECASE),
    "internal-mount-path": re.compile(re.escape(MOUNT_ROOT) + r"""(?:/[^\s"'`<>|;:,)\]}]+)*"""),
    "nvidia-api-key": re.compile(r"\bnvapi-[A-Za-z0-9_-]{40,}\b"),
    "private-key": re.compile(r"-----BEGIN (?:EC |OPENSSH |PGP |RSA )?PRIVATE KEY-----"),
    "specific-s3-location": re.compile(
        re.escape(S3_SCHEME)
        + r"""(?P<bucket>[A-Za-z0-9][A-Za-z0-9._-]*)(?:/[^\s"'`<>|;:,)\]}]+)*"""
    ),
}

# Existing intentional references are baselined by exact path suffix. Suffixes
# account for the skill copies under .agents, .claude, and .codex without
# permitting the marker elsewhere in OSS.
CONTENT_ALLOWLIST: dict[str, frozenset[Path]] = {
    "confidentiality-marker": frozenset(
        {
            Path("LICENSES/LicenseRef-NvidiaProprietary.txt"),
        }
    ),
}

# These are exact suffixes following the guarded mount root. The root is
# intentionally assembled by the scanner so this baseline does not contain and
# then exempt its own references.
# Each file lists only the active, directory-level runtime values it needs. A
# new reference in an existing file still fails CI.
MOUNT_REFERENCE_ALLOWLIST: dict[Path, set[str]] = {
    # Runtime cache roots and filesystem probes for supported cluster types.
    Path("groot/core/data/hydrator/service.py"): {
        "/amlfs-07/shared",
        "/amlfs-07/shared/hydrator_cache",
        "/aws-lfs-02/shared",
        "/aws-lfs-02/shared/hydrator_cache",
        "/aws-lfs-03/shared",
        "/aws-lfs-03/shared/hydrator_cache",
    },
    # Runtime glob that discovers an available LFS mount.
    Path("groot/core/training/experiment/utils.py"): {"", "/*lfs*"},
    # Runtime prefixes used to relocate checkpoint dependencies between clusters.
    Path("groot/core/training/models/n2/utils/cosmos_model_deps.py"): {
        "/amlfs-02/shared/ckpts/",
        "/amlfs-05/shared/ckpts/",
        "/amlfs-07/shared/ckpts/",
        "/aws-lfs-02/shared/ckpts/",
        "/aws-lfs-03/shared/ckpts/",
    },
    # Runtime default for the shared dataset directory.
    Path("groot/core/utils/storage/data_paths.py"): {"/aws-lfs-01/shared/datasets"},
}

# Folder-scoped exceptions remain available for duplicated skill packages, but
# no skill currently requires one.
MOUNT_FOLDER_REFERENCE_ALLOWLIST: dict[Path, set[str]] = {}

S3_REFERENCE_ALLOWLIST: dict[Path, set[str]] = {
    # Existing model-dependency download root.
    Path("groot/core/training/models/n2/utils/cosmos_model_deps.py"): {
        S3_SCHEME + "GearCheckpoints/model_deps/"
    },
}


def _is_excluded_path(relative_path: Path, excluded_paths: frozenset[Path]) -> bool:
    return any(
        relative_path == excluded_path or excluded_path in relative_path.parents
        for excluded_path in excluded_paths
    )


def _iter_text_files(
    root: Path,
    *,
    excluded_paths: frozenset[Path] = frozenset(),
) -> Iterator[tuple[Path, str]]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        current = Path(dirpath)
        current_relative = current.relative_to(root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIPPED_DIRECTORIES
            and not _is_excluded_path(current_relative / name, excluded_paths)
        )
        for filename in sorted(filenames):
            path = current / filename
            relative_path = path.relative_to(root)
            if _is_excluded_path(relative_path, excluded_paths):
                continue
            try:
                if path.stat().st_size > MAX_TEXT_FILE_BYTES:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\0" in raw:
                continue
            try:
                yield relative_path, raw.decode("utf-8")
            except UnicodeDecodeError:
                continue


def _is_allowed(
    rule_name: str,
    relative_path: Path,
    allowlist: Mapping[str, frozenset[Path]],
) -> bool:
    for suffix in allowlist.get(rule_name, frozenset()):
        if relative_path == suffix or relative_path.parts[-len(suffix.parts) :] == suffix.parts:
            return True
    return False


def _contains_path_parts(relative_path: Path, directory: Path) -> bool:
    directory_parts = directory.parts
    return any(
        relative_path.parts[index : index + len(directory_parts)] == directory_parts
        for index in range(len(relative_path.parts) - len(directory_parts) + 1)
    )


def _mount_reference_scope(
    relative_path: Path,
    file_allowlist: Mapping[Path, set[str]],
    folder_allowlist: Mapping[Path, set[str]],
) -> tuple[tuple[str, Path], set[str]] | None:
    for suffix, references in file_allowlist.items():
        if relative_path == suffix or relative_path.parts[-len(suffix.parts) :] == suffix.parts:
            return ("file", suffix), references
    for directory, references in folder_allowlist.items():
        if _contains_path_parts(relative_path, directory):
            return ("folder", directory), references
    return None


def _find_export_leaks(
    root: Path,
    *,
    allowlist: Mapping[str, frozenset[Path]] | None = None,
    mount_reference_allowlist: Mapping[Path, set[str]] | None = None,
    mount_folder_reference_allowlist: Mapping[Path, set[str]] | None = None,
    s3_reference_allowlist: Mapping[Path, set[str]] | None = None,
    excluded_paths: frozenset[Path] = frozenset(),
) -> list[str]:
    use_default_allowlists = (
        allowlist is None
        and mount_reference_allowlist is None
        and mount_folder_reference_allowlist is None
        and s3_reference_allowlist is None
    )
    allowed = CONTENT_ALLOWLIST if allowlist is None else allowlist
    allowed_mount_file_references = (
        MOUNT_REFERENCE_ALLOWLIST if use_default_allowlists else (mount_reference_allowlist or {})
    )
    allowed_mount_folder_references = (
        MOUNT_FOLDER_REFERENCE_ALLOWLIST
        if use_default_allowlists
        else (mount_folder_reference_allowlist or {})
    )
    allowed_s3_references = (
        S3_REFERENCE_ALLOWLIST if use_default_allowlists else (s3_reference_allowlist or {})
    )
    seen_mount_references: dict[tuple[str, Path], set[str]] = {}
    seen_s3_references: dict[Path, set[str]] = {}
    findings = []
    for relative_path, content in _iter_text_files(root, excluded_paths=excluded_paths):
        mount_scope = _mount_reference_scope(
            relative_path,
            allowed_mount_file_references,
            allowed_mount_folder_references,
        )
        allowed_mount_references = mount_scope[1] if mount_scope else set()
        s3_scope = _mount_reference_scope(relative_path, allowed_s3_references, {})
        allowed_file_s3_references = s3_scope[1] if s3_scope else set()
        if relative_path.name == "AGENTS.md" and relative_path != Path("AGENTS.md"):
            findings.append(f"nested-agent-guidance: {relative_path}")

        for rule_name, pattern in CONTENT_RULES.items():
            if rule_name not in {"internal-mount-path", "specific-s3-location"} and _is_allowed(
                rule_name, relative_path, allowed
            ):
                continue
            for match in pattern.finditer(content):
                line_number = content.count("\n", 0, match.start()) + 1
                if rule_name == "specific-s3-location":
                    if match.group("bucket") in S3_BUCKET_ALLOWLIST:
                        continue
                    reference = match.group(0).rstrip(".")
                    if s3_scope and reference in allowed_file_s3_references:
                        seen_s3_references.setdefault(s3_scope[0][1], set()).add(reference)
                        continue
                    findings.append(f"{rule_name}: {relative_path}:{line_number}: {reference}")
                    continue
                if rule_name == "internal-mount-path":
                    reference = match.group(0).removeprefix(MOUNT_ROOT)
                    if mount_scope and reference in allowed_mount_references:
                        seen_mount_references.setdefault(mount_scope[0], set()).add(reference)
                        continue
                    findings.append(f"{rule_name}: {relative_path}:{line_number}: {match.group(0)}")
                    continue
                findings.append(f"{rule_name}: {relative_path}:{line_number}")

    for scope_kind, scoped_allowlist in (
        ("file", allowed_mount_file_references),
        ("folder", allowed_mount_folder_references),
    ):
        for scope_path, references in scoped_allowlist.items():
            unseen_references = references - seen_mount_references.get(
                (scope_kind, scope_path), set()
            )
            for reference in sorted(unseen_references):
                findings.append(
                    "stale-internal-mount-allowlist: "
                    f"{scope_kind} {scope_path}: {MOUNT_ROOT}{reference}"
                )
    for relative_path, references in allowed_s3_references.items():
        for reference in sorted(references - seen_s3_references.get(relative_path, set())):
            findings.append(f"stale-s3-allowlist: file {relative_path}: {reference}")
    return findings


def test_oss_export_contains_no_internal_data_leaks() -> None:
    findings = _find_export_leaks(OSS_ROOT, excluded_paths=PUBLIC_EXPORT_EXCLUDED_PATHS)

    assert findings == [], "OSS export boundary violations:\n" + "\n".join(findings)


def test_export_scanner_detects_high_signal_leaks(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/AGENTS.md").write_text("internal guidance\n")
    (tmp_path / "credentials.txt").write_text(
        "token="
        "gl"
        "pat-"
        + "A" * 20
        + "\nhost=gitlab-master"
        + ".nvidia.com\nmount=/"
        + "mnt/internal/checkpoint\n"
        + "artifact="
        + S3_SCHEME
        + "production-models/checkpoints/model.pt\n"
    )

    findings = _find_export_leaks(tmp_path, allowlist={})

    assert any(finding.startswith("nested-agent-guidance:") for finding in findings)
    assert any(finding.startswith("gitlab-token:") for finding in findings)
    assert any(finding.startswith("internal-gitlab-host:") for finding in findings)
    assert any(finding.startswith("internal-mount-path:") for finding in findings)
    assert any(finding.startswith("specific-s3-location:") for finding in findings)


def test_export_scanner_ignores_only_explicitly_excluded_paths(tmp_path: Path) -> None:
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/credentials.txt").write_text("mount=/" + "mnt/internal/checkpoint\n")
    (tmp_path / "ci-copy").mkdir()
    (tmp_path / "ci-copy/credentials.txt").write_text("mount=/" + "mnt/internal/checkpoint\n")

    findings = _find_export_leaks(
        tmp_path,
        allowlist={},
        excluded_paths=frozenset({Path("ci")}),
    )

    assert findings == [
        "internal-mount-path: ci-copy/credentials.txt:1: " + MOUNT_ROOT + "/internal/checkpoint"
    ]


def test_export_scanner_allows_only_whitelisted_dummy_s3_locations(tmp_path: Path) -> None:
    (tmp_path / "examples.txt").write_text(
        "\n".join(f"{S3_SCHEME}{bucket}/path/to/object" for bucket in S3_BUCKET_ALLOWLIST)
    )

    assert _find_export_leaks(tmp_path, allowlist={}) == []


def test_s3_allowlist_is_scoped_to_an_exact_file_and_location(tmp_path: Path) -> None:
    approved = S3_SCHEME + "existing-bucket/path/to/object"
    unapproved = S3_SCHEME + "existing-bucket/path/to/other-object"
    (tmp_path / "build.txt").write_text(f"{approved}\n{unapproved}\n")

    findings = _find_export_leaks(
        tmp_path,
        allowlist={},
        s3_reference_allowlist={Path("build.txt"): {approved}},
    )

    assert findings == [f"specific-s3-location: build.txt:2: {unapproved}"]


def test_mount_allowlist_ignores_occurrence_count(tmp_path: Path) -> None:
    mount_root = "/" + "mnt"
    (tmp_path / "mounts.txt").write_text(
        f"approved={mount_root}/approved\nduplicate={mount_root}/approved\nnew={mount_root}/new\n"
    )

    findings = _find_export_leaks(
        tmp_path,
        allowlist={},
        mount_reference_allowlist={Path("mounts.txt"): {"/approved"}},
    )

    assert findings == [f"internal-mount-path: mounts.txt:3: {mount_root}/new"]


def test_mount_allowlist_supports_skill_folders(tmp_path: Path) -> None:
    mount_root = "/" + "mnt"
    skill_folder = tmp_path / "agent-copy/skills/example"
    skill_folder.mkdir(parents=True)
    (skill_folder / "one.md").write_text(f"{mount_root}/approved\n")
    (skill_folder / "two.md").write_text(f"{mount_root}/approved\n{mount_root}/new\n")

    findings = _find_export_leaks(
        tmp_path,
        allowlist={},
        mount_folder_reference_allowlist={Path("skills/example"): {"/approved"}},
    )

    assert findings == [
        f"internal-mount-path: agent-copy/skills/example/two.md:2: {mount_root}/new"
    ]


def test_mount_allowlist_reports_stale_references(tmp_path: Path) -> None:
    (tmp_path / "mounts.txt").write_text("no mount here\n")

    findings = _find_export_leaks(
        tmp_path,
        allowlist={},
        mount_reference_allowlist={Path("mounts.txt"): {"/removed"}},
    )

    assert findings == [f"stale-internal-mount-allowlist: file mounts.txt: {MOUNT_ROOT}/removed"]
