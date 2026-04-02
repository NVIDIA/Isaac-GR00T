"""README code block helpers for tests."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import re


@dataclass
class CodeBlock:
    language: str
    code: str
    heading: str
    line_number: int


def extract_code_blocks(readme_path: pathlib.Path) -> list[CodeBlock]:
    """Extract all fenced code blocks from a markdown file with context."""
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    current_heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        heading_match = re.match(r"^#{1,6}\s+(.*)", line)
        if heading_match:
            current_heading = heading_match.group(1).strip()
        fence_match = re.match(r"^```(\w*)", line)
        if fence_match:
            lang = fence_match.group(1)
            start_line = i + 1
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append(
                CodeBlock(
                    language=lang,
                    code="\n".join(code_lines),
                    heading=current_heading,
                    line_number=start_line,
                )
            )
        i += 1
    return blocks


def normalize_ws(text: str) -> str:
    """Normalize whitespace for robust command matching."""
    return " ".join(text.replace("\\\n", " ").replace("\\", " ").split())


def assert_snippets_in_readme(readme_path: pathlib.Path, snippets: list[tuple[str, str]]) -> None:
    """Assert that each (name, snippet) appears in the README (whitespace-normalised)."""
    readme_text = readme_path.read_text(encoding="utf-8")
    normalized = normalize_ws(readme_text)
    missing = [name for name, snippet in snippets if normalize_ws(snippet) not in normalized]
    assert not missing, (
        f"README drift detected in {readme_path.name} for sections: "
        f"{', '.join(missing)}. Update tests or README to keep them aligned."
    )


def find_block(
    blocks: list[CodeBlock],
    keyword: str,
    *,
    language: str | None = None,
    occurrence: int = 1,
) -> CodeBlock:
    """Return the Nth block (1-based) whose code contains keyword."""
    count = 0
    for block in blocks:
        if language is not None and block.language != language:
            continue
        if keyword in block.code:
            count += 1
            if count == occurrence:
                return block
    raise ValueError(
        f"No block containing {keyword!r} found (language={language!r}, occurrence={occurrence})"
    )


def replace_once(text: str, old: str, new: str) -> str:
    """Replace ``old`` with ``new`` in ``text``, asserting exactly one occurrence."""
    assert text.count(old) == 1, f"Expected exactly one occurrence of {old!r}"
    return text.replace(old, new)


def run_bash_blocks(
    blocks: list[CodeBlock | str],
    *,
    cwd: pathlib.Path | None = None,
    env: dict | None = None,
    force_yes: bool = False,
) -> None:
    """Run bash/sh code blocks extracted from a README via subprocess.

    If ``force_yes`` is True, ``apt-get`` commands are automatically given
    the ``-y`` flag so they do not prompt for confirmation.
    """
    import re
    import subprocess

    for block in blocks:
        if isinstance(block, str):
            code = block
        elif block.language in ("bash", "sh"):
            code = block.code
        else:
            continue
        if force_yes:
            # Strip `sudo` from apt/apt-get lines (CI often runs as root).
            code = re.sub(r"\bsudo\s+(apt(?:-get)?)\b", r"\1", code)
            # Insert -y after `apt`/`apt-get <subcommand>` when not already present.
            code = re.sub(r"\bapt(-get)?(\s+\S+)(?!\s+-y)", r"apt\1\2 -y", code)
        print(f"[readme] running bash block:\n{code}\n", flush=True)
        result = subprocess.run(
            ["bash", "-c", code], cwd=cwd, env=env, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", flush=True)
        if result.returncode != 0:
            output = (result.stdout or "") + (result.stderr or "")
            raise AssertionError(
                f"bash block failed (returncode={result.returncode})\noutput:\n{output}"
            )


def run_readme_python_blocks(
    blocks: list[CodeBlock | str],
    readme_path: pathlib.Path,
    repo_root: pathlib.Path,
) -> None:
    """Combine blocks (CodeBlock or raw string) and execute as a single Python script."""
    parts = [b if isinstance(b, str) else b.code for b in blocks]
    combined = CodeBlock(language="python", code="\n".join(parts), heading="", line_number=0)
    run_python_blocks([combined], readme_path=readme_path, repo_root=repo_root)


def run_python_blocks(
    blocks: list[CodeBlock],
    *,
    readme_path: pathlib.Path,
    repo_root: pathlib.Path,
) -> None:
    """Execute Python code blocks extracted from a README."""
    for block in [b for b in blocks if b.language == "python"]:
        globs: dict = {
            "_REPO_ROOT": repo_root,
            "_README_DIR": readme_path.parent,
        }
        filename = f"<{readme_path.name}:{block.line_number}>"
        exec(compile(block.code, filename, "exec"), globs)  # noqa: S102
