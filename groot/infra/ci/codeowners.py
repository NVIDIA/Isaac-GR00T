# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import fnmatch
from pathlib import Path

from groot.infra.ci import codeowners_entry


class Codeowners:
    def __init__(
        self, entries: list[codeowners_entry.CodeownersEntry], path: Path | None = None
    ) -> None:
        self._entries = entries
        self._path = path

    @classmethod
    def init_from_file(cls, path: Path) -> "Codeowners":
        entries = []
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                pattern = parts[0]
                owners = parts[1:]
                entries.append(codeowners_entry.CodeownersEntry(pattern=pattern, owners=owners))
        return cls(entries, path=path)

    def root_owners(self) -> list[str]:
        for entry in self._entries:
            if entry.pattern == "*":
                return entry.owners
        return []

    def _matches(self, pattern: str, file_path: str) -> bool:
        """Check if pattern matches file_path using gitignore-style rules."""
        # Normalize file_path to not start with /
        file_path = file_path.lstrip("/")

        # Handle the catch-all
        if pattern == "*":
            return True

        # Anchored patterns start with /
        if pattern.startswith("/"):
            # Remove leading slash for matching
            pat = pattern.lstrip("/")
            # Directory pattern ends with /
            if pat.endswith("/"):
                dir_pat = pat.rstrip("/")
                return file_path.startswith(dir_pat + "/") or file_path == dir_pat
            else:
                # Use fnmatch-style but anchored
                return fnmatch.fnmatch(file_path, pat) or file_path.startswith(pat + "/")
        # Non-anchored: can match anywhere in path
        elif pattern.endswith("/"):
            dir_pat = pattern.rstrip("/")
            # Match if any path component equals dir_pat or path starts with it
            parts = file_path.split("/")
            return any(fnmatch.fnmatch(part, dir_pat) for part in parts)
        else:
            # Match basename or full path
            basename = file_path.split("/")[-1]
            return fnmatch.fnmatch(basename, pattern) or fnmatch.fnmatch(file_path, pattern)

    def owners_for(self, file_path: str) -> list[str]:
        """Return owners for a file. Last matching rule wins."""
        matched_owners: list[str] = []
        for entry in self._entries:
            if self._matches(entry.pattern, file_path):
                matched_owners = entry.owners
        return matched_owners

    def is_orphan(self, file_path: str) -> bool:
        """True if only the root * rule covers this file (no specific entry)."""
        last_match_pattern = None
        for entry in self._entries:
            if self._matches(entry.pattern, file_path):
                last_match_pattern = entry.pattern
        return last_match_pattern == "*"

    def propagate_ownership(self, source: str, dest: str) -> str | None:
        """Propagate CODEOWNERS ownership from *source* to *dest*.

        Inserts a CODEOWNERS entry when:
        - *source* has a specific (non-root) ownership rule, AND
        - *dest* is only covered by the root ``*`` rule (orphan).

        When the source is covered by a directory rule (ending with ``/``), a directory
        entry is inserted for *dest*'s parent directory.  Otherwise a file-specific entry
        is inserted for *dest* itself.

        Requires the instance to have been created via ``init_from_file``.

        Args:
            source: Repo-relative source path, e.g. ``"groot/foo/bar.py"``.
            dest:   Repo-relative destination path, e.g. ``"groot/internal/foo/bar.py"``.

        Returns:
            The inserted line, or ``None`` if nothing was done.

        Raises:
            ValueError: If the instance was not created via ``init_from_file``.
        """
        if self._path is None:
            raise ValueError("propagate_ownership requires an instance created via init_from_file")

        # Find the last non-root entry that covers source.
        matching_entry = None
        for entry in self._entries:
            if entry.pattern != "*" and self._matches(entry.pattern, source):
                matching_entry = entry

        if matching_entry is None or matching_entry.owners == self.root_owners():
            return None

        if not self.is_orphan(dest):
            return None

        # Prefer a directory entry when source is covered by a directory rule.
        if matching_entry.pattern.endswith("/"):
            dest_dir = str(Path(dest).parent) + "/"
            new_line = f"/{dest_dir} {' '.join(matching_entry.owners)}"
        else:
            new_line = f"/{dest} {' '.join(matching_entry.owners)}"

        text = self._path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)

        # Insert after the last /groot/ rule, or after the last ownership rule.
        insert_after = -1
        fallback = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fallback = i
            if stripped.startswith("/groot/"):
                insert_after = i

        if insert_after == -1:
            insert_after = fallback

        lines.insert(insert_after + 1, new_line + "\n")
        self._path.write_text("".join(lines), encoding="utf-8")
        return new_line
