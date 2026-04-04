from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Protocol

from vtm.fingerprints import RepoFingerprint


class GitFingerprintAdapter(Protocol):
    def collect(self, repo_root: str) -> RepoFingerprint: ...


class GitRepoFingerprintCollector:
    def collect(self, repo_root: str) -> RepoFingerprint:
        root = Path(repo_root).resolve()
        branch_output = self._git_text(root, "branch", "--show-current")
        head_commit = self._git_text(root, "rev-parse", "HEAD")
        tree_digest = self._git_text(root, "rev-parse", "HEAD^{tree}")
        normalized_status = self._normalize_status(
            self._git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
        )
        diff_bytes = self._git_bytes(root, "diff", "--no-ext-diff", "--binary", "HEAD", "--")
        untracked_manifest = self._untracked_manifest(root)
        dirty_digest = hashlib.sha256(
            normalized_status.encode("utf-8")
            + b"\0"
            + diff_bytes
            + b"\0"
            + untracked_manifest
        ).hexdigest()
        return RepoFingerprint(
            repo_root=str(root),
            branch=branch_output or None,
            head_commit=head_commit,
            tree_digest=tree_digest,
            dirty_digest=dirty_digest,
        )

    def _git_text(self, repo_root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _git_bytes(self, repo_root: Path, *args: str) -> bytes:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=False,
        )
        return result.stdout.replace(b"\r\n", b"\n")

    def _normalize_status(self, status_output: str) -> str:
        return "\n".join(line.rstrip() for line in status_output.replace("\r\n", "\n").splitlines())

    def _untracked_manifest(self, repo_root: Path) -> bytes:
        untracked_paths = sorted(
            self._git_paths(repo_root, "ls-files", "--others", "--exclude-standard")
        )
        manifest = bytearray()
        for relative_path in untracked_paths:
            content = (repo_root / relative_path).read_bytes()
            manifest.extend(relative_path.as_posix().encode("utf-8"))
            manifest.extend(b"\0")
            manifest.extend(hashlib.sha256(content).hexdigest().encode("ascii"))
            manifest.extend(b"\n")
        return bytes(manifest)

    def _git_paths(self, repo_root: Path, *args: str) -> tuple[Path, ...]:
        output = self._git_bytes(repo_root, *args, "-z")
        if not output:
            return ()
        return tuple(Path(path.decode("utf-8")) for path in output.split(b"\0") if path)
