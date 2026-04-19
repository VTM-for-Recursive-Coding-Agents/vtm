"""Repository materialization helpers for benchmark suites."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from vtm.benchmarks.models import RepoSpec
from vtm.benchmarks.synthetic import (
    SyntheticControlledCodingDriftCorpus,
    SyntheticPythonSmokeCorpus,
)


class RepoWorkspaceCommandError(RuntimeError):
    """Command failure that preserves stdout and stderr for debugging."""

    def __init__(
        self,
        *,
        command: list[str] | tuple[str, ...],
        cwd: Path | None,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = tuple(command)
        self.cwd = str(cwd) if cwd is not None else None
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(self._build_message())

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)

    def _build_message(self) -> str:
        stdout = self.stdout.rstrip() or "<empty>"
        stderr = self.stderr.rstrip() or "<empty>"
        cwd = self.cwd or "<none>"
        return (
            "command failed during repo materialization\n"
            f"command: {self.command_text}\n"
            f"cwd: {cwd}\n"
            f"return code: {self.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )


class RepoWorkspaceManager:
    """Clones, updates, and diffs benchmark repositories on disk."""

    def materialize_repo(self, repo_spec: RepoSpec, corpus_root: Path) -> Path:
        """Materialize a repo source into the local benchmark corpus directory."""
        corpus_root.mkdir(parents=True, exist_ok=True)
        repo_root = corpus_root / repo_spec.repo_name
        if repo_spec.source_kind == "synthetic_python_smoke":
            SyntheticPythonSmokeCorpus().materialize(repo_root)
            return repo_root
        if repo_spec.source_kind == "synthetic_python_controlled_coding_drift":
            SyntheticControlledCodingDriftCorpus().materialize(repo_root)
            return repo_root

        if not repo_root.exists():
            remote_url = self._resolve_clone_remote(repo_spec.remote_url or "")
            self.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--filter=blob:none",
                    "--branch",
                    repo_spec.branch,
                    remote_url,
                    str(repo_root),
                ]
            )
        else:
            self._validate_existing_repo_root(repo_root)
            self.run(["git", "fetch", "--quiet", "--all", "--tags", "--prune"], cwd=repo_root)
        return repo_root

    def git_checkout(self, repo_root: Path, ref: str) -> None:
        """Check out the requested git ref inside a materialized repo."""
        try:
            self.run(["git", "checkout", "--quiet", ref], cwd=repo_root)
        except RepoWorkspaceCommandError:
            # Prepared benchmark refs can live outside normal branch heads, so a
            # local clone may need an explicit fetch before the ref is visible.
            self.run(["git", "fetch", "--quiet", "origin", f"{ref}:{ref}"], cwd=repo_root)
            self.run(["git", "checkout", "--quiet", ref], cwd=repo_root)

    def git_diff_paths(self, repo_root: Path, base_ref: str, head_ref: str) -> tuple[str, ...]:
        """Return changed paths between the two refs."""
        completed = self.run(
            ["git", "diff", "--name-only", f"{base_ref}..{head_ref}"],
            cwd=repo_root,
            check=False,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return tuple(lines)

    def git_diff_text(self, repo_root: Path, base_ref: str, head_ref: str) -> str:
        """Return the diff text between the two refs."""
        return self.run(
            ["git", "diff", "--binary", "--no-ext-diff", f"{base_ref}..{head_ref}", "--"],
            cwd=repo_root,
            check=False,
        ).stdout

    def run(
        self,
        command: list[str] | tuple[str, ...],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                cwd=cwd,
                check=check,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise RepoWorkspaceCommandError(
                command=list(command),
                cwd=cwd,
                returncode=error.returncode,
                stdout=error.stdout or error.output or "",
                stderr=error.stderr or "",
            ) from error

    def _resolve_clone_remote(self, remote_url: str) -> str:
        """Resolve local filesystem remotes to absolute paths before cloning."""
        if "://" in remote_url or remote_url.startswith("git@"):
            return remote_url
        local_path = Path(remote_url)
        if local_path.exists():
            return str(local_path.resolve())
        return remote_url

    def _validate_existing_repo_root(self, repo_root: Path) -> None:
        """Ensure an existing destination is a usable git working tree."""
        if not repo_root.is_dir():
            raise RuntimeError(
                f"existing repo root is not a directory: {repo_root}"
            )
        completed = self.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip() == "true":
            return
        detail = completed.stderr.rstrip() or completed.stdout.rstrip() or "<empty>"
        raise RuntimeError(
            "existing repo root is not a complete git repository: "
            f"{repo_root}\n"
            "delete the directory and rerun materialization\n"
            f"git reported: {detail}"
        )
