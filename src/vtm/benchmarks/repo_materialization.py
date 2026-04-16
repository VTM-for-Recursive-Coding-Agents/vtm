"""Repository materialization helpers for benchmark suites."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vtm.benchmarks.models import RepoSpec
from vtm.benchmarks.synthetic import SyntheticPythonSmokeCorpus
from vtm.benchmarks.synthetic_terminal import SyntheticTerminalSmokeCorpus


class RepoWorkspaceManager:
    """Clones, updates, and diffs benchmark repositories on disk."""

    def materialize_repo(self, repo_spec: RepoSpec, corpus_root: Path) -> Path:
        """Materialize a repo source into the local benchmark corpus directory."""
        corpus_root.mkdir(parents=True, exist_ok=True)
        repo_root = corpus_root / repo_spec.repo_name
        if repo_spec.source_kind == "synthetic_python_smoke":
            SyntheticPythonSmokeCorpus().materialize(repo_root)
            return repo_root
        if repo_spec.source_kind == "synthetic_terminal_smoke":
            SyntheticTerminalSmokeCorpus().materialize(repo_root)
            return repo_root

        if not repo_root.exists():
            self.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--filter=blob:none",
                    "--branch",
                    repo_spec.branch,
                    repo_spec.remote_url or "",
                    str(repo_root),
                ]
            )
        else:
            self.run(["git", "fetch", "--quiet", "--all", "--tags", "--prune"], cwd=repo_root)
        return repo_root

    def git_checkout(self, repo_root: Path, ref: str) -> None:
        """Check out the requested git ref inside a materialized repo."""
        try:
            self.run(["git", "checkout", "--quiet", ref], cwd=repo_root)
        except subprocess.CalledProcessError:
            # Prepared SWE-bench refs live outside normal branch heads, so a
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
        return subprocess.run(
            list(command),
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True,
        )
