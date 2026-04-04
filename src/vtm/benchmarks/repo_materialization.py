from __future__ import annotations

import subprocess
from pathlib import Path

from vtm.benchmarks.models import RepoSpec
from vtm.benchmarks.synthetic import SyntheticPythonSmokeCorpus


class RepoWorkspaceManager:
    def materialize_repo(self, repo_spec: RepoSpec, corpus_root: Path) -> Path:
        corpus_root.mkdir(parents=True, exist_ok=True)
        repo_root = corpus_root / repo_spec.repo_name
        if repo_spec.source_kind == "synthetic_python_smoke":
            SyntheticPythonSmokeCorpus().materialize(repo_root)
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
        self.run(["git", "checkout", "--quiet", ref], cwd=repo_root)

    def git_diff_paths(self, repo_root: Path, base_ref: str, head_ref: str) -> tuple[str, ...]:
        completed = self.run(
            ["git", "diff", "--name-only", f"{base_ref}..{head_ref}"],
            cwd=repo_root,
            check=False,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return tuple(lines)

    def git_diff_text(self, repo_root: Path, base_ref: str, head_ref: str) -> str:
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
