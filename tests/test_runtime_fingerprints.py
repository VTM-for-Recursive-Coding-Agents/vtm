from __future__ import annotations

import subprocess
from pathlib import Path

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.services.fingerprints import DependencyFingerprintBuilder


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run(repo, "git", "init")
    _run(repo, "git", "config", "user.name", "VTM Tests")
    _run(repo, "git", "config", "user.email", "vtm@example.com")
    (repo / "example.txt").write_text("alpha\n", encoding="utf-8")
    _run(repo, "git", "add", "example.txt")
    _run(repo, "git", "commit", "-m", "initial")


def test_git_repo_fingerprint_changes_with_branch_commit_and_dirty_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    collector = GitRepoFingerprintCollector()

    initial = collector.collect(str(repo))
    initial_branch = _run(repo, "git", "branch", "--show-current")
    _run(repo, "git", "checkout", "-b", "feature/runtime")
    changed_branch = collector.collect(str(repo))

    assert initial.branch == initial_branch
    assert changed_branch.branch == "feature/runtime"
    assert changed_branch.head_commit == initial.head_commit
    assert changed_branch.tree_digest == initial.tree_digest

    (repo / "example.txt").write_text("beta\n", encoding="utf-8")
    dirty = collector.collect(str(repo))
    assert dirty.dirty_digest != changed_branch.dirty_digest

    _run(repo, "git", "add", "example.txt")
    _run(repo, "git", "commit", "-m", "update")
    committed = collector.collect(str(repo))
    assert committed.head_commit != initial.head_commit
    assert committed.tree_digest != initial.tree_digest


def test_git_repo_fingerprint_changes_for_untracked_file_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    collector = GitRepoFingerprintCollector()

    clean = collector.collect(str(repo))
    untracked_path = repo / "notes.txt"
    untracked_path.write_text("alpha\n", encoding="utf-8")
    created = collector.collect(str(repo))
    untracked_path.write_text("beta\n", encoding="utf-8")
    modified = collector.collect(str(repo))

    assert created.dirty_digest != clean.dirty_digest
    assert modified.dirty_digest != created.dirty_digest


def test_runtime_env_fingerprint_is_stable_for_fixed_inputs() -> None:
    collector = RuntimeEnvFingerprintCollector(
        python_version="3.12.9",
        platform_name="test-os-x86_64",
    )
    first = collector.collect(
        tool_probes={
            "custom-tool": ("python3", "-c", "print('custom-tool 1.0.0')"),
        }
    )
    second = collector.collect(
        tool_probes={
            "custom-tool": ("python3", "-c", "print('custom-tool 1.0.0')"),
        }
    )

    assert first == second
    assert first.python_version == "3.12.9"
    assert first.platform == "test-os-x86_64"
    assert {tool.name for tool in first.tool_versions} == {"custom-tool", "git", "python3"}


def test_dependency_fingerprint_builder_uses_collectors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    builder = DependencyFingerprintBuilder(
        repo_collector=GitRepoFingerprintCollector(),
        env_collector=RuntimeEnvFingerprintCollector(
            python_version="3.12.9",
            platform_name="test-os-x86_64",
        ),
    )

    fingerprint = builder.build(
        str(repo),
        dependency_ids=("artifact:1", "memory:2"),
        input_digests=("input-a",),
        tool_probes={
            "custom-tool": ("python3", "-c", "print('custom-tool 1.0.0')"),
        },
    )

    assert fingerprint.repo.repo_root == str(repo.resolve())
    assert fingerprint.dependency_ids == ("artifact:1", "memory:2")
    assert fingerprint.input_digests == ("input-a",)
    assert fingerprint.env.python_version == "3.12.9"
