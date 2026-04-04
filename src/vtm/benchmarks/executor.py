from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vtm.benchmarks.repo_materialization import RepoWorkspaceManager


@dataclass(frozen=True)
class ExecutorOutcome:
    command: tuple[str, ...]
    command_exit_code: int | None
    command_stdout_path: str | None
    command_stderr_path: str | None
    runtime_ms: float
    workspace: str
    task_file: str
    test_command: tuple[str, ...]
    test_exit_code: int | None
    test_stdout_path: str | None
    test_stderr_path: str | None
    produced_patch_path: str | None
    produced_patch_digest: str | None
    produced_patch_text: str
    produced_changed_paths: tuple[str, ...]


class BenchmarkExecutor(Protocol):
    def execute(
        self,
        *,
        case_id: str,
        command: tuple[str, ...],
        workspace_root: Path,
        task_file: Path,
        test_command: tuple[str, ...],
    ) -> ExecutorOutcome: ...


class SubprocessBenchmarkExecutor:
    def __init__(self, *, repo_manager: RepoWorkspaceManager, output_root: Path) -> None:
        self._repo_manager = repo_manager
        self._output_root = output_root

    def execute(
        self,
        *,
        case_id: str,
        command: tuple[str, ...],
        workspace_root: Path,
        task_file: Path,
        test_command: tuple[str, ...],
    ) -> ExecutorOutcome:
        artifact_root = self._output_root / "executor-artifacts" / case_id
        artifact_root.mkdir(parents=True, exist_ok=True)

        started = self._now()
        command_result = self._repo_manager.run(command, cwd=workspace_root, check=False)
        runtime_ms = self._elapsed_ms(started)
        command_stdout_path = artifact_root / "command.stdout"
        command_stderr_path = artifact_root / "command.stderr"
        command_stdout_path.write_text(command_result.stdout, encoding="utf-8")
        command_stderr_path.write_text(command_result.stderr, encoding="utf-8")

        test_exit_code: int | None = None
        test_stdout_path: Path | None = None
        test_stderr_path: Path | None = None
        if test_command:
            test_result = self._repo_manager.run(test_command, cwd=workspace_root, check=False)
            test_exit_code = test_result.returncode
            test_stdout_path = artifact_root / "test.stdout"
            test_stderr_path = artifact_root / "test.stderr"
            test_stdout_path.write_text(test_result.stdout, encoding="utf-8")
            test_stderr_path.write_text(test_result.stderr, encoding="utf-8")

        produced_patch = self._repo_manager.run(
            ["git", "diff", "--binary", "--no-ext-diff", "--"],
            cwd=workspace_root,
            check=False,
        ).stdout
        changed_paths = tuple(
            line.strip()
            for line in self._repo_manager.run(
                ["git", "diff", "--name-only", "--"],
                cwd=workspace_root,
                check=False,
            ).stdout.splitlines()
            if line.strip()
        )
        produced_patch_path = artifact_root / "produced.patch"
        produced_patch_path.write_text(produced_patch, encoding="utf-8")
        produced_patch_digest = hashlib.sha256(produced_patch.encode("utf-8")).hexdigest()

        return ExecutorOutcome(
            command=command,
            command_exit_code=command_result.returncode,
            command_stdout_path=str(command_stdout_path),
            command_stderr_path=str(command_stderr_path),
            runtime_ms=runtime_ms,
            workspace=str(workspace_root),
            task_file=str(task_file),
            test_command=test_command,
            test_exit_code=test_exit_code,
            test_stdout_path=str(test_stdout_path) if test_stdout_path is not None else None,
            test_stderr_path=str(test_stderr_path) if test_stderr_path is not None else None,
            produced_patch_path=str(produced_patch_path),
            produced_patch_digest=produced_patch_digest,
            produced_patch_text=produced_patch,
            produced_changed_paths=changed_paths,
        )

    def _now(self) -> float:
        import time

        return time.perf_counter()

    def _elapsed_ms(self, started: float) -> float:
        import time

        return (time.perf_counter() - started) * 1000
