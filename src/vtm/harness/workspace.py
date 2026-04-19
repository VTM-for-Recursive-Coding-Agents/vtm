"""Workspace preparation and local driver implementations for harness runs."""

from __future__ import annotations

import json
import os
import pty
import re
import select
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from vtm.base import utc_now
from vtm.harness.models import HarnessWorkspaceBackend


@dataclass(frozen=True)
class CommandResult:
    """Normalized result of a workspace command or file operation."""

    operation: str
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    output: str
    duration_ms: float
    timed_out: bool
    truncated: bool
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class PreparedWorkspace:
    """Prepared benchmark workspace plus its driver and artifact paths."""

    workspace_root: Path
    artifact_root: Path
    backend_name: HarnessWorkspaceBackend
    attempt_index: int
    command_events_path: Path
    driver: WorkspaceDriver
    metadata: dict[str, str] = field(default_factory=dict)


class WorkspaceDriver(Protocol):
    """Abstract interface for operating on an isolated benchmark workspace."""

    def run_terminal(
        self,
        command: str,
        *,
        timeout_seconds: int | None = None,
        max_output_chars: int | None = None,
    ) -> CommandResult: ...

    def search(self, pattern: str, *, path: str = ".") -> CommandResult: ...

    def capture_patch(self) -> str: ...

    def capture_changed_paths(self) -> tuple[str, ...]: ...

    def git_status(self) -> str: ...

    def run_verification(
        self,
        command: tuple[str, ...],
        *,
        label: str = "verification",
    ) -> CommandResult: ...

    def close(self) -> None: ...


class WorkspaceBackend(Protocol):
    """Prepares runnable workspaces for benchmark cases."""

    def prepare_workspace(
        self,
        *,
        case_id: str,
        attempt_index: int,
        repo_root: Path,
        base_ref: str,
        output_root: Path,
        mode: str,
        command_timeout_seconds: int,
        max_output_chars: int,
    ) -> PreparedWorkspace: ...


class LocalWorkspaceDriver:
    """Local git-backed workspace driver used by the reference harness."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_root: Path,
        shell: str | None = None,
        default_command_timeout_seconds: int = 120,
        default_max_output_chars: int = 20000,
    ) -> None:
        """Open a persistent interactive shell plus subprocess helpers."""
        self.workspace_root = workspace_root
        self.artifact_root = artifact_root
        self.command_events_path = artifact_root / "command-events.jsonl"
        self.command_events_path.parent.mkdir(parents=True, exist_ok=True)
        self.command_events_path.touch()
        self._shell = shell if shell is not None else "/bin/sh"
        self._default_command_timeout_seconds = default_command_timeout_seconds
        self._default_max_output_chars = default_max_output_chars
        self._master_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._closed = False
        self._start_session()

    def run_terminal(
        self,
        command: str,
        *,
        timeout_seconds: int | None = None,
        max_output_chars: int | None = None,
    ) -> CommandResult:
        """Run a command in the persistent interactive shell session."""
        stripped = command.strip()
        if not stripped:
            started_at = utc_now().isoformat()
            completed_at = utc_now().isoformat()
            result = CommandResult(
                operation="terminal",
                command="",
                exit_code=None,
                stdout="",
                stderr="",
                output="",
                duration_ms=0.0,
                timed_out=False,
                truncated=False,
                started_at=started_at,
                completed_at=completed_at,
            )
            self._append_event(result)
            return result

        if self._master_fd is None:
            self._start_session()
        assert self._master_fd is not None

        start_marker = f"__VTM_START_{uuid4().hex}__"
        end_marker = f"__VTM_DONE_{uuid4().hex}__"
        payload = (
            f"printf '%s\\n' '{start_marker}'\n"
            f"{stripped}\n"
            f"printf '%s:%s\\n' '{end_marker}' $?\n"
        )
        started = utc_now()
        started_monotonic = time.perf_counter()
        os.write(self._master_fd, payload.encode("utf-8"))
        buffer = bytearray()
        timed_out = False
        timeout = timeout_seconds or self._default_command_timeout_seconds
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._master_fd], [], [], 0.2)
            if not ready:
                continue
            chunk = os.read(self._master_fd, 4096)
            if not chunk:
                break
            buffer.extend(chunk)
            if self._has_terminal_exit_marker(
                buffer.decode("utf-8", errors="replace"),
                start_marker=start_marker,
                end_marker=end_marker,
            ):
                break
        else:
            timed_out = True

        raw_text = buffer.decode("utf-8", errors="replace")
        if timed_out:
            exit_code = None
            output = self._normalize_terminal_output(raw_text)
            self._restart_session()
        else:
            output, exit_code = self._extract_terminal_output(
                raw_text,
                start_marker=start_marker,
                end_marker=end_marker,
            )
        capped_output, truncated = self._truncate_output(
            output,
            max_output_chars=(
                self._default_max_output_chars
                if max_output_chars is None
                else max_output_chars
            ),
        )
        completed = utc_now()
        result = CommandResult(
            operation="terminal",
            command=stripped,
            exit_code=exit_code,
            stdout=capped_output,
            stderr="",
            output=capped_output,
            duration_ms=(time.perf_counter() - started_monotonic) * 1000,
            timed_out=timed_out,
            truncated=truncated,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
        )
        self._append_event(result)
        return result

    def search(self, pattern: str, *, path: str = ".") -> CommandResult:
        """Search the workspace with `rg`, falling back to `grep`."""
        target_path = self._resolve_workspace_path(path)
        try:
            return self._run_subprocess(
                ("rg", "-n", pattern, str(target_path)),
                operation="search",
                max_output_chars=self._default_max_output_chars,
            )
        except FileNotFoundError:
            return self._run_subprocess(
                ("grep", "-R", "-n", pattern, str(target_path)),
                operation="search",
                max_output_chars=self._default_max_output_chars,
            )

    def capture_patch(self) -> str:
        """Return the current diff from `HEAD` for the workspace."""
        return self._run_subprocess(
            ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"),
            operation="capture_patch",
            max_output_chars=None,
        ).stdout

    def capture_changed_paths(self) -> tuple[str, ...]:
        """Return the set of changed paths relative to `HEAD`."""
        output = self._run_subprocess(
            ("git", "diff", "--name-only", "HEAD", "--"),
            operation="capture_changed_paths",
        ).stdout
        return tuple(line.strip() for line in output.splitlines() if line.strip())

    def git_status(self) -> str:
        """Return a short git status snapshot for the workspace."""
        return self._run_subprocess(
            ("git", "status", "--short", "--branch"),
            operation="git_status",
            max_output_chars=self._default_max_output_chars,
        ).stdout

    def run_verification(
        self,
        command: tuple[str, ...],
        *,
        label: str = "verification",
    ) -> CommandResult:
        """Run a one-shot verification command in the workspace."""
        return self._run_subprocess(
            command,
            operation=f"{label}_command",
            max_output_chars=self._default_max_output_chars,
        )

    def close(self) -> None:
        """Terminate the interactive shell session and release resources."""
        if self._closed:
            return
        self._closed = True
        if self._master_fd is None:
            return
        try:
            os.write(self._master_fd, b"exit\n")
        except OSError:
            pass
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
        os.close(self._master_fd)
        self._master_fd = None
        self._process = None

    def _start_session(self) -> None:
        self._closed = False
        master_fd, slave_fd = pty.openpty()
        env = {
            **os.environ,
            "PS1": "",
            "TERM": "dumb",
        }
        process = subprocess.Popen(
            [self._shell, "-i"],
            cwd=self.workspace_root,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)
        self._master_fd = master_fd
        self._process = process
        self._drain_initial_output()
        try:
            os.write(self._master_fd, b"stty -echo\n")
            self._drain_initial_output()
        except OSError:
            pass

    def _restart_session(self) -> None:
        self.close()
        self._start_session()

    def _drain_initial_output(self) -> None:
        if self._master_fd is None:
            return
        ready, _, _ = select.select([self._master_fd], [], [], 0.1)
        if ready:
            try:
                os.read(self._master_fd, 4096)
            except OSError:
                return

    def _run_subprocess(
        self,
        command: tuple[str, ...],
        *,
        operation: str,
        max_output_chars: int | None = None,
        timeout_seconds: int | None = None,
    ) -> CommandResult:
        started = utc_now()
        started_monotonic = time.perf_counter()
        try:
            completed = subprocess.run(
                list(command),
                cwd=self.workspace_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds or self._default_command_timeout_seconds,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = self._normalize_subprocess_output(exc.stdout)
            stderr = self._normalize_subprocess_output(exc.stderr)
            exit_code = None
            timed_out = True
        output = "\n".join(part for part in (stdout, stderr) if part).strip()
        capped_output, truncated = self._truncate_output(
            output,
            max_output_chars=max_output_chars,
        )
        if truncated:
            if stdout:
                stdout = capped_output
                stderr = ""
            else:
                stderr = capped_output
        completed_at = utc_now()
        result = CommandResult(
            operation=operation,
            command=shlex.join(command),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            output=capped_output,
            duration_ms=(time.perf_counter() - started_monotonic) * 1000,
            timed_out=timed_out,
            truncated=truncated,
            started_at=started.isoformat(),
            completed_at=completed_at.isoformat(),
        )
        self._append_event(result)
        return result

    def _extract_terminal_output(
        self,
        raw_text: str,
        *,
        start_marker: str,
        end_marker: str,
    ) -> tuple[str, int | None]:
        normalized = self._normalize_terminal_output(raw_text)
        start_token = f"{start_marker}\n"
        start_index = normalized.find(start_token)
        if start_index == -1:
            return normalized.strip(), None
        content_start = start_index + len(start_token)
        end_match = re.search(
            rf"{re.escape(end_marker)}:(?P<code>-?\d+)(?:\n|$)",
            normalized[content_start:],
        )
        if end_match is None:
            return normalized[content_start:], None
        end_index = content_start + end_match.start()
        content = normalized[content_start:end_index]
        raw_exit_code = end_match.group("code")
        try:
            exit_code = int(raw_exit_code)
        except ValueError:
            exit_code = None
        return content, exit_code

    def _has_terminal_exit_marker(
        self,
        raw_text: str,
        *,
        start_marker: str,
        end_marker: str,
    ) -> bool:
        normalized = self._normalize_terminal_output(raw_text)
        start_token = f"{start_marker}\n"
        start_index = normalized.find(start_token)
        if start_index == -1:
            return False
        content_start = start_index + len(start_token)
        return (
            re.search(
                rf"{re.escape(end_marker)}:(?P<code>-?\d+)(?:\n|$)",
                normalized[content_start:],
            )
            is not None
        )

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        if not raw_path:
            raise ValueError("path must be non-empty")
        candidate = (self.workspace_root / raw_path).resolve()
        workspace = self.workspace_root.resolve()
        if candidate != workspace and workspace not in candidate.parents:
            raise ValueError(f"path escapes workspace: {raw_path}")
        return candidate

    def _truncate_output(
        self,
        output: str,
        *,
        max_output_chars: int | None,
    ) -> tuple[str, bool]:
        if max_output_chars is None:
            return output, False
        limit = max_output_chars
        if limit <= 0:
            return "", bool(output)
        if len(output) <= limit:
            return output, False
        return output[:limit], True

    def _append_event(self, result: CommandResult) -> None:
        with self.command_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result), sort_keys=True))
            handle.write("\n")

    def _normalize_subprocess_output(self, value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def _normalize_terminal_output(self, output: str) -> str:
        return output.replace("\r\n", "\n").replace("\r", "\n")


class LocalWorkspaceBackend:
    """Reference backend that clones the repo locally per benchmark case."""

    backend_name: HarnessWorkspaceBackend = "local_workspace"

    def prepare_workspace(
        self,
        *,
        case_id: str,
        attempt_index: int,
        repo_root: Path,
        base_ref: str,
        output_root: Path,
        mode: str,
        command_timeout_seconds: int,
        max_output_chars: int,
    ) -> PreparedWorkspace:
        """Clone the repo, check out the base ref, and return a local driver."""
        attempt_dir = f"attempt-{attempt_index:02d}"
        workspace_root = output_root / "workspaces" / mode / case_id / attempt_dir
        artifact_root = output_root / "executor-artifacts" / case_id / attempt_dir
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
        workspace_root.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._run(("git", "clone", "--quiet", str(repo_root), str(workspace_root)))
        self._checkout_prepared_ref(workspace_root, base_ref)
        driver = LocalWorkspaceDriver(
            workspace_root=workspace_root,
            artifact_root=artifact_root,
            default_command_timeout_seconds=command_timeout_seconds,
            default_max_output_chars=max_output_chars,
        )
        return PreparedWorkspace(
            workspace_root=workspace_root,
            artifact_root=artifact_root,
            backend_name=self.backend_name,
            attempt_index=attempt_index,
            command_events_path=driver.command_events_path,
            driver=driver,
            metadata={},
        )

    def _checkout_prepared_ref(self, workspace_root: Path, base_ref: str) -> None:
        try:
            self._run(("git", "checkout", "--quiet", base_ref), cwd=workspace_root)
        except subprocess.CalledProcessError:
            # Prepared benchmark refs can live outside normal branch heads, so a
            # fresh clone may need an explicit fetch before the ref becomes visible.
            self._run(
                ("git", "fetch", "--quiet", "origin", f"{base_ref}:{base_ref}"),
                cwd=workspace_root,
            )
            self._run(("git", "checkout", "--quiet", base_ref), cwd=workspace_root)

    def _run(self, command: tuple[str, ...], *, cwd: Path | None = None) -> None:
        subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )


__all__ = [
    "CommandResult",
    "LocalWorkspaceBackend",
    "LocalWorkspaceDriver",
    "PreparedWorkspace",
    "WorkspaceBackend",
    "WorkspaceDriver",
]
