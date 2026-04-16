"""Docker-backed workspace preparation and driver implementations."""

from __future__ import annotations

import os
import pty
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from vtm.base import utc_now
from vtm.harness.workspace import CommandResult, LocalWorkspaceDriver, PreparedWorkspace

DEFAULT_DOCKER_PIDS_LIMIT = 256
DEFAULT_DOCKER_MEMORY_LIMIT = "2g"
DEFAULT_DOCKER_CPU_LIMIT = 2.0
DEFAULT_DOCKER_TMPFS_MOUNTS = (
    "/tmp:rw,noexec,nosuid,nodev,size=64m",
    "/run:rw,noexec,nosuid,nodev,size=16m",
)


class DockerWorkspaceDriver(LocalWorkspaceDriver):
    """Docker-backed workspace driver that mirrors the local workspace contract."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_root: Path,
        docker_binary: str,
        docker_image: str,
        docker_network: str,
        container_name: str,
        container_id: str,
        shell: str | None = None,
        default_command_timeout_seconds: int = 120,
        default_max_output_chars: int = 20000,
    ) -> None:
        self._docker_binary = docker_binary
        self._docker_image = docker_image
        self._docker_network = docker_network
        self._container_name = container_name
        self._container_id = container_id
        self._container_removed = False
        super().__init__(
            workspace_root=workspace_root,
            artifact_root=artifact_root,
            shell=shell,
            default_command_timeout_seconds=default_command_timeout_seconds,
            default_max_output_chars=default_max_output_chars,
        )

    @property
    def docker_image(self) -> str:
        return self._docker_image

    @property
    def docker_network(self) -> str:
        return self._docker_network

    @property
    def container_id(self) -> str:
        return self._container_id

    @property
    def container_name(self) -> str:
        return self._container_name

    def close(self) -> None:
        """Terminate the active exec session and remove the backing container."""
        self._close_shell_session()
        self._remove_container()

    def _restart_session(self) -> None:
        self._close_shell_session()
        self._start_session()

    def _start_session(self) -> None:
        self._closed = False
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            self._docker_exec_command((self._shell or "/bin/sh", "-i"), interactive=True),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        self._master_fd = master_fd
        self._process = process
        self._drain_initial_output()
        try:
            os.write(self._master_fd, b"export PS1=\nstty -echo\n")
            self._drain_initial_output()
        except OSError:
            pass

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
                self._docker_exec_command(command),
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
            command=" ".join(command),
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

    def _docker_exec_command(
        self,
        command: tuple[str, ...],
        *,
        interactive: bool = False,
    ) -> list[str]:
        args = [self._docker_binary, "exec"]
        if interactive:
            args.append("-i")
        args.extend(["-w", str(self.workspace_root), self._container_name])
        args.extend(command)
        return args

    def _close_shell_session(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._master_fd is None:
            self._process = None
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

    def _remove_container(self) -> None:
        if self._container_removed:
            return
        self._container_removed = True
        subprocess.run(
            [self._docker_binary, "rm", "-f", self._container_name],
            check=False,
            capture_output=True,
            text=True,
        )


class DockerWorkspaceBackend:
    """Reference backend that runs each prepared workspace inside Docker."""

    backend_name: Literal["docker_workspace"] = "docker_workspace"

    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        docker_image: str,
        docker_network: str = "none",
        docker_read_only_rootfs: bool = True,
        docker_pids_limit: int = DEFAULT_DOCKER_PIDS_LIMIT,
        docker_memory_limit: str = DEFAULT_DOCKER_MEMORY_LIMIT,
        docker_cpu_limit: float = DEFAULT_DOCKER_CPU_LIMIT,
    ) -> None:
        """Configure the Docker-backed workspace sandbox defaults."""
        self._docker_binary = docker_binary
        self._docker_image = docker_image
        self._docker_network = docker_network
        self._docker_read_only_rootfs = docker_read_only_rootfs
        self._docker_pids_limit = docker_pids_limit
        self._docker_memory_limit = docker_memory_limit
        self._docker_cpu_limit = docker_cpu_limit

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
        """Clone the repo, check out the base ref, and start a Docker container."""
        attempt_dir = f"attempt-{attempt_index:02d}"
        workspace_root = (output_root / "workspaces" / mode / case_id / attempt_dir).resolve()
        artifact_root = (output_root / "executor-artifacts" / case_id / attempt_dir).resolve()
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
        workspace_root.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._run(("git", "clone", "--quiet", str(repo_root), str(workspace_root)))
        self._checkout_prepared_ref(workspace_root, base_ref)
        return self.prepare_existing_workspace(
            case_id=case_id,
            attempt_index=attempt_index,
            workspace_root=workspace_root,
            artifact_root=artifact_root,
            command_timeout_seconds=command_timeout_seconds,
            max_output_chars=max_output_chars,
        )

    def prepare_existing_workspace(
        self,
        *,
        case_id: str,
        attempt_index: int,
        workspace_root: Path,
        artifact_root: Path,
        command_timeout_seconds: int,
        max_output_chars: int,
    ) -> PreparedWorkspace:
        """Start a Docker container around an already materialized workspace."""
        workspace_root = workspace_root.resolve()
        artifact_root = artifact_root.resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)

        container_name = self._container_name(case_id=case_id, attempt_index=attempt_index)
        startup_stdout_path = artifact_root / "docker-run.stdout"
        startup_stderr_path = artifact_root / "docker-run.stderr"
        run_command = [
            self._docker_binary,
            "run",
            "-d",
            "--name",
            container_name,
            "--network",
            self._docker_network,
        ]
        if self._docker_read_only_rootfs:
            run_command.append("--read-only")
        run_command.extend(
            [
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(self._docker_pids_limit),
                "--memory",
                self._docker_memory_limit,
                "--cpus",
                f"{self._docker_cpu_limit:g}",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
            ]
        )
        for mount in DEFAULT_DOCKER_TMPFS_MOUNTS:
            run_command.extend(["--tmpfs", mount])
        run_command.extend(
            [
                "-v",
                f"{workspace_root}:{workspace_root}:rw",
                "-v",
                f"{artifact_root}:{artifact_root}:rw",
                "-w",
                str(workspace_root),
                self._docker_image,
                "/bin/sh",
                "-lc",
                "trap 'exit 0' TERM INT; while true; do sleep 3600; done",
            ]
        )
        completed = subprocess.run(
            run_command,
            check=False,
            capture_output=True,
            text=True,
        )
        startup_stdout_path.write_text(completed.stdout, encoding="utf-8")
        startup_stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                "docker workspace startup failed "
                f"(exit_code={completed.returncode}); see "
                f"{startup_stdout_path} and {startup_stderr_path}"
            )
        container_id = completed.stdout.strip()
        if not container_id:
            raise RuntimeError(
                "docker workspace startup returned an empty container id; see "
                f"{startup_stdout_path} and {startup_stderr_path}"
            )
        driver = DockerWorkspaceDriver(
            workspace_root=workspace_root,
            artifact_root=artifact_root,
            docker_binary=self._docker_binary,
            docker_image=self._docker_image,
            docker_network=self._docker_network,
            container_name=container_name,
            container_id=container_id,
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
            metadata={
                "docker_binary": self._docker_binary,
                "docker_container_id": container_id,
                "docker_container_name": container_name,
                "docker_image": self._docker_image,
                "docker_network": self._docker_network,
                "docker_read_only_rootfs": str(self._docker_read_only_rootfs).lower(),
                "docker_pids_limit": str(self._docker_pids_limit),
                "docker_memory_limit": self._docker_memory_limit,
                "docker_cpu_limit": f"{self._docker_cpu_limit:g}",
                "docker_run_stdout_path": str(startup_stdout_path),
                "docker_run_stderr_path": str(startup_stderr_path),
            },
        )

    def _checkout_prepared_ref(self, workspace_root: Path, base_ref: str) -> None:
        try:
            self._run(("git", "checkout", "--quiet", base_ref), cwd=workspace_root)
        except subprocess.CalledProcessError:
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

    def _container_name(self, *, case_id: str, attempt_index: int) -> str:
        safe_case = re.sub(r"[^a-zA-Z0-9_.-]+", "-", case_id).strip("-") or "case"
        return f"vtm-{safe_case}-attempt-{attempt_index:02d}-{uuid4().hex[:8]}"


__all__ = [
    "DockerWorkspaceBackend",
    "DockerWorkspaceDriver",
]
