"""Procedure-validation services backed by external commands."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from vtm.artifacts import ArtifactRecord
from vtm.base import utc_now
from vtm.enums import MemoryKind, ValidityStatus
from vtm.harness.workspace_docker import DockerWorkspaceBackend
from vtm.memory_items import (
    CommandValidatorConfig,
    MemoryItem,
    ProcedurePayload,
    ValidatorSpec,
)
from vtm.stores.base import ArtifactStore
from vtm.verification import ProcedureValidationResult

_resource: Any
try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows-only fallback
    _resource = None


@dataclass(frozen=True)
class _CommandExecutionResult:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    terminated_process_group: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcedureValidator(Protocol):
    """Contract for validating a procedure memory item."""

    def validate(
        self,
        procedure: MemoryItem,
        *,
        repo_root: str | None = None,
    ) -> ProcedureValidationResult: ...


class CommandProcedureValidator:
    """Runs a configured command and captures stdout/stderr as artifacts."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        """Create a validator that writes durable artifacts for each run."""
        self._artifact_store = artifact_store

    def validate(
        self,
        procedure: MemoryItem,
        *,
        repo_root: str | None = None,
    ) -> ProcedureValidationResult:
        """Execute the configured validator command for the given procedure."""
        payload = self._require_procedure_payload(procedure)
        validator = payload.validator
        if validator is None:
            raise ValueError("procedure validation requires a validator spec")
        config = self._require_command_config(validator)

        command = list(config.command)
        inherit_parent_env = config.inherit_parent_env
        restrict_cwd_to_repo = config.restrict_cwd_to_repo
        cwd = self._resolve_cwd(
            config,
            repo_root,
            restrict_cwd_to_repo=restrict_cwd_to_repo,
        )
        env = self._resolve_env(config, inherit_parent_env=inherit_parent_env)
        expected_exit_code = config.expected_exit_code
        timeout_seconds = config.timeout_seconds
        max_output_bytes = config.max_output_bytes
        resource_limits = config.resource_limits()
        preexec_fn = self._build_preexec_fn(resource_limits)
        result_metadata: dict[str, Any] = {
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": max_output_bytes,
            "timed_out": False,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "inherit_parent_env": inherit_parent_env,
            "restrict_cwd_to_repo": restrict_cwd_to_repo,
            "resource_limits": resource_limits,
            "terminated_process_group": False,
            "restricted_mode": (
                not inherit_parent_env or restrict_cwd_to_repo or bool(resource_limits)
            ),
        }
        capture_group_id = f"capgrp_{uuid4().hex}"

        try:
            execution = self._execute_command(
                procedure=procedure,
                command=command,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
                preexec_fn=preexec_fn,
                repo_root=repo_root,
            )
            exit_code = execution.exit_code
            stdout = execution.stdout
            stderr = execution.stderr
            result_metadata.update(execution.metadata)
            result_metadata["timed_out"] = execution.timed_out
            result_metadata["terminated_process_group"] = execution.terminated_process_group
            if execution.timed_out:
                success = False
                status = ValidityStatus.UNKNOWN
                reason = f"procedure validator timed out after {timeout_seconds} seconds"
            else:
                success = exit_code == expected_exit_code
                status = ValidityStatus.VERIFIED if success else ValidityStatus.REFUTED
                reason = (
                    "procedure validator exit code matched expected"
                    if success
                    else (
                        f"procedure validator exit code {exit_code} != "
                        f"expected {expected_exit_code}"
                    )
                )
        except (OSError, RuntimeError) as exc:
            stdout = b""
            stderr = str(exc).encode("utf-8")
            exit_code = None
            success = False
            status = ValidityStatus.UNKNOWN
            reason = f"procedure validator execution failed: {exc}"

        stdout, stdout_truncated = self._truncate_output(stdout, max_output_bytes)
        stderr, stderr_truncated = self._truncate_output(stderr, max_output_bytes)
        result_metadata["stdout_truncated"] = stdout_truncated
        result_metadata["stderr_truncated"] = stderr_truncated

        captured_artifact_ids: list[str] = []
        try:
            stdout_record = self._capture_stream_artifact(
                procedure=procedure,
                stream="stdout",
                payload=stdout,
                command=command,
                truncated=stdout_truncated,
                timed_out=bool(result_metadata["timed_out"]),
                max_output_bytes=max_output_bytes,
                capture_group_id=capture_group_id,
            )
            captured_artifact_ids.append(stdout_record.artifact_id)
            stderr_record = self._capture_stream_artifact(
                procedure=procedure,
                stream="stderr",
                payload=stderr,
                command=command,
                truncated=stderr_truncated,
                timed_out=bool(result_metadata["timed_out"]),
                max_output_bytes=max_output_bytes,
                capture_group_id=capture_group_id,
            )
            captured_artifact_ids.append(stderr_record.artifact_id)
        except Exception:
            self._best_effort_abandon_artifacts(
                captured_artifact_ids,
                reason="procedure_validator_capture_failed",
            )
            raise

        return ProcedureValidationResult(
            memory_id=procedure.memory_id,
            validator_spec=validator,
            success=success,
            exit_code=exit_code,
            stdout_artifact_id=stdout_record.artifact_id,
            stderr_artifact_id=stderr_record.artifact_id,
            checked_at=utc_now(),
            status=status,
            reason=reason,
            metadata=result_metadata,
        )

    def _execute_command(
        self,
        *,
        procedure: MemoryItem,
        command: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_seconds: float | None,
        preexec_fn: Any | None,
        repo_root: str | None,
    ) -> _CommandExecutionResult:
        del procedure, repo_root
        exit_code, stdout, stderr, timed_out, terminated_process_group = self._run_command(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout_seconds,
            preexec_fn=preexec_fn,
        )
        return _CommandExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            terminated_process_group=terminated_process_group,
        )

    def _capture_stream_artifact(
        self,
        *,
        procedure: MemoryItem,
        stream: str,
        payload: bytes,
        command: list[str],
        truncated: bool,
        timed_out: bool,
        max_output_bytes: int | None,
        capture_group_id: str,
    ) -> ArtifactRecord:
        prepared = None
        try:
            prepared = self._artifact_store.prepare_bytes(
                payload,
                content_type="application/octet-stream",
                tool_name="procedure-validator",
                capture_group_id=capture_group_id,
                actor="procedure-validator",
                metadata={
                    "memory_id": procedure.memory_id,
                    "stream": stream,
                    "command": command,
                    "truncated": truncated,
                    "timed_out": timed_out,
                    "max_output_bytes": max_output_bytes,
                },
            )
            return self._artifact_store.commit_artifact(prepared.artifact_id)
        except Exception:
            if prepared is not None:
                self._best_effort_abandon_artifacts(
                    [prepared.artifact_id],
                    reason="procedure_validator_capture_failed",
                )
            raise

    def _require_procedure_payload(self, procedure: MemoryItem) -> ProcedurePayload:
        if procedure.kind is not MemoryKind.PROCEDURE or not isinstance(
            procedure.payload,
            ProcedurePayload,
        ):
            raise ValueError("procedure validator requires a procedure memory item")
        return procedure.payload

    def _require_command_config(self, validator: ValidatorSpec) -> CommandValidatorConfig:
        return validator.command_config()

    def _resolve_cwd(
        self,
        config: CommandValidatorConfig,
        repo_root: str | None,
        *,
        restrict_cwd_to_repo: bool,
    ) -> str | None:
        cwd = config.cwd
        if cwd is None:
            resolved = Path(repo_root).resolve() if repo_root is not None else None
        elif repo_root is not None and not Path(cwd).is_absolute():
            resolved = (Path(repo_root) / cwd).resolve()
        else:
            resolved = Path(cwd).resolve() if Path(cwd).is_absolute() else Path(cwd)

        if restrict_cwd_to_repo:
            if repo_root is None:
                raise ValueError("command validator restrict_cwd_to_repo requires repo_root")
            repo_root_path = Path(repo_root).resolve()
            if resolved is None:
                return str(repo_root_path)
            if not resolved.is_relative_to(repo_root_path):
                raise ValueError(
                    "command validator cwd must stay within repo_root when "
                    "restrict_cwd_to_repo=true"
                )

        if resolved is None:
            return None
        return str(resolved)

    def _resolve_env(
        self,
        config: CommandValidatorConfig,
        *,
        inherit_parent_env: bool,
    ) -> dict[str, str] | None:
        raw_env = config.env
        allowlist = config.env_allowlist
        denylist = config.env_denylist
        if raw_env is None and allowlist is None and denylist is None:
            if inherit_parent_env:
                return None
            return {}
        resolved_env = dict(os.environ) if inherit_parent_env else {}
        if allowlist is not None:
            allowed = set(allowlist)
            resolved_env = {key: value for key, value in resolved_env.items() if key in allowed}
        if denylist is not None:
            for key in denylist:
                resolved_env.pop(key, None)
        if raw_env is not None:
            resolved_env.update(dict(raw_env))
        return resolved_env

    def _run_command(
        self,
        command: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout: float | None,
        preexec_fn: Any | None,
    ) -> tuple[int | None, bytes, bytes, bool, bool]:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            close_fds=True,
            start_new_session=os.name != "nt",
            preexec_fn=preexec_fn,
        )
        try:
            if timeout is None:
                stdout, stderr = process.communicate()
            else:
                stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr, False, False
        except subprocess.TimeoutExpired:
            terminated_process_group = self._terminate_process(process)
            stdout, stderr = process.communicate()
            return None, stdout, stderr, True, terminated_process_group

    def _terminate_process(self, process: subprocess.Popen[bytes]) -> bool:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
                return True
            except (PermissionError, ProcessLookupError):
                pass
        with suppress(ProcessLookupError):
            process.kill()
        return False

    def _build_preexec_fn(self, resource_limits: Mapping[str, int]) -> Any | None:
        if not resource_limits:
            return None
        if os.name == "nt" or _resource is None:
            raise ValueError("command validator resource limits require a POSIX runtime")
        if "rlimit_memory_bytes" in resource_limits and not hasattr(_resource, "RLIMIT_AS"):
            raise ValueError(
                "command validator rlimit_memory_bytes is unsupported on this platform"
            )
        if "rlimit_process_count" in resource_limits and not hasattr(_resource, "RLIMIT_NPROC"):
            raise ValueError(
                "command validator rlimit_process_count is unsupported on this platform"
            )
        if "rlimit_file_size_bytes" in resource_limits and not hasattr(_resource, "RLIMIT_FSIZE"):
            raise ValueError(
                "command validator rlimit_file_size_bytes is unsupported on this platform"
            )

        def apply_limits() -> None:
            assert _resource is not None
            if "rlimit_cpu_seconds" in resource_limits:
                limit = resource_limits["rlimit_cpu_seconds"]
                _resource.setrlimit(_resource.RLIMIT_CPU, (limit, limit))
            if "rlimit_memory_bytes" in resource_limits:
                limit = resource_limits["rlimit_memory_bytes"]
                _resource.setrlimit(_resource.RLIMIT_AS, (limit, limit))
            if "rlimit_process_count" in resource_limits:
                limit = resource_limits["rlimit_process_count"]
                _resource.setrlimit(_resource.RLIMIT_NPROC, (limit, limit))
            if "rlimit_file_size_bytes" in resource_limits:
                limit = resource_limits["rlimit_file_size_bytes"]
                _resource.setrlimit(_resource.RLIMIT_FSIZE, (limit, limit))

        return apply_limits

    def _best_effort_abandon_artifacts(
        self,
        artifact_ids: list[str],
        *,
        reason: str,
    ) -> None:
        for artifact_id in artifact_ids:
            with suppress(Exception):
                self._artifact_store.abandon_artifact(
                    artifact_id,
                    reason=reason,
                    provenance={
                        "origin": "procedure_validator",
                        "stage": "artifact_capture",
                    },
                )

    def _truncate_output(
        self,
        payload: bytes,
        max_output_bytes: int | None,
    ) -> tuple[bytes, bool]:
        if max_output_bytes is None or len(payload) <= max_output_bytes:
            return payload, False
        return payload[:max_output_bytes], True


class DockerProcedureValidator(CommandProcedureValidator):
    """Runs procedure validation commands inside a Docker-backed workspace sandbox."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        workspace_backend: DockerWorkspaceBackend,
    ) -> None:
        super().__init__(artifact_store)
        self._workspace_backend = workspace_backend

    def _execute_command(
        self,
        *,
        procedure: MemoryItem,
        command: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_seconds: float | None,
        preexec_fn: Any | None,
        repo_root: str | None,
    ) -> _CommandExecutionResult:
        del preexec_fn
        if repo_root is None:
            raise ValueError("docker procedure validator requires repo_root")
        repo_root_path = Path(repo_root).resolve()
        if not repo_root_path.exists() or not repo_root_path.is_dir():
            raise ValueError("docker procedure validator repo_root must exist")

        host_cwd = self._resolve_sandbox_cwd(cwd, repo_root_path)
        sandbox_env = self._resolve_sandbox_env(env)

        with tempfile.TemporaryDirectory(prefix="vtm-procedure-validator-") as tmp_dir:
            temp_root = Path(tmp_dir)
            workspace_root = temp_root / "workspace"
            artifact_root = temp_root / "artifacts"
            self._snapshot_repo(repo_root_path, workspace_root)
            sandbox_cwd = workspace_root / host_cwd.relative_to(repo_root_path)
            wrapper_path = self._write_wrapper_script(
                workspace_root=workspace_root,
                sandbox_cwd=sandbox_cwd,
                env=sandbox_env,
            )
            prepared = None
            try:
                try:
                    prepared = self._workspace_backend.prepare_existing_workspace(
                        case_id=procedure.memory_id,
                        attempt_index=1,
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                        command_timeout_seconds=(
                            max(int(timeout_seconds), 1) if timeout_seconds is not None else 120
                        ),
                        max_output_chars=10_000_000,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        self._describe_startup_failure(exc, artifact_root)
                    ) from exc
                result = self._run_docker_command(
                    docker_binary=prepared.metadata["docker_binary"],
                    container_name=prepared.metadata["docker_container_name"],
                    workspace_root=workspace_root,
                    wrapper_path=wrapper_path,
                    command=command,
                    timeout_seconds=timeout_seconds,
                )
                return _CommandExecutionResult(
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    timed_out=result.timed_out,
                    metadata={
                        "workspace_backend": prepared.backend_name,
                        "sandboxed": True,
                        "restrict_cwd_to_repo": True,
                        "restricted_mode": True,
                        "docker_binary": prepared.metadata["docker_binary"],
                        "docker_container_id": prepared.metadata["docker_container_id"],
                        "docker_container_name": prepared.metadata["docker_container_name"],
                        "docker_image": prepared.metadata["docker_image"],
                        "docker_network": prepared.metadata["docker_network"],
                        "docker_read_only_rootfs": (
                            prepared.metadata["docker_read_only_rootfs"] == "true"
                        ),
                        "docker_pids_limit": int(prepared.metadata["docker_pids_limit"]),
                        "docker_memory_limit": prepared.metadata["docker_memory_limit"],
                        "docker_cpu_limit": float(prepared.metadata["docker_cpu_limit"]),
                        "sandbox_timeout_enforced_via_container_cleanup": result.timed_out,
                    },
                )
            finally:
                if prepared is not None:
                    with suppress(Exception):
                        prepared.driver.close()

    def _resolve_sandbox_cwd(self, cwd: str | None, repo_root: Path) -> Path:
        resolved = repo_root if cwd is None else Path(cwd).resolve()
        if not resolved.is_relative_to(repo_root):
            raise ValueError("docker procedure validator cwd must stay within repo_root")
        return resolved

    def _resolve_sandbox_env(self, env: dict[str, str] | None) -> dict[str, str]:
        resolved = dict(os.environ) if env is None else dict(env)
        if "PATH" not in resolved:
            resolved["PATH"] = os.defpath
        return resolved

    def _snapshot_repo(self, repo_root: Path, workspace_root: Path) -> None:
        shutil.copytree(repo_root, workspace_root, symlinks=True)

    def _write_wrapper_script(
        self,
        *,
        workspace_root: Path,
        sandbox_cwd: Path,
        env: dict[str, str],
    ) -> Path:
        wrapper_path = workspace_root / ".vtm-procedure-validator.sh"
        env_args = " ".join(shlex.quote(f"{key}={value}") for key, value in sorted(env.items()))
        wrapper_path.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            f"cd {shlex.quote(str(sandbox_cwd))}\n"
            f'exec env -i {env_args} "$@"\n',
            encoding="utf-8",
        )
        wrapper_path.chmod(0o700)
        return wrapper_path

    def _run_docker_command(
        self,
        *,
        docker_binary: str,
        container_name: str,
        workspace_root: Path,
        wrapper_path: Path,
        command: list[str],
        timeout_seconds: float | None,
    ) -> _CommandExecutionResult:
        exec_command = [
            docker_binary,
            "exec",
            "-w",
            str(workspace_root),
            container_name,
            "/bin/sh",
            str(wrapper_path),
            *command,
        ]
        try:
            completed = subprocess.run(
                exec_command,
                check=False,
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
            )
            return _CommandExecutionResult(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return _CommandExecutionResult(
                exit_code=None,
                stdout=self._normalize_timeout_output(exc.stdout),
                stderr=self._normalize_timeout_output(exc.stderr),
                timed_out=True,
            )

    def _normalize_timeout_output(self, payload: bytes | str | None) -> bytes:
        if payload is None:
            return b""
        if isinstance(payload, bytes):
            return payload
        return payload.encode("utf-8", errors="replace")

    def _describe_startup_failure(self, exc: RuntimeError, artifact_root: Path) -> str:
        stdout_path = artifact_root / "docker-run.stdout"
        stderr_path = artifact_root / "docker-run.stderr"
        stdout = self._read_text_if_present(stdout_path)
        stderr = self._read_text_if_present(stderr_path)
        details = [str(exc)]
        if stdout:
            details.append(f"stdout={stdout!r}")
        if stderr:
            details.append(f"stderr={stderr!r}")
        return "; ".join(details)

    def _read_text_if_present(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()
