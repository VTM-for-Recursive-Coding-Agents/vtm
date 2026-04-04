from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from vtm.artifacts import ArtifactRecord
from vtm.base import utc_now
from vtm.enums import MemoryKind, ValidityStatus
from vtm.memory_items import MemoryItem, ProcedurePayload, ValidatorSpec
from vtm.stores.base import ArtifactStore
from vtm.verification import ProcedureValidationResult


class ProcedureValidator(Protocol):
    def validate(
        self,
        procedure: MemoryItem,
        *,
        repo_root: str | None = None,
    ) -> ProcedureValidationResult: ...


class CommandProcedureValidator:
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    def validate(
        self,
        procedure: MemoryItem,
        *,
        repo_root: str | None = None,
    ) -> ProcedureValidationResult:
        payload = self._require_procedure_payload(procedure)
        validator = payload.validator
        if validator is None:
            raise ValueError("procedure validation requires a validator spec")
        if validator.kind != "command":
            raise ValueError(f"unsupported validator kind: {validator.kind}")

        command = self._require_command(validator)
        cwd = self._resolve_cwd(validator, repo_root)
        env = self._resolve_env(validator)
        expected_exit_code = self._require_expected_exit_code(validator)
        timeout_seconds = self._resolve_timeout_seconds(validator)
        max_output_bytes = self._resolve_max_output_bytes(validator)
        result_metadata: dict[str, Any] = {
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": max_output_bytes,
            "timed_out": False,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
        capture_group_id = f"capgrp_{uuid4().hex}"

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=False,
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = completed.returncode
            success = exit_code == expected_exit_code
            status = ValidityStatus.VERIFIED if success else ValidityStatus.REFUTED
            reason = (
                "procedure validator exit code matched expected"
                if success
                else f"procedure validator exit code {exit_code} != expected {expected_exit_code}"
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, bytes) else b""
            stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
            exit_code = None
            success = False
            status = ValidityStatus.UNKNOWN
            reason = f"procedure validator timed out after {timeout_seconds} seconds"
            result_metadata["timed_out"] = True
        except OSError as exc:
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
        return self._artifact_store.prepare_bytes(
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

    def _require_procedure_payload(self, procedure: MemoryItem) -> ProcedurePayload:
        if procedure.kind is not MemoryKind.PROCEDURE or not isinstance(
            procedure.payload,
            ProcedurePayload,
        ):
            raise ValueError("procedure validator requires a procedure memory item")
        return procedure.payload

    def _require_command(self, validator: ValidatorSpec) -> list[str]:
        command = validator.config.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError("command validator config requires a non-empty command list[str]")
        return list(command)

    def _resolve_cwd(self, validator: ValidatorSpec, repo_root: str | None) -> str | None:
        cwd = validator.config.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("command validator cwd must be a string")
        if cwd is None:
            return repo_root
        if repo_root is not None and not Path(cwd).is_absolute():
            return str(Path(repo_root) / cwd)
        return cwd

    def _resolve_env(self, validator: ValidatorSpec) -> dict[str, str] | None:
        raw_env = validator.config.get("env")
        allowlist = self._resolve_env_name_list(validator, "env_allowlist")
        denylist = self._resolve_env_name_list(validator, "env_denylist")
        if raw_env is None and allowlist is None and denylist is None:
            return None
        if raw_env is not None and (
            not isinstance(raw_env, Mapping)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_env.items()
            )
        ):
            raise ValueError("command validator env must be a dict[str, str]")

        resolved_env = dict(os.environ)
        if allowlist is not None:
            allowed = set(allowlist)
            resolved_env = {
                key: value for key, value in resolved_env.items() if key in allowed
            }
        if denylist is not None:
            for key in denylist:
                resolved_env.pop(key, None)
        if raw_env is not None:
            resolved_env.update(dict(raw_env))
        return resolved_env

    def _require_expected_exit_code(self, validator: ValidatorSpec) -> int:
        expected_exit_code = validator.config.get("expected_exit_code", 0)
        if not isinstance(expected_exit_code, int):
            raise ValueError("command validator expected_exit_code must be an int")
        return expected_exit_code

    def _resolve_timeout_seconds(self, validator: ValidatorSpec) -> float | None:
        timeout_seconds = validator.config.get("timeout_seconds")
        if timeout_seconds is None:
            return None
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("command validator timeout_seconds must be a positive number")
        return float(timeout_seconds)

    def _resolve_max_output_bytes(self, validator: ValidatorSpec) -> int | None:
        max_output_bytes = validator.config.get("max_output_bytes")
        if max_output_bytes is None:
            return None
        if not isinstance(max_output_bytes, int) or max_output_bytes <= 0:
            raise ValueError("command validator max_output_bytes must be a positive int")
        return max_output_bytes

    def _resolve_env_name_list(
        self,
        validator: ValidatorSpec,
        field_name: str,
    ) -> tuple[str, ...] | None:
        raw_value = validator.config.get(field_name)
        if raw_value is None:
            return None
        if not isinstance(raw_value, list) or not all(
            isinstance(entry, str) and entry for entry in raw_value
        ):
            raise ValueError(f"command validator {field_name} must be a list[str]")
        return tuple(raw_value)

    def _truncate_output(
        self,
        payload: bytes,
        max_output_bytes: int | None,
    ) -> tuple[bytes, bool]:
        if max_output_bytes is None or len(payload) <= max_output_bytes:
            return payload, False
        return payload[:max_output_bytes], True
