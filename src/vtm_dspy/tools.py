"""Thin VTM tool wrappers exposed to DSPy-style agents."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtm.enums import EvidenceBudget, EvidenceKind, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.retrieval import RetrieveCandidate, RetrieveRequest
from vtm.services.memory_kernel import MemoryKernel

VERIFIED_MEMORY_STATUSES = (ValidityStatus.VERIFIED, ValidityStatus.RELOCATED)


@dataclass(frozen=True)
class MemoryWriteProposal:
    """One agent-authored memory proposal awaiting host-side verification."""

    proposal_kind: str
    memory_role: str
    title: str
    summary: str
    rationale: str | None
    function_name: str | None
    repair_kind: str | None
    interface_mode: str | None
    bug_class: str | None
    failure_signature: str | None
    transfer_terms: tuple[str, ...]


def memory_tooling_supported(
    *,
    kernel: MemoryKernel | None,
    scopes: Sequence[VisibilityScope] = (),
    dependency_provider: Callable[[], DependencyFingerprint | None] | None = None,
    memory_lookup: Callable[[str], MemoryItem | None] | None = None,
) -> bool:
    """Whether the full dynamic memory tool surface can operate safely."""
    return (
        kernel is not None
        and bool(tuple(scopes))
        and dependency_provider is not None
        and memory_lookup is not None
    )


@dataclass
class VTMMemoryTools:
    """Thin memory lookup tools suitable for DSPy ReAct surfaces."""

    kernel: MemoryKernel | None
    scopes: tuple[VisibilityScope, ...] = ()
    dependency_provider: Callable[[], DependencyFingerprint | None] | None = None
    memory_lookup: Callable[[str], MemoryItem | None] | None = None
    enable_lookup_tools: bool = True
    enable_write_tools: bool = True

    def __init__(
        self,
        *,
        kernel: MemoryKernel | None,
        scopes: Sequence[VisibilityScope] = (),
        dependency_provider: Callable[[], DependencyFingerprint | None] | None = None,
        memory_lookup: Callable[[str], MemoryItem | None] | None = None,
        enable_lookup_tools: bool = True,
        enable_write_tools: bool = True,
    ) -> None:
        self.kernel = kernel
        self.scopes = tuple(scopes)
        self.dependency_provider = dependency_provider
        self.memory_lookup = memory_lookup
        self.enable_lookup_tools = bool(enable_lookup_tools)
        self.enable_write_tools = bool(enable_write_tools)
        self._write_proposals: list[MemoryWriteProposal] = []

    @property
    def enabled(self) -> bool:
        """Whether any dynamic memory operations are available."""
        return self.lookup_enabled or self.write_enabled

    @property
    def lookup_enabled(self) -> bool:
        """Whether retrieval, expansion, and verification are available."""
        return (
            self.enable_lookup_tools
            and self.kernel is not None
            and bool(self.scopes)
        )

    @property
    def write_enabled(self) -> bool:
        """Whether agent-side memory proposals are available."""
        return self.enable_write_tools and bool(self.scopes)

    def tool_mapping(self) -> dict[str, Callable[..., Any]]:
        """Return named tool callables for DSPy agent construction."""
        mapping: dict[str, Callable[..., Any]] = {}
        if self.lookup_enabled:
            mapping.update(
                {
                    "search_verified_memory": self.search_verified_memory,
                    "search_naive_memory": self.search_naive_memory,
                    "expand_memory_evidence": self.expand_memory_evidence,
                    "verify_memory": self.verify_memory,
                }
            )
        if self.write_enabled:
            mapping.update(
                {
                    "propose_memory_lesson": self.propose_memory_lesson,
                    "propose_failure_pattern": self.propose_failure_pattern,
                    "propose_solution_pattern": self.propose_solution_pattern,
                }
            )
        if not self.enabled:
            return {}
        return mapping

    def clear_write_proposals(self) -> None:
        """Forget any previously buffered proposals."""
        self._write_proposals.clear()

    def drain_write_proposals(self) -> list[dict[str, Any]]:
        """Return buffered proposals and clear the buffer."""
        drained = [
            {
                "proposal_kind": proposal.proposal_kind,
                "memory_role": proposal.memory_role,
                "title": proposal.title,
                "summary": proposal.summary,
                "rationale": proposal.rationale,
                "function_name": proposal.function_name,
                "repair_kind": proposal.repair_kind,
                "interface_mode": proposal.interface_mode,
                "bug_class": proposal.bug_class,
                "failure_signature": proposal.failure_signature,
                "transfer_terms": list(proposal.transfer_terms),
            }
            for proposal in self._write_proposals
        ]
        self._write_proposals.clear()
        return drained

    def search_verified_memory(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search memory using verified-only semantics when a dependency is available."""
        if not self.lookup_enabled or not query.strip():
            return []
        assert self.kernel is not None
        verification_mode = "stored_status_only"
        current_dependency = self._current_dependency()
        if current_dependency is not None:
            result = self.kernel.retrieve(
                RetrieveRequest(
                    query=query,
                    scopes=self.scopes,
                    evidence_budget=EvidenceBudget.SUMMARY_ONLY,
                    limit=self._bounded_limit(k),
                    current_dependency=current_dependency,
                    verify_on_read=True,
                    return_verified_only=True,
                )
            )
            verification_mode = "verify_on_read"
        else:
            result = self.kernel.retrieve(
                RetrieveRequest(
                    query=query,
                    scopes=self.scopes,
                    statuses=VERIFIED_MEMORY_STATUSES,
                    evidence_budget=EvidenceBudget.SUMMARY_ONLY,
                    limit=self._bounded_limit(k),
                )
            )
        return [
            self._serialize_candidate(candidate, verification_mode=verification_mode)
            for candidate in result.candidates
        ]

    def search_naive_memory(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search memory without read-time verification."""
        if not self.lookup_enabled or not query.strip():
            return []
        assert self.kernel is not None
        result = self.kernel.retrieve(
            RetrieveRequest(
                query=query,
                scopes=self.scopes,
                evidence_budget=EvidenceBudget.SUMMARY_ONLY,
                limit=self._bounded_limit(k),
            )
        )
        return [self._serialize_candidate(candidate) for candidate in result.candidates]

    def expand_memory_evidence(self, memory_id: str) -> dict[str, Any]:
        """Return one memory item's evidence plus a compact summary."""
        if not self.lookup_enabled or not memory_id.strip():
            return {
                "id": memory_id,
                "summary": "memory expansion unavailable",
                "status": "unavailable",
                "evidence_summary": [],
            }
        assert self.kernel is not None
        evidence = self.kernel.expand(memory_id)
        item = self._lookup_memory(memory_id)
        payload = self._serialize_memory_item(item, evidence=evidence)
        payload["id"] = memory_id
        payload["evidence"] = [self._serialize_evidence_ref(ref) for ref in evidence]
        return payload

    def verify_memory(self, memory_id: str) -> dict[str, Any]:
        """Verify one memory item against the current dependency fingerprint."""
        if not self.lookup_enabled or not memory_id.strip():
            return {
                "id": memory_id,
                "summary": "verification unavailable",
                "status": "unavailable",
                "evidence_summary": [],
            }
        assert self.kernel is not None
        current_dependency = self._current_dependency()
        if current_dependency is None:
            return {
                "id": memory_id,
                "summary": "verify_memory requires a current dependency fingerprint",
                "status": "unavailable",
                "evidence_summary": [],
            }
        item, result = self.kernel.verify_memory(memory_id, current_dependency)
        payload = self._serialize_memory_item(
            item,
            evidence=result.updated_evidence or item.evidence,
        )
        payload.update(
            {
                "id": item.memory_id,
                "status": result.current_status.value,
                "verification_reasons": list(result.reasons),
                "verification_skipped": result.skipped,
            }
        )
        return payload

    def propose_memory_lesson(
        self,
        title: str,
        summary: str,
        rationale: str = "",
        memory_role: str = "repair_lesson",
        transfer_terms: str = "",
        function_name: str = "",
        repair_kind: str = "",
        interface_mode: str = "",
        bug_class: str = "",
        failure_signature: str = "",
    ) -> dict[str, Any]:
        """Buffer one reusable lesson for host-side verification and promotion."""
        return self._buffer_write_proposal(
            proposal_kind="memory_lesson",
            memory_role=memory_role,
            title=title,
            summary=summary,
            rationale=rationale,
            transfer_terms=transfer_terms,
            function_name=function_name,
            repair_kind=repair_kind,
            interface_mode=interface_mode,
            bug_class=bug_class,
            failure_signature=failure_signature,
        )

    def propose_failure_pattern(
        self,
        failure_signature: str,
        lesson: str,
        rationale: str = "",
        transfer_terms: str = "",
        function_name: str = "",
        repair_kind: str = "",
        interface_mode: str = "",
        bug_class: str = "",
    ) -> dict[str, Any]:
        """Buffer one reusable failure pattern as a repair lesson proposal."""
        title = f"Failure pattern: {self._compact(failure_signature, limit=72)}"
        return self._buffer_write_proposal(
            proposal_kind="failure_pattern",
            memory_role="repair_lesson",
            title=title,
            summary=lesson,
            rationale=rationale,
            transfer_terms=transfer_terms,
            function_name=function_name,
            repair_kind=repair_kind,
            interface_mode=interface_mode,
            bug_class=bug_class,
            failure_signature=failure_signature,
        )

    def propose_solution_pattern(
        self,
        summary: str,
        rationale: str = "",
        transfer_terms: str = "",
        function_name: str = "",
        interface_mode: str = "",
    ) -> dict[str, Any]:
        """Buffer one generic successful solution pattern for host-side promotion."""
        title = f"Solution pattern: {self._compact(summary, limit=72)}"
        return self._buffer_write_proposal(
            proposal_kind="solution_pattern",
            memory_role="successful_solution",
            title=title,
            summary=summary,
            rationale=rationale,
            transfer_terms=transfer_terms,
            function_name=function_name,
            repair_kind="successful_initial_solution",
            interface_mode=interface_mode,
            bug_class="",
            failure_signature="",
        )

    def _current_dependency(self) -> DependencyFingerprint | None:
        if self.dependency_provider is None:
            return None
        return self.dependency_provider()

    def _buffer_write_proposal(
        self,
        *,
        proposal_kind: str,
        memory_role: str,
        title: str,
        summary: str,
        rationale: str,
        transfer_terms: str,
        function_name: str,
        repair_kind: str,
        interface_mode: str,
        bug_class: str,
        failure_signature: str,
    ) -> dict[str, Any]:
        if not self.write_enabled:
            return {"status": "unavailable", "reason": "memory tooling disabled"}
        normalized_summary = self._compact(summary, limit=240)
        if not normalized_summary:
            return {"status": "rejected", "reason": "summary is required"}
        proposal = MemoryWriteProposal(
            proposal_kind=proposal_kind,
            memory_role=self._normalize_memory_role(memory_role),
            title=self._compact(title, limit=96) or "Agent memory lesson",
            summary=normalized_summary,
            rationale=self._compact(rationale, limit=600) or None,
            function_name=self._compact(function_name, limit=64) or None,
            repair_kind=self._compact(repair_kind, limit=64) or None,
            interface_mode=self._compact(interface_mode, limit=64) or None,
            bug_class=self._compact(bug_class, limit=64) or None,
            failure_signature=self._compact(failure_signature, limit=220) or None,
            transfer_terms=self._normalize_transfer_terms(transfer_terms),
        )
        self._write_proposals.append(proposal)
        return {
            "status": "buffered",
            "proposal_index": len(self._write_proposals),
            "proposal_kind": proposal.proposal_kind,
            "memory_role": proposal.memory_role,
            "summary": proposal.summary,
        }

    def _lookup_memory(self, memory_id: str) -> MemoryItem | None:
        if self.memory_lookup is None:
            return None
        return self.memory_lookup(memory_id)

    def _normalize_memory_role(self, raw_role: str) -> str:
        normalized = raw_role.strip().lower()
        if normalized in {"repair_lesson", "successful_solution"}:
            return normalized
        return "repair_lesson"

    def _normalize_transfer_terms(self, raw_terms: str) -> tuple[str, ...]:
        terms = []
        seen: set[str] = set()
        for token in raw_terms.replace("|", ",").split(","):
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            terms.append(normalized)
            seen.add(normalized)
        return tuple(terms)

    def _compact(self, raw: str, *, limit: int) -> str:
        normalized = " ".join(raw.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)].rstrip() + "..."

    def _serialize_candidate(
        self,
        candidate: RetrieveCandidate,
        *,
        verification_mode: str | None = None,
    ) -> dict[str, Any]:
        payload = self._serialize_memory_item(candidate.memory, evidence=candidate.evidence)
        payload.update(
            {
                "id": candidate.memory.memory_id,
                "score": round(candidate.score, 6),
            }
        )
        if verification_mode is not None:
            payload["verification_mode"] = verification_mode
        return payload

    def _serialize_memory_item(
        self,
        item: MemoryItem | None,
        *,
        evidence: Sequence[EvidenceRef],
    ) -> dict[str, Any]:
        path, symbol = self._anchor_location(item, evidence)
        payload: dict[str, Any] = {
            "title": item.title if item is not None else None,
            "summary": item.summary if item is not None else None,
            "status": item.validity.status.value if item is not None else "unknown",
            "evidence_summary": self._summarize_evidence(evidence),
            "path": path,
            "symbol": symbol,
        }
        return payload

    def _anchor_location(
        self,
        item: MemoryItem | None,
        evidence: Sequence[EvidenceRef],
    ) -> tuple[str | None, str | None]:
        for ref in (*evidence, *(item.evidence if item is not None else ())):
            if ref.code_anchor is not None:
                return ref.code_anchor.path, ref.code_anchor.symbol
        return None, None

    def _summarize_evidence(self, evidence: Sequence[EvidenceRef]) -> list[str]:
        return [
            self._summarize_evidence_ref(ref)
            for ref in evidence[:3]
        ]

    def _summarize_evidence_ref(self, evidence: EvidenceRef) -> str:
        if evidence.summary:
            return evidence.summary
        if evidence.code_anchor is not None:
            return f"{evidence.code_anchor.path}::{evidence.code_anchor.symbol}"
        if evidence.kind is EvidenceKind.ARTIFACT and evidence.artifact_ref is not None:
            return f"artifact:{evidence.artifact_ref.artifact_id}"
        if evidence.kind is EvidenceKind.MEMORY and evidence.memory_id is not None:
            return f"memory:{evidence.memory_id}"
        return evidence.ref_id

    def _serialize_evidence_ref(self, evidence: EvidenceRef) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": evidence.kind.value,
            "ref_id": evidence.ref_id,
            "label": evidence.label,
            "summary": evidence.summary,
        }
        if evidence.code_anchor is not None:
            payload["path"] = evidence.code_anchor.path
            payload["symbol"] = evidence.code_anchor.symbol
            payload["start_line"] = evidence.code_anchor.start_line
            payload["end_line"] = evidence.code_anchor.end_line
        if evidence.artifact_ref is not None:
            payload["artifact_id"] = evidence.artifact_ref.artifact_id
            payload["content_type"] = evidence.artifact_ref.content_type
        if evidence.memory_id is not None:
            payload["memory_id"] = evidence.memory_id
        return payload

    def _bounded_limit(self, k: int) -> int:
        return max(1, min(int(k), 20))


@dataclass
class WorkspaceTools:
    """Minimal workspace tools constrained to one explicit repository root."""

    workspace_root: Path
    command_timeout_seconds: int = 120
    max_output_chars: int = 20000

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        command_timeout_seconds: int = 120,
        max_output_chars: int = 20000,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.command_timeout_seconds = command_timeout_seconds
        self.max_output_chars = max_output_chars

    def tool_mapping(self) -> dict[str, Callable[..., Any]]:
        """Return named tool callables for controlled coding workflows."""
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "run_command": self.run_command,
            "git_diff": self.git_diff,
        }

    def read_file(self, path: str, *, max_chars: int | None = None) -> dict[str, Any]:
        """Read one workspace-relative file path."""
        resolved = self._resolve_path(path)
        content = resolved.read_text(encoding="utf-8")
        limit = self.max_output_chars if max_chars is None else max_chars
        truncated = limit >= 0 and len(content) > limit
        if truncated:
            content = content[:limit]
        return {
            "path": str(resolved.relative_to(self.workspace_root)),
            "content": content,
            "truncated": truncated,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write one workspace-relative file path."""
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return {
            "path": str(resolved.relative_to(self.workspace_root)),
            "bytes_written": len(content.encode("utf-8")),
            "status": "written",
        }

    def run_command(self, command: str) -> dict[str, Any]:
        """Run a shell command inside the configured workspace root."""
        stripped = command.strip()
        if not stripped:
            raise ValueError("command must be non-empty")
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                ["/bin/sh", "-lc", stripped],
                cwd=self.workspace_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
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
        stdout, stdout_truncated = self._truncate(stdout)
        stderr, stderr_truncated = self._truncate(stderr)
        return {
            "command": stripped,
            "cwd": str(self.workspace_root),
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "truncated": stdout_truncated or stderr_truncated,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def git_diff(self) -> dict[str, Any]:
        """Capture the current workspace diff relative to `HEAD`."""
        result = self.run_command("git diff --binary --no-ext-diff HEAD --")
        result["diff"] = result["stdout"]
        return result

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ValueError("path must be non-empty")
        candidate = (self.workspace_root / raw_path).resolve()
        if candidate != self.workspace_root and self.workspace_root not in candidate.parents:
            raise ValueError(f"path escapes workspace: {raw_path}")
        return candidate

    def _truncate(self, content: str) -> tuple[str, bool]:
        if self.max_output_chars <= 0:
            return "", bool(content)
        if len(content) <= self.max_output_chars:
            return content, False
        return content[: self.max_output_chars], True

    def _normalize_subprocess_output(self, raw: str | bytes | None) -> str:
        if raw is None:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return raw


__all__ = ["VTMMemoryTools", "WorkspaceTools", "VERIFIED_MEMORY_STATUSES"]
