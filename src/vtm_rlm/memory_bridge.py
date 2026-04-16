"""VTM memory helpers exposed into the vendored RLM runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vtm.enums import EvidenceBudget, EvidenceKind
from vtm.evidence import EvidenceRef
from vtm.harness.models import TaskMemoryContextItem
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.retrieval import RetrieveRequest
from vtm.services import TransactionalMemoryKernel


class VTMMemoryBridge:
    """Small bridge that exposes VTM retrieval and evidence expansion as Python tools."""

    def __init__(
        self,
        *,
        kernel: TransactionalMemoryKernel | None,
        scopes: Sequence[VisibilityScope] = (),
    ) -> None:
        self._kernel = kernel
        self._scopes = tuple(scopes)

    @property
    def enabled(self) -> bool:
        """Whether dynamic memory operations are available."""
        return self._kernel is not None and bool(self._scopes)

    def custom_tools(self) -> dict[str, dict[str, object]]:
        """Return the custom-tool mapping expected by the vendored RLM runtime."""
        if not self.enabled:
            return {}
        return {
            "search_memory": {
                "tool": self.search_memory,
                "description": (
                    "Search VTM memory for repository-specific context. "
                    "Arguments: query (str), limit (int, optional)."
                ),
            },
            "expand_memory": {
                "tool": self.expand_memory,
                "description": (
                    "Expand one VTM memory item into its raw evidence references. "
                    "Arguments: memory_id (str)."
                ),
            },
        }

    def search_memory(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve memory candidates for the provided query."""
        if not self.enabled or not query.strip():
            return []
        bounded_limit = max(1, min(int(limit), 20))
        assert self._kernel is not None
        result = self._kernel.retrieve(
            RetrieveRequest(
                query=query,
                scopes=self._scopes,
                evidence_budget=EvidenceBudget.SUMMARY_ONLY,
                limit=bounded_limit,
            )
        )
        return [
            {
                "memory_id": candidate.memory.memory_id,
                "title": candidate.memory.title,
                "summary": candidate.memory.summary,
                "score": candidate.score,
                "status": candidate.memory.validity.status.value,
                "tags": list(candidate.memory.tags),
                "path": self._anchor_path(candidate.memory),
                "symbol": self._anchor_symbol(candidate.memory),
            }
            for candidate in result.candidates
        ]

    def expand_memory(self, memory_id: str) -> list[dict[str, Any]]:
        """Return a JSON-serializable representation of one memory item's evidence."""
        if not self.enabled or not memory_id:
            return []
        assert self._kernel is not None
        return [self._serialize_evidence(evidence) for evidence in self._kernel.expand(memory_id)]

    def _serialize_evidence(self, evidence: EvidenceRef) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": evidence.kind.value,
            "ref_id": evidence.ref_id,
            "label": evidence.label,
            "summary": evidence.summary,
        }
        if evidence.kind is EvidenceKind.ARTIFACT and evidence.artifact_ref is not None:
            payload["artifact"] = {
                "artifact_id": evidence.artifact_ref.artifact_id,
                "sha256": evidence.artifact_ref.sha256,
                "content_type": evidence.artifact_ref.content_type,
            }
        elif evidence.kind is EvidenceKind.CODE_ANCHOR and evidence.code_anchor is not None:
            payload["code_anchor"] = {
                "path": evidence.code_anchor.path,
                "symbol": evidence.code_anchor.symbol,
                "kind": evidence.code_anchor.kind,
                "language": evidence.code_anchor.language,
                "start_line": evidence.code_anchor.start_line,
                "end_line": evidence.code_anchor.end_line,
            }
        elif evidence.kind is EvidenceKind.MEMORY:
            payload["memory_id"] = evidence.memory_id
        return payload

    def _anchor_path(self, item: MemoryItem) -> str | None:
        for evidence in item.evidence:
            if evidence.code_anchor is not None:
                return evidence.code_anchor.path
        return None

    def _anchor_symbol(self, item: MemoryItem) -> str | None:
        for evidence in item.evidence:
            if evidence.code_anchor is not None:
                return evidence.code_anchor.symbol
        return None


def summarize_memory_context(items: Sequence[TaskMemoryContextItem]) -> str:
    """Render pre-retrieved task memory into a compact prompt block."""
    if not items:
        return "No preloaded VTM memory."
    lines = []
    for index, item in enumerate(items, start=1):
        location = ""
        if item.relative_path is not None:
            location = f" path={item.relative_path}"
        if item.symbol is not None:
            location = f"{location} symbol={item.symbol}".rstrip()
        lines.extend(
            [
                f"{index}. [{item.status}] {item.title} (score={item.score:.3f})",
                f"   summary: {item.summary}",
                f"   memory_id: {item.memory_id}{location}",
            ]
        )
    return "\n".join(lines)
