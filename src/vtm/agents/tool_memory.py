"""Built-in memory tools that integrate the native agent with VTM."""

from __future__ import annotations

import json

from vtm.agents.models import AgentToolResult, AgentToolSpec
from vtm.agents.tool_base import AgentTool, ToolExecutionContext
from vtm.agents.tool_utils import first_anchor_path, write_local_text_artifact, write_text_artifact
from vtm.base import utc_now
from vtm.enums import EvidenceBudget, MemoryKind, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.memory_items import (
    ClaimPayload,
    ConstraintPayload,
    DecisionPayload,
    MemoryItem,
    ProcedurePayload,
    ProcedureStep,
    ValidityState,
)
from vtm.retrieval import RetrieveRequest


def build_retrieve_memory_tool() -> AgentTool:
    """Build the tool that retrieves task and durable memory."""
    return AgentTool(
        spec=AgentToolSpec(
            name="retrieve_memory",
            description="Retrieve repository or task memory from VTM.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        ),
        handler=_retrieve_memory,
    )


def build_record_task_memory_tool() -> AgentTool:
    """Build the tool that writes task-scoped memory items."""
    return AgentTool(
        spec=AgentToolSpec(
            name="record_task_memory",
            description="Persist a verified task-scoped claim, constraint, or decision.",
            input_schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "claim": {"type": "string"},
                    "statement": {"type": "string"},
                    "severity": {"type": "string"},
                    "rationale": {"type": "string"},
                    "tags": {"type": "array"},
                    "raw_content": {"type": "string"},
                },
                "required": ["kind", "title", "summary"],
            },
        ),
        handler=_record_task_memory,
    )


def build_promote_procedure_tool() -> AgentTool:
    """Build the tool that promotes task memory into a durable procedure."""
    return AgentTool(
        spec=AgentToolSpec(
            name="promote_procedure",
            description="Promote task memories into a durable procedure.",
            input_schema={
                "type": "object",
                "properties": {
                    "source_memory_ids": {"type": "array"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "goal": {"type": "string"},
                    "steps": {"type": "array"},
                },
                "required": ["source_memory_ids", "title", "summary", "goal", "steps"],
            },
        ),
        handler=_promote_procedure,
    )


def _retrieve_memory(
    arguments: dict[str, object],
    context: ToolExecutionContext,
    call_id: str,
) -> AgentToolResult:
    if context.kernel is None or context.durable_scope is None:
        return AgentToolResult(success=False, content="memory retrieval is not enabled")
    query = str(arguments.get("query", "")).strip()
    limit = _coerce_int(arguments.get("limit"), default=5)
    if not query:
        return AgentToolResult(success=False, content="query must be non-empty")
    scopes = [context.durable_scope]
    if context.task_scope is not None:
        scopes.append(context.task_scope)
    result = context.kernel.retrieve(
        RetrieveRequest(
            query=query,
            scopes=tuple(scopes),
            evidence_budget=EvidenceBudget.SUMMARY_ONLY,
            limit=limit,
        )
    )
    payload = {
        "query": query,
        "candidates": [
            {
                "memory_id": candidate.memory.memory_id,
                "title": candidate.memory.title,
                "summary": candidate.memory.summary,
                "score": candidate.score,
                "status": candidate.memory.validity.status.value,
                "relative_path": first_anchor_path(candidate.memory),
            }
            for candidate in result.candidates
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    artifact_path, artifact_id = write_text_artifact(
        context=context,
        call_id=call_id,
        suffix=".retrieve.json",
        text=text,
        content_type="application/json",
        metadata={"query": query, "candidate_count": len(result.candidates)},
    )
    return AgentToolResult(
        success=True,
        content=text,
        metadata={"query": query, "candidate_count": len(result.candidates)},
        artifact_path=artifact_path,
        artifact_id=artifact_id,
    )


def _record_task_memory(
    arguments: dict[str, object],
    context: ToolExecutionContext,
    call_id: str,
) -> AgentToolResult:
    if (
        context.kernel is None
        or context.task_scope is None
        or context.dependency_builder is None
    ):
        return AgentToolResult(success=False, content="task memory is not enabled")

    kind = str(arguments.get("kind", "")).strip()
    title = str(arguments.get("title", "")).strip()
    summary = str(arguments.get("summary", "")).strip()
    if not kind or not title or not summary:
        return AgentToolResult(
            success=False,
            content="record_task_memory requires kind, title, and summary",
        )
    raw_content = str(arguments.get("raw_content", summary))
    artifact_path = write_local_text_artifact(
        context=context,
        call_id=call_id,
        suffix=".memory-source.txt",
        text=raw_content,
    )
    dependency = context.dependency_builder.build(
        str(context.workspace_root),
        dependency_ids=(f"task:{context.task_payload.get('case_id', 'unknown')}",),
        input_digests=(title, summary),
    )
    evidence: tuple[EvidenceRef, ...] = ()
    artifact_id: str | None = None
    artifact_record = context.kernel.capture_artifact(
        raw_content.encode("utf-8"),
        content_type="text/plain",
        tool_name=context.tool_name_prefix,
        metadata={"title": title, "kind": kind, "call_id": call_id},
    )
    artifact_id = artifact_record.artifact_id
    evidence = (
        context.kernel.artifact_evidence(
            artifact_record,
            label="task-memory-source",
            summary="Captured agent task-memory source text",
        ),
    )

    payload: ClaimPayload | ConstraintPayload | DecisionPayload
    memory_kind: MemoryKind
    if kind == "claim":
        payload = ClaimPayload(claim=str(arguments.get("claim", summary)))
        memory_kind = MemoryKind.CLAIM
    elif kind == "constraint":
        payload = ConstraintPayload(
            statement=str(arguments.get("statement", summary)),
            severity=str(arguments.get("severity", "info")),
        )
        memory_kind = MemoryKind.CONSTRAINT
    elif kind == "decision":
        payload = DecisionPayload(
            summary=summary,
            rationale=str(arguments.get("rationale", "")) or None,
        )
        memory_kind = MemoryKind.DECISION
    else:
        return AgentToolResult(
            success=False,
            content=f"unsupported task memory kind: {kind}",
            artifact_path=artifact_path,
            artifact_id=artifact_id,
        )
    memory = MemoryItem(
        kind=memory_kind,
        title=title,
        summary=summary,
        payload=payload,
        evidence=evidence,
        tags=tuple(str(tag) for tag in _string_items(arguments.get("tags"))),
        visibility=context.task_scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=dependency,
            checked_at=utc_now(),
            reason="agent-recorded task memory",
        ),
    )
    tx = context.kernel.begin_transaction(context.task_scope, metadata={"call_id": call_id})
    context.kernel.stage_memory_item(tx.tx_id, memory)
    context.kernel.commit_transaction(tx.tx_id)
    return AgentToolResult(
        success=True,
        content=json.dumps(
            {"memory_id": memory.memory_id, "title": title, "kind": kind},
            sort_keys=True,
        ),
        metadata={"memory_id": memory.memory_id, "kind": kind},
        artifact_path=artifact_path,
        artifact_id=artifact_id,
    )


def _promote_procedure(
    arguments: dict[str, object],
    context: ToolExecutionContext,
    call_id: str,
) -> AgentToolResult:
    if context.kernel is None or context.durable_scope is None:
        return AgentToolResult(success=False, content="procedure promotion is not enabled")
    source_memory_ids = tuple(str(item) for item in _items(arguments.get("source_memory_ids")))
    title = str(arguments.get("title", "")).strip()
    summary = str(arguments.get("summary", "")).strip()
    goal = str(arguments.get("goal", "")).strip()
    raw_steps = _items(arguments.get("steps"))
    if not source_memory_ids or not title or not summary or not goal or not raw_steps:
        return AgentToolResult(
            success=False,
            content="promote_procedure requires sources, title, summary, goal, and steps",
        )
    steps = []
    for index, raw_step in enumerate(raw_steps):
        if isinstance(raw_step, dict):
            instruction = str(raw_step.get("instruction", "")).strip()
            expected_outcome = str(raw_step.get("expected_outcome", "")).strip() or None
        else:
            instruction = str(raw_step).strip()
            expected_outcome = None
        if not instruction:
            return AgentToolResult(
                success=False,
                content="procedure steps must contain a non-empty instruction",
            )
        steps.append(
            ProcedureStep(
                order=index,
                instruction=instruction,
                expected_outcome=expected_outcome,
            )
        )
    procedure = MemoryItem(
        kind=MemoryKind.PROCEDURE,
        title=title,
        summary=summary,
        payload=ProcedurePayload(goal=goal, steps=tuple(steps)),
        visibility=context.durable_scope,
    )
    promoted = context.kernel.promote_to_procedure(source_memory_ids, procedure)
    text = json.dumps(
        {"memory_id": promoted.memory_id, "source_memory_ids": list(source_memory_ids)},
        sort_keys=True,
    )
    artifact_path, artifact_id = write_text_artifact(
        context=context,
        call_id=call_id,
        suffix=".procedure.json",
        text=text,
        content_type="application/json",
        metadata={"memory_id": promoted.memory_id},
    )
    return AgentToolResult(
        success=True,
        content=text,
        metadata={"memory_id": promoted.memory_id},
        artifact_path=artifact_path,
        artifact_id=artifact_id,
    )


def _coerce_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError("expected an int-compatible value")


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def _string_items(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _items(value))


__all__ = [
    "build_promote_procedure_tool",
    "build_record_task_memory_tool",
    "build_retrieve_memory_tool",
]
