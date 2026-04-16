"""Writeback helpers for persisting successful vendored-RLM runs into VTM."""

from __future__ import annotations

import json
from pathlib import Path

from vtm.base import utc_now
from vtm.enums import MemoryKind, ValidityStatus
from vtm.memory_items import DecisionPayload, MemoryItem, ValidityState, VisibilityScope
from vtm.services import DependencyFingerprintBuilder, TransactionalMemoryKernel
from vtm_rlm.execution import VendoredRLMRunResult


def write_success_memory(
    *,
    kernel: TransactionalMemoryKernel | None,
    dependency_builder: DependencyFingerprintBuilder | None,
    workspace_root: Path,
    task_statement: str,
    case_id: str,
    scope: VisibilityScope | None,
    produced_patch_text: str,
    run_result: VendoredRLMRunResult,
) -> str | None:
    """Persist a verified decision memory for one successful RLM run."""
    if kernel is None or dependency_builder is None or scope is None:
        return None

    response_artifact = kernel.capture_artifact(
        run_result.response.encode("utf-8"),
        content_type="text/plain",
        tool_name="vendored-rlm",
        metadata={"case_id": case_id, "artifact": "final_response"},
    )
    completion_artifact = kernel.capture_artifact(
        json.dumps(
            {
                "response": run_result.response,
                "usage_summary": run_result.usage_summary,
                "metadata": run_result.metadata,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ).encode("utf-8"),
        content_type="application/json",
        tool_name="vendored-rlm",
        metadata={"case_id": case_id, "artifact": "completion"},
    )
    evidence = [
        kernel.artifact_evidence(
            response_artifact,
            label="rlm-final-response",
            summary="Captured final response from vendored RLM",
        ),
        kernel.artifact_evidence(
            completion_artifact,
            label="rlm-completion",
            summary="Captured completion payload and usage summary from vendored RLM",
        ),
    ]
    if produced_patch_text.strip():
        patch_artifact = kernel.capture_artifact(
            produced_patch_text.encode("utf-8"),
            content_type="text/x-diff",
            tool_name="vendored-rlm",
            metadata={"case_id": case_id, "artifact": "produced_patch"},
        )
        evidence.append(
            kernel.artifact_evidence(
                patch_artifact,
                label="rlm-produced-patch",
                summary="Patch produced by vendored RLM",
            )
        )

    dependency = dependency_builder.build(
        str(workspace_root),
        dependency_ids=(f"rlm-case:{case_id}",),
        input_digests=(task_statement, produced_patch_text),
    )
    summary = f"Successful vendored-RLM run for task: {task_statement}"
    memory = MemoryItem(
        kind=MemoryKind.DECISION,
        title=f"Vendored RLM result for {case_id}",
        summary=summary,
        payload=DecisionPayload(summary=summary, rationale=run_result.response),
        evidence=tuple(evidence),
        tags=("rlm", "execution", "successful_run"),
        visibility=scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=dependency,
            checked_at=utc_now(),
            reason="successful vendored-RLM execution with passing final verification",
        ),
        metadata={
            "case_id": case_id,
            "generated_by": "vendored_rlm",
            "usage_summary": run_result.usage_summary,
        },
    )
    tx = kernel.begin_transaction(scope, metadata={"origin": "vendored_rlm_writeback"})
    kernel.stage_memory_item(tx.tx_id, memory)
    kernel.commit_transaction(tx.tx_id)
    return memory.memory_id
