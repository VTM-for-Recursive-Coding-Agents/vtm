"""Native single-agent runtime for local coding tasks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vtm.agents.compaction import ContextCompactor, DeterministicContextCompactor
from vtm.agents.models import (
    AgentConversationMessage,
    AgentMode,
    AgentModelTurnRequest,
    AgentRunRequest,
    AgentRunResult,
    AgentRunStatus,
    AgentSessionRecord,
    AgentTurnRecord,
    CompactionRecord,
    PromptPack,
    ToolCallRecord,
)
from vtm.agents.permissions import (
    BenchmarkAutonomousPermissionPolicy,
    InteractiveGuardedPermissionPolicy,
    ToolPermissionPolicy,
)
from vtm.agents.tools import BuiltInToolProvider, ToolExecutionContext, ToolProvider
from vtm.base import utc_now
from vtm.harness.workspace import WorkspaceDriver
from vtm.memory_items import VisibilityScope
from vtm.services import DependencyFingerprintBuilder, TransactionalMemoryKernel

if TYPE_CHECKING:
    from vtm.adapters.agent_model import AgentModelAdapter


@dataclass(frozen=True)
class AgentRuntimeContext:
    """Execution-time dependencies supplied to the native agent."""

    task_file: Path
    workspace_root: Path
    artifact_root: Path
    task_payload: dict[str, Any]
    test_command: tuple[str, ...]
    workspace_driver: WorkspaceDriver
    kernel: TransactionalMemoryKernel | None = None
    task_scope: VisibilityScope | None = None
    durable_scope: VisibilityScope | None = None
    dependency_builder: DependencyFingerprintBuilder | None = None


class TerminalCodingAgent:
    """Single-agent loop that alternates model turns with tool execution."""

    def __init__(
        self,
        *,
        model_adapter: AgentModelAdapter,
        prompt_pack: PromptPack | None = None,
        permission_policy: ToolPermissionPolicy | None = None,
        compactor: ContextCompactor | None = None,
        tool_provider: ToolProvider | None = None,
    ) -> None:
        """Create an agent with pluggable model, tools, policies, and compaction."""
        self._model_adapter = model_adapter
        self._prompt_pack = prompt_pack or PromptPack()
        self._permission_policy = permission_policy
        self._compactor = compactor or DeterministicContextCompactor()
        self._tool_provider = tool_provider or BuiltInToolProvider()

    def run(self, request: AgentRunRequest, context: AgentRuntimeContext) -> AgentRunResult:
        """Execute the full agent loop and write durable trace artifacts."""
        session_id = request.session_id or f"agent_{uuid4().hex}"
        context.artifact_root.mkdir(parents=True, exist_ok=True)
        tool_results_root = context.artifact_root / "tool-results"
        tool_results_root.mkdir(parents=True, exist_ok=True)
        policy = self._permission_policy or self._default_permission_policy(request.mode)
        tool_context = ToolExecutionContext(
            workspace_root=context.workspace_root,
            task_file=context.task_file,
            task_payload=context.task_payload,
            artifact_root=context.artifact_root,
            workspace_driver=context.workspace_driver,
            command_timeout_seconds=request.command_timeout_seconds,
            max_command_output_chars=request.max_command_output_chars,
            kernel=context.kernel,
            task_scope=context.task_scope,
            durable_scope=context.durable_scope,
            dependency_builder=context.dependency_builder,
        )
        tools = self._tool_provider.build_tools(tool_context)
        started = utc_now()
        messages = self._initial_messages(context)
        tool_call_count = 0
        tool_failure_count = 0
        terminal_command_count = 0
        command_timeout_count = 0
        test_iterations = 0
        first_passing_turn: int | None = None
        memory_write_count = 0
        memory_promotion_count = 0
        guardrail_blocks = 0
        turn_records: list[AgentTurnRecord] = []
        tool_call_records: list[ToolCallRecord] = []
        compaction_records: list[CompactionRecord] = []
        status = AgentRunStatus.COMPLETED
        final_message: str | None = None
        deadline = time.monotonic() + request.max_runtime_seconds
        tool_registry = tuple(sorted(tools))

        session_record = AgentSessionRecord(
            session_id=session_id,
            case_id=request.case_id,
            parent_session_id=request.parent_session_id,
            model_id=request.model_id,
            mode=request.mode,
            workspace=str(context.workspace_root),
            task_file=str(context.task_file),
            prompt_profile=request.prompt_profile,
            tool_registry=tool_registry,
            metadata={
                "attempt_index": request.attempt_index,
                "max_turns": request.max_turns,
                "max_tool_failures": request.max_tool_failures,
                "max_runtime_seconds": request.max_runtime_seconds,
                "compaction_window": request.compaction_window,
                "command_timeout_seconds": request.command_timeout_seconds,
                "max_command_output_chars": request.max_command_output_chars,
                "sampling_temperature": request.sampling_temperature,
                "sampling_seed": request.sampling_seed,
            },
        )
        self._write_model(context.artifact_root / "session.json", session_record)

        for turn_index in range(1, request.max_turns + 1):
            if time.monotonic() >= deadline:
                status = AgentRunStatus.TIMEOUT
                break
            compacted_messages, compaction = self._compactor.compact(
                messages=tuple(messages),
                turn_index=turn_index,
                window=request.compaction_window,
            )
            if compaction is not None:
                messages = list(compacted_messages)
                compaction_records.append(compaction)
            turn_started = utc_now()
            prompt_chars = sum(len(message.content) for message in messages)
            try:
                model_response = self._model_adapter.complete_turn(
                    AgentModelTurnRequest(
                        mode=request.mode,
                        prompt_profile=request.prompt_profile,
                        workspace=str(context.workspace_root),
                        task_payload=context.task_payload,
                        messages=tuple(messages),
                        tools=tuple(tool.spec for tool in tools.values()),
                        max_tool_calls=4,
                        sampling_temperature=request.sampling_temperature,
                        sampling_seed=request.sampling_seed,
                    )
                )
            except Exception as exc:
                status = AgentRunStatus.MODEL_ERROR
                final_message = str(exc)
                turn_records.append(
                    AgentTurnRecord(
                        turn_index=turn_index,
                        started_at=turn_started.isoformat(),
                        completed_at=utc_now().isoformat(),
                        prompt_chars=prompt_chars,
                        assistant_message=str(exc),
                        tool_call_count=0,
                        status="model_error",
                    )
                )
                break

            if model_response.assistant_message:
                messages.append(
                    AgentConversationMessage(
                        role="assistant",
                        content=model_response.assistant_message,
                    )
                )
                final_message = model_response.assistant_message

            for tool_offset, tool_call in enumerate(model_response.tool_calls, start=1):
                call_id = f"turn-{turn_index:04d}-call-{tool_offset:02d}"
                tool_call_count += 1
                started_at = utc_now()
                decision = policy.authorize(
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    workspace_root=context.workspace_root,
                )
                base_metadata = {
                    "command": None,
                    "duration_ms": None,
                    "exit_code": None,
                    "timed_out": False,
                    "truncated": False,
                }
                if not decision.allowed:
                    guardrail_blocks += 1
                    blocked_content = decision.reason or "tool call blocked by policy"
                    messages.append(
                        AgentConversationMessage(
                            role="tool",
                            tool_name=tool_call.tool_name,
                            content=blocked_content,
                        )
                    )
                    tool_call_records.append(
                        ToolCallRecord(
                            call_id=call_id,
                            turn_index=turn_index,
                            tool_name=tool_call.tool_name,
                            arguments=tool_call.arguments,
                            allowed=False,
                            success=False,
                            started_at=started_at.isoformat(),
                            completed_at=utc_now().isoformat(),
                            output_summary=blocked_content,
                            metadata={**base_metadata, "reason": decision.reason},
                        )
                    )
                    tool_failure_count += 1
                    continue

                tool = tools.get(tool_call.tool_name)
                if tool is None:
                    result_content = f"unknown tool: {tool_call.tool_name}"
                    result_success = False
                    result_artifact_path = None
                    result_artifact_id = None
                    result_metadata = dict(base_metadata)
                else:
                    result = tool.execute(tool_call.arguments, tool_context, call_id)
                    result_content = result.content
                    result_success = result.success
                    result_artifact_path = result.artifact_path
                    result_artifact_id = result.artifact_id
                    result_metadata = {**base_metadata, **result.metadata}
                    if tool_call.tool_name == "record_task_memory" and result.success:
                        memory_write_count += 1
                    if tool_call.tool_name == "promote_procedure" and result.success:
                        memory_promotion_count += 1
                    if tool_call.tool_name == "terminal":
                        terminal_command_count += 1
                        if result_metadata.get("timed_out"):
                            command_timeout_count += 1
                        command = str(tool_call.arguments.get("command", "")).strip()
                        expected_test = " ".join(context.test_command).strip()
                        if expected_test and command == expected_test:
                            test_iterations += 1
                            if (
                                result.metadata.get("exit_code") == 0
                                and first_passing_turn is None
                            ):
                                first_passing_turn = turn_index
                if not result_success:
                    tool_failure_count += 1
                messages.append(
                    AgentConversationMessage(
                        role="tool",
                        tool_name=tool_call.tool_name,
                        content=result_content,
                    )
                )
                tool_call_records.append(
                    ToolCallRecord(
                        call_id=call_id,
                        turn_index=turn_index,
                        tool_name=tool_call.tool_name,
                        arguments=tool_call.arguments,
                        allowed=True,
                        success=result_success,
                        started_at=started_at.isoformat(),
                        completed_at=utc_now().isoformat(),
                        output_summary=result_content[:240],
                        result_artifact_path=result_artifact_path,
                        result_artifact_id=result_artifact_id,
                        metadata=result_metadata,
                    )
                )
                if tool_failure_count >= request.max_tool_failures:
                    status = AgentRunStatus.MAX_TOOL_FAILURES
                    break
            turn_records.append(
                AgentTurnRecord(
                    turn_index=turn_index,
                    started_at=turn_started.isoformat(),
                    completed_at=utc_now().isoformat(),
                    prompt_chars=prompt_chars,
                    assistant_message=model_response.assistant_message,
                    tool_call_count=len(model_response.tool_calls),
                    status="completed" if status is AgentRunStatus.COMPLETED else status.value,
                )
            )
            if status is AgentRunStatus.MAX_TOOL_FAILURES:
                break
            if model_response.done:
                break
        else:
            status = AgentRunStatus.MAX_TURNS

        completed = utc_now()
        self._write_jsonl(context.artifact_root / "turns.jsonl", turn_records)
        self._write_jsonl(context.artifact_root / "tool_calls.jsonl", tool_call_records)
        self._write_jsonl(context.artifact_root / "compactions.jsonl", compaction_records)
        return AgentRunResult(
            session_id=session_id,
            status=status,
            model_id=request.model_id,
            mode=request.mode,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            final_message=final_message,
            turn_count=len(turn_records),
            tool_call_count=tool_call_count,
            tool_failure_count=tool_failure_count,
            terminal_command_count=terminal_command_count,
            command_timeout_count=command_timeout_count,
            compaction_count=len(compaction_records),
            test_iterations=test_iterations,
            first_passing_turn=first_passing_turn,
            memory_write_count=memory_write_count,
            memory_promotion_count=memory_promotion_count,
            guardrail_blocks=guardrail_blocks,
            artifacts={
                "session": str(context.artifact_root / "session.json"),
                "turns_jsonl": str(context.artifact_root / "turns.jsonl"),
                "tool_calls_jsonl": str(context.artifact_root / "tool_calls.jsonl"),
                "compactions_jsonl": str(context.artifact_root / "compactions.jsonl"),
                "tool_results_dir": str(tool_results_root),
            },
            metadata={"tool_registry": list(tool_registry)},
        )

    def _default_permission_policy(self, mode: AgentMode) -> ToolPermissionPolicy:
        if mode is AgentMode.INTERACTIVE_GUARDED:
            return InteractiveGuardedPermissionPolicy()
        return BenchmarkAutonomousPermissionPolicy()

    def _initial_messages(self, context: AgentRuntimeContext) -> list[AgentConversationMessage]:
        """Build the initial system and task-pack messages."""
        task_json = json.dumps(context.task_payload, indent=2, sort_keys=True)
        custom_rules = "\n".join(f"- {rule}" for rule in self._prompt_pack.custom_rules)
        return [
            AgentConversationMessage(
                role="system",
                content="\n".join(
                    part
                    for part in (
                        self._prompt_pack.system_instructions,
                        custom_rules,
                    )
                    if part
                ),
            ),
            AgentConversationMessage(
                role="user",
                content=f"Task pack:\n{task_json}",
            ),
        ]

    def _write_model(self, path: Path, model: Any) -> None:
        """Write a single durable model artifact."""
        path.write_text(model.to_json(), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[Any]) -> None:
        """Write a sequence of durable models as JSONL."""
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row.to_json())
                handle.write("\n")


__all__ = ["AgentRuntimeContext", "TerminalCodingAgent"]
