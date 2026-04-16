"""Records used by the native single-agent runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from vtm.base import VTMModel, utc_now


class AgentMode(StrEnum):
    """Execution policy preset for the native agent runtime."""

    INTERACTIVE_GUARDED = "interactive_guarded"
    BENCHMARK_AUTONOMOUS = "benchmark_autonomous"


class AgentRunStatus(StrEnum):
    """Terminal outcome for an agent run."""

    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    MAX_TOOL_FAILURES = "max_tool_failures"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"


AgentToolPolicy = Literal["full", "no_file_mutation"]


class AgentConversationMessage(VTMModel):
    """One message in the model-visible conversation transcript."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_name: str | None = None


class AgentToolSpec(VTMModel):
    """Tool schema exposed to the model."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AgentToolCall(VTMModel):
    """Single tool invocation requested by the model."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentModelTurnRequest(VTMModel):
    """Input passed to a model adapter for one agent turn."""

    mode: AgentMode
    prompt_profile: str
    workspace: str
    task_payload: dict[str, Any] = Field(default_factory=dict)
    messages: tuple[AgentConversationMessage, ...]
    tools: tuple[AgentToolSpec, ...]
    max_tool_calls: int = Field(default=4, ge=1, le=16)
    sampling_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    sampling_seed: int | None = None


class AgentModelTurnResponse(VTMModel):
    """Model response for one agent turn."""

    assistant_message: str | None = None
    tool_calls: tuple[AgentToolCall, ...] = Field(default_factory=tuple)
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(VTMModel):
    """Configuration and limits for a complete native-agent run."""

    session_id: str | None = None
    case_id: str
    task_file: str
    workspace: str
    model_id: str
    attempt_index: int = Field(default=1, ge=1)
    mode: AgentMode = AgentMode.BENCHMARK_AUTONOMOUS
    prompt_profile: str = "vtm-native-agent-v1"
    tool_policy: AgentToolPolicy = "full"
    task_payload: dict[str, Any] = Field(default_factory=dict)
    max_turns: int = Field(default=12, ge=1, le=128)
    max_tool_failures: int = Field(default=8, ge=1, le=128)
    max_runtime_seconds: int = Field(default=600, ge=1, le=7200)
    compaction_window: int = Field(default=10, ge=4, le=128)
    command_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    max_command_output_chars: int = Field(default=20000, ge=256, le=200000)
    sampling_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    sampling_seed: int | None = None
    parent_session_id: str | None = None


class AgentSessionRecord(VTMModel):
    """Durable metadata written once per agent session."""

    session_id: str
    case_id: str
    parent_session_id: str | None = None
    model_id: str
    mode: AgentMode
    workspace: str
    task_file: str
    prompt_profile: str
    tool_registry: tuple[str, ...] = Field(default_factory=tuple)
    started_at: str = Field(default_factory=lambda: utc_now().isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTurnRecord(VTMModel):
    """Durable summary of one completed turn."""

    turn_index: int = Field(ge=1)
    started_at: str
    completed_at: str
    prompt_chars: int = Field(ge=0)
    assistant_message: str | None = None
    tool_call_count: int = Field(default=0, ge=0)
    status: str


class ToolCallRecord(VTMModel):
    """Durable record of one tool invocation."""

    call_id: str
    turn_index: int = Field(ge=1)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    allowed: bool = True
    success: bool = True
    started_at: str
    completed_at: str
    output_summary: str | None = None
    result_artifact_path: str | None = None
    result_artifact_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompactionRecord(VTMModel):
    """Durable record of a transcript compaction event."""

    compaction_id: str
    turn_index: int = Field(ge=1)
    created_at: str
    trigger_message_count: int = Field(ge=0)
    dropped_message_count: int = Field(ge=0)
    kept_message_count: int = Field(ge=0)
    summary: str


class AgentRunResult(VTMModel):
    """Aggregate result and metrics for a completed agent run."""

    session_id: str
    status: AgentRunStatus
    model_id: str
    mode: AgentMode
    started_at: str
    completed_at: str
    final_message: str | None = None
    turn_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    tool_failure_count: int = Field(default=0, ge=0)
    terminal_command_count: int = Field(default=0, ge=0)
    command_timeout_count: int = Field(default=0, ge=0)
    compaction_count: int = Field(default=0, ge=0)
    test_iterations: int = Field(default=0, ge=0)
    first_passing_turn: int | None = Field(default=None, ge=1)
    memory_write_count: int = Field(default=0, ge=0)
    memory_promotion_count: int = Field(default=0, ge=0)
    guardrail_blocks: int = Field(default=0, ge=0)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentToolResult(VTMModel):
    """Normalized tool output returned to the runtime."""

    success: bool
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    artifact_id: str | None = None


class PromptPack(VTMModel):
    """System instructions and custom rules for the native agent."""

    profile_id: str = "vtm-native-agent-v1"
    system_instructions: str = (
        "You are a single-agent coding runtime operating in a local repository workspace."
    )
    custom_rules: tuple[str, ...] = Field(default_factory=tuple)


__all__ = [
    "AgentConversationMessage",
    "AgentMode",
    "AgentModelTurnRequest",
    "AgentModelTurnResponse",
    "AgentRunRequest",
    "AgentRunResult",
    "AgentRunStatus",
    "AgentSessionRecord",
    "AgentToolCall",
    "AgentToolResult",
    "AgentToolSpec",
    "AgentToolPolicy",
    "AgentTurnRecord",
    "CompactionRecord",
    "PromptPack",
    "ToolCallRecord",
]
