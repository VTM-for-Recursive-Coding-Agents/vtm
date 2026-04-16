"""Public typed contracts for benchmark task packs and execution traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from vtm.base import VTMModel

HarnessMemoryMode = Literal["no_memory", "lexical", "lexical_rlm_rerank", "embedding"]
HarnessCodingExecutor = Literal["external_command", "native_agent"]
HarnessEvaluationBackend = Literal["local_subprocess", "swebench_harness"]
HarnessExecutionStyle = Literal["mixed_patch", "shell_command"]
HarnessWorkspaceBackend = Literal["local_workspace", "docker_workspace"]


class TaskMemoryContextItem(VTMModel):
    """One retrieved memory entry embedded into a coding task pack."""

    memory_id: str
    title: str
    summary: str
    score: float
    status: str
    relative_path: str | None = None
    symbol: str | None = None
    slice_name: str | None = None
    raw_anchor_path: str | None = None


class HarnessTaskPack(VTMModel):
    """Self-contained task description consumed by coding executors."""

    case_id: str
    repo_name: str
    commit_pair_id: str
    evaluation_backend: HarnessEvaluationBackend
    instance_id: str | None = None
    dataset_name: str | None = None
    base_ref: str
    head_ref: str
    commit_pair_label: str | None = None
    task_statement: str
    problem_statement: str | None = None
    hints_text: str | None = None
    failing_tests: tuple[str, ...] = Field(default_factory=tuple)
    fail_to_pass_tests: tuple[str, ...] = Field(default_factory=tuple)
    pass_to_pass_tests: tuple[str, ...] = Field(default_factory=tuple)
    expected_changed_paths: tuple[str, ...] = Field(default_factory=tuple)
    touched_paths: tuple[str, ...] = Field(default_factory=tuple)
    retrieval_query: str | None = None
    test_command: tuple[str, ...] = Field(default_factory=tuple)
    target_patch_digest: str
    gold_test_patch_digest: str | None = None
    memory_mode: HarnessMemoryMode
    top_k: int = Field(ge=1, le=100)
    task_kind: str | None = None
    difficulty: str | None = None
    execution_style: HarnessExecutionStyle = "mixed_patch"
    memory_context: tuple[TaskMemoryContextItem, ...] = Field(default_factory=tuple)
    coding_executor: HarnessCodingExecutor


class ExecutorRequest(VTMModel):
    """Normalized executor input built from a harness task pack."""

    case_id: str
    task_file: str
    workspace: str
    artifact_root: str = ""
    coding_executor: HarnessCodingExecutor
    attempt_index: int = Field(default=1, ge=1)
    workspace_backend: HarnessWorkspaceBackend = "local_workspace"
    command: tuple[str, ...] = Field(default_factory=tuple)
    test_command: tuple[str, ...] = Field(default_factory=tuple)


class TraceManifest(VTMModel):
    """Paths to native-agent trace artifacts emitted during execution."""

    session: str
    turns_jsonl: str
    tool_calls_jsonl: str
    compactions_jsonl: str
    tool_results_dir: str


class ExecutorResult(VTMModel):
    """Normalized executor output used by coding-benchmark evaluation."""

    command: tuple[str, ...]
    command_exit_code: int | None
    command_stdout_path: str | None
    command_stderr_path: str | None
    attempt_index: int = Field(default=1, ge=1)
    command_timed_out: bool = False
    runtime_ms: float = 0.0
    workspace: str = ""
    task_file: str = ""
    test_command: tuple[str, ...] = Field(default_factory=tuple)
    test_exit_code: int | None = None
    test_stdout_path: str | None = None
    test_stderr_path: str | None = None
    final_verification_runtime_ms: float | None = None
    final_verification_timed_out: bool = False
    final_git_status_path: str | None = None
    command_events_path: str | None = None
    workspace_backend: HarnessWorkspaceBackend = "local_workspace"
    produced_patch_path: str | None = None
    produced_patch_digest: str | None = None
    produced_patch_text: str = ""
    produced_changed_paths: tuple[str, ...] = Field(default_factory=tuple)
    docker_image: str | None = None
    docker_container_id: str | None = None
    docker_container_name: str | None = None
    docker_network: Literal["none", "bridge"] | None = None
    trace_manifest: TraceManifest | None = None
    agent_metrics: dict[str, Any] = Field(default_factory=dict)
    agent_artifacts: dict[str, str] = Field(default_factory=dict)
    agent_metadata: dict[str, Any] = Field(default_factory=dict)
