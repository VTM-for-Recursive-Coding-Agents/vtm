"""Typed manifest, config, and result records for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from vtm.agents.models import AgentMode
from vtm.base import VTMModel
from vtm.enums import ValidityStatus

BenchmarkSuite = Literal["retrieval", "drift", "coding"]
BenchmarkMode = Literal["no_memory", "lexical", "lexical_rlm_rerank", "embedding"]
RepoSourceKind = Literal["git", "synthetic_python_smoke", "synthetic_terminal_smoke"]
CodingEvaluationBackend = Literal["local_subprocess", "swebench_harness"]
CodingExecutor = Literal["external_command", "native_agent"]
SWEbenchHarnessCacheLevel = Literal["none", "base", "env", "instance"]


class CommitPair(VTMModel):
    """Pair of refs describing the base and target repository state."""

    pair_id: str
    base_ref: str
    head_ref: str
    label: str | None = None
    description: str | None = None


class RepoSpec(VTMModel):
    """Repository source plus the commit pairs evaluated from it."""

    repo_name: str
    source_kind: RepoSourceKind = "git"
    remote_url: str | None = None
    branch: str = "main"
    commit_pairs: tuple[CommitPair, ...]

    @model_validator(mode="after")
    def validate_source(self) -> RepoSpec:
        """Validate repo source invariants and commit-pair uniqueness."""
        if self.source_kind == "git" and not self.remote_url:
            raise ValueError("git repo specs require remote_url")
        if self.source_kind != "git" and self.remote_url is not None:
            raise ValueError("synthetic repo specs must not set remote_url")
        pair_ids = [pair.pair_id for pair in self.commit_pairs]
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("repo commit_pairs require unique pair_id values")
        return self


class RetrievalCase(VTMModel):
    """Benchmark case for retrieval evaluation against seeded memory."""

    case_type: Literal["retrieval"] = "retrieval"
    case_id: str
    repo_name: str
    commit_pair_id: str
    slice_name: str = "taskish_behavior"
    memory_id: str
    query: str
    expected_memory_ids: tuple[str, ...]
    relative_path: str
    symbol: str


class DriftCase(VTMModel):
    """Benchmark case for verification drift detection."""

    case_type: Literal["drift"] = "drift"
    case_id: str
    repo_name: str
    commit_pair_id: str
    memory_id: str
    relative_path: str
    symbol: str
    expected_status: ValidityStatus


class CodingTaskCase(VTMModel):
    """Benchmark case for code-change generation and validation."""

    case_type: Literal["coding_task"] = "coding_task"
    case_id: str
    repo_name: str
    commit_pair_id: str
    evaluation_backend: CodingEvaluationBackend = "local_subprocess"
    instance_id: str | None = None
    dataset_name: str | None = None
    task_statement: str
    problem_statement: str | None = None
    hints_text: str | None = None
    failing_tests: tuple[str, ...] = Field(default_factory=tuple)
    fail_to_pass_tests: tuple[str, ...] = Field(default_factory=tuple)
    pass_to_pass_tests: tuple[str, ...] = Field(default_factory=tuple)
    touched_paths: tuple[str, ...] = Field(default_factory=tuple)
    expected_changed_paths: tuple[str, ...] = Field(default_factory=tuple)
    retrieval_query: str | None = None
    test_command: tuple[str, ...] = Field(default_factory=tuple)
    target_patch: str | None = None
    gold_test_patch_digest: str | None = None
    task_kind: str | None = None
    difficulty: str | None = None

    @model_validator(mode="after")
    def populate_expected_changed_paths(self) -> CodingTaskCase:
        """Default expected changed paths to the touched-path list."""
        if not self.expected_changed_paths:
            object.__setattr__(self, "expected_changed_paths", self.touched_paths)
        return self


class BenchmarkManifest(VTMModel):
    """Benchmark corpus definition including repos and coding tasks."""

    manifest_id: str
    description: str | None = None
    repos: tuple[RepoSpec, ...]
    coding_tasks: tuple[CodingTaskCase, ...] = Field(default_factory=tuple)
    seed: int = 0

    @model_validator(mode="after")
    def validate_references(self) -> BenchmarkManifest:
        """Ensure coding tasks reference known repos and commit pairs."""
        repo_map = {repo.repo_name: repo for repo in self.repos}
        for task in self.coding_tasks:
            repo = repo_map.get(task.repo_name)
            if repo is None:
                raise ValueError(f"coding task references unknown repo: {task.repo_name}")
            if not any(pair.pair_id == task.commit_pair_id for pair in repo.commit_pairs):
                raise ValueError(
                    "coding task references unknown commit pair: "
                    f"{task.repo_name}:{task.commit_pair_id}"
                )
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> BenchmarkManifest:
        """Load a manifest from a JSON file on disk."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class BenchmarkRunConfig(VTMModel):
    """Configuration for executing one benchmark run."""

    manifest_path: str
    suite: BenchmarkSuite
    mode: BenchmarkMode = "lexical"
    output_dir: str
    top_k: int = Field(default=5, ge=1, le=100)
    max_cases: int | None = Field(default=None, ge=1)
    seed: int = 0
    repo_filters: tuple[str, ...] = Field(default_factory=tuple)
    pair_filters: tuple[str, ...] = Field(default_factory=tuple)
    coding_executor: CodingExecutor = "external_command"
    executor_command: tuple[str, ...] = Field(default_factory=tuple)
    attempt_count: int = Field(default=1, ge=1, le=32)
    pass_k_values: tuple[int, ...] = (1,)
    agent_model_id: str | None = None
    agent_mode: AgentMode = AgentMode.BENCHMARK_AUTONOMOUS
    agent_prompt_profile: str = "vtm-native-agent-v1"
    agent_max_turns: int = Field(default=12, ge=1, le=128)
    agent_max_tool_failures: int = Field(default=8, ge=1, le=128)
    agent_max_runtime_seconds: int = Field(default=600, ge=1, le=7200)
    agent_compaction_window: int = Field(default=10, ge=4, le=128)
    agent_command_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    agent_max_output_chars: int = Field(default=20000, ge=256, le=200000)
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent_seed_base: int | None = None
    swebench_dataset_name: str | None = None
    swebench_harness_workers: int = Field(default=4, ge=0)
    swebench_harness_cache_level: SWEbenchHarnessCacheLevel = "env"
    swebench_harness_run_id: str | None = None

    @model_validator(mode="after")
    def validate_attempt_controls(self) -> BenchmarkRunConfig:
        pass_k_values = tuple(int(value) for value in self.pass_k_values)
        if not pass_k_values:
            raise ValueError("pass_k_values must contain at least one k value")
        if any(value <= 0 for value in pass_k_values):
            raise ValueError("pass_k_values must contain only positive integers")
        if len(set(pass_k_values)) != len(pass_k_values):
            raise ValueError("pass_k_values must not contain duplicates")
        if any(value > self.attempt_count for value in pass_k_values):
            raise ValueError("pass_k_values must be less than or equal to attempt_count")
        if self.suite != "coding":
            if self.attempt_count != 1:
                raise ValueError("attempt_count > 1 is only supported for coding suites")
            if pass_k_values != (1,):
                raise ValueError("pass_k_values are only supported for coding suites")
        object.__setattr__(self, "pass_k_values", pass_k_values)
        return self


class BenchmarkCaseResult(VTMModel):
    """Per-case metrics and metadata emitted by a suite."""

    suite: BenchmarkSuite
    mode: BenchmarkMode
    case_id: str
    repo_name: str
    commit_pair_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkAttemptResult(VTMModel):
    suite: BenchmarkSuite
    mode: BenchmarkMode
    case_id: str
    repo_name: str
    commit_pair_id: str
    attempt_index: int = Field(ge=1)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunResult(VTMModel):
    """Aggregate benchmark run metadata, summary metrics, and artifacts."""

    run_id: str
    manifest_id: str
    manifest_digest: str
    suite: BenchmarkSuite
    mode: BenchmarkMode
    case_count: int = Field(ge=0)
    started_at: str
    completed_at: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
