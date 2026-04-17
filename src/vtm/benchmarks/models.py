"""Typed manifest, config, and result records for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from vtm.base import VTMModel
from vtm.enums import ValidityStatus

BenchmarkSuite = Literal["retrieval", "drift", "coding"]
BenchmarkMode = Literal[
    "no_memory",
    "naive_lexical",
    "verified_lexical",
    "lexical_rlm_rerank",
]
RepoSourceKind = Literal["git", "synthetic_python_smoke"]
CodingEvaluationBackend = Literal["local_subprocess", "swebench_harness"]
CodingExecutionStyle = Literal["mixed_patch", "shell_command"]
CodingExecutionEngine = Literal["vendored_rlm"]
WorkspaceBackendName = Literal["local_workspace", "docker_workspace"]
DockerNetworkMode = Literal["none", "bridge"]
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
    expected_head_status: ValidityStatus | None = None
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
    verifier_output: str | None = None
    localization_notes: tuple[str, ...] = Field(default_factory=tuple)
    debug_expected_changed_paths: bool = False
    test_command: tuple[str, ...] = Field(default_factory=tuple)
    target_patch: str | None = None
    gold_test_patch_digest: str | None = None
    task_kind: str | None = None
    difficulty: str | None = None
    execution_style: CodingExecutionStyle = "mixed_patch"

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
    mode: BenchmarkMode = "verified_lexical"
    output_dir: str
    top_k: int = Field(default=5, ge=1, le=100)
    max_cases: int | None = Field(default=None, ge=1)
    seed: int = 0
    repo_filters: tuple[str, ...] = Field(default_factory=tuple)
    pair_filters: tuple[str, ...] = Field(default_factory=tuple)
    seed_on_base_query_on_head: bool = False
    workspace_backend: WorkspaceBackendName = "local_workspace"
    coding_engine: CodingExecutionEngine = "vendored_rlm"
    docker_image: str | None = None
    docker_binary: str = "docker"
    docker_network: DockerNetworkMode = "none"
    attempt_count: int = Field(default=1, ge=1, le=32)
    pass_k_values: tuple[int, ...] = (1,)
    rlm_model_id: str | None = None
    rlm_max_iterations: int = Field(default=12, ge=1, le=128)
    rlm_max_runtime_seconds: int = Field(default=600, ge=1, le=7200)
    workspace_command_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    workspace_max_output_chars: int = Field(default=20000, ge=256, le=200000)
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
        if self.seed_on_base_query_on_head and self.suite != "retrieval":
            raise ValueError(
                "seed_on_base_query_on_head is only supported for retrieval suites"
            )
        if self.suite != "coding":
            if self.attempt_count != 1:
                raise ValueError("attempt_count > 1 is only supported for coding suites")
            if pass_k_values != (1,):
                raise ValueError("pass_k_values are only supported for coding suites")
        if self.workspace_backend == "docker_workspace" and not self.docker_image:
            raise ValueError("docker_workspace requires docker_image")
        if self.workspace_backend == "local_workspace":
            if self.docker_image is not None:
                raise ValueError("docker_image is only supported with docker_workspace")
            if self.docker_binary != "docker":
                raise ValueError("docker_binary is only supported with docker_workspace")
            if self.docker_network != "none":
                raise ValueError("docker_network is only supported with docker_workspace")
        object.__setattr__(self, "pass_k_values", pass_k_values)
        return self


def resolved_benchmark_mode(mode: BenchmarkMode) -> BenchmarkMode:
    """Resolve deprecated aliases to the effective benchmark mode."""
    return mode


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
    """Per-attempt metrics and metadata for coding-suite runs."""

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


class BenchmarkComparisonResult(VTMModel):
    """Paired comparison metadata, metrics, and artifacts for two benchmark runs."""

    comparison_id: str
    suite: BenchmarkSuite
    baseline_run_id: str
    baseline_manifest_id: str
    baseline_mode: BenchmarkMode
    baseline_case_count: int = Field(ge=0)
    candidate_run_id: str
    candidate_manifest_id: str
    candidate_mode: BenchmarkMode
    candidate_case_count: int = Field(ge=0)
    common_case_count: int = Field(ge=0)
    baseline_only_case_count: int = Field(ge=0)
    candidate_only_case_count: int = Field(ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class BenchmarkMatrixResult(VTMModel):
    """Aggregate result for one maintained benchmark matrix execution."""

    matrix_id: str
    preset_name: str | None = None
    manifest_path: str
    suite: BenchmarkSuite
    output_dir: str
    baseline_mode: BenchmarkMode
    modes: tuple[BenchmarkMode, ...]
    run_results: dict[str, BenchmarkRunResult] = Field(default_factory=dict)
    comparison_results: dict[str, BenchmarkComparisonResult] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
