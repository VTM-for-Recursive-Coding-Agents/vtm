from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from vtm.base import VTMModel
from vtm.enums import ValidityStatus

BenchmarkSuite = Literal["retrieval", "drift", "coding"]
BenchmarkMode = Literal["no_memory", "lexical", "lexical_rlm_rerank", "embedding"]
RepoSourceKind = Literal["git", "synthetic_python_smoke"]
CodingEvaluationBackend = Literal["local_subprocess", "swebench_harness"]
SWEbenchHarnessCacheLevel = Literal["none", "base", "env", "instance"]


class CommitPair(VTMModel):
    pair_id: str
    base_ref: str
    head_ref: str
    label: str | None = None
    description: str | None = None


class RepoSpec(VTMModel):
    repo_name: str
    source_kind: RepoSourceKind = "git"
    remote_url: str | None = None
    branch: str = "main"
    commit_pairs: tuple[CommitPair, ...]

    @model_validator(mode="after")
    def validate_source(self) -> RepoSpec:
        if self.source_kind == "git" and not self.remote_url:
            raise ValueError("git repo specs require remote_url")
        if self.source_kind != "git" and self.remote_url is not None:
            raise ValueError("synthetic repo specs must not set remote_url")
        pair_ids = [pair.pair_id for pair in self.commit_pairs]
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("repo commit_pairs require unique pair_id values")
        return self


class RetrievalCase(VTMModel):
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
    case_type: Literal["drift"] = "drift"
    case_id: str
    repo_name: str
    commit_pair_id: str
    memory_id: str
    relative_path: str
    symbol: str
    expected_status: ValidityStatus


class CodingTaskCase(VTMModel):
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
    test_command: tuple[str, ...] = Field(default_factory=tuple)
    target_patch: str | None = None
    gold_test_patch_digest: str | None = None
    task_kind: str | None = None
    difficulty: str | None = None

    @model_validator(mode="after")
    def populate_expected_changed_paths(self) -> CodingTaskCase:
        if not self.expected_changed_paths:
            object.__setattr__(self, "expected_changed_paths", self.touched_paths)
        return self


class BenchmarkManifest(VTMModel):
    manifest_id: str
    description: str | None = None
    repos: tuple[RepoSpec, ...]
    coding_tasks: tuple[CodingTaskCase, ...] = Field(default_factory=tuple)
    seed: int = 0

    @model_validator(mode="after")
    def validate_references(self) -> BenchmarkManifest:
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
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class BenchmarkRunConfig(VTMModel):
    manifest_path: str
    suite: BenchmarkSuite
    mode: BenchmarkMode = "lexical"
    output_dir: str
    top_k: int = Field(default=5, ge=1, le=100)
    max_cases: int | None = Field(default=None, ge=1)
    seed: int = 0
    repo_filters: tuple[str, ...] = Field(default_factory=tuple)
    pair_filters: tuple[str, ...] = Field(default_factory=tuple)
    executor_command: tuple[str, ...] = Field(default_factory=tuple)
    swebench_dataset_name: str | None = None
    swebench_harness_workers: int = Field(default=4, ge=0)
    swebench_harness_cache_level: SWEbenchHarnessCacheLevel = "env"
    swebench_harness_run_id: str | None = None


class BenchmarkCaseResult(VTMModel):
    suite: BenchmarkSuite
    mode: BenchmarkMode
    case_id: str
    repo_name: str
    commit_pair_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunResult(VTMModel):
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
