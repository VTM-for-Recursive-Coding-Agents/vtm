"""Coding benchmark suite execution and task evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtm.benchmarks.kernel_factory import BenchmarkKernelFactory
from vtm.benchmarks.models import (
    BenchmarkAttemptResult,
    BenchmarkCaseResult,
    BenchmarkManifest,
    BenchmarkMode,
    BenchmarkRunConfig,
    CodingTaskCase,
    CommitPair,
    RepoSpec,
    resolved_benchmark_mode,
)
from vtm.benchmarks.openrouter import execution_model, openrouter_api_key, openrouter_base_url
from vtm.benchmarks.repo_materialization import RepoWorkspaceManager
from vtm.benchmarks.swebench_harness import SWEbenchHarnessRunner
from vtm.benchmarks.symbol_index import SymbolIndexer
from vtm.enums import EvidenceBudget, EvidenceKind, MemoryKind, ScopeKind, ValidityStatus
from vtm.harness.executors import (
    RLMBenchmarkExecutor,
)
from vtm.harness.models import ExecutorRequest, HarnessTaskPack, TaskMemoryContextItem
from vtm.harness.scoring import changed_path_metrics, patch_similarity
from vtm.harness.workspace import (
    LocalWorkspaceBackend,
    PreparedWorkspace,
    WorkspaceBackend,
)
from vtm.harness.workspace_docker import DockerWorkspaceBackend
from vtm.memory_items import ClaimPayload, MemoryItem, ValidityState, VisibilityScope
from vtm.retrieval import RetrieveRequest
from vtm.services import TransactionalMemoryKernel
from vtm.stores import (
    FilesystemArtifactStore,
    SqliteCacheStore,
    SqliteMetadataStore,
)
from vtm_rlm.context import RLMRuntimeContext


@dataclass(frozen=True)
class PreparedCodingTask:
    """Canonical task-pack payload written once and reused across attempts."""

    task_pack: HarnessTaskPack
    task_file: Path
    expected_changed_paths: tuple[str, ...]
    touched_paths: tuple[str, ...]
    target_patch: str
    target_patch_digest: str
    memory_context: tuple[TaskMemoryContextItem, ...]
    context_chars: int
    retrieval_verified_count: int = 0
    retrieval_relocated_count: int = 0
    retrieval_stale_filtered_count: int = 0
    retrieval_stale_hit_rate: float = 0.0


@dataclass(frozen=True)
class VisibleTaskContext:
    """Visible, non-oracle task signals used for retrieval and prompting."""

    retrieval_query: str
    verifier_output: str | None
    localization_notes: tuple[str, ...]


def run_coding_suite(
    *,
    output_dir: Path,
    manifest: BenchmarkManifest,
    config: BenchmarkRunConfig,
    selected_repo_pairs: list[tuple[RepoSpec, CommitPair]],
    repo_manager: RepoWorkspaceManager,
    symbol_indexer: SymbolIndexer,
    kernel_factory: BenchmarkKernelFactory,
    require_pair: Callable[[RepoSpec, str], CommitPair],
    swebench_harness_runner: Any | None = None,
) -> tuple[list[CodingTaskCase], list[BenchmarkCaseResult], list[BenchmarkAttemptResult]]:
    """Run all selected coding-task cases and emit per-attempt plus aggregate rows."""
    allowed_pairs = {
        (repo_spec.repo_name, pair.pair_id)
        for repo_spec, pair in selected_repo_pairs
    }
    cases = [
        task
        for task in manifest.coding_tasks
        if not allowed_pairs or (task.repo_name, task.commit_pair_id) in allowed_pairs
    ]
    if config.max_cases is not None:
        cases = cases[: config.max_cases]

    repo_map = {repo.repo_name: repo for repo in manifest.repos}
    workspace_backend = build_workspace_backend(config)
    repo_roots: dict[str, Path] = {}
    attempt_results_by_case: dict[str, list[BenchmarkAttemptResult]] = {}
    for task in cases:
        repo_spec = repo_map[task.repo_name]
        pair = require_pair(repo_spec, task.commit_pair_id)
        repo_root = repo_roots.setdefault(
            repo_spec.repo_name,
            repo_manager.materialize_repo(repo_spec, output_dir / "corpus"),
        )
        prepared_task = prepare_coding_task(
            repo_root=repo_root,
            repo_spec=repo_spec,
            pair=pair,
            task=task,
            output_dir=output_dir,
            config=config,
            repo_manager=repo_manager,
            symbol_indexer=symbol_indexer,
            kernel_factory=kernel_factory,
        )
        attempt_results_by_case[task.case_id] = [
            evaluate_coding_attempt(
                repo_root=repo_root,
                repo_spec=repo_spec,
                pair=pair,
                task=task,
                prepared_task=prepared_task,
                output_dir=output_dir,
                config=config,
                workspace_backend=workspace_backend,
                kernel_factory=kernel_factory,
                attempt_index=attempt_index,
            )
            for attempt_index in range(1, config.attempt_count + 1)
        ]

    harness_cases = [task for task in cases if task.evaluation_backend == "swebench_harness"]
    if harness_cases:
        harness_runner = swebench_harness_runner or SWEbenchHarnessRunner()
        attempt_results_by_case = merge_swebench_attempt_results(
            cases=harness_cases,
            attempt_results_by_case=attempt_results_by_case,
            config=config,
            output_dir=output_dir,
            harness_runner=harness_runner,
        )

    results = [
        aggregate_attempt_results(
            task=task,
            attempts=attempt_results_by_case[task.case_id],
        )
        for task in cases
    ]
    attempt_results = [
        attempt
        for task in cases
        for attempt in attempt_results_by_case[task.case_id]
    ]
    return cases, results, attempt_results


def prepare_coding_task(
    *,
    repo_root: Path,
    repo_spec: RepoSpec,
    pair: CommitPair,
    task: CodingTaskCase,
    output_dir: Path,
    config: BenchmarkRunConfig,
    repo_manager: RepoWorkspaceManager,
    symbol_indexer: SymbolIndexer,
    kernel_factory: BenchmarkKernelFactory,
) -> PreparedCodingTask:
    """Build the canonical task pack for one coding case."""
    repo_manager.git_checkout(repo_root, pair.base_ref)
    memory_context: tuple[TaskMemoryContextItem, ...] = ()
    expected_changed_paths = task.expected_changed_paths or task.touched_paths
    visible_task_context = build_visible_task_context(
        repo_root=repo_root,
        task=task,
        timeout_seconds=config.workspace_command_timeout_seconds,
    )
    retrieval_verified_count = 0
    retrieval_relocated_count = 0
    retrieval_stale_filtered_count = 0
    retrieval_stale_hit_rate = 0.0
    kernel: TransactionalMemoryKernel | None = None
    metadata: SqliteMetadataStore | None = None
    artifacts: FilesystemArtifactStore | None = None
    cache: SqliteCacheStore | None = None
    durable_scope: VisibilityScope | None = None
    if config.mode != "no_memory":
        base_symbols = symbol_indexer.extract_symbols(repo_root)
        kernel, metadata, artifacts, cache, durable_scope = kernel_factory.open_kernel(
            repo_root=repo_root,
            repo_name=repo_spec.repo_name,
            pair=pair,
            output_dir=output_dir,
        )
        kernel_factory.seed_memories(
            kernel,
            repo_root,
            repo_spec.repo_name,
            pair,
            base_symbols,
            durable_scope,
        )
        current_dependency = benchmark_dependency_fingerprint(
            kernel_factory=kernel_factory,
            repo_root=repo_root,
            pair=pair,
        )
        seed_task_conditioned_memories(
            kernel=kernel,
            scope=durable_scope,
            task=task,
            current_dependency=current_dependency,
            visible_task_context=visible_task_context,
        )
        query = visible_task_context.retrieval_query
        retrieval_result = kernel.retrieve(
            coding_retrieve_request(
                mode=config.mode,
                query=query,
                scope=durable_scope,
                limit=config.top_k,
                current_dependency=current_dependency,
            )
        )
        memory_context = tuple(
            memory_context_payload(
                candidate.memory,
                candidate.score,
                candidate.explanation.metadata,
            )
            for candidate in retrieval_result.candidates
        )
        retrieval_verified_count = retrieval_result.verified_count
        retrieval_relocated_count = retrieval_result.relocated_count
        retrieval_stale_filtered_count = retrieval_result.stale_filtered_count
        retrieval_stale_hit_rate = retrieval_result.stale_hit_rate

    try:
        touched_paths = task.touched_paths or repo_manager.git_diff_paths(
            repo_root,
            pair.base_ref,
            pair.head_ref,
        )
        target_patch = task.target_patch or repo_manager.git_diff_text(
            repo_root,
            pair.base_ref,
            pair.head_ref,
        )
    finally:
        if metadata is not None and artifacts is not None and cache is not None:
            kernel_factory.close_kernel_stores(metadata, artifacts, cache)

    target_patch_digest = hashlib.sha256(target_patch.encode("utf-8")).hexdigest()
    task_pack = HarnessTaskPack(
        case_id=task.case_id,
        repo_name=task.repo_name,
        commit_pair_id=task.commit_pair_id,
        evaluation_backend=task.evaluation_backend,
        instance_id=task.instance_id,
        dataset_name=task.dataset_name,
        base_ref=pair.base_ref,
        head_ref=pair.head_ref,
        commit_pair_label=pair.label,
        task_statement=task.task_statement,
        problem_statement=task.problem_statement,
        hints_text=task.hints_text,
        failing_tests=task.failing_tests,
        fail_to_pass_tests=task.fail_to_pass_tests,
        pass_to_pass_tests=task.pass_to_pass_tests,
        expected_changed_paths=expected_changed_paths,
        touched_paths=touched_paths,
        retrieval_query=visible_task_context.retrieval_query,
        verifier_output=visible_task_context.verifier_output,
        localization_notes=visible_task_context.localization_notes,
        debug_expected_changed_paths=task.debug_expected_changed_paths,
        test_command=task.test_command,
        target_patch_digest=target_patch_digest,
        gold_test_patch_digest=task.gold_test_patch_digest,
        memory_mode=config.mode,
        top_k=config.top_k,
        task_kind=task.task_kind,
        difficulty=task.difficulty,
        execution_style=task.execution_style,
        memory_context=memory_context,
    )
    tasks_dir = output_dir / "task-packs"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / f"{task.case_id}.json"
    task_file.write_text(
        json.dumps(task_pack.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    context_chars = len(
        json.dumps(
            [item.model_dump(mode="json") for item in memory_context],
            sort_keys=True,
        )
    )
    return PreparedCodingTask(
        task_pack=task_pack,
        task_file=task_file,
        expected_changed_paths=expected_changed_paths,
        touched_paths=touched_paths,
        target_patch=target_patch,
        target_patch_digest=target_patch_digest,
        memory_context=memory_context,
        context_chars=context_chars,
        retrieval_verified_count=retrieval_verified_count,
        retrieval_relocated_count=retrieval_relocated_count,
        retrieval_stale_filtered_count=retrieval_stale_filtered_count,
        retrieval_stale_hit_rate=retrieval_stale_hit_rate,
    )


def coding_retrieve_request(
    *,
    mode: BenchmarkMode,
    query: str,
    scope: VisibilityScope,
    limit: int,
    current_dependency: Any | None,
) -> RetrieveRequest:
    """Build the retrieval request that matches the configured benchmark mode."""
    resolved_mode = resolved_benchmark_mode(mode)
    if resolved_mode in {"verified_lexical", "lexical_rlm_rerank"}:
        return RetrieveRequest(
            query=query,
            scopes=(scope,),
            statuses=tuple(ValidityStatus),
            evidence_budget=EvidenceBudget.SUMMARY_ONLY,
            limit=limit,
            current_dependency=current_dependency,
            verify_on_read=True,
            return_verified_only=True,
        )
    return RetrieveRequest(
        query=query,
        scopes=(scope,),
        evidence_budget=EvidenceBudget.SUMMARY_ONLY,
        limit=limit,
    )


def benchmark_dependency_fingerprint(
    *,
    kernel_factory: BenchmarkKernelFactory,
    repo_root: Path,
    pair: CommitPair,
) -> Any:
    """Build the benchmark dependency fingerprint for the current base checkout."""
    return kernel_factory.dependency_builder().build(
        str(repo_root),
        dependency_ids=(f"benchmark:{pair.pair_id}",),
        input_digests=(pair.base_ref,),
    )


def build_visible_task_context(
    *,
    repo_root: Path,
    task: CodingTaskCase,
    timeout_seconds: int,
) -> VisibleTaskContext:
    """Collect visible task signals without relying on oracle scoring fields."""
    verifier_output = collect_verifier_output(
        repo_root=repo_root,
        test_command=task.test_command,
        timeout_seconds=timeout_seconds,
    )
    localization_notes = tuple(
        dict.fromkeys(
            [
                *task.localization_notes,
                *infer_localization_notes(repo_root=repo_root, task=task),
            ]
        )
    )
    query = task.retrieval_query or " ".join(
        part
        for part in (
            _compact_visible_text(task.task_statement),
            _compact_visible_text(task.problem_statement),
            " ".join(task.failing_tests),
            " ".join(task.fail_to_pass_tests),
            " ".join(task.pass_to_pass_tests),
            " ".join(localization_notes),
            _compact_visible_text(verifier_output),
        )
        if part
    ).strip()
    return VisibleTaskContext(
        retrieval_query=query,
        verifier_output=verifier_output,
        localization_notes=localization_notes,
    )


def collect_verifier_output(
    *,
    repo_root: Path,
    test_command: tuple[str, ...],
    timeout_seconds: int,
) -> str | None:
    """Capture a compact verifier excerpt from the visible failing command."""
    if not test_command:
        return None
    try:
        completed = subprocess.run(
            list(test_command),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        signal = completed.stderr.strip() or completed.stdout.strip()
    except subprocess.TimeoutExpired as exc:
        signal = (
            _coerce_timeout_text(exc.stderr)
            or _coerce_timeout_text(exc.stdout)
            or "Verifier command timed out before producing output."
        )
    if not signal:
        return None
    return _clip_visible_text(signal, max_chars=700)


def infer_localization_notes(
    *,
    repo_root: Path,
    task: CodingTaskCase,
) -> tuple[str, ...]:
    """Infer deterministic localization notes from visible tests and imports."""
    notes: list[str] = []
    for test_name in (*task.fail_to_pass_tests, *task.failing_tests, *task.pass_to_pass_tests):
        test_path = resolve_test_path(repo_root=repo_root, test_name=test_name)
        if test_path is None:
            continue
        try:
            relative_test_path = test_path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        notes.append(f"Failing test file: {relative_test_path}")
        notes.extend(imported_module_notes(repo_root=repo_root, test_path=test_path))
    return tuple(dict.fromkeys(notes))


def resolve_test_path(*, repo_root: Path, test_name: str) -> Path | None:
    """Resolve a pytest-style or dotted unittest identifier to a repository file."""
    candidate = test_name.split("::", 1)[0].strip()
    direct_path = repo_root / candidate
    if candidate and direct_path.exists():
        return direct_path
    dotted = candidate.replace(".", "/")
    if dotted.endswith(".py"):
        dotted_path = repo_root / dotted
    else:
        dotted_path = repo_root / f"{dotted}.py"
    if candidate and dotted_path.exists():
        return dotted_path
    return None


def imported_module_notes(*, repo_root: Path, test_path: Path) -> tuple[str, ...]:
    """Extract simple source-file hints from test imports."""
    notes: list[str] = []
    for raw_line in test_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("from "):
            module_name = line[5:].split(" import ", 1)[0].strip()
        elif line.startswith("import "):
            module_name = line[7:].split(",", 1)[0].strip()
        else:
            continue
        if not module_name or module_name.startswith("tests"):
            continue
        module_path = repo_root / f"{module_name.replace('.', '/')}.py"
        if module_path.exists():
            relative_module_path = module_path.relative_to(repo_root).as_posix()
            notes.append(f"Referenced module from test import: {relative_module_path}")
    return tuple(dict.fromkeys(notes))


def seed_task_conditioned_memories(
    *,
    kernel: TransactionalMemoryKernel,
    scope: VisibilityScope,
    task: CodingTaskCase,
    current_dependency: Any,
    visible_task_context: VisibleTaskContext,
) -> None:
    """Seed compact task-conditioned memory cards from visible benchmark inputs."""
    cards = [
        (
            "problem",
            f"Task problem for {task.case_id}",
            "Visible issue summary derived from the task statement and problem statement.",
            " ".join(
                part
                for part in (
                    _compact_visible_text(task.task_statement),
                    _compact_visible_text(task.problem_statement),
                )
                if part
            ),
        ),
        (
            "tests",
            f"Failing tests for {task.case_id}",
            "Visible failing and stability tests attached to this task.",
            " ".join(
                part
                for part in (
                    f"Failing tests: {', '.join(task.fail_to_pass_tests or task.failing_tests)}"
                    if (task.fail_to_pass_tests or task.failing_tests)
                    else "",
                    f"Pass-to-pass tests: {', '.join(task.pass_to_pass_tests)}"
                    if task.pass_to_pass_tests
                    else "",
                )
                if part
            ),
        ),
        (
            "verifier",
            f"Verifier signal for {task.case_id}",
            "Visible verifier output captured from the failing task command.",
            visible_task_context.verifier_output or "",
        ),
        (
            "localization",
            f"Localization notes for {task.case_id}",
            "Deterministic localization notes inferred from visible test paths and imports.",
            " ".join(visible_task_context.localization_notes),
        ),
    ]
    tx = kernel.begin_transaction(scope)
    for card_kind, title, summary, claim in cards:
        cleaned_claim = claim.strip()
        if not cleaned_claim:
            continue
        artifact = kernel.capture_artifact(
            cleaned_claim.encode("utf-8"),
            content_type="text/plain",
            tool_name="benchmark-task-card",
            metadata={"case_id": task.case_id, "card_kind": card_kind},
        )
        kernel.stage_memory_item(
            tx.tx_id,
            MemoryItem(
                memory_id=task_memory_id(task.case_id, card_kind),
                kind=MemoryKind.CLAIM,
                title=title,
                summary=_clip_visible_text(summary, max_chars=180),
                payload=ClaimPayload(claim=_clip_visible_text(cleaned_claim, max_chars=700)),
                evidence=(
                    kernel.artifact_evidence(
                        artifact,
                        label=f"task-{card_kind}",
                        summary="Visible benchmark task signal",
                    ),
                ),
                tags=("benchmark_task", task.case_id, card_kind),
                visibility=scope,
                validity=ValidityState(
                    status=ValidityStatus.PENDING,
                    dependency_fingerprint=current_dependency,
                ),
                metadata={"generated_by": "benchmark_task_card"},
            ),
        )
    kernel.commit_transaction(tx.tx_id)


def task_memory_id(case_id: str, card_kind: str) -> str:
    """Return a deterministic id for a task-conditioned memory card."""
    digest = hashlib.sha256(f"{case_id}:{card_kind}".encode()).hexdigest()
    return f"task_{digest[:24]}"


def _compact_visible_text(text: str | None) -> str:
    if not text:
        return ""
    collapsed = " ".join(line.strip() for line in text.splitlines() if line.strip())
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return _clip_visible_text(collapsed, max_chars=280)


def _clip_visible_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _coerce_timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return value.strip()


def evaluate_coding_attempt(
    *,
    repo_root: Path,
    repo_spec: RepoSpec,
    pair: CommitPair,
    task: CodingTaskCase,
    prepared_task: PreparedCodingTask,
    output_dir: Path,
    config: BenchmarkRunConfig,
    workspace_backend: WorkspaceBackend,
    kernel_factory: BenchmarkKernelFactory,
    attempt_index: int,
) -> BenchmarkAttemptResult:
    """Execute one concrete attempt for a coding task."""
    executed = False
    passed = False
    resolved = False
    patch_similarity_score: float | None = None
    runtime_ms = 0.0
    produced_patch_nonempty = False
    executor_succeeded = False
    evaluated = False
    patch_applied = False
    changed_path_precision: float | None = None
    changed_path_recall: float | None = None
    changed_path_f1: float | None = None
    produced_changed_paths: tuple[str, ...] = ()
    produced_patch_text = ""
    testable = task.evaluation_backend == "swebench_harness" or bool(task.test_command)
    incomplete = not testable
    executor_metadata: dict[str, Any] = {}
    outcome = None
    infra_failure = False
    final_verification_passed: bool | None = None
    final_verification_runtime_ms: float | None = None
    prepared_workspace: PreparedWorkspace | None = None
    kernel: TransactionalMemoryKernel | None = None
    metadata: SqliteMetadataStore | None = None
    artifacts: FilesystemArtifactStore | None = None
    cache: SqliteCacheStore | None = None
    durable_scope: VisibilityScope | None = None

    try:
        prepared_workspace = workspace_backend.prepare_workspace(
            case_id=task.case_id,
            attempt_index=attempt_index,
            repo_root=repo_root,
            base_ref=pair.base_ref,
            output_root=output_dir,
            mode=config.mode,
            command_timeout_seconds=config.workspace_command_timeout_seconds,
            max_output_chars=config.workspace_max_output_chars,
        )
        executor_request = ExecutorRequest(
            case_id=task.case_id,
            task_file=str(prepared_task.task_file),
            workspace=str(prepared_workspace.workspace_root),
            artifact_root=str(prepared_workspace.artifact_root),
            attempt_index=prepared_workspace.attempt_index,
            workspace_backend=prepared_workspace.backend_name,
            test_command=task.test_command,
        )
        session_id = f"{task.case_id}-{config.mode}-attempt-{attempt_index:02d}"
        task_scope = VisibilityScope(kind=ScopeKind.TASK, scope_id=session_id)
        if config.mode != "no_memory":
            kernel, metadata, artifacts, cache, durable_scope = kernel_factory.open_kernel(
                repo_root=repo_root,
                repo_name=repo_spec.repo_name,
                pair=pair,
                output_dir=output_dir,
            )
        runtime_context = RLMRuntimeContext(
            kernel=kernel,
            task_scope=task_scope if kernel is not None else None,
            durable_scope=durable_scope,
            dependency_builder=(
                kernel_factory.dependency_builder() if kernel is not None else None
            ),
        )
        model_id = execution_model(config.rlm_model_id)
        outcome = RLMBenchmarkExecutor(
            model_id=model_id,
            base_url=openrouter_base_url(),
            api_key=openrouter_api_key(),
            max_iterations=config.rlm_max_iterations,
            max_timeout_seconds=config.rlm_max_runtime_seconds,
        ).execute(
            request=executor_request,
            prepared_workspace=prepared_workspace,
            runtime_context=runtime_context,
        )
        executed = True

        if outcome is not None:
            assert prepared_workspace is not None
            incomplete = (
                task.evaluation_backend == "local_subprocess"
                and not bool(task.test_command)
            )
            runtime_ms = outcome.runtime_ms
            executor_succeeded = outcome.command_exit_code == 0 and not outcome.command_timed_out
            produced_patch_nonempty = bool(outcome.produced_patch_text.strip())
            final_verification_runtime_ms = outcome.final_verification_runtime_ms
            if task.test_command:
                passed = outcome.test_exit_code == 0 and not outcome.final_verification_timed_out
                resolved = (
                    passed if task.evaluation_backend == "local_subprocess" else False
                )
                evaluated = True
                patch_applied = produced_patch_nonempty
                final_verification_passed = passed
            patch_similarity_score = patch_similarity(
                prepared_task.target_patch,
                outcome.produced_patch_text,
            )
            produced_changed_paths = outcome.produced_changed_paths
            produced_patch_text = outcome.produced_patch_text
            (
                changed_path_precision,
                changed_path_recall,
                changed_path_f1,
            ) = changed_path_metrics(
                expected_changed_paths=prepared_task.expected_changed_paths,
                produced_changed_paths=produced_changed_paths,
            )
            executor_metadata = {
                "execution_engine": config.coding_engine,
                "attempt_index": attempt_index,
                "artifact_root": str(prepared_workspace.artifact_root),
                "execution_style": task.execution_style,
                "executor_exit_code": outcome.command_exit_code,
                "executor_timed_out": outcome.command_timed_out,
                "executor_runtime_ms": outcome.runtime_ms,
                "executor_stdout_path": outcome.command_stdout_path,
                "executor_stderr_path": outcome.command_stderr_path,
                "workspace": outcome.workspace,
                "workspace_backend": outcome.workspace_backend,
                "task_file": outcome.task_file,
                "test_command": list(outcome.test_command),
                "test_exit_code": outcome.test_exit_code,
                "test_stdout_path": outcome.test_stdout_path,
                "test_stderr_path": outcome.test_stderr_path,
                "final_verification_stdout_path": outcome.test_stdout_path,
                "final_verification_stderr_path": outcome.test_stderr_path,
                "final_verification_runtime_ms": outcome.final_verification_runtime_ms,
                "final_verification_timed_out": outcome.final_verification_timed_out,
                "command_events_path": outcome.command_events_path,
                "final_git_status_path": outcome.final_git_status_path,
                "produced_patch_path": outcome.produced_patch_path,
                "produced_patch_digest": outcome.produced_patch_digest,
                "produced_changed_paths": list(outcome.produced_changed_paths),
                "produced_patch_text": outcome.produced_patch_text,
                "docker_image": outcome.docker_image,
                "docker_container_id": outcome.docker_container_id,
                "docker_container_name": outcome.docker_container_name,
                "docker_network": outcome.docker_network,
                **prepared_workspace.metadata,
                **outcome.agent_artifacts,
                **outcome.agent_metadata,
            }
        else:
            incomplete = True
            executor_metadata = {
                "execution_engine": config.coding_engine,
                "attempt_index": attempt_index,
                "execution_style": task.execution_style,
                "workspace_backend": config.workspace_backend,
            }
            if prepared_workspace is not None:
                executor_metadata["workspace"] = str(prepared_workspace.workspace_root)
                executor_metadata["artifact_root"] = str(prepared_workspace.artifact_root)
                executor_metadata.update(prepared_workspace.metadata)
    except Exception as exc:
        infra_failure = True
        incomplete = True
        executor_metadata = {
            "execution_engine": config.coding_engine,
            "attempt_index": attempt_index,
            "infra_error": str(exc),
            "execution_style": task.execution_style,
            "workspace_backend": config.workspace_backend,
        }
        if prepared_workspace is not None:
            executor_metadata["workspace"] = str(prepared_workspace.workspace_root)
            executor_metadata["artifact_root"] = str(prepared_workspace.artifact_root)
            executor_metadata.update(prepared_workspace.metadata)
    finally:
        if metadata is not None and artifacts is not None and cache is not None:
            kernel_factory.close_kernel_stores(metadata, artifacts, cache)
        if prepared_workspace is not None and outcome is None:
            prepared_workspace.driver.close()

    return BenchmarkAttemptResult(
        suite="coding",
        mode=config.mode,
        case_id=task.case_id,
        repo_name=task.repo_name,
        commit_pair_id=task.commit_pair_id,
        attempt_index=attempt_index,
        metrics={
            "executed": executed,
            "evaluated": evaluated,
            "passed": passed,
            "resolved": resolved,
            "testable": testable,
            "incomplete": incomplete,
            "executor_succeeded": executor_succeeded,
            "produced_patch_nonempty": produced_patch_nonempty,
            "patch_applied": patch_applied,
            "changed_path_precision": changed_path_precision,
            "changed_path_recall": changed_path_recall,
            "changed_path_f1": changed_path_f1,
            "final_verification_passed": final_verification_passed,
            "final_verification_runtime_ms": final_verification_runtime_ms,
            "infra_failure": infra_failure,
            "runtime_ms": runtime_ms,
            "patch_similarity": patch_similarity_score,
            "retrieval_usage_rate": 0.0 if not prepared_task.memory_context else 1.0,
            "verified_count": prepared_task.retrieval_verified_count,
            "relocated_count": prepared_task.retrieval_relocated_count,
            "stale_filtered_count": prepared_task.retrieval_stale_filtered_count,
            "stale_hit_rate": prepared_task.retrieval_stale_hit_rate,
            "context_chars": prepared_task.context_chars,
            **(outcome.agent_metrics if outcome is not None else {}),
        },
        metadata={
            "task_file": str(prepared_task.task_file),
            "touched_paths": list(prepared_task.touched_paths),
            "expected_changed_paths": list(prepared_task.expected_changed_paths),
            "target_patch_digest": prepared_task.target_patch_digest,
            "memory_context_count": len(prepared_task.memory_context),
            "memory_mode": config.mode,
            "top_k": config.top_k,
            "task_kind": task.task_kind,
            "difficulty": task.difficulty,
            "execution_style": task.execution_style,
            "evaluation_backend": task.evaluation_backend,
            "instance_id": task.instance_id,
            "dataset_name": task.dataset_name,
            "problem_statement": task.problem_statement,
            "hints_text": task.hints_text,
            "fail_to_pass_tests": list(task.fail_to_pass_tests),
            "pass_to_pass_tests": list(task.pass_to_pass_tests),
            "verifier_output": prepared_task.task_pack.verifier_output,
            "localization_notes": list(prepared_task.task_pack.localization_notes),
            "gold_test_patch_digest": task.gold_test_patch_digest,
            "retrieval_query": prepared_task.task_pack.retrieval_query,
            "base_ref": pair.base_ref,
            "head_ref": pair.head_ref,
            "commit_pair_label": pair.label,
            "produced_changed_paths": list(produced_changed_paths),
            "produced_patch_text": produced_patch_text,
            "execution_engine": config.coding_engine,
            **executor_metadata,
        },
    )


def merge_swebench_attempt_results(
    *,
    cases: list[CodingTaskCase],
    attempt_results_by_case: dict[str, list[BenchmarkAttemptResult]],
    config: BenchmarkRunConfig,
    output_dir: Path,
    harness_runner: SWEbenchHarnessRunner,
) -> dict[str, list[BenchmarkAttemptResult]]:
    """Overlay official SWE-bench harness outcomes onto attempt rows."""
    case_map = {case.case_id: case for case in cases}
    for attempt_index in range(1, config.attempt_count + 1):
        attempt_results = [
            next(
                attempt
                for attempt in attempt_results_by_case[case.case_id]
                if attempt.attempt_index == attempt_index
            )
            for case in cases
        ]
        normalized_results, _ = harness_runner.evaluate_predictions(
            cases=cases,
            results=[
                BenchmarkCaseResult(
                    suite=attempt.suite,
                    mode=attempt.mode,
                    case_id=attempt.case_id,
                    repo_name=attempt.repo_name,
                    commit_pair_id=attempt.commit_pair_id,
                    metrics=dict(attempt.metrics),
                    metadata=dict(attempt.metadata),
                )
                for attempt in attempt_results
            ],
            config=config,
            output_dir=harness_attempt_output_dir(
                output_dir=output_dir,
                attempt_index=attempt_index,
                attempt_count=config.attempt_count,
            ),
        )
        updated_attempts: list[BenchmarkAttemptResult] = []
        for attempt in attempt_results:
            case = case_map[attempt.case_id]
            harness = normalized_results[case.instance_id or case.case_id]
            updated_attempts.append(
                attempt.model_copy(
                    update={
                        "metrics": {
                            **attempt.metrics,
                            "evaluated": True,
                            "passed": harness.resolved,
                            "resolved": harness.resolved,
                            "incomplete": False,
                            "patch_applied": harness.patch_applied,
                        },
                        "metadata": {
                            **attempt.metadata,
                            "harness_status": harness.harness_status,
                            "evaluation_log_path": harness.evaluation_log_path,
                        },
                    }
                )
            )
        for attempt in updated_attempts:
            case_attempts = attempt_results_by_case[attempt.case_id]
            attempt_results_by_case[attempt.case_id] = [
                attempt if item.attempt_index == attempt_index else item
                for item in case_attempts
            ]
    return attempt_results_by_case


def aggregate_attempt_results(
    *,
    task: CodingTaskCase,
    attempts: list[BenchmarkAttemptResult],
) -> BenchmarkCaseResult:
    """Collapse many attempt rows into the single aggregate case row."""
    if not attempts:
        raise ValueError(f"missing attempt rows for case {task.case_id}")

    best_attempt = select_best_attempt(attempts)
    any_executed = any(bool(attempt.metrics.get("executed", False)) for attempt in attempts)
    successful_attempt_indices = [
        attempt.attempt_index
        for attempt in attempts
        if bool(attempt.metrics.get("passed", False))
        or bool(attempt.metrics.get("resolved", False))
    ]
    aggregate_metrics = {
        **best_attempt.metrics,
        "executed": any_executed,
        "evaluated": any(bool(attempt.metrics.get("evaluated", False)) for attempt in attempts),
        "passed": any(bool(attempt.metrics.get("passed", False)) for attempt in attempts),
        "resolved": any(bool(attempt.metrics.get("resolved", False)) for attempt in attempts),
        "testable": any(bool(attempt.metrics.get("testable", False)) for attempt in attempts),
        "incomplete": all(bool(attempt.metrics.get("incomplete", False)) for attempt in attempts),
        "executor_succeeded": any(
            bool(attempt.metrics.get("executor_succeeded", False)) for attempt in attempts
        ),
        "produced_patch_nonempty": any(
            bool(attempt.metrics.get("produced_patch_nonempty", False)) for attempt in attempts
        ),
        "patch_applied": any(
            bool(attempt.metrics.get("patch_applied", False)) for attempt in attempts
        ),
        "infra_failure": all(
            bool(attempt.metrics.get("infra_failure", False)) for attempt in attempts
        ),
    }
    aggregate_metadata = {
        **best_attempt.metadata,
        "attempt_count": len(attempts),
        "successful_attempt_indices": successful_attempt_indices,
        "best_attempt_index": best_attempt.attempt_index if any_executed else None,
        "attempt_artifact_roots": [
            str(attempt.metadata["artifact_root"])
            for attempt in attempts
            if attempt.metadata.get("artifact_root") is not None
        ],
    }
    return BenchmarkCaseResult(
        suite=best_attempt.suite,
        mode=best_attempt.mode,
        case_id=best_attempt.case_id,
        repo_name=best_attempt.repo_name,
        commit_pair_id=best_attempt.commit_pair_id,
        metrics=aggregate_metrics,
        metadata=aggregate_metadata,
    )
def build_workspace_backend(config: BenchmarkRunConfig) -> WorkspaceBackend:
    """Construct the configured workspace backend for coding execution."""
    if config.workspace_backend == "docker_workspace":
        return DockerWorkspaceBackend(
            docker_binary=config.docker_binary,
            docker_image=config.docker_image or "",
            docker_network=config.docker_network,
        )
    return LocalWorkspaceBackend()


def harness_attempt_output_dir(
    *,
    output_dir: Path,
    attempt_index: int,
    attempt_count: int,
) -> Path:
    """Keep single-attempt harness output stable while isolating repeated attempts."""
    if attempt_count == 1:
        return output_dir
    return output_dir / "swebench-harness" / f"attempt-{attempt_index:02d}"


def select_best_attempt(attempts: list[BenchmarkAttemptResult]) -> BenchmarkAttemptResult:
    """Pick the best attempt, preferring resolved and earlier successful runs."""
    return max(attempts, key=best_attempt_sort_key)


def best_attempt_sort_key(attempt: BenchmarkAttemptResult) -> tuple[float, ...]:
    """Rank attempts by outcome quality and then by earliest attempt index."""
    metrics = attempt.metrics
    return (
        1.0 if bool(metrics.get("resolved", False)) else 0.0,
        1.0 if bool(metrics.get("passed", False)) else 0.0,
        1.0 if bool(metrics.get("patch_applied", False)) else 0.0,
        1.0 if bool(metrics.get("executor_succeeded", False)) else 0.0,
        float(metrics.get("patch_similarity") or 0.0),
        float(metrics.get("changed_path_f1") or 0.0),
        float(-attempt.attempt_index),
    )


def memory_context_payload(
    memory: MemoryItem,
    score: float,
    explanation_metadata: dict[str, Any],
) -> TaskMemoryContextItem:
    """Convert a retrieved memory into the normalized task-pack context item."""
    anchor = first_anchor(memory)
    return TaskMemoryContextItem(
        memory_id=memory.memory_id,
        title=memory.title,
        summary=memory.summary,
        score=score,
        status=memory.validity.status.value,
        relative_path=anchor.path if anchor is not None else None,
        symbol=anchor.symbol if anchor is not None else None,
        slice_name=str(explanation_metadata.get("slice_name"))
        if explanation_metadata.get("slice_name") is not None
        else None,
        raw_anchor_path=anchor.path if anchor is not None else None,
    )


def first_anchor(memory: MemoryItem) -> Any | None:
    """Return the first code-anchor evidence attached to a memory item."""
    for evidence in memory.evidence:
        if evidence.kind is EvidenceKind.CODE_ANCHOR and evidence.code_anchor is not None:
            return evidence.code_anchor
    return None


__all__ = ["run_coding_suite"]
