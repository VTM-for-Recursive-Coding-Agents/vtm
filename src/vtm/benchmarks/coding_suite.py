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
from vtm.benchmarks.symbol_index import SymbolIndexer, SymbolSnapshot
from vtm.enums import EvidenceBudget, EvidenceKind, MemoryKind, ValidityStatus
from vtm.harness.executors import DSPyReActBenchmarkExecutor, ExecutorMemoryRuntime
from vtm.harness.models import ExecutorRequest, HarnessTaskPack, TaskMemoryContextItem
from vtm.harness.scoring import changed_path_metrics, patch_similarity
from vtm.harness.workspace import (
    LocalWorkspaceBackend,
    PreparedWorkspace,
    WorkspaceBackend,
)
from vtm.harness.workspace_docker import DockerWorkspaceBackend
from vtm.memory_items import (
    ClaimPayload,
    CommandValidatorConfig,
    ConstraintPayload,
    DecisionPayload,
    MemoryItem,
    ProcedurePayload,
    ProcedureStep,
    ValidatorSpec,
    ValidityState,
    VisibilityScope,
)
from vtm.retrieval import RetrieveCandidate, RetrieveExplanation, RetrieveRequest
from vtm.services import TransactionalMemoryKernel
from vtm.stores import (
    FilesystemArtifactStore,
    SqliteCacheStore,
    SqliteMetadataStore,
)

PATH_HINT_RE = re.compile(r"[A-Za-z0-9_./-]+\.py")
IDENTIFIER_HINT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
FAILURE_SIGNATURE_RE = re.compile(
    r"(traceback|error|exception|failed|assert|cannot import name|no module named|not found)",
    re.IGNORECASE,
)


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
    visible_task_context: VisibleTaskContext | None = None
    seed_symbols: dict[tuple[str, str], SymbolSnapshot] | None = None
    memory_seed_ref: str | None = None


@dataclass(frozen=True)
class AttemptTaskMaterialization:
    """Attempt-local task-pack payload and retrieval metadata."""

    task_pack: HarnessTaskPack
    task_file: Path
    memory_context: tuple[TaskMemoryContextItem, ...]
    context_chars: int
    retrieval_verified_count: int = 0
    retrieval_relocated_count: int = 0
    retrieval_stale_filtered_count: int = 0
    retrieval_stale_hit_rate: float = 0.0
    failure_handoff_count: int = 0


@dataclass(frozen=True)
class VisibleTaskContext:
    """Visible, non-oracle task signals used for retrieval and prompting."""

    retrieval_query: str
    retrieval_query_parts: tuple[str, ...]
    verifier_output: str | None
    localization_notes: tuple[str, ...]
    path_hints: tuple[str, ...] = ()
    module_hints: tuple[str, ...] = ()
    symbol_hints: tuple[str, ...] = ()
    failure_signatures: tuple[str, ...] = ()


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
        case_attempts: list[BenchmarkAttemptResult] = []
        for attempt_index in range(1, config.attempt_count + 1):
            case_attempts.append(
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
                    previous_attempts=tuple(case_attempts),
                )
            )
        attempt_results_by_case[task.case_id] = case_attempts

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
    memory_seed_ref = task.memory_seed_ref or pair.base_ref
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
        repo_manager.git_checkout(repo_root, memory_seed_ref)
        seed_symbols = symbol_indexer.extract_symbols(repo_root)
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
            seed_symbols,
            durable_scope,
            dependency_ref=memory_seed_ref,
        )
        repo_manager.git_checkout(repo_root, pair.base_ref)
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
                limit=coding_candidate_pool_limit(config.top_k),
                current_dependency=current_dependency,
            )
        )
        selected_candidates = rerank_coding_candidates(
            retrieval_result.candidates,
            visible_task_context=visible_task_context,
            top_k=config.top_k,
        )
        memory_context = tuple(
            memory_context_payload(
                candidate.memory,
                candidate.score,
                candidate.explanation,
            )
            for candidate in selected_candidates
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
        visible_task_context=visible_task_context,
        seed_symbols=seed_symbols if config.mode != "no_memory" else None,
        memory_seed_ref=memory_seed_ref,
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
    if resolved_mode == "verified_lexical":
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


def coding_candidate_pool_limit(top_k: int) -> int:
    """Expand the lexical candidate pool before task-specific reranking."""
    return min(25, max(10, top_k * 3))


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
    path_hints = _extract_path_hints(
        repo_root=repo_root,
        texts=(
            task.task_statement,
            task.problem_statement,
            task.hints_text,
            *task.failing_tests,
            *task.fail_to_pass_tests,
            *task.pass_to_pass_tests,
            *localization_notes,
            verifier_output,
        ),
    )
    module_hints = _extract_module_hints(path_hints)
    symbol_hints = _extract_symbol_hints(
        task=task,
        localization_notes=localization_notes,
        verifier_output=verifier_output,
    )
    failure_signatures = _extract_failure_signatures(verifier_output)
    query_parts = _unique_visible_parts(
        task.retrieval_query,
        _compact_visible_text(task.task_statement),
        _compact_visible_text(task.problem_statement),
        _compact_visible_text(task.hints_text),
        " ".join(task.failing_tests),
        " ".join(task.fail_to_pass_tests),
        " ".join(task.pass_to_pass_tests),
        " ".join(localization_notes),
        " ".join(path_hints),
        " ".join(module_hints),
        " ".join(symbol_hints),
        " ".join(failure_signatures),
        _compact_visible_text(verifier_output),
    )
    query = " ".join(query_parts).strip()
    return VisibleTaskContext(
        retrieval_query=query,
        retrieval_query_parts=query_parts,
        verifier_output=verifier_output,
        localization_notes=localization_notes,
        path_hints=path_hints,
        module_hints=module_hints,
        symbol_hints=symbol_hints,
        failure_signatures=failure_signatures,
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
    dotted_candidates = [candidate]
    if "." in candidate:
        segments = candidate.split(".")
        dotted_candidates.extend(
            ".".join(segments[:end])
            for end in range(len(segments) - 1, 0, -1)
        )
    for dotted_candidate in dict.fromkeys(
        option for option in dotted_candidates if option and "/" not in option
    ):
        dotted = dotted_candidate.replace(".", "/")
        dotted_path = repo_root / (dotted if dotted.endswith(".py") else f"{dotted}.py")
        if dotted_path.exists():
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
            _render_visible_card(
                f"Task statement: {_compact_visible_text(task.task_statement)}",
                (
                    f"Problem statement: {_compact_visible_text(task.problem_statement)}"
                    if task.problem_statement
                    else ""
                ),
                f"Hints: {_compact_visible_text(task.hints_text)}" if task.hints_text else "",
                _render_hint_line("Candidate symbols", visible_task_context.symbol_hints),
            ),
        ),
        (
            "tests",
            f"Failing tests for {task.case_id}",
            "Visible failing and stability tests attached to this task.",
            _render_visible_card(
                (
                    f"Failing tests: {', '.join(task.fail_to_pass_tests or task.failing_tests)}"
                    if (task.fail_to_pass_tests or task.failing_tests)
                    else ""
                ),
                (
                    f"Pass-to-pass tests: {', '.join(task.pass_to_pass_tests)}"
                    if task.pass_to_pass_tests
                    else ""
                ),
                _render_hint_line("Candidate paths", visible_task_context.path_hints),
                _render_hint_line("Candidate modules", visible_task_context.module_hints),
            ),
        ),
        (
            "verifier",
            f"Verifier signal for {task.case_id}",
            "Visible verifier output captured from the failing task command.",
            _render_visible_card(
                _render_hint_line("Failure signatures", visible_task_context.failure_signatures),
                visible_task_context.verifier_output or "",
            ),
        ),
        (
            "localization",
            f"Localization notes for {task.case_id}",
            "Deterministic localization notes inferred from visible test paths and imports.",
            _render_visible_card(
                "\n".join(visible_task_context.localization_notes),
                _render_hint_line("Candidate paths", visible_task_context.path_hints),
                _render_hint_line("Candidate modules", visible_task_context.module_hints),
                _render_hint_line("Candidate symbols", visible_task_context.symbol_hints),
            ),
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
    if task.seed_validation_procedure_memory and task.test_command:
        instruction = f"Run {' '.join(task.test_command)} from the repository root."
        validation_procedure_text = _build_validation_procedure_evidence_text(
            task=task,
            instruction=instruction,
        )
        validation_artifact = kernel.capture_artifact(
            validation_procedure_text.encode("utf-8"),
            content_type="text/plain",
            tool_name="benchmark-task-validation-procedure",
            metadata={"case_id": task.case_id, "card_kind": "validation_procedure"},
        )
        kernel.stage_memory_item(
            tx.tx_id,
            MemoryItem(
                memory_id=task_memory_id(task.case_id, "validation_procedure"),
                kind=MemoryKind.PROCEDURE,
                title=f"Validation procedure for {task.case_id}",
                summary="Current maintained validation command for this coding task.",
                payload=ProcedurePayload(
                    goal="Validate the repository change with the maintained benchmark command.",
                    steps=(
                        ProcedureStep(
                            order=0,
                            instruction=instruction,
                            expected_outcome="The task test command exits with status 0.",
                        ),
                    ),
                    validator=ValidatorSpec(
                        name="benchmark-task-validation",
                        kind="command",
                        config=CommandValidatorConfig(
                            command=task.test_command,
                            cwd=".",
                            expected_exit_code=0,
                            restrict_cwd_to_repo=True,
                        ),
                    ),
                ),
                evidence=(
                    kernel.artifact_evidence(
                        validation_artifact,
                        label="task-validation-procedure",
                        summary="Benchmark-visible validation command and failing tests",
                    ),
                ),
                tags=("benchmark_task", task.case_id, "validation_procedure"),
                visibility=scope,
                validity=ValidityState(
                    status=ValidityStatus.VERIFIED,
                    dependency_fingerprint=current_dependency,
                ),
                metadata={"generated_by": "benchmark_task_validation_procedure"},
            ),
        )
    kernel.commit_transaction(tx.tx_id)


def task_memory_id(case_id: str, card_kind: str) -> str:
    """Return a deterministic id for a task-conditioned memory card."""
    digest = hashlib.sha256(f"{case_id}:{card_kind}".encode()).hexdigest()
    return f"task_{digest[:24]}"


def _build_validation_procedure_evidence_text(
    *,
    task: CodingTaskCase,
    instruction: str,
) -> str:
    failing_tests = task.fail_to_pass_tests or task.failing_tests
    lines = [
        f"Task statement: {task.task_statement}",
        f"Instruction: {instruction}",
        f"Validation command: {' '.join(task.test_command)}",
    ]
    if failing_tests:
        lines.append(f"Failing tests: {', '.join(failing_tests)}")
    if task.pass_to_pass_tests:
        lines.append(f"Pass-to-pass tests: {', '.join(task.pass_to_pass_tests)}")
    return "\n".join(lines)


def _compact_visible_text(text: str | None) -> str:
    if not text:
        return ""
    collapsed = " ".join(line.strip() for line in text.splitlines() if line.strip())
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return _clip_visible_text(collapsed, max_chars=280)


def _unique_visible_parts(*parts: str | None) -> tuple[str, ...]:
    unique_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _compact_visible_text(part)
        if not cleaned or cleaned in seen:
            continue
        unique_parts.append(cleaned)
        seen.add(cleaned)
    return tuple(unique_parts)


def _render_visible_card(*parts: str | None) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _render_hint_line(label: str, values: tuple[str, ...]) -> str:
    if not values:
        return ""
    preview = ", ".join(values[:6])
    if len(values) > 6:
        preview = f"{preview}, ..."
    return f"{label}: {preview}"


def _extract_path_hints(
    *,
    repo_root: Path,
    texts: tuple[str | None, ...],
) -> tuple[str, ...]:
    hints: list[str] = []
    for text in texts:
        if not text:
            continue
        for raw_match in PATH_HINT_RE.findall(text):
            candidate = raw_match.strip()
            if not candidate:
                continue
            raw_path = Path(candidate)
            if raw_path.is_absolute():
                try:
                    normalized = raw_path.relative_to(repo_root).as_posix()
                except ValueError:
                    normalized = raw_path.as_posix()
            else:
                normalized = candidate.removeprefix("./")
            hints.append(normalized)
    return tuple(dict.fromkeys(hints))


def _extract_module_hints(path_hints: tuple[str, ...]) -> tuple[str, ...]:
    hints: list[str] = []
    for path_hint in path_hints:
        if not path_hint.endswith(".py"):
            continue
        normalized = path_hint[:-3].replace("/", ".")
        normalized = normalized.removesuffix(".__init__")
        if normalized:
            hints.append(normalized)
        stem = Path(path_hint).stem
        if stem and stem != "__init__":
            hints.append(stem)
    return tuple(dict.fromkeys(hints))


def _extract_symbol_hints(
    *,
    task: CodingTaskCase,
    localization_notes: tuple[str, ...],
    verifier_output: str | None,
) -> tuple[str, ...]:
    hints: list[str] = []
    for test_name in (*task.fail_to_pass_tests, *task.failing_tests, *task.pass_to_pass_tests):
        for segment in re.split(r"[:.]+", test_name):
            normalized = segment.strip()
            if normalized.startswith("test_") and len(normalized) > 5:
                hints.append(normalized[5:])
            if _is_symbol_hint(normalized):
                hints.append(normalized)
    for text in (
        task.task_statement,
        task.problem_statement,
        task.hints_text,
        *localization_notes,
        verifier_output,
    ):
        if not text:
            continue
        for token in IDENTIFIER_HINT_RE.findall(text):
            if token.startswith("test_") and len(token) > 5:
                hints.append(token[5:])
            if _is_symbol_hint(token):
                hints.append(token)
    return tuple(dict.fromkeys(hints))


def _is_symbol_hint(token: str) -> bool:
    normalized = token.strip("_")
    if len(normalized) < 3:
        return False
    if normalized.lower() in {"test", "tests", "none"}:
        return False
    return "_" in normalized


def _extract_failure_signatures(verifier_output: str | None) -> tuple[str, ...]:
    if not verifier_output:
        return ()
    lines: list[str] = []
    for raw_line in verifier_output.splitlines():
        line = raw_line.strip()
        if not line or not FAILURE_SIGNATURE_RE.search(line):
            continue
        lines.append(_clip_visible_text(line, max_chars=180))
    if not lines:
        first_line = next(
            (line.strip() for line in verifier_output.splitlines() if line.strip()),
            "",
        )
        if first_line:
            lines.append(_clip_visible_text(first_line, max_chars=180))
    return tuple(dict.fromkeys(lines[:4]))


def rerank_coding_candidates(
    candidates: tuple[RetrieveCandidate, ...],
    *,
    visible_task_context: VisibleTaskContext,
    top_k: int,
) -> tuple[RetrieveCandidate, ...]:
    """Prefer memories that align with visible file, module, symbol, and failure hints."""
    if not candidates:
        return ()

    path_hints = set(visible_task_context.path_hints)
    filename_hints = {Path(path_hint).name for path_hint in path_hints if path_hint}
    module_hints = set(visible_task_context.module_hints)
    symbol_hints = set(visible_task_context.symbol_hints)
    failure_terms = _hint_terms(*visible_task_context.failure_signatures)
    ranked: list[tuple[RetrieveCandidate, int]] = []

    for candidate in candidates:
        bonus, reasons = _candidate_structural_bonus(
            candidate,
            path_hints=path_hints,
            filename_hints=filename_hints,
            module_hints=module_hints,
            symbol_hints=symbol_hints,
            failure_terms=failure_terms,
        )
        final_score = candidate.score + bonus
        explanation_metadata = dict(candidate.explanation.metadata)
        explanation_metadata["coding_structural_bonus"] = bonus
        explanation_metadata["coding_structural_matches"] = reasons
        explanation_reason = candidate.explanation.reason
        if reasons:
            explanation_reason = (
                f"{explanation_reason}; structural boost: {', '.join(reasons)}"
            )
        ranked_candidate = candidate.model_copy(
            update={
                "score": final_score,
                "explanation": candidate.explanation.model_copy(
                    update={
                        "score": final_score,
                        "reason": explanation_reason,
                        "metadata": explanation_metadata,
                    }
                ),
            }
        )
        ranked.append((ranked_candidate, len(reasons)))

    ranked.sort(
        key=lambda entry: (
            -entry[0].score,
            -entry[1],
            entry[0].memory.memory_id,
        )
    )
    return tuple(entry[0] for entry in ranked[:top_k])


def _candidate_structural_bonus(
    candidate: RetrieveCandidate,
    *,
    path_hints: set[str],
    filename_hints: set[str],
    module_hints: set[str],
    symbol_hints: set[str],
    failure_terms: set[str],
) -> tuple[float, tuple[str, ...]]:
    bonus = 0.0
    reasons: list[str] = []
    anchor = first_anchor(candidate.memory)
    if anchor is not None:
        if anchor.path in path_hints:
            bonus += 6.0
            reasons.append("path")
        elif Path(anchor.path).name in filename_hints:
            bonus += 3.0
            reasons.append("filename")

        anchor_module = ""
        if anchor.path.endswith(".py"):
            anchor_module = anchor.path[:-3].replace("/", ".").removesuffix(".__init__")
        anchor_stem = Path(anchor.path).stem
        if anchor_module in module_hints or anchor_stem in module_hints:
            bonus += 2.5
            reasons.append("module")

        if anchor.symbol and anchor.symbol in symbol_hints:
            bonus += 5.0
            reasons.append("symbol")

    if failure_terms:
        failure_overlap = len(failure_terms & _candidate_hint_terms(candidate.memory, anchor))
        if failure_overlap:
            bonus += min(2.0, 0.5 * float(failure_overlap))
            reasons.append("failure_signature")

    distinct_reasons = tuple(dict.fromkeys(reasons))
    if len(distinct_reasons) > 1:
        bonus += 0.5 * float(len(distinct_reasons) - 1)
    return bonus, distinct_reasons


def _candidate_hint_terms(memory: MemoryItem, anchor: Any | None) -> set[str]:
    texts = [memory.title, memory.summary, " ".join(memory.tags)]
    payload = memory.payload
    if isinstance(payload, ClaimPayload):
        texts.append(payload.claim)
    elif isinstance(payload, ProcedurePayload):
        texts.append(payload.goal)
        texts.extend(step.instruction for step in payload.steps)
    elif isinstance(payload, ConstraintPayload):
        texts.append(payload.statement)
    elif isinstance(payload, DecisionPayload):
        texts.append(payload.summary)
        if payload.rationale is not None:
            texts.append(payload.rationale)
    if anchor is not None:
        texts.append(anchor.path)
        if anchor.symbol is not None:
            texts.append(anchor.symbol)
    return _hint_terms(*texts)


def _hint_terms(*texts: str | None) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        if not text:
            continue
        terms.update(token.lower() for token in TEXT_TOKEN_RE.findall(text))
    return terms


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


def materialize_attempt_task(
    *,
    prepared_task: PreparedCodingTask,
    prepared_workspace: PreparedWorkspace,
    task: CodingTaskCase,
    config: BenchmarkRunConfig,
    kernel: TransactionalMemoryKernel | None,
    scope: VisibilityScope | None,
    current_dependency: Any | None,
    previous_attempts: tuple[BenchmarkAttemptResult, ...],
) -> AttemptTaskMaterialization:
    """Create the attempt-local task pack consumed by the executor."""
    if (
        config.mode == "no_memory"
        or kernel is None
        or scope is None
        or current_dependency is None
        or prepared_task.visible_task_context is None
    ):
        return AttemptTaskMaterialization(
            task_pack=prepared_task.task_pack,
            task_file=prepared_task.task_file,
            memory_context=prepared_task.memory_context,
            context_chars=prepared_task.context_chars,
            retrieval_verified_count=prepared_task.retrieval_verified_count,
            retrieval_relocated_count=prepared_task.retrieval_relocated_count,
            retrieval_stale_filtered_count=prepared_task.retrieval_stale_filtered_count,
            retrieval_stale_hit_rate=prepared_task.retrieval_stale_hit_rate,
        )

    visible_task_context = attempt_visible_task_context(
        base_context=prepared_task.visible_task_context,
        previous_attempts=previous_attempts,
    )
    seed_task_conditioned_memories(
        kernel=kernel,
        scope=scope,
        task=task,
        current_dependency=current_dependency,
        visible_task_context=visible_task_context,
    )
    retrieval_result = kernel.retrieve(
        coding_retrieve_request(
            mode=config.mode,
            query=visible_task_context.retrieval_query,
            scope=scope,
            limit=coding_candidate_pool_limit(config.top_k),
            current_dependency=current_dependency,
        )
    )
    selected_candidates = rerank_coding_candidates(
        retrieval_result.candidates,
        visible_task_context=visible_task_context,
        top_k=config.top_k,
    )
    retrieved_memory_context = tuple(
        memory_context_payload(
            candidate.memory,
            candidate.score,
            candidate.explanation,
        )
        for candidate in selected_candidates
    )
    handoff_context = _previous_attempt_memory_context(previous_attempts)
    memory_context = _merge_attempt_memory_context(
        handoff_context,
        retrieved_memory_context,
        limit=config.top_k,
    )
    task_pack = prepared_task.task_pack.model_copy(
        update={
            "retrieval_query": visible_task_context.retrieval_query,
            "verifier_output": visible_task_context.verifier_output,
            "localization_notes": visible_task_context.localization_notes,
            "memory_context": memory_context,
        }
    )
    task_file = prepared_workspace.artifact_root / "task-pack.json"
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
    return AttemptTaskMaterialization(
        task_pack=task_pack,
        task_file=task_file,
        memory_context=memory_context,
        context_chars=context_chars,
        retrieval_verified_count=retrieval_result.verified_count,
        retrieval_relocated_count=retrieval_result.relocated_count,
        retrieval_stale_filtered_count=retrieval_result.stale_filtered_count,
        retrieval_stale_hit_rate=retrieval_result.stale_hit_rate,
        failure_handoff_count=len(handoff_context),
    )


def attempt_visible_task_context(
    *,
    base_context: VisibleTaskContext,
    previous_attempts: tuple[BenchmarkAttemptResult, ...],
) -> VisibleTaskContext:
    """Augment visible retrieval signals with the latest failed-attempt feedback."""
    if not previous_attempts:
        return base_context

    latest_attempt = previous_attempts[-1]
    if bool(latest_attempt.metrics.get("passed", False)) or bool(
        latest_attempt.metrics.get("resolved", False)
    ):
        return base_context

    failure_excerpt = _latest_attempt_failure_excerpt(latest_attempt)
    previous_changed_paths = tuple(
        str(item).strip()
        for item in latest_attempt.metadata.get("produced_changed_paths", ())
        if str(item).strip()
    )
    if not failure_excerpt and not previous_changed_paths:
        return base_context

    path_hints = tuple(dict.fromkeys((*base_context.path_hints, *previous_changed_paths)))
    module_hints = tuple(
        dict.fromkeys((*base_context.module_hints, *_extract_module_hints(previous_changed_paths)))
    )
    failure_signatures = tuple(
        dict.fromkeys(
            (*base_context.failure_signatures, *_extract_failure_signatures(failure_excerpt))
        )
    )
    localization_notes = list(base_context.localization_notes)
    if previous_changed_paths:
        localization_notes.append(
            "Previous failed attempt changed: " + ", ".join(previous_changed_paths)
        )
    if failure_excerpt:
        localization_notes.append(
            "Previous failed attempt verifier signal: "
            + _clip_visible_text(failure_excerpt.replace("\n", " "), max_chars=220)
        )
    localization_tuple = tuple(dict.fromkeys(note for note in localization_notes if note))
    query_parts = _unique_visible_parts(
        *base_context.retrieval_query_parts,
        " ".join(previous_changed_paths),
        failure_excerpt,
    )
    return VisibleTaskContext(
        retrieval_query=" ".join(query_parts).strip(),
        retrieval_query_parts=query_parts,
        verifier_output=failure_excerpt or base_context.verifier_output,
        localization_notes=localization_tuple,
        path_hints=path_hints,
        module_hints=module_hints,
        symbol_hints=base_context.symbol_hints,
        failure_signatures=failure_signatures,
    )


def _latest_attempt_failure_excerpt(attempt: BenchmarkAttemptResult) -> str | None:
    for key in (
        "final_verification_stderr_path",
        "final_verification_stdout_path",
        "executor_stderr_path",
    ):
        excerpt = _read_attempt_text_excerpt(attempt.metadata.get(key))
        if excerpt:
            return excerpt
    infra_error = str(attempt.metadata.get("infra_error") or "").strip()
    if infra_error:
        return _clip_visible_text(infra_error, max_chars=700)
    return None


def _read_attempt_text_excerpt(path_value: Any) -> str | None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    contents = path.read_text(encoding="utf-8").strip()
    if not contents:
        return None
    return _clip_visible_text(contents, max_chars=700)


def _previous_attempt_memory_context(
    previous_attempts: tuple[BenchmarkAttemptResult, ...],
) -> tuple[TaskMemoryContextItem, ...]:
    if not previous_attempts:
        return ()
    latest_attempt = previous_attempts[-1]
    if bool(latest_attempt.metrics.get("passed", False)) or bool(
        latest_attempt.metrics.get("resolved", False)
    ):
        return ()

    changed_paths = tuple(
        str(item).strip()
        for item in latest_attempt.metadata.get("produced_changed_paths", ())
        if str(item).strip()
    )
    failure_excerpt = _latest_attempt_failure_excerpt(latest_attempt)
    if not changed_paths and not failure_excerpt:
        return ()

    attempt_index = latest_attempt.attempt_index
    summary_parts: list[str] = [f"Attempt {attempt_index} failed and needs repair."]
    if changed_paths:
        summary_parts.append("Changed files: " + ", ".join(changed_paths[:4]))
    if failure_excerpt:
        summary_parts.append(
            "Verifier signal: "
            + _clip_visible_text(failure_excerpt.replace("\n", " "), max_chars=180)
        )
    matched_terms = tuple(
        sorted(_hint_terms(" ".join(changed_paths), failure_excerpt))[:8]
    )
    return (
        TaskMemoryContextItem(
            memory_id=f"attempt-handoff:{latest_attempt.case_id}:{attempt_index:02d}",
            title=f"Repair Handoff From Attempt {attempt_index}",
            summary=_clip_visible_text(" ".join(summary_parts), max_chars=260),
            score=10_000.0,
            status="working_memory",
            relative_path=changed_paths[0] if changed_paths else None,
            matched_terms=matched_terms,
            matched_fields=("produced_changed_paths", "verifier_output"),
            relevance_reason=(
                "Derived from the latest failed attempt so the next pass can repair "
                "the in-flight solution instead of restarting."
            ),
        ),
    )


def _merge_attempt_memory_context(
    handoff_context: tuple[TaskMemoryContextItem, ...],
    retrieved_context: tuple[TaskMemoryContextItem, ...],
    *,
    limit: int,
) -> tuple[TaskMemoryContextItem, ...]:
    if not handoff_context:
        return retrieved_context
    remaining = max(0, limit - len(handoff_context))
    return (*handoff_context, *retrieved_context[:remaining])


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
    previous_attempts: tuple[BenchmarkAttemptResult, ...] = (),
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
    testable = bool(task.test_command)
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
    current_dependency: Any | None = None
    memory_runtime: ExecutorMemoryRuntime | None = None
    attempt_task = AttemptTaskMaterialization(
        task_pack=prepared_task.task_pack,
        task_file=prepared_task.task_file,
        memory_context=prepared_task.memory_context,
        context_chars=prepared_task.context_chars,
        retrieval_verified_count=prepared_task.retrieval_verified_count,
        retrieval_relocated_count=prepared_task.retrieval_relocated_count,
        retrieval_stale_filtered_count=prepared_task.retrieval_stale_filtered_count,
        retrieval_stale_hit_rate=prepared_task.retrieval_stale_hit_rate,
    )

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
        if config.mode != "no_memory":
            kernel, metadata, artifacts, cache, durable_scope = kernel_factory.open_kernel(
                repo_root=repo_root,
                repo_name=repo_spec.repo_name,
                pair=pair,
                output_dir=output_dir,
            )
            if prepared_task.seed_symbols is None:
                raise ValueError("memory-enabled coding attempts require seeded symbols")
            kernel_factory.seed_memories(
                kernel,
                repo_root,
                repo_spec.repo_name,
                pair,
                prepared_task.seed_symbols,
                durable_scope,
                dependency_ref=prepared_task.memory_seed_ref,
            )
            current_dependency = benchmark_dependency_fingerprint(
                kernel_factory=kernel_factory,
                repo_root=repo_root,
                pair=pair,
            )
            if metadata is not None and durable_scope is not None:
                memory_runtime = ExecutorMemoryRuntime(
                    kernel=kernel,
                    scopes=(durable_scope,),
                    dependency_provider=lambda: current_dependency,
                    memory_lookup=metadata.get_memory_item,
                )
            attempt_task = materialize_attempt_task(
                prepared_task=prepared_task,
                prepared_workspace=prepared_workspace,
                task=task,
                config=config,
                kernel=kernel,
                scope=durable_scope,
                current_dependency=current_dependency,
                previous_attempts=previous_attempts,
            )
        executor_request = ExecutorRequest(
            case_id=task.case_id,
            task_file=str(attempt_task.task_file),
            workspace=str(prepared_workspace.workspace_root),
            artifact_root=str(prepared_workspace.artifact_root),
            attempt_index=prepared_workspace.attempt_index,
            workspace_backend=prepared_workspace.backend_name,
            test_command=task.test_command,
        )
        model_id = execution_model(config.execution_model_id)
        outcome = DSPyReActBenchmarkExecutor(
            model_id=model_id,
            base_url=openrouter_base_url(),
            api_key=openrouter_api_key(),
            max_iterations=config.agent_max_iterations,
            command_timeout_seconds=config.workspace_command_timeout_seconds,
            max_output_chars=config.workspace_max_output_chars,
        ).execute(
            request=executor_request,
            prepared_workspace=prepared_workspace,
            memory_runtime=memory_runtime,
        )
        executed = True

        if outcome is not None:
            assert prepared_workspace is not None
            incomplete = not bool(task.test_command)
            runtime_ms = outcome.runtime_ms
            executor_succeeded = outcome.command_exit_code == 0 and not outcome.command_timed_out
            produced_patch_nonempty = bool(outcome.produced_patch_text.strip())
            final_verification_runtime_ms = outcome.final_verification_runtime_ms
            if task.test_command:
                passed = outcome.test_exit_code == 0 and not outcome.final_verification_timed_out
                resolved = passed
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
            "retrieval_usage_rate": 0.0 if not attempt_task.memory_context else 1.0,
            "verified_count": attempt_task.retrieval_verified_count,
            "relocated_count": attempt_task.retrieval_relocated_count,
            "stale_filtered_count": attempt_task.retrieval_stale_filtered_count,
            "stale_hit_rate": attempt_task.retrieval_stale_hit_rate,
            "context_chars": attempt_task.context_chars,
            "failure_handoff_count": attempt_task.failure_handoff_count,
            **(outcome.agent_metrics if outcome is not None else {}),
        },
        metadata={
            "task_file": str(attempt_task.task_file),
            "touched_paths": list(prepared_task.touched_paths),
            "expected_changed_paths": list(prepared_task.expected_changed_paths),
            "target_patch_digest": prepared_task.target_patch_digest,
            "memory_context_count": len(attempt_task.memory_context),
            "failure_handoff_count": attempt_task.failure_handoff_count,
            "memory_mode": config.mode,
            "top_k": config.top_k,
            "task_kind": task.task_kind,
            "difficulty": task.difficulty,
            "execution_style": task.execution_style,
            "evaluation_backend": task.evaluation_backend,
            "problem_statement": task.problem_statement,
            "hints_text": attempt_task.task_pack.hints_text,
            "fail_to_pass_tests": list(task.fail_to_pass_tests),
            "pass_to_pass_tests": list(task.pass_to_pass_tests),
            "verifier_output": attempt_task.task_pack.verifier_output,
            "localization_notes": list(attempt_task.task_pack.localization_notes),
            "gold_test_patch_digest": task.gold_test_patch_digest,
            "retrieval_query": attempt_task.task_pack.retrieval_query,
            "base_ref": pair.base_ref,
            "head_ref": pair.head_ref,
            "commit_pair_label": pair.label,
            "produced_changed_paths": list(produced_changed_paths),
            "produced_patch_text": produced_patch_text,
            "execution_engine": config.coding_engine,
            **executor_metadata,
        },
    )


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
    explanation: RetrieveExplanation,
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
        slice_name=str(explanation.metadata.get("slice_name"))
        if explanation.metadata.get("slice_name") is not None
        else None,
        raw_anchor_path=anchor.path if anchor is not None else None,
        matched_terms=explanation.matched_tokens,
        matched_fields=explanation.matched_fields,
        relevance_reason=explanation.reason,
    )


def first_anchor(memory: MemoryItem) -> Any | None:
    """Return the first code-anchor evidence attached to a memory item."""
    for evidence in memory.evidence:
        if evidence.kind is EvidenceKind.CODE_ANCHOR and evidence.code_anchor is not None:
            return evidence.code_anchor
    return None


__all__ = ["run_coding_suite"]
