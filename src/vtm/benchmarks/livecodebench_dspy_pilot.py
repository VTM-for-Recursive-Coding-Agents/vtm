"""Small LiveCodeBench pilot comparing direct and DSPy ReAct flows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, Final, Literal

from vtm.adapters.openai_chat import OpenAICompatibleChatClient, OpenAICompatibleChatConfig
from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.base import utc_now
from vtm.benchmarks.livecodebench_sources import (
    LiveCodeBenchCheckoutSource,
    LiveCodeBenchProblem,
    PilotScenario,
    ProblemFileSource,
    ProblemSource,
    discover_problem_source,
)
from vtm.benchmarks.openrouter import execution_model, openrouter_api_key, openrouter_base_url
from vtm.enums import ClaimStrength, DetailLevel, EvidenceBudget, EvidenceKind, MemoryKind, ScopeKind, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint, EnvFingerprint, RepoFingerprint, ToolVersion
from vtm.memory_items import (
    ClaimPayload,
    DecisionPayload,
    MemoryItem,
    SummaryCardPayload,
    ValidityState,
    VisibilityScope,
)
from vtm.retrieval import RetrieveRequest
from vtm.services.memory_kernel import TransactionalMemoryKernel
from vtm.services.procedures import CommandProcedureValidator
from vtm.services.retriever import LexicalRetriever
from vtm.services.verifier import BasicVerifier
from vtm.stores.artifact_store import FilesystemArtifactStore
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.sqlite_store import SqliteMetadataStore
from vtm_dspy import dspy_available
from vtm_dspy.config import DEFAULT_DSPY_TIMEOUT_SECONDS, DSPyOpenRouterConfig, resolve_dspy_timeout_seconds
from vtm_dspy.react_agent import VTMReActCodingAgent
from vtm_dspy.tools import memory_tooling_supported

PilotMethod = Literal[
    "direct",
    "dspy_baseline",
    "dspy_vtm_local_only",
    "dspy_vtm_persistent_only",
    "dspy_vtm",
]

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / ".benchmarks" / "livecodebench-dspy"
DEFAULT_BENCHMARK_ROOT: Final[Path] = PROJECT_ROOT / "benchmarks" / "LiveCodeBench"
DEFAULT_MAX_PROBLEMS: Final[int] = 3
DEFAULT_MAX_TOKENS: Final[int] = 65536
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_SCENARIO: Final[PilotScenario] = "self_repair"
DEFAULT_PROVIDER_ONLY: Final[tuple[str, ...]] = ("ionstream/fp8",)
DEFAULT_SELF_REPAIR_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_DIRECT_EMPTY_RESPONSE_RETRIES: Final[int] = 2
PILOT_ATTEMPT_ONE_REACT_MAX_ITERS: Final[int] = 8
PILOT_REPAIR_REACT_MAX_ITERS: Final[int] = 10
METHODS: Final[tuple[PilotMethod, ...]] = (
    "direct",
    "dspy_baseline",
    "dspy_vtm_local_only",
    "dspy_vtm_persistent_only",
    "dspy_vtm",
)
PILOT_LIMITATION_NOTES: Final[tuple[str, ...]] = (
    "LiveCodeBench DSPy output is a scaffolded pilot, not the maintained VTM regression surface.",
    "Pilot scoring uses repo-local public tests instead of the official hidden LiveCodeBench evaluation loop.",
    "Use controlled_coding_drift for VTM regression checks and repo-memory comparisons.",
)
CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
PROMPT_CARD_ROLE_PRIORITY: Final[dict[str, int]] = {
    "repair_handoff": 0,
    "repair_constraint": 1,
    "feedback_item": 2,
    "feedback": 3,
    "canonical_repair_lesson": 4,
    "repair_lesson": 5,
    "public_tests": 6,
    "function_contract": 7,
    "refuted_answer": 8,
    "attempt_summary": 9,
    "successful_solution": 10,
    "problem_summary": 20,
}
ATTEMPT_ONE_MEMORY_ROLES: Final[frozenset[str]] = frozenset({"function_contract", "public_tests"})
REPAIR_LOCAL_MEMORY_ROLES: Final[frozenset[str]] = frozenset(
    {
        "repair_handoff",
        "repair_constraint",
        "feedback_item",
        "feedback",
        "refuted_answer",
        "attempt_summary",
    }
)
CHEAP_REPAIR_LOCAL_MEMORY_ROLES: Final[frozenset[str]] = frozenset(
    {"repair_handoff", "repair_constraint", "feedback_item", "feedback"}
)
REPAIR_PERSISTENT_MEMORY_ROLES: Final[frozenset[str]] = frozenset(
    {
        "canonical_repair_lesson",
        "repair_lesson",
        "successful_solution",
        "function_contract",
        "public_tests",
    }
)
CHEAP_REPAIR_PERSISTENT_MEMORY_ROLES: Final[frozenset[str]] = frozenset(
    {"canonical_repair_lesson", "repair_lesson"}
)
ATTEMPT_ONE_PROMPT_CARD_LIMIT: Final[int] = 2
REPAIR_PROMPT_CARD_LIMIT: Final[int] = 4
CHEAP_REPAIR_PROMPT_CARD_LIMIT: Final[int] = 3


@dataclass(frozen=True)
class RetrievalPlan:
    """Role-scoped retrieval strategy for one attempt/store pair."""

    allowed_roles: frozenset[str]
    limit: int


@dataclass(frozen=True)
class RepairHandoff:
    """Public self-repair handoff carried across attempts."""

    previous_response: str = ""
    previous_code: str | None = None
    visible_feedback: tuple[str, ...] = ()
    attempt_index: int = 1
    failure_signature: str = ""
    failure_kind: str = "generic_feedback_guided_repair"
    bug_class: str = "unknown"
    repair_objective: str = ""
    preserve_constraints: tuple[str, ...] = ()
    public_signal_summary: str = ""
    local_query: str = ""
    persistent_query: str = ""


RepairContext = RepairHandoff


@dataclass(frozen=True)
class FailureSignature:
    """Compact parsed failure features used in prompts and retrieval queries."""

    summary: str
    failure_kind: str
    bug_class: str
    exception_type: str | None
    expected_value: str | None
    actual_value: str | None
    function_name: str | None
    missing_symbol: str | None
    wrong_callable_shape: bool
    output_format_issue: bool
    empty_or_missing_output: bool
    syntax_error: bool
    timeout_or_runtime_abort: bool
    repair_target: str | None
    keywords: tuple[str, ...]
    raw_feedback: tuple[str, ...]


@dataclass(frozen=True)
class AttemptCandidate:
    """One generated candidate plus its public-test outcome."""

    response_text: str
    extracted_code: str | None
    evaluation: Mapping[str, Any] | None
    usage: dict[str, Any] | None
    response_error: str | None
    direct_retry_count: int
    dspy_tool_calls: int
    memory_write_proposals: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PilotRunConfig:
    """Resolved execution config for one pilot invocation."""

    methods: tuple[PilotMethod, ...]
    requested_scenario: PilotScenario
    resolved_scenario: PilotScenario
    model: str
    base_url: str
    api_key: str | None
    temperature: float
    max_tokens: int
    candidates_per_attempt: int
    problem_offset: int
    max_problems: int
    execute: bool
    output_root: Path
    persistent_memory_root: Path
    benchmark_root: Path
    problem_file: Path | None
    run_id: str
    dspy_timeout_seconds: float = DEFAULT_DSPY_TIMEOUT_SECONDS
    provider_only: tuple[str, ...] = DEFAULT_PROVIDER_ONLY

    @property
    def dry_run(self) -> bool:
        return not self.execute


@dataclass(frozen=True)
class PilotRunPaths:
    """Stable on-disk locations for one method run."""

    run_dir: Path
    problems_jsonl: Path
    summary_json: Path


@dataclass(frozen=True)
class MethodRuntime:
    """Dry-run metadata describing one configured pilot method."""

    method: PilotMethod
    uses_dspy: bool
    uses_vtm_memory: bool
    tool_names: tuple[str, ...]
    dspy_available: bool
    memory_tools_enabled: bool


@dataclass
class PilotMemorySession:
    """Benchmark-local VTM session for one LiveCodeBench problem."""

    kernel: TransactionalMemoryKernel
    metadata_store: SqliteMetadataStore
    artifact_store: FilesystemArtifactStore
    cache_store: SqliteCacheStore
    scope: VisibilityScope
    dependency: DependencyFingerprint

    def close(self) -> None:
        self.cache_store.close()
        self.artifact_store.close()
        self.metadata_store.close()


def memory_tools_enabled_for_session(session: PilotMemorySession | None) -> bool:
    """Whether one pilot session can expose the full dynamic memory tool set."""
    if session is None:
        return False
    if session.dependency is None:
        return False
    memory_lookup = getattr(session.metadata_store, "get_memory_item", None)
    if not callable(memory_lookup):
        return False
    return memory_tooling_supported(
        kernel=session.kernel,
        scopes=(session.scope,),
        dependency_provider=lambda: session.dependency,
        memory_lookup=memory_lookup,
    )


def method_uses_local_memory(method: PilotMethod) -> bool:
    return method in {"dspy_vtm_local_only", "dspy_vtm"}


def method_uses_persistent_memory(method: PilotMethod) -> bool:
    return method in {"dspy_vtm_persistent_only", "dspy_vtm"}


def method_uses_vtm_memory(method: PilotMethod) -> bool:
    return method_uses_local_memory(method) or method_uses_persistent_memory(method)


def build_dspy_agent(
    *,
    method: PilotMethod,
    session: PilotMemorySession | None,
    model_config: DSPyOpenRouterConfig,
    workspace_root: Path | None = None,
    enable_memory_lookup_tools: bool | None = None,
    enable_memory_write_tools: bool | None = None,
    max_iters: int | None = None,
) -> VTMReActCodingAgent:
    """Construct the configured DSPy agent for one pilot method."""
    if method_uses_vtm_memory(method):
        assert session is not None
        tooling_supported = memory_tools_enabled_for_session(session)
        resolved_enable_memory_lookup_tools = (
            tooling_supported
            if enable_memory_lookup_tools is None
            else bool(enable_memory_lookup_tools and tooling_supported)
        )
        resolved_enable_memory_write_tools = (
            tooling_supported
            if enable_memory_write_tools is None
            else bool(enable_memory_write_tools and tooling_supported)
        )
        return VTMReActCodingAgent(
            kernel=session.kernel,
            scopes=(session.scope,),
            enable_memory_tools=resolved_enable_memory_lookup_tools,
            enable_memory_write_tools=resolved_enable_memory_write_tools,
            workspace_root=workspace_root,
            dependency_provider=lambda: session.dependency,
            memory_lookup=session.metadata_store.get_memory_item,
            model_config=model_config,
            max_iters=max_iters or PILOT_REPAIR_REACT_MAX_ITERS,
        )
    assert method == "dspy_baseline"
    return VTMReActCodingAgent(
        kernel=None,
        scopes=(),
        enable_memory_tools=False,
        enable_memory_write_tools=False,
        workspace_root=workspace_root,
        model_config=model_config,
        max_iters=max_iters or PILOT_REPAIR_REACT_MAX_ITERS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small LiveCodeBench pilot comparing direct OpenRouter, DSPy, "
            "and DSPy plus VTM verified memory."
        )
    )
    parser.add_argument(
        "--method",
        choices=(
            "direct",
            "dspy_baseline",
            "dspy_vtm_local_only",
            "dspy_vtm_persistent_only",
            "dspy_vtm",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--scenario",
        choices=("code_generation", "self_repair"),
        default=DEFAULT_SCENARIO,
    )
    parser.add_argument("--problem-offset", type=int, default=0)
    parser.add_argument("--max-problems", type=int, default=DEFAULT_MAX_PROBLEMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--persistent-memory-root",
        type=Path,
        default=None,
        help=(
            "Stable VTM store root used by VTM methods to persist successful solves across runs. "
            "Defaults under --output-root."
        ),
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--provider-only",
        default=",".join(DEFAULT_PROVIDER_ONLY),
        help=(
            "Comma-separated OpenRouter provider slugs to allow for this pilot. "
            "Defaults to ionstream/fp8 with fallbacks disabled."
        ),
    )
    parser.add_argument(
        "--dspy-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Per-request timeout for DSPy/OpenRouter calls. "
            "Defaults to VTM_DSPY_TIMEOUT_SECONDS or 180 seconds."
        ),
    )
    parser.add_argument(
        "--candidates-per-attempt",
        type=int,
        default=1,
        help=(
            "Generate k candidates per attempt and select the best public-test result. "
            "Applies equally to all pilot methods."
        ),
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--problem-file",
        type=Path,
        default=None,
        help="Optional JSON or JSONL file with public LiveCodeBench problems.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compatibility flag. Dry-run is the default unless --execute is passed.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def resolve_methods(raw_method: str) -> tuple[PilotMethod, ...]:
    if raw_method == "all":
        return METHODS
    return (raw_method,)  # type: ignore[return-value]


def _parse_provider_only(raw_value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return parts


def _openrouter_provider_preferences(provider_only: Sequence[str]) -> dict[str, Any] | None:
    resolved_only = [provider for provider in provider_only if str(provider).strip()]
    if not resolved_only:
        return None
    return {
        "only": resolved_only,
        "allow_fallbacks": False,
    }


def method_run_dir(
    output_root: Path,
    *,
    scenario: PilotScenario,
    model: str,
    run_id: str,
    method: PilotMethod,
) -> Path:
    return output_root / scenario / _slugify(model) / run_id / method


def method_run_paths(
    output_root: Path,
    *,
    scenario: PilotScenario,
    model: str,
    run_id: str,
    method: PilotMethod,
) -> PilotRunPaths:
    run_dir = method_run_dir(
        output_root,
        scenario=scenario,
        model=model,
        run_id=run_id,
        method=method,
    )
    return PilotRunPaths(
        run_dir=run_dir,
        problems_jsonl=run_dir / "problems.jsonl",
        summary_json=run_dir / "summary.json",
    )


def default_persistent_memory_root(
    output_root: Path,
    *,
    scenario: PilotScenario,
    model: str,
) -> Path:
    return output_root / "_persistent_vtm_memory" / scenario / _slugify(model)


def resolve_scenario(
    requested: PilotScenario,
    *,
    supported_scenarios: set[PilotScenario],
) -> PilotScenario:
    if requested in supported_scenarios:
        return requested
    if requested == "self_repair" and "code_generation" in supported_scenarios:
        return "code_generation"
    return requested


def resolve_config(
    args: argparse.Namespace,
    *,
    source: ProblemSource,
) -> PilotRunConfig:
    max_problems = max(1, int(args.max_problems))
    max_tokens = max(1, int(args.max_tokens))
    candidates_per_attempt = max(1, int(args.candidates_per_attempt))
    problem_offset = max(0, int(args.problem_offset))
    model = execution_model(args.model or None)
    base_url = (args.base_url or openrouter_base_url()).strip()
    api_key = (args.api_key or openrouter_api_key() or "").strip() or None
    requested_scenario = args.scenario  # type: ignore[assignment]
    supported = source.supported_scenarios()
    resolved_scenario = resolve_scenario(requested_scenario, supported_scenarios=supported)
    run_id = args.run_id or f"lcb_dspy_pilot_{utc_now().strftime('%Y%m%d_%H%M%S')}"
    dspy_timeout_seconds = resolve_dspy_timeout_seconds(args.dspy_timeout_seconds)
    provider_only = _parse_provider_only(str(args.provider_only))
    persistent_memory_root = (
        args.persistent_memory_root
        if args.persistent_memory_root is not None
        else default_persistent_memory_root(
            args.output_root,
            scenario=resolved_scenario,
            model=model,
        )
    )
    return PilotRunConfig(
        methods=resolve_methods(args.method),
        requested_scenario=requested_scenario,
        resolved_scenario=resolved_scenario,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=float(args.temperature),
        max_tokens=max_tokens,
        candidates_per_attempt=candidates_per_attempt,
        problem_offset=problem_offset,
        max_problems=max_problems,
        execute=bool(args.execute),
        output_root=args.output_root,
        persistent_memory_root=persistent_memory_root,
        benchmark_root=args.benchmark_root,
        problem_file=args.problem_file,
        run_id=run_id,
        dspy_timeout_seconds=dspy_timeout_seconds,
        provider_only=provider_only,
    )


def describe_method_runtime(
    method: PilotMethod,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float | None = None,
    provider_only: Sequence[str] = DEFAULT_PROVIDER_ONLY,
) -> MethodRuntime:
    provider_preferences = _openrouter_provider_preferences(provider_only)
    model_config = DSPyOpenRouterConfig.from_env(
        base_url_value=base_url,
        api_key_value=api_key,
        execution_model_name=model,
        dspy_model_name=model,
        timeout_seconds=timeout_seconds,
        extra_body=(
            {"provider": provider_preferences}
            if provider_preferences is not None
            else None
        ),
    )
    if method_uses_vtm_memory(method):
        with TemporaryDirectory(prefix="vtm-lcb-dspy-dry-run-") as temp_dir:
            session = (
                open_persistent_memory_session(
                    state_root=Path(temp_dir),
                    scenario="self_repair",
                    model=model,
                    workspace_root=PROJECT_ROOT,
                )
                if method == "dspy_vtm_persistent_only"
                else open_memory_session(
                    state_root=Path(temp_dir),
                    problem_id="dry-run",
                    workspace_root=PROJECT_ROOT,
                )
            )
            try:
                agent = build_dspy_agent(
                    method=method,
                    session=session,
                    model_config=model_config,
                )
                memory_tools_enabled = agent.memory_tools.enabled
                tool_names = agent.tool_names()
            finally:
                session.close()
        return MethodRuntime(
            method=method,
            uses_dspy=True,
            uses_vtm_memory=True,
            tool_names=tool_names,
            dspy_available=dspy_available(),
            memory_tools_enabled=memory_tools_enabled,
        )
    if method == "dspy_baseline":
        agent = build_dspy_agent(
            method=method,
            session=None,
            model_config=model_config,
        )
        return MethodRuntime(
            method=method,
            uses_dspy=True,
            uses_vtm_memory=False,
            tool_names=agent.tool_names(),
            dspy_available=dspy_available(),
            memory_tools_enabled=agent.memory_tools.enabled,
        )
    assert method == "direct"
    return MethodRuntime(
        method=method,
        uses_dspy=False,
        uses_vtm_memory=False,
        tool_names=(),
        dspy_available=dspy_available(),
        memory_tools_enabled=False,
    )


def build_attempt_prompt(
    problem: LiveCodeBenchProblem,
    *,
    attempt_index: int,
    agent_mode: Literal["direct", "dspy"] = "direct",
    require_memory_tooling: bool = False,
    suggested_memory_query: str | None = None,
    memory_cards: Sequence[Mapping[str, Any]] = (),
    visible_feedback: Sequence[str] = (),
    repair_context: RepairHandoff | None = None,
    compact_repair: bool = False,
) -> str:
    required_func_name = _required_function_name(problem)
    parsed_failure = (
        _parse_failure_signature(problem, visible_feedback)
        if repair_context is None
        else _failure_signature_from_handoff(problem, repair_context)
    )
    sections = [
        (
            "Repair the previous LiveCodeBench attempt using the visible failure signal."
            if compact_repair and repair_context is not None
            else "Solve the following LiveCodeBench problem."
        ),
        (
            "Return the final answer as a single ```python fenced code block and nothing else."
            if agent_mode == "direct"
            else (
                "Use tools only if needed. When you finish, put the solution in the final "
                "`response` field as a single ```python fenced code block."
            )
        ),
        f"Problem ID: {problem.problem_id}",
    ]
    if require_memory_tooling:
        sections.extend(
            [
                "",
                "Memory Workflow:",
                (
                    "Before writing the final solution on repair attempts, call "
                    "`search_verified_memory` with the visible failure signature."
                ),
                (
                    "If you get relevant hits, call `expand_memory_evidence` on the top 1-2 "
                    "repair-oriented memories before deciding on the code change."
                ),
                (
                    "Use `verify_memory` when a retrieved lesson looks applicable but you need "
                    "to confirm it still matches the current dependency fingerprint."
                ),
                (
                    "When you discover a reusable fix or failure pattern, call "
                    "`propose_memory_lesson` or `propose_failure_pattern` before the final "
                    "response so the host can decide whether to promote it."
                ),
                (
                    "Use `propose_solution_pattern` only for generic solution structure, not "
                    "for problem-specific code dumps."
                ),
                "Do not skip memory lookup on repair attempts when memory tools are available.",
            ]
        )
        if attempt_index > 1 and repair_context is not None:
            sections.extend(
                [
                    "",
                    "Repair Plan:",
                    (
                        "Before writing code, produce a 3-line repair plan in your reasoning "
                        "or tool flow."
                    ),
                    "Line 1: `failure kind: ...`",
                    "Line 2: `likely cause: ...`",
                    "Line 3: `smallest fix: ...` and preserve constraints.",
                    (
                        "Use retrieved memory and the repair handoff to ground the plan before "
                        "you change the code."
                    ),
                    (
                        "Do not include the repair plan in the final `response`; the final "
                        "response must remain only the Python solution."
                    ),
                ]
            )
    if not compact_repair:
        _append_repair_handoff_section(
            sections,
            repair_context=repair_context,
            memory_cards=memory_cards,
            suggested_memory_query=suggested_memory_query,
        )
        sections.extend(["", "Problem Statement:", problem.prompt.strip()])
    else:
        _append_repair_handoff_section(
            sections,
            repair_context=repair_context,
            memory_cards=memory_cards,
            suggested_memory_query=suggested_memory_query,
        )
    if required_func_name is not None and not compact_repair:
        sections.extend(
            [
                "",
                "Implementation Contract:",
                (
                    f"Define a top-level function named `{required_func_name}` that matches "
                    "the expected signature."
                ),
                (
                    "Do not rely on a `class Solution` wrapper unless you also expose the same "
                    "callable at module scope."
                ),
            ]
        )
    if problem.starter_code and not compact_repair:
        sections.extend(["", "Starter Code:", "```python", problem.starter_code.strip(), "```"])
    if problem.prompt_metadata and not compact_repair:
        sections.extend(
            [
                "",
                "Problem Metadata:",
                json.dumps(problem.prompt_metadata, indent=2, sort_keys=True),
            ]
        )
    if compact_repair:
        _append_minimal_contract_snapshot(sections, problem=problem, memory_cards=memory_cards)
        _append_public_test_snapshot(sections, problem=problem, memory_cards=memory_cards)
    if repair_context is not None:
        sections.append("")
        sections.append("Previous Candidate:")
        if repair_context.previous_code:
            sections.extend(
                [
                    "```python",
                    repair_context.previous_code.strip(),
                    "```",
                ]
            )
        else:
            sections.append(compact_text(repair_context.previous_response, limit=600))
    if not compact_repair:
        _append_contract_section(sections, problem=problem, memory_cards=memory_cards)
        _append_public_test_section(
            sections,
            problem=problem,
            memory_cards=memory_cards,
        )
        _append_visible_failure_section(
            sections,
            parsed_failure=parsed_failure,
            visible_feedback=visible_feedback,
        )
        _append_repair_lesson_section(
            sections,
            memory_cards=memory_cards,
            limit=2,
        )
        _append_supporting_memory_section(sections, memory_cards=memory_cards)
    else:
        _append_visible_failure_section(
            sections,
            parsed_failure=parsed_failure,
            visible_feedback=visible_feedback,
        )
        _append_repair_lesson_section(
            sections,
            memory_cards=memory_cards,
            limit=1,
        )
    if attempt_index > 1 and repair_context is not None:
        sections.append("")
        if compact_repair:
            sections.extend(
                [
                    f"This is final short repair attempt {attempt_index}. "
                    "Make the smallest correction that resolves the visible failure.",
                    (
                        "Do not rewrite from scratch unless the prior candidate is empty, "
                        "malformed, or syntactically broken."
                    ),
                    "Preserve any behavior not contradicted by the visible failure.",
                ]
            )
        else:
            sections.extend(
                [
                    f"This is repair attempt {attempt_index}. Fix the previous attempt.",
                    "Preserve any behavior not contradicted by the visible failure.",
                ]
            )
    return "\n".join(sections).strip() + "\n"


def extract_code(text: str) -> str | None:
    if not text.strip():
        return None
    blocks = [
        match.group(1).strip()
        for match in CODE_BLOCK_PATTERN.finditer(text)
        if match.group(1).strip()
    ]
    if blocks:
        return blocks[-1]
    stripped = text.strip()
    return stripped or None


def aggregate_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: PilotMethod,
    scenario: PilotScenario,
    model: str,
) -> dict[str, Any]:
    total = len(rows)
    evaluation_rows = [
        row.get("evaluation")
        for row in rows
        if isinstance(row.get("evaluation"), Mapping)
        and row["evaluation"].get("available") is True
        and isinstance(row["evaluation"].get("passed"), bool)
    ]
    pass_count = sum(1 for row in evaluation_rows if row.get("passed") is True)
    evaluation_available_count = len(evaluation_rows)
    pass_rate = (pass_count / evaluation_available_count) if evaluation_rows else None
    syntax_error_count = sum(
        1
        for row in rows
        if isinstance(row.get("evaluation"), Mapping) and row["evaluation"].get("syntax_error")
    )
    retrieval_usage_rate = (
        sum(
            1
            for row in rows
            if isinstance(row.get("retrieval"), Mapping) and row["retrieval"].get("invoked")
        )
        / total
        if total
        else 0.0
    )
    retrieval_hit_rate = (
        sum(
            1
            for row in rows
            if isinstance(row.get("retrieval"), Mapping) and row["retrieval"].get("used")
        )
        / total
        if total
        else 0.0
    )
    mean_verified_count = _mean_numeric(
        row.get("retrieval", {}).get("verified_count", 0)  # type: ignore[union-attr]
        if isinstance(row.get("retrieval"), Mapping)
        else 0
        for row in rows
    )
    mean_stale_filtered_count = _mean_numeric(
        row.get("retrieval", {}).get("stale_filtered_count", 0)  # type: ignore[union-attr]
        if isinstance(row.get("retrieval"), Mapping)
        else 0
        for row in rows
    )
    mean_tool_calls = _mean_numeric(
        row.get("tool_calls", 0)
        for row in rows
    )
    mean_candidates_per_attempt = _mean_numeric(
        row.get("candidates_per_attempt", 1)
        for row in rows
    )
    total_agent_memory_write_count = sum(int(row.get("agent_memory_write_count", 0) or 0) for row in rows)
    mean_agent_memory_write_count = _mean_numeric(
        row.get("agent_memory_write_count", 0)
        for row in rows
    )
    agent_memory_write_rate = (
        sum(1 for row in rows if int(row.get("agent_memory_write_count", 0) or 0) > 0) / total
        if total
        else 0.0
    )
    total_canonical_memory_hit_count = sum(int(row.get("canonical_memory_hit_count", 0) or 0) for row in rows)
    mean_canonical_memory_hit_count = _mean_numeric(
        row.get("canonical_memory_hit_count", 0)
        for row in rows
    )
    canonical_memory_hit_rate = (
        sum(1 for row in rows if int(row.get("canonical_memory_hit_count", 0) or 0) > 0) / total
        if total
        else 0.0
    )
    total_consolidated_memory_card_count = sum(
        int(row.get("consolidated_memory_card_count", 0) or 0)
        for row in rows
    )
    consolidated_memory_card_rate = (
        sum(1 for row in rows if int(row.get("consolidated_memory_card_count", 0) or 0) > 0) / total
        if total
        else 0.0
    )
    repair_handoff_hit_rate = (
        sum(1 for row in rows if bool(row.get("repair_handoff_memory_hit"))) / total
        if total
        else 0.0
    )
    repair_handoff_success_rate = (
        sum(
            1
            for row in rows
            if bool(row.get("repair_handoff_memory_hit"))
            and isinstance(row.get("evaluation"), Mapping)
            and row["evaluation"].get("passed") is True
        )
        / max(1, sum(1 for row in rows if bool(row.get("repair_handoff_memory_hit"))))
    )
    repair_handoff_card_in_prompt_rate = (
        sum(1 for row in rows if bool(row.get("repair_handoff_card_in_prompt"))) / total
        if total
        else 0.0
    )
    contract_card_in_prompt_rate = (
        sum(1 for row in rows if bool(row.get("contract_card_in_prompt"))) / total
        if total
        else 0.0
    )
    public_test_card_in_prompt_rate = (
        sum(1 for row in rows if bool(row.get("public_test_card_in_prompt"))) / total
        if total
        else 0.0
    )
    top_prompt_memory_role_distribution = _value_distribution(
        row.get("top_prompt_memory_role")
        for row in rows
    )
    used_candidate_selection = any(int(row.get("candidates_per_attempt", 1)) > 1 for row in rows)
    attempt1_pass_at_1_count = sum(1 for row in rows if _attempt1_candidate_passed(row, candidate_index=1))
    attempt1_pass_at_k_count = sum(1 for row in rows if _attempt1_any_candidate_passed(row))
    attempt1_pass_at_1 = (attempt1_pass_at_1_count / total) if total else None
    attempt1_pass_at_k = (attempt1_pass_at_k_count / total) if total else None
    attempt2_pass_at_1_count = sum(1 for row in rows if _attempt_candidate_passed(row, attempt_index=2, candidate_index=1))
    attempt2_pass_at_k_count = sum(1 for row in rows if _attempt_any_candidate_passed(row, attempt_index=2))
    attempt2_pass_at_1 = (attempt2_pass_at_1_count / total) if total else None
    attempt2_pass_at_k = (attempt2_pass_at_k_count / total) if total else None
    attempt3_pass_at_1_count = sum(1 for row in rows if _attempt_candidate_passed(row, attempt_index=3, candidate_index=1))
    attempt3_pass_at_k_count = sum(1 for row in rows if _attempt_any_candidate_passed(row, attempt_index=3))
    attempt3_pass_at_1 = (attempt3_pass_at_1_count / total) if total else None
    attempt3_pass_at_k = (attempt3_pass_at_k_count / total) if total else None
    attempt1_pass_curve = _attempt_pass_curve(rows, attempt_index=1)
    attempt2_pass_curve = _attempt_pass_curve(rows, attempt_index=2)
    attempt3_pass_curve = _attempt_pass_curve(rows, attempt_index=3)
    return {
        "benchmark": "livecodebench",
        "kind": "dspy_pilot",
        "status": "completed",
        "scenario": scenario,
        "method": method,
        "model": model,
        "total": total,
        "evaluation_available_count": evaluation_available_count,
        "pass_count": pass_count if evaluation_rows else None,
        "pass_rate": round(pass_rate, 6) if pass_rate is not None else None,
        "public_test_pass_rate": round(pass_rate, 6) if pass_rate is not None else None,
        "syntax_error_count": syntax_error_count,
        "retrieval_usage_rate": round(retrieval_usage_rate, 6),
        "retrieval_hit_rate": round(retrieval_hit_rate, 6),
        "mean_verified_count": round(mean_verified_count, 6),
        "mean_stale_filtered_count": round(mean_stale_filtered_count, 6),
        "mean_tool_calls": round(mean_tool_calls, 6),
        "mean_candidates_per_attempt": round(mean_candidates_per_attempt, 6),
        "total_agent_memory_write_count": total_agent_memory_write_count,
        "mean_agent_memory_write_count": round(mean_agent_memory_write_count, 6),
        "agent_memory_write_rate": round(agent_memory_write_rate, 6),
        "total_canonical_memory_hit_count": total_canonical_memory_hit_count,
        "mean_canonical_memory_hit_count": round(mean_canonical_memory_hit_count, 6),
        "canonical_memory_hit_rate": round(canonical_memory_hit_rate, 6),
        "total_consolidated_memory_card_count": total_consolidated_memory_card_count,
        "consolidated_memory_card_rate": round(consolidated_memory_card_rate, 6),
        "repair_handoff_hit_rate": round(repair_handoff_hit_rate, 6),
        "repair_handoff_success_rate": round(repair_handoff_success_rate, 6),
        "repair_handoff_card_in_prompt_rate": round(repair_handoff_card_in_prompt_rate, 6),
        "contract_card_in_prompt_rate": round(contract_card_in_prompt_rate, 6),
        "public_test_card_in_prompt_rate": round(public_test_card_in_prompt_rate, 6),
        "top_prompt_memory_role_distribution": top_prompt_memory_role_distribution,
        "candidate_selection_mode": (
            "best_of_k_public_tests" if used_candidate_selection else "single_sample"
        ),
        "attempt1_public_test_pass_at_1_count": attempt1_pass_at_1_count,
        "attempt1_public_test_pass_at_1": (
            round(attempt1_pass_at_1, 6) if attempt1_pass_at_1 is not None else None
        ),
        "attempt1_public_test_pass_at_k_count": attempt1_pass_at_k_count,
        "attempt1_public_test_pass_at_k": (
            round(attempt1_pass_at_k, 6) if attempt1_pass_at_k is not None else None
        ),
        "attempt1_public_test_pass_curve": attempt1_pass_curve,
        "attempt2_public_test_pass_at_1_count": attempt2_pass_at_1_count,
        "attempt2_public_test_pass_at_1": (
            round(attempt2_pass_at_1, 6) if attempt2_pass_at_1 is not None else None
        ),
        "attempt2_public_test_pass_at_k_count": attempt2_pass_at_k_count,
        "attempt2_public_test_pass_at_k": (
            round(attempt2_pass_at_k, 6) if attempt2_pass_at_k is not None else None
        ),
        "attempt2_public_test_pass_curve": attempt2_pass_curve,
        "attempt3_public_test_pass_at_1_count": attempt3_pass_at_1_count,
        "attempt3_public_test_pass_at_1": (
            round(attempt3_pass_at_1, 6) if attempt3_pass_at_1 is not None else None
        ),
        "attempt3_public_test_pass_at_k_count": attempt3_pass_at_k_count,
        "attempt3_public_test_pass_at_k": (
            round(attempt3_pass_at_k, 6) if attempt3_pass_at_k is not None else None
        ),
        "attempt3_public_test_pass_curve": attempt3_pass_curve,
        "attempt2_delta_over_attempt1": (
            round(attempt2_pass_at_1 - attempt1_pass_at_1, 6)
            if attempt2_pass_at_1 is not None and attempt1_pass_at_1 is not None
            else None
        ),
        "attempt3_delta_over_attempt2": (
            round(attempt3_pass_at_1 - attempt2_pass_at_1, 6)
            if attempt3_pass_at_1 is not None and attempt2_pass_at_1 is not None
            else None
        ),
        "attempt3_total_delta_over_attempt1": (
            round(attempt3_pass_at_1 - attempt1_pass_at_1, 6)
            if attempt3_pass_at_1 is not None and attempt1_pass_at_1 is not None
            else None
        ),
        "attempt2_success_with_canonical_hit_rate": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=2,
                predicate=lambda row: 2 in tuple(row.get("canonical_memory_hit_attempts", ())),
            ),
            6,
        ),
        "attempt2_success_without_canonical_hit_rate": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=2,
                predicate=lambda row: 2 not in tuple(row.get("canonical_memory_hit_attempts", ())),
            ),
            6,
        ),
        "attempt3_success_with_canonical_hit_rate": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=3,
                predicate=lambda row: 3 in tuple(row.get("canonical_memory_hit_attempts", ())),
            ),
            6,
        ),
        "attempt3_success_without_canonical_hit_rate": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=3,
                predicate=lambda row: 3 not in tuple(row.get("canonical_memory_hit_attempts", ())),
            ),
            6,
        ),
        "attempt2_success_when_repair_card_in_prompt": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=2,
                predicate=lambda row: bool(row.get("attempt2_repair_card_in_prompt")),
            ),
            6,
        ),
        "attempt2_success_when_no_repair_card_in_prompt": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=2,
                predicate=lambda row: not bool(row.get("attempt2_repair_card_in_prompt")),
            ),
            6,
        ),
        "attempt3_success_when_repair_card_in_prompt": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=3,
                predicate=lambda row: bool(row.get("attempt3_repair_card_in_prompt")),
            ),
            6,
        ),
        "attempt3_success_when_no_repair_card_in_prompt": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=3,
                predicate=lambda row: not bool(row.get("attempt3_repair_card_in_prompt")),
            ),
            6,
        ),
        "attempt2_success_when_handoff_in_prompt": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=2,
                predicate=lambda row: bool(row.get("attempt2_handoff_card_in_prompt")),
            ),
            6,
        ),
        "attempt3_success_when_handoff_in_prompt": round(
            _conditional_attempt_success_rate(
                rows,
                attempt_index=3,
                predicate=lambda row: bool(row.get("attempt3_handoff_card_in_prompt")),
            ),
            6,
        ),
        "pilot_limitations": list(PILOT_LIMITATION_NOTES),
    }


def _build_method_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: PilotMethod,
    config: PilotRunConfig,
    source: ProblemSource,
    status: Literal["running", "completed"],
    completed_problem_count: int,
    planned_problem_count: int,
) -> dict[str, Any]:
    summary = aggregate_summary(
        rows,
        method=method,
        scenario=config.resolved_scenario,
        model=config.model,
    )
    summary["status"] = status
    summary["run_id"] = config.run_id
    summary["generated_at"] = utc_now().isoformat()
    summary["problem_source"] = source.describe()
    summary["problem_offset"] = config.problem_offset
    summary["persistent_memory_root"] = str(config.persistent_memory_root)
    summary["scenario_semantics"] = _scenario_semantics(config.resolved_scenario)
    summary["candidate_selection_semantics"] = _candidate_selection_semantics(
        config.candidates_per_attempt
    )
    summary["candidates_per_attempt"] = config.candidates_per_attempt
    summary["completed_problem_count"] = completed_problem_count
    summary["planned_problem_count"] = planned_problem_count
    summary["remaining_problem_count"] = max(0, planned_problem_count - completed_problem_count)
    return summary


def skipped_summary(
    *,
    method: PilotMethod,
    scenario: PilotScenario,
    model: str,
    run_id: str,
    reason: str,
    problem_source: Mapping[str, Any],
    problem_offset: int,
    planned_problem_count: int,
) -> dict[str, Any]:
    return {
        "benchmark": "livecodebench",
        "kind": "dspy_pilot",
        "status": "skipped",
        "scenario": scenario,
        "method": method,
        "model": model,
        "run_id": run_id,
        "generated_at": utc_now().isoformat(),
        "problem_source": dict(problem_source),
        "problem_offset": problem_offset,
        "planned_problem_count": planned_problem_count,
        "scenario_semantics": _scenario_semantics(scenario),
        "total": 0,
        "evaluation_available_count": 0,
        "pass_count": None,
        "pass_rate": None,
        "public_test_pass_rate": None,
        "syntax_error_count": 0,
        "retrieval_usage_rate": 0.0,
        "retrieval_hit_rate": 0.0,
        "mean_verified_count": 0.0,
        "mean_stale_filtered_count": 0.0,
        "mean_tool_calls": 0.0,
        "total_agent_memory_write_count": 0,
        "mean_agent_memory_write_count": 0.0,
        "agent_memory_write_rate": 0.0,
        "total_canonical_memory_hit_count": 0,
        "mean_canonical_memory_hit_count": 0.0,
        "canonical_memory_hit_rate": 0.0,
        "total_consolidated_memory_card_count": 0,
        "consolidated_memory_card_rate": 0.0,
        "repair_handoff_hit_rate": 0.0,
        "repair_handoff_success_rate": 0.0,
        "repair_handoff_card_in_prompt_rate": 0.0,
        "contract_card_in_prompt_rate": 0.0,
        "public_test_card_in_prompt_rate": 0.0,
        "top_prompt_memory_role_distribution": {},
        "attempt1_public_test_pass_at_1_count": 0,
        "attempt1_public_test_pass_at_1": None,
        "attempt1_public_test_pass_at_k_count": 0,
        "attempt1_public_test_pass_at_k": None,
        "attempt1_public_test_pass_curve": {},
        "attempt2_public_test_pass_at_1_count": 0,
        "attempt2_public_test_pass_at_1": None,
        "attempt2_public_test_pass_at_k_count": 0,
        "attempt2_public_test_pass_at_k": None,
        "attempt2_public_test_pass_curve": {},
        "attempt3_public_test_pass_at_1_count": 0,
        "attempt3_public_test_pass_at_1": None,
        "attempt3_public_test_pass_at_k_count": 0,
        "attempt3_public_test_pass_at_k": None,
        "attempt3_public_test_pass_curve": {},
        "attempt2_delta_over_attempt1": None,
        "attempt3_delta_over_attempt2": None,
        "attempt3_total_delta_over_attempt1": None,
        "attempt2_success_with_canonical_hit_rate": 0.0,
        "attempt2_success_without_canonical_hit_rate": 0.0,
        "attempt3_success_with_canonical_hit_rate": 0.0,
        "attempt3_success_without_canonical_hit_rate": 0.0,
        "attempt2_success_when_repair_card_in_prompt": 0.0,
        "attempt2_success_when_no_repair_card_in_prompt": 0.0,
        "attempt3_success_when_repair_card_in_prompt": 0.0,
        "attempt3_success_when_no_repair_card_in_prompt": 0.0,
        "attempt2_success_when_handoff_in_prompt": 0.0,
        "attempt3_success_when_handoff_in_prompt": 0.0,
        "skip_reason": reason,
        "pilot_limitations": list(PILOT_LIMITATION_NOTES),
    }


def run_pilot(
    config: PilotRunConfig,
    *,
    source: ProblemSource,
) -> dict[str, Any]:
    source_error: str | None = None
    try:
        problems = source.load_problems(
            config.resolved_scenario,
            problem_offset=config.problem_offset,
            max_problems=config.max_problems,
        )
    except FileNotFoundError as exc:
        if not config.dry_run:
            raise SystemExit(_problem_source_error_message(config, source, exc)) from exc
        problems = []
        source_error = str(exc)
    payload: dict[str, Any] = {
        "dry_run": config.dry_run,
        "requested_scenario": config.requested_scenario,
        "resolved_scenario": config.resolved_scenario,
        "methods": list(config.methods),
        "model": config.model,
        "base_url": config.base_url,
        "api_key_configured": config.api_key is not None,
        "output_root": str(config.output_root),
        "persistent_memory_root": str(config.persistent_memory_root),
        "benchmark_root": str(config.benchmark_root),
        "candidates_per_attempt": config.candidates_per_attempt,
        "problem_offset": config.problem_offset,
        "problem_source": source.describe(),
        "problem_source_error": source_error,
        "problem_count": len(problems),
        "scenario_semantics": _scenario_semantics(config.resolved_scenario),
        "candidate_selection_semantics": _candidate_selection_semantics(
            config.candidates_per_attempt
        ),
        "pilot_limitations": list(PILOT_LIMITATION_NOTES),
        "direct_reference_command": _reference_command(config),
        "runs": [],
    }
    for method in config.methods:
        runtime = describe_method_runtime(
            method,
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.dspy_timeout_seconds,
            provider_only=config.provider_only,
        )
        paths = method_run_paths(
            config.output_root,
            scenario=config.resolved_scenario,
            model=config.model,
            run_id=config.run_id,
            method=method,
        )
        run_payload = {
            "method": method,
            "runtime": runtime.__dict__,
            "run_dir": str(paths.run_dir),
            "problems_jsonl": str(paths.problems_jsonl),
            "summary_json": str(paths.summary_json),
            "persistent_memory_root": str(config.persistent_memory_root),
            "candidates_per_attempt": config.candidates_per_attempt,
        }
        payload["runs"].append(run_payload)
        if config.dry_run:
            continue
        if method != "direct" and not runtime.dspy_available:
            raise RuntimeError(f"{method} requires the optional 'dspy' extra")
        print(
            f"[livecodebench-dspy-pilot] starting method={method} "
            f"problems={len(problems)} run_id={config.run_id}",
            flush=True,
        )
        rows = execute_method(
            method,
            config=config,
            problems=problems,
            source=source,
        )
        summary = _build_method_summary(
            rows,
            method=method,
            config=config,
            source=source,
            status="completed",
            completed_problem_count=len(rows),
            planned_problem_count=len(problems),
        )
        write_summary(paths.summary_json, summary)
        print(
            f"[livecodebench-dspy-pilot] finished method={method} "
            f"completed={len(rows)}/{len(problems)} summary={paths.summary_json}",
            flush=True,
        )
        run_payload["summary"] = summary
    return payload


def _problem_source_error_message(
    config: PilotRunConfig,
    source: ProblemSource,
    exc: FileNotFoundError,
) -> str:
    description = source.describe()
    benchmark_root_exists = bool(description.get("benchmark_root_exists"))
    checkout_loader_script_exists = description.get("checkout_loader_script_exists")
    problem_file = description.get("problem_file")
    supported = description.get("supported_scenarios", [])
    if problem_file:
        return (
            f"{exc}. The supplied --problem-file was not usable. "
            "Pass a readable JSON/JSONL file with public LiveCodeBench problems."
        )
    if not benchmark_root_exists:
        return (
            f"{exc}. LiveCodeBench checkout not found at {config.benchmark_root}. "
            "Run `bash scripts/livecodebench/setup_livecodebench.sh` first, or pass "
            "`--problem-file <public-problems.jsonl>`."
        )
    if checkout_loader_script_exists is False:
        return (
            f"{exc}. The repo-local checkout loader script is missing. "
            "Restore `scripts/livecodebench/checkout_problem_loader.py` or pass "
            "`--problem-file <public-problems.jsonl>`."
        )
    if supported:
        return (
            f"{exc}. Supported scenarios discovered under {config.benchmark_root}: "
            f"{', '.join(str(item) for item in supported)}. "
            "Pick one of those or pass `--problem-file <public-problems.jsonl>`."
        )
    return (
        f"{exc}. No usable LiveCodeBench checkout-backed problem source was discovered under "
        f"{config.benchmark_root}. Run `bash scripts/livecodebench/setup_livecodebench.sh` "
        "first, confirm the checkout venv is present, or pass "
        "`--problem-file <public-problems.jsonl>`."
    )


def execute_method(
    method: PilotMethod,
    *,
    config: PilotRunConfig,
    problems: Sequence[LiveCodeBenchProblem],
    source: ProblemSource,
) -> list[dict[str, Any]]:
    if config.api_key is None:
        raise RuntimeError("OpenRouter API key is required for --execute")
    client = OpenAICompatibleChatClient(
        OpenAICompatibleChatConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            extra_body=(
                {"provider": _openrouter_provider_preferences(config.provider_only)}
                if _openrouter_provider_preferences(config.provider_only) is not None
                else None
            ),
        )
    )
    provider_preferences = _openrouter_provider_preferences(config.provider_only)
    model_config = DSPyOpenRouterConfig.from_env(
        base_url_value=config.base_url,
        api_key_value=config.api_key,
        execution_model_name=config.model,
        dspy_model_name=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.dspy_timeout_seconds,
        extra_body=(
            {"provider": provider_preferences}
            if provider_preferences is not None
            else None
        ),
    )
    rows: list[dict[str, Any]] = []
    paths = method_run_paths(
        config.output_root,
        scenario=config.resolved_scenario,
        model=config.model,
        run_id=config.run_id,
        method=method,
    )
    write_problem_rows(paths.problems_jsonl, rows)
    write_summary(
        paths.summary_json,
        _build_method_summary(
            rows,
            method=method,
            config=config,
            source=source,
            status="running",
            completed_problem_count=0,
            planned_problem_count=len(problems),
        ),
    )
    persistent_session = (
        open_persistent_memory_session(
            state_root=config.persistent_memory_root,
            scenario=config.resolved_scenario,
            model=config.model,
            workspace_root=PROJECT_ROOT,
        )
        if method_uses_persistent_memory(method)
        else None
    )
    try:
        for problem_index, problem in enumerate(problems, start=1):
            print(
                f"[livecodebench-dspy-pilot] method={method} "
                f"problem={problem_index}/{len(problems)} problem_id={problem.problem_id}",
                flush=True,
            )
            row = execute_problem(
                problem,
                method=method,
                config=config,
                source=source,
                client=client,
                model_config=model_config,
                persistent_session=persistent_session,
            )
            rows.append(row)
            write_problem_rows(paths.problems_jsonl, rows)
            write_summary(
                paths.summary_json,
                _build_method_summary(
                    rows,
                    method=method,
                    config=config,
                    source=source,
                    status="running",
                    completed_problem_count=len(rows),
                    planned_problem_count=len(problems),
                ),
            )
            print(
                f"[livecodebench-dspy-pilot] method={method} "
                f"completed={len(rows)}/{len(problems)} last_problem={problem.problem_id} "
                f"passed={bool(isinstance(row.get('evaluation'), Mapping) and row['evaluation'].get('passed') is True)}",
                flush=True,
            )
    finally:
        if persistent_session is not None:
            persistent_session.close()
    return rows


def execute_problem(
    problem: LiveCodeBenchProblem,
    *,
    method: PilotMethod,
    config: PilotRunConfig,
    source: ProblemSource,
    client: OpenAICompatibleChatClient,
    model_config: DSPyOpenRouterConfig,
    persistent_session: PilotMemorySession | None = None,
) -> dict[str, Any]:
    max_attempts = DEFAULT_SELF_REPAIR_MAX_ATTEMPTS if config.resolved_scenario == "self_repair" else 1
    visible_feedback: list[str] = []
    response_text = ""
    extracted_code: str | None = None
    usage: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    retrieval_payload: dict[str, Any] | None = None
    retrieval_invocation_count = 0
    retrieval_hit_count = 0
    retrieval_verified_count = 0
    retrieval_stale_filtered_count = 0
    retrieval_queries: list[str] = []
    retrieved_memory_cards: Sequence[Mapping[str, Any]] = ()
    dspy_tool_calls = 0
    direct_retry_count = 0
    response_error: str | None = None
    repair_context: RepairHandoff | None = None
    attempts_executed = 0
    selected_candidate_indices: list[int] = []
    candidate_batches: list[dict[str, Any]] = []
    promoted_agent_memory_ids: list[str] = []
    canonical_memory_hit_count = 0
    canonical_memory_hit_attempts: list[int] = []
    consolidated_memory_card_count = 0
    repair_handoff_memory_hit = False
    handoff_failure_kind: str | None = None
    handoff_bug_class: str | None = None
    handoff_guardrail_count = 0
    attempt3_contract_snapshot_included = False
    attempt3_public_test_snapshot_included = False
    final_prompt_stats = _prompt_memory_stats(())
    prompt_stats_by_attempt: dict[int, dict[str, Any]] = {}
    attempt2_retrieval_required_satisfied = False
    attempt3_retrieval_required_satisfied = False
    local_repair_retrieval_attempted = False
    persistent_repair_retrieval_attempted = False
    repair_retrieval_success_by_attempt: dict[int, bool] = {}

    with TemporaryDirectory(prefix=f"vtm-lcb-{problem.problem_id}-") as temp_dir:
        local_session = (
            open_memory_session(
                state_root=Path(temp_dir),
                problem_id=problem.problem_id,
                workspace_root=PROJECT_ROOT,
            )
            if method_uses_local_memory(method)
            else None
        )
        try:
            if local_session is not None:
                seed_problem_memory(local_session, problem)
            for attempt_index in range(1, max_attempts + 1):
                attempts_executed = attempt_index
                memory_cards: Sequence[Mapping[str, Any]] = ()
                agent_session = (
                    local_session
                    if local_session is not None
                    else persistent_session
                )
                failure_signature = (
                    _failure_signature_from_handoff(problem, repair_context)
                    if repair_context is not None
                    else _parse_failure_signature(problem, visible_feedback)
                )
                local_retrieval_query = (
                    repair_context.local_query
                    if repair_context is not None and repair_context.local_query
                    else build_retrieval_query(
                        problem,
                        visible_feedback,
                        store_kind="local",
                        failure_signature=failure_signature,
                    )
                )
                persistent_retrieval_query = (
                    repair_context.persistent_query
                    if repair_context is not None and repair_context.persistent_query
                    else build_retrieval_query(
                        problem,
                        visible_feedback,
                        store_kind="persistent",
                        failure_signature=failure_signature,
                    )
                )
                local_cards: Sequence[Mapping[str, Any]] = ()
                persistent_cards: Sequence[Mapping[str, Any]] = ()
                local_plan = retrieval_plan(attempt_index=attempt_index, store_kind="local")
                persistent_plan = retrieval_plan(
                    attempt_index=attempt_index,
                    store_kind="persistent",
                )
                if local_session is not None and local_plan is not None:
                    if attempt_index > 1:
                        local_repair_retrieval_attempted = True
                    retrieval_payload = retrieve_verified_memory(
                        local_session,
                        query=local_retrieval_query,
                        attempt_index=attempt_index,
                        allowed_roles=local_plan.allowed_roles,
                        limit=local_plan.limit,
                        expand_top_k=1 if attempt_index > 1 else 0,
                        failure_signature=failure_signature,
                        store_kind="local",
                        interface_mode=_interface_mode(problem),
                    )
                    retrieval_invocation_count += 1
                    retrieval_queries.append(f"local:{local_retrieval_query}")
                    if retrieval_payload["used"]:
                        retrieval_hit_count += 1
                    retrieval_verified_count += int(retrieval_payload["verified_count"])
                    retrieval_stale_filtered_count += int(
                        retrieval_payload["stale_filtered_count"]
                    )
                    local_cards = tuple(retrieval_payload["cards"])
                if persistent_session is not None and persistent_plan is not None:
                    if attempt_index > 1:
                        persistent_repair_retrieval_attempted = True
                    persistent_payload = retrieve_verified_memory(
                        persistent_session,
                        query=persistent_retrieval_query,
                        attempt_index=attempt_index,
                        allowed_roles=persistent_plan.allowed_roles,
                        limit=persistent_plan.limit,
                        expand_top_k=2 if attempt_index > 1 else 0,
                        failure_signature=failure_signature,
                        store_kind="persistent",
                        interface_mode=_interface_mode(problem),
                    )
                    retrieval_invocation_count += 1
                    retrieval_queries.append(f"persistent:{persistent_retrieval_query}")
                    if persistent_payload["used"]:
                        retrieval_hit_count += 1
                    retrieval_verified_count += int(persistent_payload["verified_count"])
                    retrieval_stale_filtered_count += int(
                        persistent_payload["stale_filtered_count"]
                    )
                    persistent_cards = tuple(persistent_payload["cards"])
                memory_cards = tuple(
                    merge_memory_cards(
                        local_cards,
                        persistent_cards,
                        attempt_index=attempt_index,
                    )
                )
                final_prompt_stats = _prompt_memory_stats(memory_cards)
                prompt_stats_by_attempt[attempt_index] = dict(final_prompt_stats)
                if attempt_index > 1:
                    repair_retrieval_success_by_attempt[attempt_index] = bool(
                        local_cards or persistent_cards
                    )
                    if attempt_index == 2:
                        attempt2_retrieval_required_satisfied = bool(local_cards or persistent_cards)
                    if attempt_index == 3:
                        attempt3_retrieval_required_satisfied = bool(local_cards or persistent_cards)
                canonical_hits_this_attempt = sum(
                    1
                    for card in memory_cards
                    if _memory_card_role(card) == "canonical_repair_lesson"
                )
                canonical_memory_hit_count += canonical_hits_this_attempt
                if canonical_hits_this_attempt:
                    canonical_memory_hit_attempts.append(attempt_index)
                repair_handoff_memory_hit = repair_handoff_memory_hit or any(
                    _memory_card_role(card) == "repair_handoff" for card in memory_cards
                )
                retrieved_memory_cards = tuple(memory_cards)
                prompt = build_attempt_prompt(
                    problem,
                    attempt_index=attempt_index,
                    agent_mode="direct" if method == "direct" else "dspy",
                    require_memory_tooling=(
                        method_uses_vtm_memory(method)
                        and attempt_index > 1
                        and agent_session is not None
                    ),
                    suggested_memory_query=(
                        persistent_retrieval_query
                        if method_uses_persistent_memory(method) and attempt_index > 1
                        else (
                            local_retrieval_query
                            if method_uses_local_memory(method) and attempt_index > 1
                            else None
                        )
                    ),
                    memory_cards=memory_cards,
                    visible_feedback=visible_feedback,
                    repair_context=repair_context,
                    compact_repair=attempt_index >= 3 and repair_context is not None,
                )
                if attempt_index >= 3 and repair_context is not None:
                    attempt3_contract_snapshot_included = "Minimal Contract Snapshot:" in prompt
                    attempt3_public_test_snapshot_included = "Public-Test Snapshot:" in prompt
                candidate_count = max(1, config.candidates_per_attempt)
                attempt_candidates: list[AttemptCandidate] = []
                for _candidate_index in range(1, candidate_count + 1):
                    candidate = _generate_attempt_candidate(
                        method=method,
                        prompt=prompt,
                        problem=problem,
                        source=source,
                        client=client,
                        session=agent_session,
                        config=config,
                        model_config=model_config,
                        attempt_index=attempt_index,
                    )
                    attempt_candidates.append(candidate)
                selected_index, selected_candidate = _select_attempt_candidate(attempt_candidates)
                selected_candidate_indices.append(selected_index + 1)
                candidate_batches.append(
                    {
                        "attempt_index": attempt_index,
                        "candidate_count": len(attempt_candidates),
                        "selected_candidate_index": selected_index + 1,
                        "selection_mode": (
                            "best_of_k_public_tests" if len(attempt_candidates) > 1 else "single_sample"
                        ),
                        "candidates": [
                            {
                                "candidate_index": candidate_index,
                                "passed": (
                                    candidate.evaluation.get("passed")
                                    if isinstance(candidate.evaluation, Mapping)
                                    else None
                                ),
                                "pass_rate": (
                                    candidate.evaluation.get("pass_rate")
                                    if isinstance(candidate.evaluation, Mapping)
                                    else None
                                ),
                                "response_error": candidate.response_error,
                            }
                            for candidate_index, candidate in enumerate(attempt_candidates, start=1)
                        ],
                    }
                )
                response_text = selected_candidate.response_text
                extracted_code = selected_candidate.extracted_code
                evaluation = (
                    dict(selected_candidate.evaluation)
                    if isinstance(selected_candidate.evaluation, Mapping)
                    else selected_candidate.evaluation
                )
                for candidate in attempt_candidates:
                    usage = _sum_usage(usage, candidate.usage)
                direct_retry_count += sum(
                    candidate.direct_retry_count for candidate in attempt_candidates
                )
                dspy_tool_calls += sum(
                    candidate.dspy_tool_calls for candidate in attempt_candidates
                )
                response_error = selected_candidate.response_error
                selected_failure_signature = _parse_failure_signature(
                    problem,
                    evaluation.get("failure_feedback", ()) if isinstance(evaluation, Mapping) else (),
                    response_text=response_text,
                    extracted_code=extracted_code,
                    evaluation=evaluation,
                )
                next_repair_context = (
                    _build_repair_handoff(
                        problem,
                        attempt_index=attempt_index,
                        response_text=response_text,
                        extracted_code=extracted_code,
                        visible_feedback=tuple(
                            str(item).strip()
                            for item in (
                                evaluation.get("failure_feedback", ())
                                if isinstance(evaluation, Mapping)
                                else ()
                            )
                            if str(item).strip()
                        ),
                        failure_signature=selected_failure_signature,
                    )
                    if evaluation is not None and evaluation.get("passed") is not True
                    else None
                )
                if local_session is not None:
                    record_attempt_memory(
                        local_session,
                        problem=problem,
                        attempt_index=attempt_index,
                        response_text=response_text,
                        extracted_code=extracted_code,
                        evaluation=evaluation,
                        repair_context=next_repair_context,
                    )
                if (
                    method_uses_persistent_memory(method)
                    and persistent_session is not None
                    and evaluation is not None
                    and evaluation.get("passed") is True
                ):
                    write_result = write_persistent_success_memory(
                        persistent_session,
                        problem=problem,
                        attempt_index=attempt_index,
                        response_text=response_text,
                        extracted_code=extracted_code,
                        evaluation=evaluation,
                        visible_feedback=visible_feedback,
                        memory_write_proposals=selected_candidate.memory_write_proposals,
                    )
                    if write_result is not None:
                        promoted_agent_memory_ids.extend(write_result.get("agent_memory_ids", ()))
                        consolidated_memory_card_count += len(
                            write_result.get("consolidated_memory_ids", ())
                        )
                if (
                    not evaluation
                    or evaluation.get("passed") is True
                    or attempt_index >= max_attempts
                ):
                    break
                feedback_items = evaluation.get("failure_feedback")
                if isinstance(feedback_items, list):
                    visible_feedback = [str(item) for item in feedback_items if str(item).strip()]
                repair_context = next_repair_context
                if repair_context is not None:
                    handoff_failure_kind = repair_context.failure_kind
                    handoff_bug_class = repair_context.bug_class
                    handoff_guardrail_count = len(repair_context.preserve_constraints)
        finally:
            if local_session is not None:
                local_session.close()

    total_tool_calls = dspy_tool_calls + retrieval_invocation_count
    retrieval_summary = None
    if retrieval_invocation_count:
        retrieval_summary = {
            "invoked": True,
            "used": retrieval_hit_count > 0,
            "query": retrieval_queries[-1],
            "query_history": retrieval_queries,
            "cards": list(retrieved_memory_cards),
            "verified_count": retrieval_verified_count,
            "stale_filtered_count": retrieval_stale_filtered_count,
            "tool_calls": retrieval_invocation_count,
            "attempt_count": retrieval_invocation_count,
            "hit_count": retrieval_hit_count,
        }
    return {
        "problem_id": problem.problem_id,
        "method": method,
        "scenario": config.resolved_scenario,
        "model": config.model,
        "prompt_metadata": {
            "problem_id": problem.problem_id,
            "scenario": problem.scenario,
            "prompt_metadata": problem.prompt_metadata,
            "public_feedback": list(problem.public_feedback),
        },
        "response": response_text,
        "extracted_code": extracted_code,
        "evaluation": evaluation,
        "usage": usage,
        "retrieval": retrieval_summary,
        "tool_calls": total_tool_calls,
        "direct_retry_count": direct_retry_count,
        "response_error": response_error,
        "attempt_count": attempts_executed,
        "repair_used": attempts_executed > 1,
        "cheap_repair_used": attempts_executed > 2,
        "candidates_per_attempt": config.candidates_per_attempt,
        "candidate_selection_mode": (
            "best_of_k_public_tests" if config.candidates_per_attempt > 1 else "single_sample"
        ),
        "selected_candidate_indices": selected_candidate_indices,
        "candidate_batches": candidate_batches,
        "agent_memory_write_count": len(promoted_agent_memory_ids),
        "agent_memory_write_ids": promoted_agent_memory_ids,
        "canonical_memory_hit_count": canonical_memory_hit_count,
        "canonical_memory_hit_attempts": canonical_memory_hit_attempts,
        "consolidated_memory_card_count": consolidated_memory_card_count,
        "handoff_failure_kind": handoff_failure_kind,
        "handoff_bug_class": handoff_bug_class,
        "handoff_guardrail_count": handoff_guardrail_count,
        "repair_handoff_memory_hit": repair_handoff_memory_hit,
        "attempt3_contract_snapshot_included": attempt3_contract_snapshot_included,
        "attempt3_public_test_snapshot_included": attempt3_public_test_snapshot_included,
        "attempt2_retrieval_required_satisfied": attempt2_retrieval_required_satisfied,
        "attempt3_retrieval_required_satisfied": attempt3_retrieval_required_satisfied,
        "local_repair_retrieval_attempted": local_repair_retrieval_attempted,
        "persistent_repair_retrieval_attempted": persistent_repair_retrieval_attempted,
        "attempt2_repair_retrieval_success": repair_retrieval_success_by_attempt.get(2, False),
        "attempt3_repair_retrieval_success": repair_retrieval_success_by_attempt.get(3, False),
        **final_prompt_stats,
        "attempt2_repair_card_in_prompt": bool(
            prompt_stats_by_attempt.get(2, {}).get("repair_memory_card_count", 0)
        ),
        "attempt3_repair_card_in_prompt": bool(
            prompt_stats_by_attempt.get(3, {}).get("repair_memory_card_count", 0)
        ),
        "attempt2_handoff_card_in_prompt": bool(
            prompt_stats_by_attempt.get(2, {}).get("repair_handoff_card_in_prompt", False)
        ),
        "attempt3_handoff_card_in_prompt": bool(
            prompt_stats_by_attempt.get(3, {}).get("repair_handoff_card_in_prompt", False)
        ),
        "attempt2_contract_card_in_prompt": bool(
            prompt_stats_by_attempt.get(2, {}).get("contract_card_in_prompt", False)
        ),
        "attempt3_contract_card_in_prompt": bool(
            prompt_stats_by_attempt.get(3, {}).get("contract_card_in_prompt", False)
        ),
        "attempt2_public_test_card_in_prompt": bool(
            prompt_stats_by_attempt.get(2, {}).get("public_test_card_in_prompt", False)
        ),
        "attempt3_public_test_card_in_prompt": bool(
            prompt_stats_by_attempt.get(3, {}).get("public_test_card_in_prompt", False)
        ),
    }


def _generate_attempt_candidate(
    *,
    method: PilotMethod,
    prompt: str,
    problem: LiveCodeBenchProblem,
    source: ProblemSource,
    client: OpenAICompatibleChatClient,
    session: PilotMemorySession | None,
    config: PilotRunConfig,
    model_config: DSPyOpenRouterConfig,
    attempt_index: int,
) -> AttemptCandidate:
    response_payload: Mapping[str, Any] = {}
    memory_write_proposals: tuple[dict[str, Any], ...] = ()
    try:
        if method == "direct":
            response_payload, response_text, retry_count, response_error = _request_direct_completion(
                client,
                model=config.model,
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            usage = _normalize_usage(response_payload.get("usage"))
            dspy_tool_calls = 0
        else:
            response_payload = run_dspy_attempt(
                prompt=prompt,
                method=method,
                session=session,
                model_config=model_config,
                attempt_index=attempt_index,
            )
            response_text = response_payload["response_text"]
            retry_count = 0
            usage = _normalize_usage(response_payload.get("usage"))
            response_error = (
                response_payload["response_error"]
                if isinstance(response_payload.get("response_error"), str)
                else None
            )
            dspy_tool_calls = int(response_payload["tool_calls"])
            memory_write_proposals = tuple(response_payload.get("memory_write_proposals") or ())
    except Exception as exc:
        response_text = ""
        retry_count = 0
        usage = None
        dspy_tool_calls = 0
        response_error = _format_runtime_error(exc)
    extracted_code = extract_code(response_text)
    evaluation = source.evaluate(
        problem,
        response_text=response_text,
        extracted_code=extracted_code,
    )
    return AttemptCandidate(
        response_text=response_text,
        extracted_code=extracted_code,
        evaluation=evaluation,
        usage=usage,
        response_error=response_error,
        direct_retry_count=retry_count,
        dspy_tool_calls=dspy_tool_calls,
        memory_write_proposals=memory_write_proposals,
    )


def _format_runtime_error(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{exc.__class__.__name__}: {detail}"
    return exc.__class__.__name__


def _select_attempt_candidate(candidates: Sequence[AttemptCandidate]) -> tuple[int, AttemptCandidate]:
    assert candidates
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            -_candidate_passed(item[1]),
            -_candidate_pass_rate(item[1]),
            -_candidate_passed_test_count(item[1]),
            _candidate_has_syntax_error(item[1]),
            _candidate_has_response_error(item[1]),
            item[0],
        ),
    )
    return ranked[0]


def _request_direct_completion(
    client: OpenAICompatibleChatClient,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str, int, str | None]:
    last_payload: dict[str, Any] | None = None
    last_error: str | None = None
    for retry_count in range(DEFAULT_DIRECT_EMPTY_RESPONSE_RETRIES + 1):
        response_payload = client.create_chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        last_payload = response_payload
        try:
            response_text = client.extract_message_text(response_payload)
        except RuntimeError as exc:
            if _is_retryable_empty_chat_response(response_payload):
                last_error = str(exc)
                continue
            raise
        if response_text.strip() or not _is_retryable_empty_chat_response(response_payload):
            return response_payload, response_text, retry_count, last_error
        last_error = "OpenAI-compatible chat response returned empty text content."
    assert last_payload is not None
    return last_payload, "", DEFAULT_DIRECT_EMPTY_RESPONSE_RETRIES, last_error


def _is_retryable_empty_chat_response(response_payload: Mapping[str, Any]) -> bool:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return False
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return False
    has_text = any(
        isinstance(value, str) and value.strip()
        for value in (
            choice.get("text"),
            message.get("content"),
            message.get("refusal"),
        )
    )
    if has_text:
        return False
    usage = response_payload.get("usage")
    if isinstance(usage, Mapping):
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens == 0:
            return True
    finish_reason = choice.get("finish_reason")
    if finish_reason in (None, "", "error"):
        return True
    return message.get("content") is None


def run_dspy_attempt(
    *,
    prompt: str,
    method: PilotMethod,
    session: PilotMemorySession | None,
    model_config: DSPyOpenRouterConfig,
    attempt_index: int = 1,
) -> dict[str, Any]:
    if method_uses_vtm_memory(method):
        assert session is not None
    enable_memory_lookup_tools = method_uses_vtm_memory(method) and attempt_index > 1
    enable_memory_write_tools = method_uses_vtm_memory(method)
    react_max_iters = (
        PILOT_ATTEMPT_ONE_REACT_MAX_ITERS
        if attempt_index <= 1
        else PILOT_REPAIR_REACT_MAX_ITERS
    )
    agent = build_dspy_agent(
        method=method,
        session=session,
        model_config=model_config,
        enable_memory_lookup_tools=enable_memory_lookup_tools,
        enable_memory_write_tools=enable_memory_write_tools,
        max_iters=react_max_iters,
    )
    result = agent.run(prompt)
    response_payload = result.get("response")
    response_error = None
    if isinstance(response_payload, Mapping):
        error_value = response_payload.get("error")
        if isinstance(error_value, str) and error_value.strip():
            response_error = error_value
    if response_error is None and isinstance(result.get("trajectory"), Mapping):
        error_value = result["trajectory"].get("execution_error")
        if isinstance(error_value, str) and error_value.strip():
            response_error = error_value
    return {
        "response_text": _coerce_response_text(response_payload),
        "tool_calls": _count_result_tool_calls(result, response_payload),
        "trajectory": result.get("trajectory"),
        "usage": None,
        "response_error": response_error,
        "memory_write_proposals": (
            list(result.get("memory_write_proposals"))
            if isinstance(result.get("memory_write_proposals"), Sequence)
            and not isinstance(result.get("memory_write_proposals"), str | bytes)
            else []
        ),
    }


def open_memory_session(
    *,
    state_root: Path,
    problem_id: str,
    workspace_root: Path,
) -> PilotMemorySession:
    metadata_store = SqliteMetadataStore(
        state_root / "metadata.sqlite",
        event_log_path=state_root / "events.jsonl",
    )
    artifact_store = FilesystemArtifactStore(state_root / "artifacts")
    cache_store = SqliteCacheStore(state_root / "cache.sqlite", event_store=metadata_store)
    anchor_builder = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    kernel = TransactionalMemoryKernel(
        metadata_store=metadata_store,
        event_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        verifier=BasicVerifier(relocator=anchor_builder),
        retriever=LexicalRetriever(metadata_store),
        anchor_adapter=anchor_builder,
        procedure_validator=CommandProcedureValidator(artifact_store),
    )
    scope = VisibilityScope(kind=ScopeKind.TASK, scope_id=f"livecodebench:{problem_id}")
    dependency = DependencyFingerprint(
        repo=RepoFingerprint(
            repo_root=str(workspace_root),
            branch="livecodebench-pilot",
            head_commit=problem_id,
            tree_digest=problem_id,
            dirty_digest="clean",
        ),
        env=EnvFingerprint(
            python_version=platform.python_version(),
            platform=platform.platform(),
            tool_versions=(ToolVersion(name="vtm", version="pilot"),),
        ),
        dependency_ids=(f"livecodebench:{problem_id}",),
        input_digests=(problem_id,),
    )
    return PilotMemorySession(
        kernel=kernel,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        scope=scope,
        dependency=dependency,
    )


def persistent_memory_scope_id(*, scenario: PilotScenario, model: str) -> str:
    return f"livecodebench-persistent:{scenario}:{_slugify(model)}"


def open_persistent_memory_session(
    *,
    state_root: Path,
    scenario: PilotScenario,
    model: str,
    workspace_root: Path,
) -> PilotMemorySession:
    metadata_store = SqliteMetadataStore(
        state_root / "metadata.sqlite",
        event_log_path=state_root / "events.jsonl",
    )
    artifact_store = FilesystemArtifactStore(state_root / "artifacts")
    cache_store = SqliteCacheStore(state_root / "cache.sqlite", event_store=metadata_store)
    anchor_builder = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    kernel = TransactionalMemoryKernel(
        metadata_store=metadata_store,
        event_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        verifier=BasicVerifier(relocator=anchor_builder),
        retriever=LexicalRetriever(metadata_store),
        anchor_adapter=anchor_builder,
        procedure_validator=CommandProcedureValidator(artifact_store),
    )
    scope = VisibilityScope(
        kind=ScopeKind.REPO,
        scope_id=persistent_memory_scope_id(scenario=scenario, model=model),
    )
    dependency_key = f"livecodebench-persistent:{scenario}:{_slugify(model)}"
    dependency = DependencyFingerprint(
        repo=RepoFingerprint(
            repo_root=str(workspace_root),
            branch="livecodebench-persistent-memory",
            head_commit=dependency_key,
            tree_digest=dependency_key,
            dirty_digest="clean",
        ),
        env=EnvFingerprint(
            python_version=platform.python_version(),
            platform=platform.platform(),
            tool_versions=(ToolVersion(name="vtm", version="pilot-persistent"),),
        ),
        dependency_ids=(dependency_key,),
        input_digests=(scenario, model),
    )
    return PilotMemorySession(
        kernel=kernel,
        metadata_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        scope=scope,
        dependency=dependency,
    )


def success_memory_id(problem_id: str) -> str:
    digest = hashlib.sha256(problem_id.encode("utf-8")).hexdigest()
    return f"lcb_success_{digest[:24]}"


def seed_problem_memory(
    session: PilotMemorySession,
    problem: LiveCodeBenchProblem,
) -> None:
    summary_text = compact_text(problem.prompt, limit=220)
    contract_claim = _function_contract_claim(problem)
    public_tests_claim = _public_tests_claim(problem)
    payload = {
        "problem_id": problem.problem_id,
        "scenario": problem.scenario,
        "prompt": problem.prompt,
        "starter_code": problem.starter_code,
        "prompt_metadata": problem.prompt_metadata,
    }
    record = session.kernel.capture_artifact(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        content_type="application/json",
        tool_name="run_livecodebench_dspy_pilot",
        metadata={"kind": "problem_statement", "problem_id": problem.problem_id},
    )
    evidence = session.kernel.artifact_evidence(
        record,
        label="public_problem_statement",
        summary="Public problem statement and starter code",
    )
    tx = session.kernel.begin_transaction(
        session.scope,
        metadata={"problem_id": problem.problem_id},
    )
    platform_name = _platform_name(problem)
    difficulty_name = _difficulty_name(problem)
    interface_mode = _interface_mode(problem)
    session.kernel.stage_memory_item(
        tx.tx_id,
        MemoryItem(
            kind=MemoryKind.CLAIM,
            title=f"LiveCodeBench problem {problem.problem_id}",
            summary=summary_text,
            payload=ClaimPayload(
                claim=summary_text,
                strength=ClaimStrength.SUPPORTED,
            ),
            evidence=(evidence,),
            tags=("livecodebench", "problem"),
            visibility=session.scope,
            validity=ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=session.dependency,
                reason="Public problem statement captured for the current pilot run",
            ),
            metadata={
                "problem_id": problem.problem_id,
                "memory_role": "problem_summary",
                "platform": platform_name,
                "difficulty": difficulty_name,
                "interface_mode": interface_mode,
            },
        ),
    )
    if contract_claim:
        session.kernel.stage_memory_item(
            tx.tx_id,
            MemoryItem(
                kind=MemoryKind.CLAIM,
                title=f"Interface contract for {problem.problem_id}",
                summary=compact_text(contract_claim, limit=220),
                payload=ClaimPayload(
                    claim=contract_claim,
                    strength=ClaimStrength.SUPPORTED,
                ),
                evidence=(evidence,),
                tags=("livecodebench", "contract"),
                visibility=session.scope,
                validity=ValidityState(
                    status=ValidityStatus.VERIFIED,
                    dependency_fingerprint=session.dependency,
                    reason="Public function contract captured for the current pilot run",
                ),
                metadata={
                    "problem_id": problem.problem_id,
                    "memory_role": "function_contract",
                    "platform": platform_name,
                    "difficulty": difficulty_name,
                    "interface_mode": interface_mode,
                },
            ),
        )
    if public_tests_claim:
        session.kernel.stage_memory_item(
            tx.tx_id,
            MemoryItem(
                kind=MemoryKind.CLAIM,
                title=f"Public tests for {problem.problem_id}",
                summary=compact_text(public_tests_claim, limit=220),
                payload=ClaimPayload(
                    claim=public_tests_claim,
                    strength=ClaimStrength.SUPPORTED,
                ),
                evidence=(evidence,),
                tags=("livecodebench", "public_tests"),
                visibility=session.scope,
                validity=ValidityState(
                    status=ValidityStatus.VERIFIED,
                    dependency_fingerprint=session.dependency,
                    reason="Public tests captured for the current pilot run",
                ),
                metadata={
                    "problem_id": problem.problem_id,
                    "memory_role": "public_tests",
                    "platform": platform_name,
                    "difficulty": difficulty_name,
                    "interface_mode": interface_mode,
                },
            ),
        )
    session.kernel.commit_transaction(tx.tx_id)


def record_attempt_memory(
    session: PilotMemorySession,
    *,
    problem: LiveCodeBenchProblem,
    attempt_index: int,
    response_text: str,
    extracted_code: str | None,
    evaluation: Mapping[str, Any] | None,
    repair_context: RepairHandoff | None = None,
) -> None:
    response_record = session.kernel.capture_artifact(
        response_text.encode("utf-8"),
        content_type="text/plain",
        tool_name="run_livecodebench_dspy_pilot",
        metadata={
            "kind": "model_response",
            "problem_id": problem.problem_id,
            "attempt_index": attempt_index,
        },
    )
    response_evidence = session.kernel.artifact_evidence(
        response_record,
        label=f"attempt_{attempt_index}_response",
        summary=f"Raw response for attempt {attempt_index}",
    )
    evidence = [response_evidence]
    feedback_items = [
        str(item).strip()
        for item in (evaluation.get("failure_feedback", []) if evaluation else [])
        if str(item).strip()
    ]
    feedback_summaries: list[str] = []
    if feedback_items:
        feedback_payload = json.dumps(
            feedback_items,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        feedback_record = session.kernel.capture_artifact(
            feedback_payload,
            content_type="application/json",
            tool_name="run_livecodebench_dspy_pilot",
            metadata={
                "kind": "visible_feedback",
                "problem_id": problem.problem_id,
                "attempt_index": attempt_index,
            },
        )
        feedback_evidence = session.kernel.artifact_evidence(
            feedback_record,
            label=f"attempt_{attempt_index}_feedback",
            summary=f"Visible execution or evaluation feedback for attempt {attempt_index}",
        )
        evidence.append(feedback_evidence)
        feedback_summaries.extend(feedback_items)

    tx = session.kernel.begin_transaction(
        session.scope,
        metadata={"problem_id": problem.problem_id, "attempt_index": attempt_index},
    )
    session.kernel.stage_memory_item(
        tx.tx_id,
        MemoryItem(
            kind=MemoryKind.CLAIM,
            title=f"Attempt {attempt_index} response summary",
            summary=(
                f"Attempt {attempt_index} returned "
                f"{len((extracted_code or response_text).strip())} characters of candidate code."
            ),
            payload=ClaimPayload(
                claim=(
                    f"Attempt {attempt_index} returned a candidate solution for "
                    f"{problem.problem_id}."
                ),
                strength=ClaimStrength.TENTATIVE,
            ),
            evidence=tuple(evidence),
            tags=("livecodebench", "attempt"),
            visibility=session.scope,
            validity=ValidityState(
                status=ValidityStatus.UNKNOWN,
                reason="Attempt summaries are retained as unverified working memory",
            ),
            metadata={"problem_id": problem.problem_id, "memory_role": "attempt_summary"},
        ),
    )
    if feedback_summaries:
        feedback_summary = compact_text(" ".join(feedback_summaries), limit=220)
        failure_signature = (
            _failure_signature_from_handoff(problem, repair_context)
            if repair_context is not None
            else _parse_failure_signature(
                problem,
                feedback_summaries,
                response_text=response_text,
                extracted_code=extracted_code,
                evaluation=evaluation,
            )
        )
        if repair_context is not None:
            session.kernel.stage_memory_item(
                tx.tx_id,
                MemoryItem(
                    kind=MemoryKind.CLAIM,
                    title=f"Repair Handoff From Attempt {attempt_index}",
                    summary=compact_text(
                        " ".join(
                            part
                            for part in (
                                f"Attempt {attempt_index} failed and needs repair.",
                                f"Failure kind: {repair_context.failure_kind}.",
                                f"Repair objective: {repair_context.repair_objective}",
                                "Preserve: "
                                + "; ".join(repair_context.preserve_constraints[:2])
                                if repair_context.preserve_constraints
                                else "",
                                f"Visible failure: {repair_context.public_signal_summary}.",
                            )
                            if part
                        ),
                        limit=260,
                    ),
                    payload=ClaimPayload(
                        claim=(
                            f"Attempt {attempt_index} requires a structured repair handoff "
                            f"for {problem.problem_id}."
                        ),
                        strength=ClaimStrength.TENTATIVE,
                    ),
                    evidence=tuple(evidence),
                    tags=("livecodebench", "repair_handoff", failure_signature.failure_kind),
                    visibility=session.scope,
                    validity=ValidityState(
                        status=ValidityStatus.VERIFIED,
                        dependency_fingerprint=session.dependency,
                        reason="Repair handoff distilled from visible public feedback for the current run",
                    ),
                    metadata={
                        "problem_id": problem.problem_id,
                        "memory_role": "repair_handoff",
                        "attempt_index": attempt_index,
                        "failure_kind": repair_context.failure_kind,
                        "bug_class": repair_context.bug_class,
                        "function_name": failure_signature.function_name,
                        "feedback_signature": repair_context.failure_signature,
                    },
                ),
            )
            for constraint_index, constraint in enumerate(repair_context.preserve_constraints[:3], start=1):
                session.kernel.stage_memory_item(
                    tx.tx_id,
                    MemoryItem(
                        kind=MemoryKind.CLAIM,
                        title=f"Attempt {attempt_index} repair constraint {constraint_index}",
                        summary=compact_text(constraint, limit=220),
                        payload=ClaimPayload(
                            claim=constraint,
                            strength=ClaimStrength.SUPPORTED,
                        ),
                        evidence=tuple(evidence),
                        tags=("livecodebench", "repair_constraint"),
                        visibility=session.scope,
                        validity=ValidityState(
                            status=ValidityStatus.VERIFIED,
                            dependency_fingerprint=session.dependency,
                            reason="Repair constraints derived from the current public contract and failure",
                        ),
                        metadata={
                            "problem_id": problem.problem_id,
                            "memory_role": "repair_constraint",
                            "attempt_index": attempt_index,
                            "constraint_index": constraint_index,
                        },
                    ),
                )
        session.kernel.stage_memory_item(
            tx.tx_id,
            MemoryItem(
                kind=MemoryKind.CLAIM,
                title=f"Attempt {attempt_index} visible feedback",
                summary=feedback_summary,
                payload=ClaimPayload(
                    claim=feedback_summary,
                    strength=ClaimStrength.SUPPORTED,
                ),
                evidence=tuple(evidence),
                tags=("livecodebench", "feedback"),
                visibility=session.scope,
                validity=ValidityState(
                    status=ValidityStatus.VERIFIED,
                    dependency_fingerprint=session.dependency,
                    reason="Visible execution feedback captured during the current run",
                ),
                metadata={"problem_id": problem.problem_id, "memory_role": "feedback"},
            ),
        )
        for feedback_index, feedback_text in enumerate(feedback_summaries[:3], start=1):
            feedback_item_summary = compact_text(feedback_text, limit=220)
            session.kernel.stage_memory_item(
                tx.tx_id,
                MemoryItem(
                    kind=MemoryKind.CLAIM,
                    title=f"Attempt {attempt_index} feedback item {feedback_index}",
                    summary=feedback_item_summary,
                    payload=ClaimPayload(
                        claim=feedback_text,
                        strength=ClaimStrength.SUPPORTED,
                    ),
                    evidence=tuple(evidence),
                    tags=("livecodebench", "feedback_item"),
                    visibility=session.scope,
                    validity=ValidityState(
                        status=ValidityStatus.VERIFIED,
                        dependency_fingerprint=session.dependency,
                        reason="Visible execution feedback captured during the current run",
                    ),
                    metadata={
                        "problem_id": problem.problem_id,
                        "memory_role": "feedback_item",
                        "feedback_index": feedback_index,
                    },
                ),
            )
        session.kernel.stage_memory_item(
            tx.tx_id,
            MemoryItem(
                kind=MemoryKind.CLAIM,
                title=f"Attempt {attempt_index} refuted answer",
                summary=f"Attempt {attempt_index} was refuted by visible feedback.",
                payload=ClaimPayload(
                    claim=f"Attempt {attempt_index} was refuted by visible feedback.",
                    strength=ClaimStrength.SUPPORTED,
                ),
                evidence=tuple(evidence),
                tags=("livecodebench", "refuted"),
                visibility=session.scope,
                validity=ValidityState(
                    status=ValidityStatus.REFUTED,
                    reason="Visible feedback contradicted the prior answer",
                ),
                metadata={"problem_id": problem.problem_id, "memory_role": "refuted_answer"},
            ),
        )
    session.kernel.commit_transaction(tx.tx_id)


def write_persistent_success_memory(
    session: PilotMemorySession,
    *,
    problem: LiveCodeBenchProblem,
    attempt_index: int,
    response_text: str,
    extracted_code: str | None,
    evaluation: Mapping[str, Any] | None,
    visible_feedback: Sequence[str] = (),
    memory_write_proposals: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    if evaluation is not None and evaluation.get("passed") is not True:
        return None
    resolved_code = (extracted_code or extract_code(response_text) or response_text).strip()
    if not resolved_code:
        return None

    required_func_name = _required_function_name(problem)
    failure_signature = _parse_failure_signature(
        problem,
        visible_feedback,
        response_text=response_text,
        extracted_code=resolved_code,
        evaluation=evaluation,
    )
    response_record = session.kernel.capture_artifact(
        response_text.encode("utf-8"),
        content_type="text/plain",
        tool_name="run_livecodebench_dspy_pilot",
        metadata={"kind": "persistent_success_response", "problem_id": problem.problem_id},
    )
    code_record = session.kernel.capture_artifact(
        resolved_code.encode("utf-8"),
        content_type="text/x-python",
        tool_name="run_livecodebench_dspy_pilot",
        metadata={"kind": "persistent_success_code", "problem_id": problem.problem_id},
    )
    evaluation_record = session.kernel.capture_artifact(
        json.dumps(
            {
                "problem_id": problem.problem_id,
                "attempt_index": attempt_index,
                "evaluation": dict(evaluation or {}),
                "visible_feedback": list(visible_feedback),
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
        content_type="application/json",
        tool_name="run_livecodebench_dspy_pilot",
        metadata={"kind": "persistent_success_evaluation", "problem_id": problem.problem_id},
    )
    summary = _successful_solution_summary(
        problem,
        code=resolved_code,
        visible_feedback=visible_feedback,
    )
    rationale = _successful_solution_rationale(
        problem,
        code=resolved_code,
        response_text=response_text,
        evaluation=evaluation,
    )
    tags = ["livecodebench", "successful_solution", problem.problem_id]
    platform_name = _platform_name(problem)
    difficulty_name = _difficulty_name(problem)
    interface_mode = _interface_mode(problem)
    repair_kind = _repair_kind(problem, visible_feedback)
    transfer_terms = _persistent_transfer_terms(problem, visible_feedback)
    if platform_name:
        tags.append(platform_name)
    if difficulty_name:
        tags.append(difficulty_name)
    if required_func_name:
        tags.append(required_func_name)
    tags.append(interface_mode)
    tags.append(repair_kind)
    tags.extend(transfer_terms)

    tx = session.kernel.begin_transaction(
        session.scope,
        metadata={"problem_id": problem.problem_id, "attempt_index": attempt_index},
    )
    memory = MemoryItem(
        memory_id=success_memory_id(problem.problem_id),
        kind=MemoryKind.DECISION,
        title=(
            f"Successful {interface_mode} solution pattern"
            if not required_func_name
            else f"Successful {interface_mode} pattern for {required_func_name}"
        ),
        summary=summary,
        payload=DecisionPayload(summary=summary, rationale=rationale),
        evidence=(
            session.kernel.artifact_evidence(
                response_record,
                label="persistent-success-response",
                summary="Successful model response promoted into persistent pilot memory",
            ),
            session.kernel.artifact_evidence(
                code_record,
                label="persistent-success-code",
                summary="Successful extracted code promoted into persistent pilot memory",
            ),
            session.kernel.artifact_evidence(
                evaluation_record,
                label="persistent-success-evaluation",
                summary="Public test evaluation summary for the successful solution",
            ),
        ),
        tags=tuple(tags),
        visibility=session.scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=session.dependency,
            reason="Successful public-test solve promoted into persistent LiveCodeBench pilot memory",
        ),
        metadata={
            "problem_id": problem.problem_id,
            "memory_role": "successful_solution",
            "attempt_index": attempt_index,
            "scenario": problem.scenario,
            "function_name": required_func_name,
            "platform": platform_name,
            "difficulty": difficulty_name,
            "interface_mode": interface_mode,
            "repair_kind": repair_kind,
            "failure_kind": failure_signature.failure_kind,
            "bug_class": failure_signature.bug_class,
            "repair_target": failure_signature.repair_target,
            "transfer_terms": transfer_terms,
        },
    )
    session.kernel.stage_memory_item(tx.tx_id, memory)
    promoted_memory_ids = [memory.memory_id]
    repair_memory = _persistent_repair_memory(
        session,
        problem=problem,
        attempt_index=attempt_index,
        code=resolved_code,
        visible_feedback=visible_feedback,
        failure_signature=failure_signature,
        response_record=response_record,
        code_record=code_record,
        evaluation_record=evaluation_record,
    )
    if repair_memory is not None:
        session.kernel.stage_memory_item(tx.tx_id, repair_memory)
        promoted_memory_ids.append(repair_memory.memory_id)
    agent_memory_ids = _stage_agent_memory_write_proposals(
        session,
        tx_id=tx.tx_id,
        problem=problem,
        attempt_index=attempt_index,
        response_record=response_record,
        code_record=code_record,
        evaluation_record=evaluation_record,
        visible_feedback=visible_feedback,
        memory_write_proposals=memory_write_proposals,
    )
    promoted_memory_ids.extend(agent_memory_ids)
    session.kernel.commit_transaction(tx.tx_id)
    consolidation_result = consolidate_persistent_repair_memory(session)
    return {
        "success_memory_id": memory.memory_id,
        "promoted_memory_ids": promoted_memory_ids,
        "agent_memory_ids": agent_memory_ids,
        "consolidated_memory_ids": consolidation_result["created_memory_ids"],
    }


def merge_memory_cards(
    *card_groups: Sequence[Mapping[str, Any]],
    attempt_index: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group in card_groups:
        for card in group:
            card_id = str(card.get("id") or "").strip()
            if card_id and card_id in seen_ids:
                continue
            merged.append(dict(card))
            if card_id:
                seen_ids.add(card_id)
    merged.sort(
        key=lambda card: (
            PROMPT_CARD_ROLE_PRIORITY.get(str(card.get("role") or "").strip(), 10),
            -float(card.get("score") or 0.0),
            str(card.get("title") or ""),
        )
    )
    return _budget_memory_cards(merged, attempt_index=attempt_index, limit=limit)


def _successful_solution_summary(
    problem: LiveCodeBenchProblem,
    *,
    code: str,
    visible_feedback: Sequence[str] = (),
) -> str:
    transfer_terms = _persistent_transfer_terms(problem, visible_feedback)
    parts = [
        "Reusable successful solution pattern.",
        f"Interface mode: {_interface_mode(problem)}.",
        f"Repair kind: {_repair_kind(problem, visible_feedback)}.",
    ]
    required_func_name = _required_function_name(problem)
    if required_func_name:
        parts.append(f"Expose top-level callable `{required_func_name}`.")
    platform_name = _platform_name(problem)
    if platform_name:
        parts.append(f"Platform: {platform_name}.")
    difficulty_name = _difficulty_name(problem)
    if difficulty_name:
        parts.append(f"Difficulty: {difficulty_name}.")
    if visible_feedback:
        parts.append(
            "Resolved visible feedback: "
            + compact_text(" ".join(str(item) for item in visible_feedback), limit=90)
        )
    if transfer_terms:
        parts.append(f"Transfer terms: {' '.join(transfer_terms[:8])}.")
    parts.append(f"Public contract: {compact_text(_function_contract_claim(problem), limit=100)}")
    parts.append(f"Code shape: {compact_text(code, limit=120)}")
    return compact_text(" ".join(parts), limit=220)


def repair_memory_id(problem_id: str, visible_feedback: Sequence[str]) -> str:
    signature = " | ".join(compact_text(str(item), limit=160) for item in visible_feedback if str(item).strip())
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return f"livecodebench-repair-{problem_id}-{digest}"


def _persistent_repair_memory(
    session: PilotMemorySession,
    *,
    problem: LiveCodeBenchProblem,
    attempt_index: int,
    code: str,
    visible_feedback: Sequence[str],
    failure_signature: FailureSignature,
    response_record: Any,
    code_record: Any,
    evaluation_record: Any,
) -> MemoryItem | None:
    if attempt_index <= 1 or not any(str(item).strip() for item in visible_feedback):
        return None
    required_func_name = _required_function_name(problem)
    summary = _repair_lesson_summary(
        problem,
        code=code,
        visible_feedback=visible_feedback,
    )
    rationale = _repair_lesson_rationale(
        problem,
        code=code,
        visible_feedback=visible_feedback,
    )
    tags = ["livecodebench", "repair_lesson", problem.problem_id]
    platform_name = _platform_name(problem)
    difficulty_name = _difficulty_name(problem)
    interface_mode = _interface_mode(problem)
    repair_kind = _repair_kind(problem, visible_feedback)
    transfer_terms = _persistent_transfer_terms(problem, visible_feedback)
    if required_func_name:
        tags.append(required_func_name)
    if platform_name:
        tags.append(platform_name)
    if difficulty_name:
        tags.append(difficulty_name)
    tags.append(interface_mode)
    tags.append(repair_kind)
    tags.append(failure_signature.failure_kind)
    if failure_signature.bug_class and failure_signature.bug_class != "unknown":
        tags.append(failure_signature.bug_class)
    if failure_signature.repair_target:
        tags.append(str(failure_signature.repair_target))
    tags.extend(_feedback_signature_tags(visible_feedback))
    tags.extend(transfer_terms)
    return MemoryItem(
        memory_id=repair_memory_id(problem.problem_id, visible_feedback),
        kind=MemoryKind.DECISION,
        title=f"Repair lesson: {repair_kind} on {interface_mode}",
        summary=summary,
        payload=DecisionPayload(summary=summary, rationale=rationale),
        evidence=(
            session.kernel.artifact_evidence(
                response_record,
                label="persistent-repair-response",
                summary="Successful repaired response used to derive a repair lesson",
            ),
            session.kernel.artifact_evidence(
                code_record,
                label="persistent-repair-code",
                summary="Successful repaired code used to derive a repair lesson",
            ),
            session.kernel.artifact_evidence(
                evaluation_record,
                label="persistent-repair-evaluation",
                summary="Visible failure signature resolved by the successful repair",
            ),
        ),
        tags=tuple(dict.fromkeys(tags)),
        visibility=session.scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=session.dependency,
            reason="Verified public-test repair promoted into persistent distilled repair memory",
        ),
        metadata={
            "problem_id": problem.problem_id,
            "memory_role": "repair_lesson",
            "attempt_index": attempt_index,
            "scenario": problem.scenario,
            "function_name": required_func_name,
            "platform": platform_name,
            "difficulty": difficulty_name,
            "interface_mode": interface_mode,
            "repair_kind": repair_kind,
            "failure_kind": failure_signature.failure_kind,
            "bug_class": failure_signature.bug_class,
            "repair_target": failure_signature.repair_target,
            "transfer_terms": transfer_terms,
            "feedback_signature": " | ".join(
                compact_text(str(item), limit=160)
                for item in visible_feedback
                if str(item).strip()
            ),
        },
    )


def _stage_agent_memory_write_proposals(
    session: PilotMemorySession,
    *,
    tx_id: str,
    problem: LiveCodeBenchProblem,
    attempt_index: int,
    response_record: Any,
    code_record: Any,
    evaluation_record: Any,
    visible_feedback: Sequence[str],
    memory_write_proposals: Sequence[Mapping[str, Any]],
) -> list[str]:
    promoted_ids: list[str] = []
    failure_signature = _parse_failure_signature(problem, visible_feedback)
    for proposal in _validated_agent_memory_write_proposals(
        problem,
        attempt_index=attempt_index,
        visible_feedback=visible_feedback,
        memory_write_proposals=memory_write_proposals,
    ):
        proposal_record = session.kernel.capture_artifact(
            json.dumps(proposal, indent=2, sort_keys=True).encode("utf-8"),
            content_type="application/json",
            tool_name="run_livecodebench_dspy_pilot",
            metadata={
                "kind": "agent_memory_write_proposal",
                "problem_id": problem.problem_id,
                "attempt_index": attempt_index,
                "memory_role": proposal["memory_role"],
            },
        )
        memory = MemoryItem(
            memory_id=_agent_memory_proposal_id(proposal),
            kind=MemoryKind.DECISION,
            title=str(proposal["title"]),
            summary=str(proposal["summary"]),
            payload=DecisionPayload(
                summary=str(proposal["summary"]),
                rationale=str(proposal["rationale"]) if proposal.get("rationale") else None,
            ),
            evidence=(
                session.kernel.artifact_evidence(
                    proposal_record,
                    label="agent-memory-proposal",
                    summary="Agent-authored memory proposal accepted by the host after a verified solve",
                ),
                session.kernel.artifact_evidence(
                    response_record,
                    label="agent-memory-response",
                    summary="Successful response associated with the promoted agent memory proposal",
                ),
                session.kernel.artifact_evidence(
                    code_record,
                    label="agent-memory-code",
                    summary="Successful code associated with the promoted agent memory proposal",
                ),
                session.kernel.artifact_evidence(
                    evaluation_record,
                    label="agent-memory-evaluation",
                    summary="Public test evaluation used to verify the promoted agent memory proposal",
                ),
            ),
            tags=tuple(
                dict.fromkeys(
                    [
                        "livecodebench",
                        "agent_memory_proposal",
                        str(proposal["memory_role"]),
                        str(proposal["interface_mode"]),
                        str(proposal["repair_kind"]),
                        *tuple(proposal["transfer_terms"]),
                    ]
                )
            ),
            visibility=session.scope,
            validity=ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=session.dependency,
                reason=(
                    "Agent-authored lesson promoted only after a verified public-test solve"
                ),
            ),
            metadata={
                "problem_id": problem.problem_id,
                "memory_role": proposal["memory_role"],
                "attempt_index": attempt_index,
                "scenario": problem.scenario,
                "function_name": proposal["function_name"],
                "platform": _platform_name(problem),
                "difficulty": _difficulty_name(problem),
                "interface_mode": proposal["interface_mode"],
                "repair_kind": proposal["repair_kind"],
                "failure_kind": failure_signature.failure_kind,
                "repair_target": failure_signature.repair_target,
                "transfer_terms": proposal["transfer_terms"],
                "feedback_signature": proposal["failure_signature"],
                "agent_memory_proposal": True,
                "proposal_kind": proposal["proposal_kind"],
                "bug_class": proposal["bug_class"],
            },
        )
        session.kernel.stage_memory_item(tx_id, memory)
        promoted_ids.append(memory.memory_id)
    return promoted_ids


def _validated_agent_memory_write_proposals(
    problem: LiveCodeBenchProblem,
    *,
    attempt_index: int,
    visible_feedback: Sequence[str],
    memory_write_proposals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not memory_write_proposals:
        return []
    default_function_name = _required_function_name(problem)
    default_interface_mode = _interface_mode(problem)
    default_repair_kind = _repair_kind(problem, visible_feedback)
    default_transfer_terms = _persistent_transfer_terms(problem, visible_feedback)
    default_failure_signature = _parse_failure_signature(problem, visible_feedback)
    accepted: list[dict[str, Any]] = []
    for raw_proposal in memory_write_proposals[:3]:
        if not isinstance(raw_proposal, Mapping):
            continue
        proposal_kind = str(raw_proposal.get("proposal_kind") or "memory_lesson").strip().lower()
        memory_role = _normalized_agent_memory_role(raw_proposal, attempt_index=attempt_index)
        if memory_role == "repair_lesson" and attempt_index <= 1:
            continue
        summary = compact_text(str(raw_proposal.get("summary") or "").strip(), limit=240)
        if not summary or _looks_like_code_dump(summary):
            continue
        rationale_raw = str(raw_proposal.get("rationale") or "").strip()
        rationale = None if _looks_like_code_dump(rationale_raw) else compact_text(rationale_raw, limit=600)
        failure_signature = compact_text(
            str(raw_proposal.get("failure_signature") or " | ".join(visible_feedback)).strip(),
            limit=220,
        )
        interface_mode = compact_text(
            str(raw_proposal.get("interface_mode") or default_interface_mode).strip(),
            limit=64,
        )
        repair_kind = compact_text(
            str(raw_proposal.get("repair_kind") or default_repair_kind).strip(),
            limit=64,
        )
        function_name = compact_text(
            str(raw_proposal.get("function_name") or default_function_name or "").strip(),
            limit=64,
        )
        bug_class = compact_text(
            str(raw_proposal.get("bug_class") or default_failure_signature.bug_class).strip(),
            limit=64,
        )
        transfer_terms = _normalize_agent_transfer_terms(raw_proposal.get("transfer_terms"))
        merged_transfer_terms = tuple(
            dict.fromkeys(
                term
                for term in (
                    *transfer_terms,
                    interface_mode,
                    repair_kind,
                    function_name.lower() if function_name else "",
                    *default_transfer_terms,
                )
                if term
            )
        )
        title = _normalized_agent_memory_title(
            raw_proposal=raw_proposal,
            problem=problem,
            memory_role=memory_role,
            interface_mode=interface_mode,
            repair_kind=repair_kind,
            summary=summary,
        )
        accepted.append(
            {
                "proposal_kind": proposal_kind,
                "memory_role": memory_role,
                "title": title,
                "summary": summary,
                "rationale": rationale,
                "function_name": function_name or None,
                "repair_kind": repair_kind,
                "interface_mode": interface_mode,
                "bug_class": bug_class or None,
                "failure_signature": failure_signature or None,
                "transfer_terms": merged_transfer_terms,
            }
        )
    return accepted


def _normalized_agent_memory_role(
    proposal: Mapping[str, Any],
    *,
    attempt_index: int,
) -> str:
    explicit = str(proposal.get("memory_role") or "").strip().lower()
    if explicit in {"repair_lesson", "successful_solution"}:
        return explicit
    proposal_kind = str(proposal.get("proposal_kind") or "").strip().lower()
    if proposal_kind == "solution_pattern" and attempt_index <= 1:
        return "successful_solution"
    return "repair_lesson"


def _normalized_agent_memory_title(
    *,
    raw_proposal: Mapping[str, Any],
    problem: LiveCodeBenchProblem,
    memory_role: str,
    interface_mode: str,
    repair_kind: str,
    summary: str,
) -> str:
    title = compact_text(str(raw_proposal.get("title") or "").strip(), limit=96)
    if title and problem.problem_id.lower() not in title.lower() and not _looks_like_code_dump(title):
        return title
    if memory_role == "successful_solution":
        return f"Agent solution pattern: {interface_mode}"
    return f"Agent repair lesson: {repair_kind} on {interface_mode}"


def _normalize_agent_transfer_terms(raw_terms: Any) -> tuple[str, ...]:
    if isinstance(raw_terms, str):
        candidates = raw_terms.replace("|", ",").split(",")
    elif isinstance(raw_terms, Sequence) and not isinstance(raw_terms, str | bytes):
        candidates = [str(item) for item in raw_terms]
    else:
        candidates = []
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        token = compact_text(str(candidate).strip().lower(), limit=48)
        if not token or token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    return tuple(normalized)


def _looks_like_code_dump(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return "```" in normalized or bool(
        re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", normalized)
        or re.search(r"\bclass\s+[A-Za-z_]\w*", normalized)
    )


def _agent_memory_proposal_id(proposal: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "memory_role": proposal.get("memory_role"),
                "title": proposal.get("title"),
                "summary": proposal.get("summary"),
                "function_name": proposal.get("function_name"),
                "repair_kind": proposal.get("repair_kind"),
                "interface_mode": proposal.get("interface_mode"),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"lcb_agent_memory_{digest[:24]}"


def consolidate_persistent_repair_memory(session: PilotMemorySession) -> dict[str, Any]:
    groups = _persistent_repair_consolidation_groups(session)
    created_memory_ids: list[str] = []
    updated_memory_ids: list[str] = []
    for group_key, group in groups.items():
        summary_card = _build_canonical_repair_summary_card(
            session,
            group_key=group_key,
            memories=group,
        )
        existing = session.metadata_store.get_memory_item(summary_card.memory_id)
        if _same_summary_card(existing, summary_card):
            continue
        session.metadata_store.save_memory_item(summary_card)
        if existing is None:
            created_memory_ids.append(summary_card.memory_id)
        else:
            updated_memory_ids.append(summary_card.memory_id)
    return {
        "group_count": len(groups),
        "created_memory_ids": created_memory_ids,
        "updated_memory_ids": updated_memory_ids,
    }


def _persistent_repair_consolidation_groups(
    session: PilotMemorySession,
) -> dict[tuple[str, ...], list[MemoryItem]]:
    groups: dict[tuple[str, ...], list[MemoryItem]] = {}
    for memory in session.metadata_store.list_memory_items():
        if _memory_role(memory) != "repair_lesson":
            continue
        if memory.validity.status not in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}:
            continue
        group_key = _canonical_repair_group_key(memory)
        if group_key is None:
            continue
        groups.setdefault(group_key, []).append(memory)
    return {
        key: sorted(
            group,
            key=lambda item: (-item.updated_at.timestamp(), item.memory_id),
        )
        for key, group in groups.items()
        if len(group) >= 2
    }


def _canonical_repair_group_key(memory: MemoryItem) -> tuple[str, ...] | None:
    metadata = memory.metadata
    memory_role = str(metadata.get("memory_role") or "").strip()
    if memory_role != "repair_lesson":
        return None
    scenario = str(metadata.get("scenario") or "").strip().lower() or "self_repair"
    interface_mode = str(metadata.get("interface_mode") or "").strip().lower()
    repair_kind = str(metadata.get("repair_kind") or "").strip().lower()
    failure_kind = str(metadata.get("failure_kind") or "").strip().lower()
    bug_class = str(metadata.get("bug_class") or "").strip().lower()
    function_name = str(metadata.get("function_name") or "").strip().lower()
    if not interface_mode or not repair_kind or not failure_kind:
        return None
    return (
        scenario,
        interface_mode,
        repair_kind,
        failure_kind,
        bug_class,
        function_name,
    )


def _canonical_repair_summary_id(group_key: tuple[str, ...]) -> str:
    digest = hashlib.sha256("|".join(group_key).encode("utf-8")).hexdigest()
    return f"lcb_canonical_repair_{digest[:24]}"


def _build_canonical_repair_summary_card(
    session: PilotMemorySession,
    *,
    group_key: tuple[str, ...],
    memories: Sequence[MemoryItem],
) -> MemoryItem:
    scenario, interface_mode, repair_kind, failure_kind, bug_class, function_name = group_key
    supporting_ids = tuple(memory.memory_id for memory in memories)
    top_terms = _top_group_transfer_terms(memories)
    title = f"Canonical repair lesson: {repair_kind} on {interface_mode}"
    if function_name:
        title += f" for {function_name}"
    summary_parts = [
        "Canonical repair lesson distilled from verified LiveCodeBench repairs.",
        f"Repair kind: {repair_kind}.",
        f"Failure kind: {failure_kind}.",
        f"Interface mode: {interface_mode}.",
        f"Supports {len(memories)} related repairs.",
    ]
    if function_name:
        summary_parts.append(f"Function shape: {function_name}.")
    if bug_class:
        summary_parts.append(f"Bug class: {bug_class}.")
    if top_terms:
        summary_parts.append(f"Frequent transfer terms: {' '.join(top_terms[:8])}.")
    summary_parts.append(
        "Representative lessons: "
        + " | ".join(compact_text(memory.summary, limit=90) for memory in memories[:2])
    )
    summary = compact_text(" ".join(summary_parts), limit=240)
    evidence = tuple(
        EvidenceRef(
            kind=EvidenceKind.MEMORY,
            ref_id=f"memory:{memory.memory_id}",
            memory_id=memory.memory_id,
            summary=compact_text(memory.summary, limit=120),
        )
        for memory in memories
    )
    rationale_lines = [
        f"Scenario: {scenario}",
        f"Repair kind: {repair_kind}",
        f"Failure kind: {failure_kind}",
        f"Interface mode: {interface_mode}",
        f"Supporting memories: {', '.join(supporting_ids)}",
    ]
    if top_terms:
        rationale_lines.extend(["", "Frequent Transfer Terms:", *[f"- {term}" for term in top_terms[:10]]])
    rationale_lines.extend(
        [
            "",
            "Supporting Lesson Summaries:",
            *[f"- {compact_text(memory.summary, limit=220)}" for memory in memories[:4]],
        ]
    )
    dependency = memories[0].validity.dependency_fingerprint or session.dependency
    return MemoryItem(
        memory_id=_canonical_repair_summary_id(group_key),
        kind=MemoryKind.SUMMARY_CARD,
        title=title,
        summary=summary,
        payload=SummaryCardPayload(
            summary=summary,
            detail_level=DetailLevel.SUMMARY,
            supporting_memory_ids=supporting_ids,
        ),
        evidence=evidence,
        tags=tuple(
            dict.fromkeys(
                [
                    "livecodebench",
                    "canonical_repair_lesson",
                    repair_kind,
                    failure_kind,
                    interface_mode,
                    *top_terms[:10],
                ]
            )
        ),
        visibility=session.scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=dependency,
            checked_at=utc_now(),
            reason="generated by livecodebench persistent repair consolidation",
        ),
        metadata={
            "memory_role": "canonical_repair_lesson",
            "scenario": scenario,
            "function_name": function_name or None,
            "platform": str(memories[0].metadata.get("platform") or "").strip() or None,
            "difficulty": str(memories[0].metadata.get("difficulty") or "").strip() or None,
            "interface_mode": interface_mode,
            "repair_kind": repair_kind,
            "failure_kind": failure_kind,
            "bug_class": bug_class or None,
            "transfer_terms": top_terms,
            "canonical_support_count": len(memories),
            "canonical_supporting_memory_ids": supporting_ids,
            "generated_by": "livecodebench_repair_consolidator",
            "consolidation_group_key": group_key,
            "rationale_preview": compact_text("\n".join(rationale_lines), limit=220),
        },
    )


def _top_group_transfer_terms(memories: Sequence[MemoryItem], *, limit: int = 10) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for memory in memories:
        raw_terms = memory.metadata.get("transfer_terms") or ()
        if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, str | bytes):
            continue
        seen: set[str] = set()
        for raw_term in raw_terms:
            term = str(raw_term).strip().lower()
            if not term or term in seen:
                continue
            counts[term] = counts.get(term, 0) + 1
            seen.add(term)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(term for term, count in ranked if count >= 2)[:limit]


def _same_summary_card(current: MemoryItem | None, candidate: MemoryItem) -> bool:
    if current is None:
        return False
    current_support_ids = tuple(getattr(current.payload, "supporting_memory_ids", ()))
    candidate_support_ids = tuple(getattr(candidate.payload, "supporting_memory_ids", ()))
    return (
        current.kind is MemoryKind.SUMMARY_CARD
        and current.summary == candidate.summary
        and current.title == candidate.title
        and current_support_ids == candidate_support_ids
        and tuple(current.tags) == tuple(candidate.tags)
    )


def _repair_lesson_summary(
    problem: LiveCodeBenchProblem,
    *,
    code: str,
    visible_feedback: Sequence[str],
) -> str:
    required_func_name = _required_function_name(problem)
    transfer_terms = _persistent_transfer_terms(problem, visible_feedback)
    parts = [
        "Reusable repair lesson.",
        f"Repair kind: {_repair_kind(problem, visible_feedback)}.",
        f"Interface mode: {_interface_mode(problem)}.",
        "Resolved visible feedback: "
        + compact_text(" ".join(str(item) for item in visible_feedback), limit=120),
    ]
    if required_func_name:
        parts.append(f"Keep the public interface as top-level `{required_func_name}`.")
    platform_name = _platform_name(problem)
    if platform_name:
        parts.append(f"Platform: {platform_name}.")
    difficulty_name = _difficulty_name(problem)
    if difficulty_name:
        parts.append(f"Difficulty: {difficulty_name}.")
    public_tests = _public_tests_claim(problem)
    if public_tests:
        parts.append(f"Check against public examples: {compact_text(public_tests, limit=120)}")
    if transfer_terms:
        parts.append(f"Transfer terms: {' '.join(transfer_terms[:8])}.")
    parts.append(f"Successful code shape: {compact_text(code, limit=100)}")
    return compact_text(" ".join(parts), limit=240)


def _repair_lesson_rationale(
    problem: LiveCodeBenchProblem,
    *,
    code: str,
    visible_feedback: Sequence[str],
) -> str:
    sections = [
        f"Problem ID: {problem.problem_id}",
        "Resolved Feedback:",
        *[f"- {compact_text(str(item), limit=220)}" for item in visible_feedback if str(item).strip()],
    ]
    contract_claim = _function_contract_claim(problem)
    if contract_claim:
        sections.extend(["", "Interface Contract:", contract_claim])
    public_tests_claim = _public_tests_claim(problem)
    if public_tests_claim:
        sections.extend(["", "Public-Test Hints:", public_tests_claim])
    sections.extend(["", "Successful Code Shape:", compact_text(code, limit=400)])
    return "\n".join(sections).strip()


def _feedback_signature_tags(visible_feedback: Sequence[str], *, limit: int = 8) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z0-9_]+", " ".join(str(item) for item in visible_feedback))
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "public",
        "attempt",
        "expected",
        "actual",
    }
    selected: list[str] = []
    seen: set[str] = set()
    for raw_token in tokens:
        token = raw_token.lower()
        if len(token) < 3 or token in stopwords or token in seen:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _prompt_transfer_terms(problem: LiveCodeBenchProblem, *, limit: int = 10) -> tuple[str, ...]:
    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9_]+",
        " ".join(
            part
            for part in (
                problem.prompt,
                problem.starter_code or "",
                _function_contract_claim(problem) or "",
            )
        ),
    )
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "return",
        "implement",
        "provided",
        "following",
        "problem",
        "public",
        "tests",
        "code",
        "callable",
        "named",
        "module",
        "scope",
        "function",
        "input",
        "output",
        "define",
        "using",
        "same",
    }
    selected: list[str] = []
    seen: set[str] = set()
    for raw_token in tokens:
        token = raw_token.lower()
        if len(token) < 4 or token in stopwords or token in seen:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _persistent_transfer_terms(
    problem: LiveCodeBenchProblem,
    visible_feedback: Sequence[str],
) -> tuple[str, ...]:
    failure_signature = _parse_failure_signature(problem, visible_feedback)
    parts: list[str] = []
    function_name = _required_function_name(problem)
    if function_name:
        parts.append(function_name.lower())
    platform_name = _platform_name(problem)
    if platform_name:
        parts.append(platform_name)
    difficulty_name = _difficulty_name(problem)
    if difficulty_name:
        parts.append(difficulty_name)
    parts.append(_interface_mode(problem))
    parts.append(_repair_kind(problem, visible_feedback))
    parts.append(failure_signature.failure_kind)
    if failure_signature.bug_class and failure_signature.bug_class != "unknown":
        parts.append(failure_signature.bug_class)
    if failure_signature.repair_target:
        parts.append(failure_signature.repair_target)
    parts.extend(_public_test_modes(problem))
    parts.extend(_feedback_signature_tags(visible_feedback, limit=6))
    parts.extend(_prompt_transfer_terms(problem, limit=8))
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = str(part).strip().lower()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return tuple(deduped)


def _platform_name(problem: LiveCodeBenchProblem) -> str | None:
    platform_name = str(problem.prompt_metadata.get("platform", "")).strip().lower()
    return platform_name or None


def _difficulty_name(problem: LiveCodeBenchProblem) -> str | None:
    difficulty_name = str(problem.prompt_metadata.get("difficulty", "")).strip().lower()
    return difficulty_name or None


def _public_test_modes(problem: LiveCodeBenchProblem) -> tuple[str, ...]:
    public_tests = problem.evaluator_payload.get("public_tests")
    if not isinstance(public_tests, list | tuple):
        return ()
    modes: list[str] = []
    for item in public_tests:
        if not isinstance(item, Mapping):
            continue
        mode = str(item.get("testtype", "stdin")).strip().lower()
        if mode and mode not in modes:
            modes.append(mode)
    return tuple(modes)


def _interface_mode(problem: LiveCodeBenchProblem) -> str:
    modes = _public_test_modes(problem)
    if "functional" in modes:
        return "top_level_function"
    if "stdin" in modes:
        return "stdin_stdout"
    return "unknown"


def _repair_kind(problem: LiveCodeBenchProblem, visible_feedback: Sequence[str]) -> str:
    signature = _parse_failure_signature(problem, visible_feedback)
    if signature.failure_kind == "runtime_exception" and signature.exception_type is not None:
        return f"runtime_{signature.exception_type.lower()}"
    if signature.failure_kind == "public_test_logic_mismatch":
        return "public_test_logic_mismatch"
    if signature.failure_kind == "syntax_error":
        return "syntax_error_repair"
    if signature.failure_kind == "missing_top_level_function":
        return "missing_top_level_function"
    if signature.failure_kind == "wrong_interface_shape":
        return "wrong_interface_shape"
    if signature.failure_kind == "output_format_mismatch":
        return "output_format_mismatch"
    if signature.failure_kind == "empty_response_or_no_code":
        return "empty_response_or_no_code"
    if visible_feedback:
        return "feedback_guided_repair"
    return "successful_initial_solution"


def _successful_solution_rationale(
    problem: LiveCodeBenchProblem,
    *,
    code: str,
    response_text: str,
    evaluation: Mapping[str, Any] | None,
) -> str:
    sections = [
        f"Problem ID: {problem.problem_id}",
        "Problem Statement:",
        problem.prompt.strip(),
    ]
    contract_claim = _function_contract_claim(problem)
    if contract_claim:
        sections.extend(["", "Implementation Contract:", contract_claim])
    public_tests_claim = _public_tests_claim(problem)
    if public_tests_claim:
        sections.extend(["", "Public Tests:", public_tests_claim])
    if evaluation:
        sections.extend(
            [
                "",
                "Evaluation Summary:",
                json.dumps(dict(evaluation), indent=2, sort_keys=True),
            ]
        )
    sections.extend(
        [
            "",
            "Successful Response:",
            response_text.strip(),
            "",
            "Successful Extracted Code:",
            code.strip(),
        ]
    )
    return "\n".join(section for section in sections if section is not None).strip()


def build_retrieval_query(
    problem: LiveCodeBenchProblem,
    visible_feedback: Sequence[str],
    *,
    store_kind: Literal["local", "persistent"] = "local",
    failure_signature: FailureSignature | None = None,
) -> str:
    failure_signature = failure_signature or _parse_failure_signature(problem, visible_feedback)
    interface_mode = _interface_mode(problem)
    platform_name = _platform_name(problem)
    difficulty_name = _difficulty_name(problem)
    query_parts = (
        [problem.problem_id, compact_text(problem.prompt, limit=140)]
        if store_kind == "local"
        else [problem.scenario, interface_mode]
    )
    if failure_signature.function_name:
        query_parts.append(f"function {failure_signature.function_name}")
    if platform_name:
        query_parts.append(f"platform {platform_name}")
    if difficulty_name:
        query_parts.append(f"difficulty {difficulty_name}")
    query_parts.extend(_public_test_modes(problem))
    if failure_signature.exception_type:
        query_parts.append(failure_signature.exception_type)
    if failure_signature.expected_value and failure_signature.actual_value:
        query_parts.append(
            f"expected {failure_signature.expected_value} actual {failure_signature.actual_value}"
        )
    else:
        if failure_signature.expected_value:
            query_parts.append(f"expected {failure_signature.expected_value}")
        if failure_signature.actual_value:
            query_parts.append(f"actual {failure_signature.actual_value}")
    if failure_signature.summary:
        query_parts.append(failure_signature.summary)
    if failure_signature.failure_kind:
        query_parts.append(f"failure_kind {failure_signature.failure_kind}")
    if failure_signature.bug_class and failure_signature.bug_class != "unknown":
        query_parts.append(f"bug_class {failure_signature.bug_class}")
    if failure_signature.repair_target:
        query_parts.append(f"repair_target {failure_signature.repair_target}")
    if failure_signature.keywords:
        query_parts.append(" ".join(failure_signature.keywords))
    if store_kind == "persistent":
        query_parts.append(f"repair_kind {_repair_kind(problem, visible_feedback)}")
        transfer_terms = _persistent_transfer_terms(problem, visible_feedback)
        if transfer_terms:
            query_parts.append(" ".join(transfer_terms))
    return " | ".join(part for part in query_parts if part)


def retrieval_plan(
    *,
    attempt_index: int,
    store_kind: Literal["local", "persistent"],
) -> RetrievalPlan | None:
    if store_kind == "local":
        if attempt_index <= 1:
            return RetrievalPlan(allowed_roles=ATTEMPT_ONE_MEMORY_ROLES, limit=2)
        if attempt_index >= 3:
            return RetrievalPlan(allowed_roles=CHEAP_REPAIR_LOCAL_MEMORY_ROLES, limit=2)
        return RetrievalPlan(allowed_roles=REPAIR_LOCAL_MEMORY_ROLES, limit=3)
    if attempt_index <= 1:
        return None
    if attempt_index >= 3:
        return RetrievalPlan(allowed_roles=CHEAP_REPAIR_PERSISTENT_MEMORY_ROLES, limit=1)
    return RetrievalPlan(allowed_roles=REPAIR_PERSISTENT_MEMORY_ROLES, limit=3)


def retrieve_verified_memory(
    session: PilotMemorySession,
    *,
    query: str,
    attempt_index: int,
    allowed_roles: frozenset[str] | None = None,
    limit: int = 5,
    expand_top_k: int = 0,
    failure_signature: FailureSignature | None = None,
    store_kind: Literal["local", "persistent"] | None = None,
    interface_mode: str | None = None,
) -> dict[str, Any]:
    candidate_limit = max(limit * 3, 6) if allowed_roles else max(limit, 5)
    result = session.kernel.retrieve(
        RetrieveRequest(
            query=query,
            scopes=(session.scope,),
            evidence_budget=EvidenceBudget.SUMMARY_ONLY,
            limit=candidate_limit,
            current_dependency=session.dependency,
            verify_on_read=True,
            return_verified_only=True,
        )
    )
    selected_candidates = _select_prompt_candidates(
        result.candidates,
        attempt_index=attempt_index,
        allowed_roles=allowed_roles,
        limit=limit,
        failure_signature=failure_signature,
        store_kind=store_kind,
        interface_mode=interface_mode,
    )
    serialized_cards = [
        _serialize_candidate(candidate) for candidate in selected_candidates
    ]
    if expand_top_k > 0:
        serialized_cards = _expand_memory_cards(
            session,
            serialized_cards,
            attempt_index=attempt_index,
            limit=expand_top_k,
        )
    return {
        "used": bool(selected_candidates),
        "query": query,
        "cards": serialized_cards,
        "verified_count": result.verified_count,
        "stale_filtered_count": result.stale_filtered_count,
        "tool_calls": 1,
    }


def write_problem_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_text(raw: str | None, *, limit: int = 240) -> str:
    if raw is None:
        return ""
    normalized = " ".join(raw.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _failure_signature_from_handoff(
    problem: LiveCodeBenchProblem,
    repair_context: RepairHandoff,
) -> FailureSignature:
    return _parse_failure_signature(
        problem,
        repair_context.visible_feedback,
        response_text=repair_context.previous_response,
        extracted_code=repair_context.previous_code,
    )


def _build_repair_handoff(
    problem: LiveCodeBenchProblem,
    *,
    attempt_index: int,
    response_text: str,
    extracted_code: str | None,
    visible_feedback: Sequence[str],
    failure_signature: FailureSignature,
) -> RepairHandoff:
    preserve_constraints = _preserve_constraints(problem)
    repair_target = failure_signature.repair_target or failure_signature.function_name or "the previous solution"
    preserve_summary = (
        compact_text("; ".join(preserve_constraints[:2]), limit=90)
        if preserve_constraints
        else "the public contract"
    )
    repair_objective = f"Fix {repair_target} while preserving {preserve_summary}."
    return RepairHandoff(
        previous_response=response_text,
        previous_code=extracted_code,
        visible_feedback=tuple(str(item).strip() for item in visible_feedback if str(item).strip()),
        attempt_index=attempt_index,
        failure_signature=failure_signature.summary,
        failure_kind=failure_signature.failure_kind,
        bug_class=failure_signature.bug_class,
        repair_objective=compact_text(repair_objective, limit=220),
        preserve_constraints=preserve_constraints,
        public_signal_summary=compact_text(" ".join(str(item) for item in visible_feedback), limit=180),
        local_query=build_retrieval_query(
            problem,
            visible_feedback,
            store_kind="local",
            failure_signature=failure_signature,
        ),
        persistent_query=build_retrieval_query(
            problem,
            visible_feedback,
            store_kind="persistent",
            failure_signature=failure_signature,
        ),
    )


def _required_function_name(problem: LiveCodeBenchProblem) -> str | None:
    metadata = problem.evaluator_payload.get("problem_metadata")
    if not isinstance(metadata, Mapping):
        return None
    func_name = metadata.get("func_name")
    if not isinstance(func_name, str):
        return None
    stripped = func_name.strip()
    return stripped or None


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def _reference_command(config: PilotRunConfig) -> str:
    scenario_token = _external_scenario_token(config.resolved_scenario)
    command = [
        str(config.benchmark_root / ".venv" / "bin" / "python"),
        "-m",
        "lcb_runner.runner.main",
        "--model",
        config.model,
        "--scenario",
        scenario_token,
        "--temperature",
        str(config.temperature),
        "--n",
        str(config.max_problems),
    ]
    return " ".join(command)


def _external_scenario_token(scenario: PilotScenario) -> str:
    if scenario == "self_repair":
        return "selfrepair"
    return "codegeneration"


def _scenario_semantics(scenario: PilotScenario) -> str:
    if scenario == "self_repair":
        return (
            "Pilot self-repair over LiveCodeBench public code-generation problems. "
            "Attempt 2 receives the previous candidate code plus visible public-test feedback. "
            "Attempt 3, if needed, is a short repair-only pass with the previous code, "
            "parsed failure, and top repair lesson."
        )
    return "Single-pass code generation over LiveCodeBench public code-generation problems."


def _candidate_selection_semantics(candidates_per_attempt: int) -> str:
    if candidates_per_attempt <= 1:
        return "single_sample"
    return (
        f"best_of_{candidates_per_attempt}_public_tests: generate {candidates_per_attempt} candidates "
        "per attempt and keep the best public-test result for all methods."
    )


def _normalize_usage(raw_usage: Any) -> dict[str, Any] | None:
    if raw_usage is None:
        return None
    if isinstance(raw_usage, Mapping):
        return {str(key): value for key, value in raw_usage.items()}
    return {"raw": raw_usage}


def _sum_usage(
    current: dict[str, Any] | None,
    addition: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if current is None:
        return dict(addition) if addition is not None else None
    if addition is None:
        return current
    merged = dict(current)
    for key, value in addition.items():
        existing = merged.get(key)
        if isinstance(existing, int | float) and isinstance(value, int | float):
            merged[key] = existing + value
        else:
            merged[key] = value
    return merged


def _coerce_response_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("response", "answer", "output", "text", "final_answer", "completion"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(payload, indent=2, sort_keys=True)
    return str(payload)


def _candidate_passed(candidate: AttemptCandidate) -> int:
    evaluation = candidate.evaluation
    if isinstance(evaluation, Mapping) and evaluation.get("passed") is True:
        return 1
    return 0


def _candidate_pass_rate(candidate: AttemptCandidate) -> float:
    evaluation = candidate.evaluation
    if isinstance(evaluation, Mapping):
        value = evaluation.get("pass_rate")
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _candidate_passed_test_count(candidate: AttemptCandidate) -> int:
    evaluation = candidate.evaluation
    if isinstance(evaluation, Mapping):
        value = evaluation.get("passed_test_count")
        if isinstance(value, int):
            return value
    return 0


def _candidate_has_syntax_error(candidate: AttemptCandidate) -> int:
    evaluation = candidate.evaluation
    if isinstance(evaluation, Mapping) and evaluation.get("syntax_error"):
        return 1
    return 0


def _candidate_has_response_error(candidate: AttemptCandidate) -> int:
    return 1 if candidate.response_error else 0


def _attempt1_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    batches = row.get("candidate_batches")
    if not isinstance(batches, Sequence) or isinstance(batches, str | bytes) or not batches:
        return []
    batch = batches[0]
    if not isinstance(batch, Mapping):
        return []
    candidates = batch.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _attempt_candidates(row: Mapping[str, Any], *, attempt_index: int) -> list[Mapping[str, Any]]:
    batches = row.get("candidate_batches")
    if not isinstance(batches, Sequence) or isinstance(batches, str | bytes):
        return []
    for batch in batches:
        if not isinstance(batch, Mapping):
            continue
        if int(batch.get("attempt_index", 0) or 0) != attempt_index:
            continue
        candidates = batch.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes):
            return []
        return [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    return []


def _attempt1_candidate_passed(row: Mapping[str, Any], *, candidate_index: int) -> bool:
    candidates = _attempt1_candidates(row)
    zero_index = candidate_index - 1
    if zero_index < 0 or zero_index >= len(candidates):
        return False
    return candidates[zero_index].get("passed") is True


def _attempt1_any_candidate_passed(row: Mapping[str, Any]) -> bool:
    candidates = _attempt1_candidates(row)
    return any(candidate.get("passed") is True for candidate in candidates)


def _attempt_candidate_passed(
    row: Mapping[str, Any],
    *,
    attempt_index: int,
    candidate_index: int,
) -> bool:
    candidates = _attempt_candidates(row, attempt_index=attempt_index)
    zero_index = candidate_index - 1
    if zero_index < 0 or zero_index >= len(candidates):
        return False
    return candidates[zero_index].get("passed") is True


def _attempt_any_candidate_passed(row: Mapping[str, Any], *, attempt_index: int) -> bool:
    candidates = _attempt_candidates(row, attempt_index=attempt_index)
    return any(candidate.get("passed") is True for candidate in candidates)


def _attempt_any_candidate_passed_upto_k(
    row: Mapping[str, Any],
    *,
    attempt_index: int,
    k: int,
) -> bool:
    if k <= 0:
        return False
    candidates = _attempt_candidates(row, attempt_index=attempt_index)
    return any(candidate.get("passed") is True for candidate in candidates[:k])


def _attempt_pass_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    attempt_index: int,
) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {}
    max_k = max(len(_attempt_candidates(row, attempt_index=attempt_index)) for row in rows)
    if max_k <= 0:
        return {}
    curve: dict[str, float] = {}
    for k in range(1, max_k + 1):
        pass_count = sum(
            1
            for row in rows
            if _attempt_any_candidate_passed_upto_k(row, attempt_index=attempt_index, k=k)
        )
        curve[str(k)] = round(pass_count / total, 6)
    return curve


def _conditional_attempt_success_rate(
    rows: Sequence[Mapping[str, Any]],
    *,
    attempt_index: int,
    predicate: Any,
) -> float:
    filtered = [
        row
        for row in rows
        if _attempt_candidates(row, attempt_index=attempt_index) and predicate(row)
    ]
    if not filtered:
        return 0.0
    passed = sum(1 for row in filtered if _attempt_any_candidate_passed(row, attempt_index=attempt_index))
    return passed / len(filtered)


def _value_distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    return dict(sorted(counts.items()))


def _count_serialized_tool_calls(payload: Any) -> int:
    if isinstance(payload, Mapping):
        for key in ("trajectory", "tool_calls", "trace"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                return len(value)
            if isinstance(value, Mapping):
                tool_call_count = sum(
                    1 for item_key in value if str(item_key).startswith("tool_name_")
                )
                if tool_call_count:
                    return tool_call_count
            if isinstance(value, int):
                return value
    return 0


def _count_result_tool_calls(result: Any, response_payload: Any) -> int:
    response_count = _count_serialized_tool_calls(response_payload)
    if response_count:
        return response_count
    if isinstance(result, Mapping):
        return _count_serialized_tool_calls(result)
    return 0


def _serialize_candidate(candidate: Any) -> dict[str, Any]:
    memory = candidate.memory
    evidence_summary = [
        evidence.summary or evidence.ref_id
        for evidence in candidate.evidence[:3]
    ]
    path = None
    symbol = None
    for evidence in candidate.evidence:
        if evidence.code_anchor is not None:
            path = evidence.code_anchor.path
            symbol = evidence.code_anchor.symbol
            break
    return {
        "id": memory.memory_id,
        "title": memory.title,
        "summary": memory.summary,
        "status": memory.validity.status.value,
        "role": _memory_role(memory),
        "score": round(candidate.score, 6),
        "path": path,
        "symbol": symbol,
        "problem_id": str(memory.metadata.get("problem_id") or "").strip(),
        "function_name": str(memory.metadata.get("function_name") or "").strip(),
        "feedback_signature": str(memory.metadata.get("feedback_signature") or "").strip(),
        "repair_kind": str(memory.metadata.get("repair_kind") or "").strip(),
        "failure_kind": str(memory.metadata.get("failure_kind") or "").strip(),
        "bug_class": str(memory.metadata.get("bug_class") or "").strip(),
        "repair_target": str(memory.metadata.get("repair_target") or "").strip(),
        "interface_mode": str(memory.metadata.get("interface_mode") or "").strip(),
        "platform": str(memory.metadata.get("platform") or "").strip(),
        "difficulty": str(memory.metadata.get("difficulty") or "").strip(),
        "transfer_terms": tuple(memory.metadata.get("transfer_terms") or ()),
        "canonical_support_count": int(memory.metadata.get("canonical_support_count") or 0),
        "evidence_summary": evidence_summary,
    }


def _expand_memory_cards(
    session: PilotMemorySession,
    cards: Sequence[Mapping[str, Any]],
    *,
    attempt_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    if attempt_index <= 1 or limit <= 0:
        return [dict(card) for card in cards]
    prioritized_roles = (
        "canonical_repair_lesson",
        "repair_lesson",
        "function_contract",
        "public_tests",
        "successful_solution",
    )
    prioritized_ids: list[str] = []
    for role in prioritized_roles:
        for raw_card in cards:
            if _memory_card_role(raw_card) != role:
                continue
            memory_id = str(raw_card.get("id") or "").strip()
            if memory_id and memory_id not in prioritized_ids:
                prioritized_ids.append(memory_id)
            if len(prioritized_ids) >= limit:
                break
        if len(prioritized_ids) >= limit:
            break
    expanded_ids: set[str] = set(prioritized_ids)
    remaining = max(0, limit - len(expanded_ids))
    expanded_cards: list[dict[str, Any]] = []
    for raw_card in cards:
        card = dict(raw_card)
        role = _memory_card_role(card)
        memory_id = str(card.get("id") or "").strip()
        should_expand = memory_id in expanded_ids
        if not should_expand and remaining > 0 and role in prioritized_roles:
            should_expand = True
            expanded_ids.add(memory_id)
            remaining -= 1
        if should_expand and memory_id:
            memory_item = session.metadata_store.get_memory_item(memory_id)
            payload = getattr(memory_item, "payload", None)
            rationale = getattr(payload, "rationale", None)
            if isinstance(rationale, str) and rationale.strip():
                card["rationale_preview"] = compact_text(rationale, limit=220)
            elif role == "canonical_repair_lesson":
                rationale_preview = str(
                    getattr(memory_item, "metadata", {}).get("rationale_preview")
                    if memory_item is not None
                    else ""
                ).strip()
                if rationale_preview:
                    card["rationale_preview"] = rationale_preview
            evidence = session.kernel.expand(memory_id)
            extra_summaries = [
                ref.summary or ref.ref_id
                for ref in evidence
                if (ref.summary or ref.ref_id)
            ]
            if extra_summaries:
                merged = list(card.get("evidence_summary") or [])
                for summary in extra_summaries[:3]:
                    if summary not in merged:
                        merged.append(summary)
                card["evidence_summary"] = merged[:4]
        expanded_cards.append(card)
    return expanded_cards


def _render_memory_card(*, index: int, card: Mapping[str, Any]) -> list[str]:
    role = str(card.get("role") or "").strip().replace("_", " ")
    title = str(card.get("title") or "Verified memory").strip()
    path = str(card.get("path") or "").strip()
    symbol = str(card.get("symbol") or "").strip()
    location = ""
    if path and symbol:
        location = f" ({path}::{symbol})"
    elif path:
        location = f" ({path})"
    header = f"{index}. "
    if role:
        header += f"[{role}] "
    header += f"{title}{location}"
    lines = [header]
    summary = str(card.get("summary") or "").strip()
    if summary:
        lines.append(f"summary: {summary}")
    canonical_support_count = int(card.get("canonical_support_count") or 0)
    if canonical_support_count > 0:
        lines.append(f"support_count: {canonical_support_count}")
    facets = [
        ("repair_kind", str(card.get("repair_kind") or "").strip()),
        ("failure_kind", str(card.get("failure_kind") or "").strip()),
        ("bug_class", str(card.get("bug_class") or "").strip()),
        ("interface_mode", str(card.get("interface_mode") or "").strip()),
        ("platform", str(card.get("platform") or "").strip()),
        ("difficulty", str(card.get("difficulty") or "").strip()),
    ]
    facet_parts = [f"{key}={value}" for key, value in facets if value]
    if facet_parts:
        lines.append(f"facets: {', '.join(facet_parts)}")
    rationale_preview = str(card.get("rationale_preview") or "").strip()
    if rationale_preview:
        lines.append(f"rationale: {rationale_preview}")
    evidence_summary = card.get("evidence_summary")
    if isinstance(evidence_summary, Sequence) and not isinstance(evidence_summary, str | bytes):
        evidence_preview = [str(item).strip() for item in evidence_summary if str(item).strip()]
        if evidence_preview:
            lines.append(f"evidence: {'; '.join(evidence_preview[:3])}")
    return lines


def _function_contract_claim(problem: LiveCodeBenchProblem) -> str | None:
    lines: list[str] = []
    required_func_name = _required_function_name(problem)
    if required_func_name:
        lines.append(f"Expose a top-level callable named `{required_func_name}`.")
    public_tests = problem.evaluator_payload.get("public_tests")
    if isinstance(public_tests, list | tuple) and public_tests:
        test_types = {
            str(item.get("testtype", "stdin")).strip().lower()
            for item in public_tests
            if isinstance(item, Mapping)
        }
        if "functional" in test_types:
            lines.append("Public evaluation may call the function directly.")
        if not test_types or "stdin" in test_types:
            lines.append("Public evaluation may execute the module from stdin/stdout.")
    starter_preview = compact_text(problem.starter_code, limit=180)
    if starter_preview:
        lines.append(f"Starter code preview: {starter_preview}")
    if not lines:
        return None
    return "\n".join(lines)


def _public_tests_claim(problem: LiveCodeBenchProblem) -> str | None:
    public_tests = problem.evaluator_payload.get("public_tests")
    if not isinstance(public_tests, list | tuple) or not public_tests:
        return None
    lines: list[str] = []
    for index, raw_test in enumerate(public_tests[:3], start=1):
        if isinstance(raw_test, Mapping):
            test_type = str(raw_test.get("testtype", "stdin")).strip().lower()
            label = "Functional example" if test_type == "functional" else "Stdin example"
            input_text = compact_text(str(raw_test.get("input", "")), limit=80)
            output_text = compact_text(str(raw_test.get("output", "")), limit=80)
            lines.append(f"{label} {index}: input={input_text!r} output={output_text!r}")
        else:
            lines.append(f"Public test {index}: {compact_text(str(raw_test), limit=120)}")
    return "\n".join(lines) if lines else None


def _missing_symbol_from_feedback(feedback_text: str) -> str | None:
    patterns = (
        r"name ['\"]?([A-Za-z_]\w*)['\"]? is not defined",
        r"undefined symbol ['\"]?([A-Za-z_]\w*)['\"]?",
        r"missing callable ['\"]?([A-Za-z_]\w*)['\"]?",
    )
    for pattern in patterns:
        match = re.search(pattern, feedback_text, re.IGNORECASE)
        if match is not None:
            return match.group(1)
    return None


def _bug_class_from_feedback(feedback_text: str, exception_type: str | None) -> str:
    normalized_exception = (exception_type or "").lower()
    if "indexerror" in normalized_exception or "out of range" in feedback_text:
        return "indexing/bounds"
    if "keyerror" in normalized_exception:
        return "indexing/bounds"
    if "exception" in normalized_exception or normalized_exception.endswith("error"):
        return "exception handling"
    if any(token in feedback_text for token in ("mutate", "modified input", "in-place")):
        return "state mutation"
    if any(token in feedback_text for token in ("parse", "format", "newline", "whitespace")):
        return "parsing/formatting"
    if any(token in feedback_text for token in ("branch", "condition", "case", "else", "loop")):
        return "control flow"
    if any(token in feedback_text for token in ("expected=", "actual=", "mismatch", "off by", "sum", "difference")):
        return "arithmetic/logic"
    return "unknown"


def _preserve_constraints(problem: LiveCodeBenchProblem) -> tuple[str, ...]:
    constraints: list[str] = []
    required_func_name = _required_function_name(problem)
    if required_func_name:
        constraints.append(f"Keep a top-level callable named `{required_func_name}`.")
    interface_mode = _interface_mode(problem)
    if interface_mode:
        constraints.append(f"Preserve interface mode `{interface_mode}`.")
    public_test_modes = _public_test_modes(problem)
    if public_test_modes:
        constraints.append(
            "Preserve public evaluation mode(s): " + ", ".join(public_test_modes) + "."
        )
    constraints.append("Preserve behavior not contradicted by the visible failure.")
    return tuple(dict.fromkeys(constraints))


def _parse_failure_signature(
    problem: LiveCodeBenchProblem,
    visible_feedback: Sequence[str],
    *,
    response_text: str | None = None,
    extracted_code: str | None = None,
    evaluation: Mapping[str, Any] | None = None,
) -> FailureSignature:
    raw_feedback = tuple(str(item).strip() for item in visible_feedback if str(item).strip())
    joined = " | ".join(raw_feedback)
    joined_lower = joined.lower()
    resolved_code = (extracted_code or extract_code(response_text or "") or response_text or "").strip()
    required_func_name = _required_function_name(problem)
    missing_symbol = _missing_symbol_from_feedback(joined)
    syntax_error = bool(
        (isinstance(evaluation, Mapping) and evaluation.get("syntax_error"))
        or "syntaxerror" in joined_lower
        or "indentationerror" in joined_lower
    )
    timeout_or_runtime_abort = any(
        token in joined_lower for token in ("timed out", "timeout", "time limit", "killed", "abort")
    )
    output_format_issue = any(
        token in joined_lower
        for token in (
            "wrong answer format",
            "output format",
            "format mismatch",
            "stdout",
            "newline",
            "whitespace",
        )
    )
    empty_or_missing_output = (not resolved_code and not raw_feedback) or any(
        token in joined_lower
        for token in ("empty response", "no code", "no output", "empty output", "missing output")
    )
    wrong_callable_shape = bool(
        required_func_name
        and resolved_code
        and required_func_name not in resolved_code
        and any(token in resolved_code.lower() for token in ("class solution", "def solve", "def main"))
    )
    if not raw_feedback:
        return FailureSignature(
            summary="",
            failure_kind=(
                "empty_response_or_no_code"
                if empty_or_missing_output
                else "generic_feedback_guided_repair"
            ),
            bug_class="function contract violation" if wrong_callable_shape else "unknown",
            exception_type=None,
            expected_value=None,
            actual_value=None,
            function_name=required_func_name,
            missing_symbol=missing_symbol,
            wrong_callable_shape=wrong_callable_shape,
            output_format_issue=output_format_issue,
            empty_or_missing_output=empty_or_missing_output,
            syntax_error=syntax_error,
            timeout_or_runtime_abort=timeout_or_runtime_abort,
            repair_target=required_func_name or missing_symbol,
            keywords=(),
            raw_feedback=(),
        )
    exception_match = re.search(
        r"\b([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b",
        joined,
    )
    expected_match = re.search(
        r"expected\s*[=:]\s*(.+?)(?=\s+actual\s*[=:]|[|,;\n]|$)",
        joined,
        re.IGNORECASE,
    )
    actual_match = re.search(r"actual\s*[=:]\s*([^|,;\n]+)", joined, re.IGNORECASE)
    keywords = _feedback_signature_tags(raw_feedback)
    summary_parts = [compact_text(raw_feedback[0], limit=120)]
    if len(raw_feedback) > 1:
        summary_parts.append(compact_text(raw_feedback[1], limit=80))
    if exception_match is not None:
        summary_parts.append(exception_match.group(1))
    summary = compact_text(" | ".join(part for part in summary_parts if part), limit=220)
    missing_top_level_function = bool(
        required_func_name
        and (
            missing_symbol == required_func_name
            or any(
                token in joined_lower
                for token in (
                    f"name '{required_func_name.lower()}' is not defined",
                    f"{required_func_name.lower()} is not defined",
                    f"missing {required_func_name.lower()}",
                    f"undefined symbol {required_func_name.lower()}",
                )
            )
        )
    )
    wrong_callable_shape = wrong_callable_shape or bool(
        required_func_name
        and not missing_top_level_function
        and any(
            token in joined_lower
            for token in ("class solution", "top-level callable", "module scope", "wrong signature")
        )
    )
    if syntax_error:
        failure_kind = "syntax_error"
        bug_class = "parsing/formatting"
    elif empty_or_missing_output:
        failure_kind = "empty_response_or_no_code"
        bug_class = "unknown"
    elif missing_top_level_function:
        failure_kind = "missing_top_level_function"
        bug_class = "function contract violation"
    elif wrong_callable_shape:
        failure_kind = "wrong_interface_shape"
        bug_class = "function contract violation"
    elif output_format_issue:
        failure_kind = "output_format_mismatch"
        bug_class = "parsing/formatting"
    elif exception_match is not None or timeout_or_runtime_abort:
        failure_kind = "runtime_exception"
        bug_class = _bug_class_from_feedback(
            joined_lower,
            exception_match.group(1) if exception_match is not None else None,
        )
    elif expected_match is not None or actual_match is not None:
        failure_kind = "public_test_logic_mismatch"
        bug_class = _bug_class_from_feedback(joined_lower, None)
    else:
        failure_kind = "generic_feedback_guided_repair"
        bug_class = _bug_class_from_feedback(joined_lower, None)
    repair_target = required_func_name or missing_symbol
    if failure_kind == "output_format_mismatch":
        repair_target = "the output format"
    elif failure_kind == "public_test_logic_mismatch":
        repair_target = repair_target or "the failing logic"
    elif failure_kind == "runtime_exception":
        repair_target = repair_target or (
            exception_match.group(1) if exception_match is not None else "the runtime failure"
        )
    return FailureSignature(
        summary=summary,
        failure_kind=failure_kind,
        bug_class=bug_class,
        exception_type=exception_match.group(1) if exception_match is not None else None,
        expected_value=(
            compact_text(expected_match.group(1).strip().strip("'\""), limit=40)
            if expected_match is not None
            else None
        ),
        actual_value=(
            compact_text(actual_match.group(1).strip().strip("'\""), limit=40)
            if actual_match is not None
            else None
        ),
        function_name=required_func_name,
        missing_symbol=missing_symbol,
        wrong_callable_shape=wrong_callable_shape,
        output_format_issue=output_format_issue,
        empty_or_missing_output=empty_or_missing_output,
        syntax_error=syntax_error,
        timeout_or_runtime_abort=timeout_or_runtime_abort,
        repair_target=repair_target,
        keywords=keywords,
        raw_feedback=raw_feedback,
    )


def _memory_role(memory: Any) -> str:
    metadata = getattr(memory, "metadata", None)
    if isinstance(metadata, Mapping):
        role = metadata.get("memory_role")
        if isinstance(role, str) and role.strip():
            return role.strip()
    return ""


def _memory_card_role(card: Mapping[str, Any]) -> str:
    return str(card.get("role") or "").strip()


def _card_dedupe_key(card: Mapping[str, Any]) -> tuple[str, ...]:
    role = _memory_card_role(card)
    problem_id = str(card.get("problem_id") or "").strip().lower()
    function_name = str(card.get("function_name") or "").strip().lower()
    feedback_signature = str(card.get("feedback_signature") or "").strip().lower()
    if role == "repair_lesson":
        return (role, function_name, feedback_signature or _normalized_card_text(card))
    if role == "canonical_repair_lesson":
        return (
            role,
            function_name,
            str(card.get("repair_kind") or "").strip().lower(),
            str(card.get("failure_kind") or "").strip().lower(),
            str(card.get("interface_mode") or "").strip().lower(),
        )
    if role == "repair_handoff":
        return (role, str(card.get("id") or "").strip().lower(), problem_id)
    if role == "successful_solution":
        return (role, problem_id, function_name, _normalized_card_text(card))
    return (role, _normalized_card_text(card))


def _normalized_card_text(card: Mapping[str, Any], *, limit: int = 16) -> str:
    text = " ".join(
        part
        for part in (
            str(card.get("title") or "").strip(),
            str(card.get("summary") or "").strip(),
        )
        if part
    )
    tokens = _feedback_signature_tags((text,), limit=limit)
    return " ".join(tokens)


def _card_metadata_match(card: Mapping[str, Any], key: str, expected: str | None) -> bool:
    if not expected:
        return False
    return str(card.get(key) or "").strip().lower() == expected.strip().lower()


def _candidate_metadata_match(candidate: Any, key: str, expected: str | None) -> bool:
    memory = getattr(candidate, "memory", None)
    metadata = getattr(memory, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    if not expected:
        return False
    return str(metadata.get(key) or "").strip().lower() == expected.strip().lower()


def _repair_relevance_score(
    candidate: Any,
    *,
    attempt_index: int,
    failure_signature: FailureSignature | None,
    store_kind: Literal["local", "persistent"] | None,
    interface_mode: str | None,
) -> tuple[Any, ...]:
    role = _memory_role(candidate.memory)
    memory = getattr(candidate, "memory", None)
    metadata = getattr(memory, "metadata", {}) if memory is not None else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    if attempt_index <= 1 or failure_signature is None:
        return (
            PROMPT_CARD_ROLE_PRIORITY.get(role, 10),
            -float(candidate.score),
            getattr(candidate.memory, "title", "") or "",
        )
    local_store_priority = {
        "repair_handoff": 0,
        "repair_constraint": 1,
        "feedback_item": 2,
        "feedback": 3,
        "public_tests": 4,
        "function_contract": 5,
        "repair_lesson": 6,
        "canonical_repair_lesson": 7,
        "successful_solution": 8,
    }
    persistent_store_priority = {
        "canonical_repair_lesson": 0,
        "repair_lesson": 1,
        "function_contract": 2,
        "public_tests": 3,
        "repair_handoff": 4,
        "repair_constraint": 5,
        "feedback_item": 6,
        "feedback": 7,
        "successful_solution": 8,
    }
    role_priority = (
        local_store_priority.get(role, PROMPT_CARD_ROLE_PRIORITY.get(role, 10))
        if store_kind == "local"
        else (
            persistent_store_priority.get(role, PROMPT_CARD_ROLE_PRIORITY.get(role, 10))
            if store_kind == "persistent"
            else PROMPT_CARD_ROLE_PRIORITY.get(role, 10)
        )
    )
    interface_match = _candidate_metadata_match(candidate, "interface_mode", interface_mode)
    failure_kind_match = _candidate_metadata_match(candidate, "failure_kind", failure_signature.failure_kind)
    bug_class_match = _candidate_metadata_match(candidate, "bug_class", failure_signature.bug_class)
    repair_target_match = _candidate_metadata_match(candidate, "repair_target", failure_signature.repair_target)
    function_match = _candidate_metadata_match(candidate, "function_name", failure_signature.function_name)
    return (
        role_priority,
        0 if failure_kind_match else 1,
        0 if bug_class_match else 1,
        0 if repair_target_match else 1,
        0 if function_match else 1,
        0 if interface_match else 1,
        -float(candidate.score),
        getattr(candidate.memory, "title", "") or "",
    )


def _budget_memory_cards(
    cards: Sequence[Mapping[str, Any]],
    *,
    attempt_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    resolved_limit = min(
        limit,
        ATTEMPT_ONE_PROMPT_CARD_LIMIT
        if attempt_index <= 1
        else (CHEAP_REPAIR_PROMPT_CARD_LIMIT if attempt_index >= 3 else REPAIR_PROMPT_CARD_LIMIT),
    )
    has_repair_lesson = any(
        _memory_card_role(card) in {"canonical_repair_lesson", "repair_lesson"} for card in cards
    )
    selected: list[dict[str, Any]] = []
    seen_dedupe_keys: set[tuple[str, ...]] = set()

    def try_add(raw_card: Mapping[str, Any]) -> bool:
        role = _memory_card_role(raw_card)
        if role == "successful_solution" and has_repair_lesson:
            return False
        dedupe_key = _card_dedupe_key(raw_card)
        if dedupe_key in seen_dedupe_keys:
            return False
        selected.append(dict(raw_card))
        seen_dedupe_keys.add(dedupe_key)
        return True

    if attempt_index <= 1:
        for raw_card in cards:
            if _memory_card_role(raw_card) not in {"function_contract", "public_tests"}:
                continue
            if try_add(raw_card) and len(selected) >= resolved_limit:
                return selected
        return selected

    reserved_groups = (
        (
            frozenset({"function_contract", "public_tests"}),
            1,
        ),
        (
            frozenset({"canonical_repair_lesson", "repair_lesson"}),
            1,
        ),
        (
            frozenset({"repair_handoff", "repair_constraint", "feedback_item", "feedback"}),
            1,
        ),
    )
    if attempt_index >= 3:
        reserved_groups = (
            (frozenset({"repair_handoff", "repair_constraint", "feedback_item", "feedback"}), 1),
            (frozenset({"canonical_repair_lesson", "repair_lesson"}), 1),
            (frozenset({"function_contract", "public_tests"}), 1),
        )

    for roles, reserved_count in reserved_groups:
        added = 0
        for raw_card in cards:
            if _memory_card_role(raw_card) not in roles:
                continue
            if try_add(raw_card):
                added += 1
            if added >= reserved_count or len(selected) >= resolved_limit:
                break
        if len(selected) >= resolved_limit:
            return selected

    for raw_card in cards:
        if try_add(raw_card) and len(selected) >= resolved_limit:
            break
    return selected


def _memory_cards_for_roles(
    memory_cards: Sequence[Mapping[str, Any]],
    roles: frozenset[str],
) -> list[Mapping[str, Any]]:
    return [card for card in memory_cards if _memory_card_role(card) in roles]


def _prompt_memory_stats(memory_cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roles = [_memory_card_role(card) for card in memory_cards]
    top_card = memory_cards[0] if memory_cards else {}
    repair_roles = {"canonical_repair_lesson", "repair_lesson"}
    return {
        "repair_memory_card_count": sum(1 for role in roles if role in repair_roles),
        "repair_lesson_card_count": sum(1 for role in roles if role == "repair_lesson"),
        "canonical_repair_card_count": sum(1 for role in roles if role == "canonical_repair_lesson"),
        "contract_card_in_prompt": any(role == "function_contract" for role in roles),
        "public_test_card_in_prompt": any(role == "public_tests" for role in roles),
        "repair_handoff_card_in_prompt": any(role == "repair_handoff" for role in roles),
        "top_prompt_memory_role": _memory_card_role(top_card) or None,
        "top_prompt_memory_failure_kind": str(top_card.get("failure_kind") or "").strip() or None,
        "top_prompt_memory_bug_class": str(top_card.get("bug_class") or "").strip() or None,
    }


def _append_memory_section(
    sections: list[str],
    *,
    title: str,
    cards: Sequence[Mapping[str, Any]],
    limit: int | None = None,
) -> None:
    if not cards:
        return
    sections.append("")
    sections.append(title)
    selected_cards = list(cards[:limit] if limit is not None else cards)
    for index, card in enumerate(selected_cards, start=1):
        sections.extend(_render_memory_card(index=index, card=card))


def _append_contract_section(
    sections: list[str],
    *,
    problem: LiveCodeBenchProblem,
    memory_cards: Sequence[Mapping[str, Any]],
) -> None:
    contract_cards = _memory_cards_for_roles(memory_cards, frozenset({"function_contract"}))
    if not contract_cards:
        return
    _append_memory_section(
        sections,
        title="Verified Contract Hints:",
        cards=contract_cards,
        limit=2,
    )


def _append_minimal_contract_snapshot(
    sections: list[str],
    *,
    problem: LiveCodeBenchProblem,
    memory_cards: Sequence[Mapping[str, Any]],
) -> bool:
    contract_cards = _memory_cards_for_roles(memory_cards, frozenset({"function_contract"}))
    contract_claim = _function_contract_claim(problem)
    if not contract_cards and not contract_claim:
        return False
    sections.append("")
    sections.append("Minimal Contract Snapshot:")
    if contract_claim:
        for line in contract_claim.splitlines():
            sections.append(f"- {line}")
    for index, card in enumerate(contract_cards[:1], start=1):
        sections.extend(_render_memory_card(index=index, card=card))
    return True


def _append_public_test_section(
    sections: list[str],
    *,
    problem: LiveCodeBenchProblem,
    memory_cards: Sequence[Mapping[str, Any]],
) -> None:
    public_test_cards = _memory_cards_for_roles(memory_cards, frozenset({"public_tests"}))
    if not problem.public_feedback and not public_test_cards:
        return
    sections.append("")
    sections.append("Public-Test Signals:")
    for feedback in problem.public_feedback:
        sections.append(f"- {feedback}")
    for index, card in enumerate(public_test_cards[:2], start=1):
        sections.extend(_render_memory_card(index=index, card=card))


def _append_public_test_snapshot(
    sections: list[str],
    *,
    problem: LiveCodeBenchProblem,
    memory_cards: Sequence[Mapping[str, Any]],
) -> bool:
    public_test_cards = _memory_cards_for_roles(memory_cards, frozenset({"public_tests"}))
    public_tests_claim = _public_tests_claim(problem)
    visible_public_feedback = [str(item).strip() for item in problem.public_feedback if str(item).strip()]
    if not public_test_cards and not public_tests_claim and not visible_public_feedback:
        return False
    sections.append("")
    sections.append("Public-Test Snapshot:")
    for feedback in visible_public_feedback[:2]:
        sections.append(f"- {feedback}")
    if public_tests_claim:
        for line in public_tests_claim.splitlines()[:2]:
            sections.append(f"- {line}")
    for index, card in enumerate(public_test_cards[:1], start=1):
        sections.extend(_render_memory_card(index=index, card=card))
    return True


def _append_visible_failure_section(
    sections: list[str],
    *,
    parsed_failure: FailureSignature,
    visible_feedback: Sequence[str],
) -> None:
    if not visible_feedback:
        return
    sections.append("")
    sections.append("Visible Failure:")
    if parsed_failure.summary:
        sections.append(f"summary: {parsed_failure.summary}")
    for feedback in visible_feedback:
        sections.append(f"- {feedback}")


def _append_repair_handoff_section(
    sections: list[str],
    *,
    repair_context: RepairHandoff | None,
    memory_cards: Sequence[Mapping[str, Any]],
    suggested_memory_query: str | None,
) -> None:
    handoff_cards = _memory_cards_for_roles(
        memory_cards,
        frozenset({"repair_handoff", "repair_constraint"}),
    )
    if repair_context is None and not handoff_cards and not suggested_memory_query:
        return
    sections.append("")
    sections.append("Repair Handoff:")
    if repair_context is not None:
        sections.append(f"Previous attempt: {repair_context.attempt_index}")
        if repair_context.failure_kind:
            sections.append(f"failure_kind: {repair_context.failure_kind}")
        if repair_context.bug_class:
            sections.append(f"bug_class: {repair_context.bug_class}")
        if repair_context.repair_objective:
            sections.append(f"repair_objective: {repair_context.repair_objective}")
        if repair_context.public_signal_summary:
            sections.append(f"public_signal_summary: {repair_context.public_signal_summary}")
        if repair_context.preserve_constraints:
            sections.append("Preserve Constraints:")
            for constraint in repair_context.preserve_constraints:
                sections.append(f"- {constraint}")
    for index, card in enumerate(handoff_cards[:2], start=1):
        sections.extend(_render_memory_card(index=index, card=card))
    resolved_query = suggested_memory_query
    if repair_context is not None:
        resolved_query = resolved_query or repair_context.persistent_query or repair_context.local_query
    if resolved_query:
        sections.append(f"Suggested memory query: {resolved_query}")


def _append_repair_lesson_section(
    sections: list[str],
    *,
    memory_cards: Sequence[Mapping[str, Any]],
    limit: int,
) -> None:
    repair_cards = _memory_cards_for_roles(
        memory_cards,
        frozenset({"canonical_repair_lesson", "repair_lesson", "successful_solution"}),
    )
    _append_memory_section(
        sections,
        title="Verified Repair Lessons:",
        cards=repair_cards,
        limit=limit,
    )


def _append_supporting_memory_section(
    sections: list[str],
    memory_cards: Sequence[Mapping[str, Any]],
) -> None:
    remaining_cards = [
        card
        for card in memory_cards
        if _memory_card_role(card)
        not in {
            "repair_handoff",
            "repair_constraint",
            "function_contract",
            "public_tests",
            "feedback",
            "feedback_item",
            "canonical_repair_lesson",
            "repair_lesson",
            "successful_solution",
        }
    ]
    _append_memory_section(
        sections,
        title="Other Verified Memory:",
        cards=remaining_cards,
        limit=2,
    )


def _select_prompt_candidates(
    candidates: Sequence[Any],
    *,
    attempt_index: int,
    allowed_roles: frozenset[str] | None = None,
    limit: int = 5,
    failure_signature: FailureSignature | None = None,
    store_kind: Literal["local", "persistent"] | None = None,
    interface_mode: str | None = None,
) -> list[Any]:
    if not candidates:
        return []
    if allowed_roles is not None:
        candidates = [
            candidate for candidate in candidates if _memory_role(candidate.memory) in allowed_roles
        ]
        if not candidates:
            return []
    role_ranked = sorted(
        candidates,
        key=lambda candidate:
            _repair_relevance_score(
                candidate,
                attempt_index=attempt_index,
                failure_signature=failure_signature,
                store_kind=store_kind,
                interface_mode=interface_mode,
            ),
    )
    non_summary = [
        candidate
        for candidate in role_ranked
        if _memory_role(candidate.memory) != "problem_summary"
    ]
    if non_summary:
        role_ranked = non_summary
    if attempt_index <= 1:
        role_ranked = [
            candidate
            for candidate in role_ranked
            if _memory_role(candidate.memory) not in {"feedback", "feedback_item", "refuted_answer"}
        ] or role_ranked
    selected: list[Any] = []
    seen_ids: set[str] = set()
    for candidate in role_ranked:
        memory_id = getattr(candidate.memory, "memory_id", "")
        if memory_id in seen_ids:
            continue
        selected.append(candidate)
        seen_ids.add(memory_id)
        if len(selected) >= limit:
            break
    return selected


def _mean_numeric(values: Iterable[Any]) -> float:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return 0.0
    return fmean(numeric_values)


def dry_run_report(config: PilotRunConfig, *, source: ProblemSource) -> dict[str, Any]:
    payload = run_pilot(config, source=source)
    payload["imports"] = {
        "dspy": importlib.util.find_spec("dspy") is not None,
        "lcb_runner": importlib.util.find_spec("lcb_runner") is not None,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = discover_problem_source(
        benchmark_root=args.benchmark_root,
        problem_file=args.problem_file,
    )
    config = resolve_config(args, source=source)
    payload = (
        dry_run_report(config, source=source)
        if config.dry_run
        else run_pilot(config, source=source)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_BENCHMARK_ROOT",
    "DEFAULT_MAX_PROBLEMS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SCENARIO",
    "LiveCodeBenchCheckoutSource",
    "METHODS",
    "LiveCodeBenchProblem",
    "MethodRuntime",
    "PilotRunConfig",
    "ProblemFileSource",
    "ProblemSource",
    "aggregate_summary",
    "build_attempt_prompt",
    "build_parser",
    "describe_method_runtime",
    "discover_problem_source",
    "execute_problem",
    "extract_code",
    "main",
    "method_run_dir",
    "method_run_paths",
    "open_memory_session",
    "resolve_config",
    "resolve_methods",
    "resolve_scenario",
]
