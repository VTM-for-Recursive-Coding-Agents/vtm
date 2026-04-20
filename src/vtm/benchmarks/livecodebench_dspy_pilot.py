"""Small LiveCodeBench pilot comparing direct, DSPy ReAct, and DSPy RLM flows."""

from __future__ import annotations

import argparse
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
from vtm.enums import ClaimStrength, EvidenceBudget, MemoryKind, ScopeKind, ValidityStatus
from vtm.fingerprints import DependencyFingerprint, EnvFingerprint, RepoFingerprint, ToolVersion
from vtm.memory_items import ClaimPayload, MemoryItem, ValidityState, VisibilityScope
from vtm.retrieval import RetrieveRequest
from vtm.services.memory_kernel import TransactionalMemoryKernel
from vtm.services.procedures import CommandProcedureValidator
from vtm.services.retriever import LexicalRetriever
from vtm.services.verifier import BasicVerifier
from vtm.stores.artifact_store import FilesystemArtifactStore
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.sqlite_store import SqliteMetadataStore
from vtm_dspy import dspy_available, require_dspy
from vtm_dspy.config import DSPyOpenRouterConfig
from vtm_dspy.react_agent import VTMReActCodingAgent
from vtm_dspy.rlm_agent import VTMRLMCodingAgent, rlm_interpreter_availability

PilotMethod = Literal[
    "direct",
    "dspy_baseline",
    "dspy_vtm",
    "dspy_rlm_baseline",
    "dspy_rlm_vtm",
]

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / ".benchmarks" / "livecodebench-dspy"
DEFAULT_BENCHMARK_ROOT: Final[Path] = PROJECT_ROOT / "benchmarks" / "LiveCodeBench"
DEFAULT_MAX_PROBLEMS: Final[int] = 3
DEFAULT_MAX_TOKENS: Final[int] = 8192
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_SCENARIO: Final[PilotScenario] = "self_repair"
DEFAULT_DIRECT_EMPTY_RESPONSE_RETRIES: Final[int] = 2
PILOT_RLM_MAX_ITERATIONS: Final[int] = 2
PILOT_RLM_MAX_LLM_CALLS: Final[int] = 4
METHODS: Final[tuple[PilotMethod, ...]] = (
    "direct",
    "dspy_baseline",
    "dspy_vtm",
    "dspy_rlm_baseline",
    "dspy_rlm_vtm",
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
    "feedback_item": 0,
    "feedback": 1,
    "public_tests": 2,
    "function_contract": 3,
    "refuted_answer": 4,
    "attempt_summary": 5,
    "problem_summary": 20,
}


@dataclass(frozen=True)
class RepairContext:
    """Public self-repair context carried across attempts."""

    previous_response: str
    previous_code: str | None
    visible_feedback: tuple[str, ...]


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
    problem_offset: int
    max_problems: int
    execute: bool
    output_root: Path
    benchmark_root: Path
    problem_file: Path | None
    run_id: str

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
    rlm_available: bool
    memory_tools_enabled: bool
    interpreter_available: bool | None = None
    interpreter_error: str | None = None


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small LiveCodeBench pilot comparing direct OpenRouter, DSPy, "
            "DSPy plus VTM verified memory, and DSPy RLM variants with and without memory."
        )
    )
    parser.add_argument(
        "--method",
        choices=("direct", "dspy_baseline", "dspy_vtm", "dspy_rlm_baseline", "dspy_rlm_vtm", "all"),
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
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
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
    problem_offset = max(0, int(args.problem_offset))
    model = execution_model(args.model or None)
    base_url = (args.base_url or openrouter_base_url()).strip()
    api_key = (args.api_key or openrouter_api_key() or "").strip() or None
    requested_scenario = args.scenario  # type: ignore[assignment]
    supported = source.supported_scenarios()
    resolved_scenario = resolve_scenario(requested_scenario, supported_scenarios=supported)
    run_id = args.run_id or f"lcb_dspy_pilot_{utc_now().strftime('%Y%m%d_%H%M%S')}"
    return PilotRunConfig(
        methods=resolve_methods(args.method),
        requested_scenario=requested_scenario,
        resolved_scenario=resolved_scenario,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=float(args.temperature),
        max_tokens=max_tokens,
        problem_offset=problem_offset,
        max_problems=max_problems,
        execute=bool(args.execute),
        output_root=args.output_root,
        benchmark_root=args.benchmark_root,
        problem_file=args.problem_file,
        run_id=run_id,
    )


def describe_method_runtime(
    method: PilotMethod,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
) -> MethodRuntime:
    model_config = DSPyOpenRouterConfig.from_env(
        base_url_value=base_url,
        api_key_value=api_key,
        execution_model_name=model,
        dspy_model_name=model,
    )
    if method in {"dspy_vtm", "dspy_rlm_vtm"}:
        interpreter_available, interpreter_error = (
            rlm_interpreter_availability()
            if method == "dspy_rlm_vtm"
            else (None, None)
        )
        with TemporaryDirectory(prefix="vtm-lcb-dspy-dry-run-") as temp_dir:
            session = open_memory_session(
                state_root=Path(temp_dir),
                problem_id="dry-run",
                workspace_root=PROJECT_ROOT,
            )
            try:
                if method == "dspy_vtm":
                    agent = VTMReActCodingAgent(
                        kernel=session.kernel,
                        scopes=(session.scope,),
                        workspace_root=None,
                        dependency_provider=lambda: session.dependency,
                        memory_lookup=session.metadata_store.get_memory_item,
                        model_config=model_config,
                    )
                    memory_tools_enabled = agent.memory_tools.enabled
                else:
                    agent = VTMRLMCodingAgent(
                        kernel=session.kernel,
                        scopes=(session.scope,),
                        dependency_provider=lambda: session.dependency,
                        memory_lookup=session.metadata_store.get_memory_item,
                        model_config=model_config,
                        max_iterations=PILOT_RLM_MAX_ITERATIONS,
                        max_llm_calls=PILOT_RLM_MAX_LLM_CALLS,
                    )
                    memory_tools_enabled = agent.context_adapter.memory_tools.enabled
                tool_names = agent.tool_names()
            finally:
                session.close()
        return MethodRuntime(
            method=method,
            uses_dspy=True,
            uses_vtm_memory=True,
            tool_names=tool_names,
            dspy_available=dspy_available(),
            rlm_available=_dspy_rlm_available(),
            memory_tools_enabled=memory_tools_enabled,
            interpreter_available=interpreter_available,
            interpreter_error=interpreter_error,
        )
    if method == "dspy_rlm_baseline":
        interpreter_available, interpreter_error = rlm_interpreter_availability()
        agent = VTMRLMCodingAgent(
            kernel=None,
            scopes=(),
            model_config=model_config,
            max_iterations=PILOT_RLM_MAX_ITERATIONS,
            max_llm_calls=PILOT_RLM_MAX_LLM_CALLS,
        )
        return MethodRuntime(
            method=method,
            uses_dspy=True,
            uses_vtm_memory=False,
            tool_names=agent.tool_names(),
            dspy_available=dspy_available(),
            rlm_available=_dspy_rlm_available(),
            memory_tools_enabled=agent.context_adapter.memory_tools.enabled,
            interpreter_available=interpreter_available,
            interpreter_error=interpreter_error,
        )
    if method == "dspy_baseline":
        agent = VTMReActCodingAgent(
            kernel=None,
            scopes=(),
            workspace_root=None,
            model_config=model_config,
        )
        return MethodRuntime(
            method=method,
            uses_dspy=True,
            uses_vtm_memory=False,
            tool_names=agent.tool_names(),
            dspy_available=dspy_available(),
            rlm_available=False,
            memory_tools_enabled=agent.memory_tools.enabled,
        )
    return MethodRuntime(
        method=method,
        uses_dspy=False,
        uses_vtm_memory=False,
        tool_names=(),
        dspy_available=dspy_available(),
        rlm_available=False,
        memory_tools_enabled=False,
    )


def build_attempt_prompt(
    problem: LiveCodeBenchProblem,
    *,
    attempt_index: int,
    agent_mode: Literal["direct", "dspy"] = "direct",
    memory_cards: Sequence[Mapping[str, Any]] = (),
    visible_feedback: Sequence[str] = (),
    repair_context: RepairContext | None = None,
) -> str:
    required_func_name = _required_function_name(problem)
    sections = [
        "Solve the following LiveCodeBench problem.",
        (
            "Return the final answer as a single ```python fenced code block and nothing else."
            if agent_mode == "direct"
            else (
                "Use tools only if needed. When you finish, put the solution in the final "
                "`response` field as a single ```python fenced code block."
            )
        ),
        f"Problem ID: {problem.problem_id}",
        "",
        "Problem Statement:",
        problem.prompt.strip(),
    ]
    if required_func_name is not None:
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
    if problem.starter_code:
        sections.extend(["", "Starter Code:", "```python", problem.starter_code.strip(), "```"])
    if problem.prompt_metadata:
        sections.extend(
            [
                "",
                "Problem Metadata:",
                json.dumps(problem.prompt_metadata, indent=2, sort_keys=True),
            ]
        )
    if memory_cards:
        sections.append("")
        sections.append("Verified Memory Cards:")
        for index, card in enumerate(memory_cards, start=1):
            sections.extend(_render_memory_card(index=index, card=card))
    if repair_context is not None:
        sections.append("")
        sections.append("Previous Attempt:")
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
    combined_feedback = tuple(problem.public_feedback) + tuple(visible_feedback)
    if combined_feedback:
        sections.append("")
        sections.append("Visible Feedback:")
        for feedback in combined_feedback:
            sections.append(f"- {feedback}")
    if attempt_index > 1 and repair_context is not None:
        sections.append("")
        sections.append(f"This is repair attempt {attempt_index}. Fix the previous attempt.")
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
        "pilot_limitations": list(PILOT_LIMITATION_NOTES),
    }


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
        "benchmark_root": str(config.benchmark_root),
        "problem_offset": config.problem_offset,
        "problem_source": source.describe(),
        "problem_source_error": source_error,
        "problem_count": len(problems),
        "scenario_semantics": _scenario_semantics(config.resolved_scenario),
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
        }
        payload["runs"].append(run_payload)
        if config.dry_run:
            continue
        if method != "direct" and not runtime.dspy_available:
            raise RuntimeError(f"{method} requires the optional 'dspy' extra")
        if method in {"dspy_rlm_baseline", "dspy_rlm_vtm"} and not runtime.rlm_available:
            raise RuntimeError(f"{method} requires a DSPy build that exposes dspy.RLM")
        if (
            method in {"dspy_rlm_baseline", "dspy_rlm_vtm"}
            and runtime.interpreter_available is False
        ):
            summary = skipped_summary(
                method=method,
                scenario=config.resolved_scenario,
                model=config.model,
                run_id=config.run_id,
                reason=runtime.interpreter_error
                or "DSPy RLM interpreter prerequisites were not available.",
                problem_source=source.describe(),
                problem_offset=config.problem_offset,
                planned_problem_count=len(problems),
            )
            write_problem_rows(paths.problems_jsonl, [])
            write_summary(paths.summary_json, summary)
            run_payload["summary"] = summary
            run_payload["skipped"] = True
            run_payload["skip_reason"] = summary["skip_reason"]
            continue
        rows = execute_method(
            method,
            config=config,
            problems=problems,
            source=source,
        )
        summary = aggregate_summary(
            rows,
            method=method,
            scenario=config.resolved_scenario,
            model=config.model,
        )
        summary["run_id"] = config.run_id
        summary["generated_at"] = utc_now().isoformat()
        summary["problem_source"] = source.describe()
        summary["problem_offset"] = config.problem_offset
        summary["scenario_semantics"] = _scenario_semantics(config.resolved_scenario)
        write_problem_rows(paths.problems_jsonl, rows)
        write_summary(paths.summary_json, summary)
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
        )
    )
    model_config = DSPyOpenRouterConfig.from_env(
        base_url_value=config.base_url,
        api_key_value=config.api_key,
        execution_model_name=config.model,
        dspy_model_name=config.model,
    )
    rows: list[dict[str, Any]] = []
    for problem in problems:
        row = execute_problem(
            problem,
            method=method,
            config=config,
            source=source,
            client=client,
            model_config=model_config,
        )
        rows.append(row)
    return rows


def execute_problem(
    problem: LiveCodeBenchProblem,
    *,
    method: PilotMethod,
    config: PilotRunConfig,
    source: ProblemSource,
    client: OpenAICompatibleChatClient,
    model_config: DSPyOpenRouterConfig,
) -> dict[str, Any]:
    max_attempts = 2 if config.resolved_scenario == "self_repair" else 1
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
    dspy_tool_calls = 0
    direct_retry_count = 0
    response_error: str | None = None
    repair_context: RepairContext | None = None

    with TemporaryDirectory(prefix=f"vtm-lcb-{problem.problem_id}-") as temp_dir:
        session = (
            open_memory_session(
                state_root=Path(temp_dir),
                problem_id=problem.problem_id,
                workspace_root=PROJECT_ROOT,
            )
            if method in {"dspy_vtm", "dspy_rlm_vtm"}
            else None
        )
        try:
            if session is not None:
                seed_problem_memory(session, problem)
            for attempt_index in range(1, max_attempts + 1):
                memory_cards: Sequence[Mapping[str, Any]] = ()
                retrieval_query = build_retrieval_query(problem, visible_feedback)
                if session is not None:
                    retrieval_payload = retrieve_verified_memory(
                        session,
                        query=retrieval_query,
                        attempt_index=attempt_index,
                    )
                    retrieval_invocation_count += 1
                    retrieval_queries.append(retrieval_query)
                    if retrieval_payload["used"]:
                        retrieval_hit_count += 1
                    retrieval_verified_count += int(retrieval_payload["verified_count"])
                    retrieval_stale_filtered_count += int(
                        retrieval_payload["stale_filtered_count"]
                    )
                    memory_cards = tuple(retrieval_payload["cards"])
                prompt = build_attempt_prompt(
                    problem,
                    attempt_index=attempt_index,
                    agent_mode="direct" if method == "direct" else "dspy",
                    memory_cards=memory_cards,
                    visible_feedback=visible_feedback,
                    repair_context=repair_context,
                )
                if method == "direct":
                    response_payload, response_text, direct_retry_count, response_error = (
                        _request_direct_completion(
                            client,
                            model=config.model,
                            prompt=prompt,
                            temperature=config.temperature,
                            max_tokens=config.max_tokens,
                        )
                    )
                    usage = _normalize_usage(response_payload.get("usage"))
                else:
                    response_payload = run_dspy_attempt(
                        prompt=prompt,
                        method=method,
                        session=session,
                        model_config=model_config,
                        memory_query=retrieval_query if session is not None else None,
                    )
                    response_text = response_payload["response_text"]
                    dspy_tool_calls += int(response_payload["tool_calls"])
                    usage = _normalize_usage(response_payload.get("usage"))
                    if isinstance(response_payload.get("response_error"), str):
                        response_error = response_payload["response_error"]
                extracted_code = extract_code(response_text)
                evaluation = source.evaluate(
                    problem,
                    response_text=response_text,
                    extracted_code=extracted_code,
                )
                if session is not None:
                    record_attempt_memory(
                        session,
                        problem=problem,
                        attempt_index=attempt_index,
                        response_text=response_text,
                        extracted_code=extracted_code,
                        evaluation=evaluation,
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
                repair_context = RepairContext(
                    previous_response=response_text,
                    previous_code=extracted_code,
                    visible_feedback=tuple(visible_feedback),
                )
        finally:
            if session is not None:
                session.close()

    total_tool_calls = dspy_tool_calls + retrieval_invocation_count
    retrieval_summary = None
    if retrieval_invocation_count:
        retrieval_summary = {
            "invoked": True,
            "used": retrieval_hit_count > 0,
            "query": retrieval_queries[-1],
            "query_history": retrieval_queries,
            "cards": retrieval_payload["cards"] if isinstance(retrieval_payload, Mapping) else [],
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
        "attempt_count": 2 if repair_context is not None else 1,
        "repair_used": repair_context is not None,
    }


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
    memory_query: str | None = None,
) -> dict[str, Any]:
    if method == "dspy_vtm":
        assert session is not None
        agent = VTMReActCodingAgent(
            kernel=session.kernel,
            scopes=(session.scope,),
            workspace_root=None,
            dependency_provider=lambda: session.dependency,
            memory_lookup=session.metadata_store.get_memory_item,
            model_config=model_config,
        )
        result = agent.run(prompt)
    elif method == "dspy_rlm_vtm":
        assert session is not None
        agent = VTMRLMCodingAgent(
            kernel=session.kernel,
            scopes=(session.scope,),
            dependency_provider=lambda: session.dependency,
            memory_lookup=session.metadata_store.get_memory_item,
            model_config=model_config,
            max_iterations=PILOT_RLM_MAX_ITERATIONS,
            max_llm_calls=PILOT_RLM_MAX_LLM_CALLS,
        )
        result = agent.run(prompt, query=memory_query)
    elif method == "dspy_rlm_baseline":
        agent = VTMRLMCodingAgent(
            kernel=None,
            scopes=(),
            model_config=model_config,
            max_iterations=PILOT_RLM_MAX_ITERATIONS,
            max_llm_calls=PILOT_RLM_MAX_LLM_CALLS,
        )
        result = agent.run(prompt, query=memory_query)
    else:
        agent = VTMReActCodingAgent(
            kernel=None,
            scopes=(),
            workspace_root=None,
            model_config=model_config,
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
            metadata={"problem_id": problem.problem_id, "memory_role": "problem_summary"},
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
                metadata={"problem_id": problem.problem_id, "memory_role": "function_contract"},
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
                metadata={"problem_id": problem.problem_id, "memory_role": "public_tests"},
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


def build_retrieval_query(
    problem: LiveCodeBenchProblem,
    visible_feedback: Sequence[str],
) -> str:
    query_parts = [
        problem.problem_id,
        compact_text(problem.prompt, limit=180),
    ]
    if visible_feedback:
        query_parts.append(compact_text(" ".join(visible_feedback), limit=120))
    return " | ".join(part for part in query_parts if part)


def retrieve_verified_memory(
    session: PilotMemorySession,
    *,
    query: str,
    attempt_index: int,
) -> dict[str, Any]:
    result = session.kernel.retrieve(
        RetrieveRequest(
            query=query,
            scopes=(session.scope,),
            evidence_budget=EvidenceBudget.SUMMARY_ONLY,
            limit=5,
            current_dependency=session.dependency,
            verify_on_read=True,
            return_verified_only=True,
        )
    )
    selected_candidates = _select_prompt_candidates(
        result.candidates,
        attempt_index=attempt_index,
    )
    return {
        "used": bool(selected_candidates),
        "query": query,
        "cards": [_serialize_candidate(candidate) for candidate in selected_candidates],
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
            "Attempt 2 receives the previous candidate code plus visible public-test feedback."
        )
    return "Single-pass code generation over LiveCodeBench public code-generation problems."


def _normalize_usage(raw_usage: Any) -> dict[str, Any] | None:
    if raw_usage is None:
        return None
    if isinstance(raw_usage, Mapping):
        return {str(key): value for key, value in raw_usage.items()}
    return {"raw": raw_usage}


def _dspy_rlm_available() -> bool:
    if not dspy_available():
        return False
    try:
        return hasattr(require_dspy(), "RLM")
    except ImportError:
        return False


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
        "evidence_summary": evidence_summary,
    }


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


def _memory_role(memory: Any) -> str:
    metadata = getattr(memory, "metadata", None)
    if isinstance(metadata, Mapping):
        role = metadata.get("memory_role")
        if isinstance(role, str) and role.strip():
            return role.strip()
    return ""


def _select_prompt_candidates(
    candidates: Sequence[Any],
    *,
    attempt_index: int,
    limit: int = 5,
) -> list[Any]:
    if not candidates:
        return []
    role_ranked = sorted(
        candidates,
        key=lambda candidate: (
            PROMPT_CARD_ROLE_PRIORITY.get(_memory_role(candidate.memory), 10),
            -float(candidate.score),
            candidate.memory.title or "",
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
