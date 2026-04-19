"""Small LiveCodeBench pilot comparing direct, DSPy, and DSPy plus VTM flows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, Final, Literal, Protocol

from vtm.adapters.openai_chat import OpenAICompatibleChatClient, OpenAICompatibleChatConfig
from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.base import utc_now
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
from vtm_dspy import dspy_available
from vtm_dspy.config import DSPyOpenRouterConfig
from vtm_dspy.react_agent import VTMReActCodingAgent

PilotMethod = Literal["direct", "dspy_baseline", "dspy_vtm"]
PilotScenario = Literal["code_generation", "self_repair"]

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / ".benchmarks" / "livecodebench-dspy"
DEFAULT_BENCHMARK_ROOT: Final[Path] = PROJECT_ROOT / "benchmarks" / "LiveCodeBench"
DEFAULT_MAX_PROBLEMS: Final[int] = 3
DEFAULT_MAX_TOKENS: Final[int] = 8192
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_SCENARIO: Final[PilotScenario] = "self_repair"
DEFAULT_DATASET_REPO: Final[str] = "livecodebench/code_generation_lite"
DEFAULT_DATASET_FILENAME: Final[str] = "test.jsonl"
METHODS: Final[tuple[PilotMethod, ...]] = ("direct", "dspy_baseline", "dspy_vtm")
CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
SCENARIO_TOKENS: Final[dict[PilotScenario, tuple[str, ...]]] = {
    "code_generation": ("code_generation", "codegeneration", "code-gen", "codegen"),
    "self_repair": ("self_repair", "selfrepair", "self-repair"),
}
PROMPT_FIELD_CANDIDATES: Final[tuple[str, ...]] = (
    "question_content",
    "prompt",
    "problem",
    "description",
    "statement",
    "question",
)
STARTER_FIELD_CANDIDATES: Final[tuple[str, ...]] = (
    "starter_code",
    "starter",
    "code_prompt",
    "prompt_starter_code",
)
ID_FIELD_CANDIDATES: Final[tuple[str, ...]] = ("question_id", "problem_id", "task_id", "id")
VISIBLE_FEEDBACK_FIELD_CANDIDATES: Final[tuple[str, ...]] = (
    "visible_feedback",
    "execution_feedback",
    "public_feedback",
    "stderr",
    "stdout",
)
PUBLIC_METADATA_KEYS: Final[tuple[str, ...]] = (
    "title",
    "difficulty",
    "language",
    "source",
    "contest_date",
    "tags",
)
HIDDEN_FIELD_FRAGMENTS: Final[tuple[str, ...]] = (
    "hidden",
    "private",
    "solution",
    "gold",
    "canonical",
    "answer",
    "expected_output",
)


class ProblemSource(Protocol):
    """Minimal LiveCodeBench source surface used by the pilot."""

    def describe(self) -> dict[str, Any]:
        """Return dry-run metadata for the underlying source."""

    def supported_scenarios(self) -> set[PilotScenario]:
        """Return which pilot scenarios this source can load."""

    def load_problems(
        self,
        scenario: PilotScenario,
        *,
        max_problems: int,
    ) -> list[LiveCodeBenchProblem]:
        """Load a bounded set of public problems."""

    def evaluate(
        self,
        problem: LiveCodeBenchProblem,
        *,
        response_text: str,
        extracted_code: str | None,
    ) -> dict[str, Any] | None:
        """Evaluate one response when the source exposes a safe evaluator."""


@dataclass(frozen=True)
class LiveCodeBenchProblem:
    """Public, non-oracle problem payload used by every pilot method."""

    problem_id: str
    scenario: PilotScenario
    prompt: str
    starter_code: str | None = None
    prompt_metadata: dict[str, Any] = field(default_factory=dict)
    public_feedback: tuple[str, ...] = field(default_factory=tuple)
    evaluator_payload: dict[str, Any] = field(default_factory=dict)
    raw_record_path: str | None = None


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


class FilesystemProblemSource:
    """Public problem loader using either a supplied JSON file or the checkout's own venv."""

    def __init__(
        self,
        *,
        benchmark_root: Path,
        problem_file: Path | None = None,
    ) -> None:
        self._benchmark_root = benchmark_root
        self._problem_file = problem_file

    def describe(self) -> dict[str, Any]:
        python_bin = self._checkout_python()
        return {
            "kind": "filesystem",
            "benchmark_root": str(self._benchmark_root),
            "benchmark_root_exists": self._benchmark_root.exists(),
            "benchmark_python": str(python_bin) if python_bin is not None else None,
            "benchmark_python_exists": python_bin.exists() if python_bin is not None else False,
            "dataset_repo": DEFAULT_DATASET_REPO,
            "dataset_filename": DEFAULT_DATASET_FILENAME,
            "problem_file": str(self._problem_file) if self._problem_file is not None else None,
            "problem_file_exists": self._problem_file.exists() if self._problem_file else False,
            "supported_scenarios": sorted(self.supported_scenarios()),
        }

    def supported_scenarios(self) -> set[PilotScenario]:
        if self._problem_file is not None:
            return {DEFAULT_SCENARIO, "code_generation"}
        if self._checkout_python() is not None:
            return {"code_generation", "self_repair"}
        supported: set[PilotScenario] = set()
        for scenario in SCENARIO_TOKENS:
            if self._resolve_data_path(scenario) is not None:
                supported.add(scenario)
        return supported

    def load_problems(
        self,
        scenario: PilotScenario,
        *,
        max_problems: int,
    ) -> list[LiveCodeBenchProblem]:
        if self._problem_file is not None:
            raw_records = _load_problem_records(self._problem_file)
            source_path = self._problem_file
        else:
            checkout_python = self._checkout_python()
            if checkout_python is not None:
                raw_records = self._load_problems_from_checkout(
                    checkout_python,
                    scenario=scenario,
                    max_problems=max_problems,
                )
                source_path = self._benchmark_root / ".venv"
            else:
                data_path = self._resolve_data_path(scenario)
                if data_path is None:
                    raise FileNotFoundError(
                        f"unable to locate LiveCodeBench data for scenario={scenario!r} under "
                        f"{self._benchmark_root}"
                    )
                raw_records = _load_problem_records(data_path)
                source_path = data_path
        problems: list[LiveCodeBenchProblem] = []
        for index, record in enumerate(raw_records):
            problem = _problem_from_record(
                record,
                scenario=scenario,
                source_path=source_path,
                fallback_id=f"{scenario}-{index + 1}",
            )
            if problem is not None:
                problems.append(problem)
            if len(problems) >= max_problems:
                break
        return problems

    def evaluate(
        self,
        problem: LiveCodeBenchProblem,
        *,
        response_text: str,
        extracted_code: str | None,
    ) -> dict[str, Any] | None:
        public_tests = problem.evaluator_payload.get("public_tests")
        if not isinstance(public_tests, list | tuple) or not public_tests:
            return None
        if any(isinstance(test, Mapping) for test in public_tests):
            return None
        code_text = extracted_code or response_text
        failures = []
        for raw_test in public_tests:
            test_text = str(raw_test).strip()
            if test_text and test_text not in code_text:
                failures.append(test_text)
        passed = not failures
        return {
            "available": True,
            "passed": passed,
            "pass_rate": 1.0 if passed else 0.0,
            "failure_feedback": failures[:3],
            "syntax_error": False,
        }

    def _resolve_data_path(self, scenario: PilotScenario) -> Path | None:
        if not self._benchmark_root.exists():
            return None
        candidates = sorted(
            path
            for path in self._benchmark_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".json", ".jsonl"}
            and any(token in path.name.lower() for token in SCENARIO_TOKENS[scenario])
        )
        if candidates:
            return candidates[0]
        return None

    def _checkout_python(self) -> Path | None:
        candidate = self._benchmark_root / ".venv" / "bin" / "python"
        if candidate.exists():
            return candidate
        return None

    def _load_problems_from_checkout(
        self,
        python_bin: Path,
        *,
        scenario: PilotScenario,
        max_problems: int,
    ) -> list[dict[str, Any]]:
        script = f"""
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id={DEFAULT_DATASET_REPO!r},
    repo_type='dataset',
    filename={DEFAULT_DATASET_FILENAME!r},
)
rows = []
with open(path, 'r', encoding='utf-8') as handle:
    for line in handle:
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
        if len(rows) >= {max_problems}:
            break
print(json.dumps(rows))
"""
        completed = subprocess.run(
            [str(python_bin), "-c", script],
            cwd=self._benchmark_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "LiveCodeBench checkout loader failed. "
                f"benchmark_root={self._benchmark_root} stderr={stderr}"
            )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list):
            raise RuntimeError("LiveCodeBench checkout loader did not return a list of problems")
        return [row for row in payload if isinstance(row, dict)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small LiveCodeBench pilot comparing direct OpenRouter, DSPy, "
            "and DSPy plus VTM verified memory."
        )
    )
    parser.add_argument(
        "--method",
        choices=("direct", "dspy_baseline", "dspy_vtm", "all"),
        default="all",
    )
    parser.add_argument(
        "--scenario",
        choices=("code_generation", "self_repair"),
        default=DEFAULT_SCENARIO,
    )
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


def discover_problem_source(
    *,
    benchmark_root: Path,
    problem_file: Path | None = None,
) -> ProblemSource:
    return FilesystemProblemSource(benchmark_root=benchmark_root, problem_file=problem_file)


def resolve_config(
    args: argparse.Namespace,
    *,
    source: ProblemSource,
) -> PilotRunConfig:
    max_problems = max(1, int(args.max_problems))
    max_tokens = max(1, int(args.max_tokens))
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
    if method == "dspy_vtm":
        with TemporaryDirectory(prefix="vtm-lcb-dspy-dry-run-") as temp_dir:
            session = open_memory_session(
                state_root=Path(temp_dir),
                problem_id="dry-run",
                workspace_root=PROJECT_ROOT,
            )
            try:
                agent = VTMReActCodingAgent(
                    kernel=session.kernel,
                    scopes=(session.scope,),
                    workspace_root=None,
                    dependency_provider=lambda: session.dependency,
                    memory_lookup=session.metadata_store.get_memory_item,
                    model_config=model_config,
                )
                tool_names = agent.tool_names()
                memory_tools_enabled = agent.memory_tools.enabled
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
            memory_tools_enabled=agent.memory_tools.enabled,
        )
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
    memory_cards: Sequence[Mapping[str, Any]] = (),
    visible_feedback: Sequence[str] = (),
) -> str:
    sections = [
        "Solve the following LiveCodeBench problem.",
        "Return the final answer as a single ```python fenced code block and nothing else.",
        f"Problem ID: {problem.problem_id}",
        "",
        "Problem Statement:",
        problem.prompt.strip(),
    ]
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
            sections.append(
                json.dumps(
                    {
                        "index": index,
                        "title": card.get("title"),
                        "summary": card.get("summary"),
                        "status": card.get("status"),
                        "path": card.get("path"),
                        "symbol": card.get("symbol"),
                        "evidence_summary": card.get("evidence_summary", []),
                    },
                    sort_keys=True,
                )
            )
    combined_feedback = tuple(problem.public_feedback) + tuple(visible_feedback)
    if combined_feedback:
        sections.append("")
        sections.append("Visible Feedback:")
        for feedback in combined_feedback:
            sections.append(f"- {feedback}")
    if attempt_index > 1:
        sections.append("")
        sections.append(
            f"This is repair attempt {attempt_index}. Incorporate the visible feedback."
        )
    return "\n".join(sections).strip() + "\n"


def extract_code(text: str) -> str | None:
    if not text.strip():
        return None
    match = CODE_BLOCK_PATTERN.search(text)
    if match is not None:
        code = match.group(1).strip()
        return code or None
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
    pass_rate = (pass_count / total) if total and evaluation_rows else None
    syntax_error_count = sum(
        1
        for row in rows
        if isinstance(row.get("evaluation"), Mapping) and row["evaluation"].get("syntax_error")
    )
    retrieval_usage_rate = (
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
        "scenario": scenario,
        "method": method,
        "model": model,
        "total": total,
        "pass_count": pass_count if evaluation_rows else None,
        "pass_rate": round(pass_rate, 6) if pass_rate is not None else None,
        "accuracy": round(pass_rate, 6) if pass_rate is not None else None,
        "syntax_error_count": syntax_error_count,
        "retrieval_usage_rate": round(retrieval_usage_rate, 6),
        "mean_verified_count": round(mean_verified_count, 6),
        "mean_stale_filtered_count": round(mean_stale_filtered_count, 6),
        "mean_tool_calls": round(mean_tool_calls, 6),
    }


def run_pilot(
    config: PilotRunConfig,
    *,
    source: ProblemSource,
) -> dict[str, Any]:
    source_error: str | None = None
    try:
        problems = source.load_problems(config.resolved_scenario, max_problems=config.max_problems)
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
        "problem_source": source.describe(),
        "problem_source_error": source_error,
        "problem_count": len(problems),
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
    if supported:
        return (
            f"{exc}. Supported scenarios discovered under {config.benchmark_root}: "
            f"{', '.join(str(item) for item in supported)}. "
            "Pick one of those or pass `--problem-file <public-problems.jsonl>`."
        )
    return (
        f"{exc}. No problem data was discovered under {config.benchmark_root}. "
        "Run `bash scripts/livecodebench/setup_livecodebench.sh` first, confirm the "
        "external checkout contains public problem JSON/JSONL files, or pass "
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
    dspy_tool_calls = 0

    with TemporaryDirectory(prefix=f"vtm-lcb-{problem.problem_id}-") as temp_dir:
        session = (
            open_memory_session(
                state_root=Path(temp_dir),
                problem_id=problem.problem_id,
                workspace_root=PROJECT_ROOT,
            )
            if method == "dspy_vtm"
            else None
        )
        try:
            if session is not None:
                seed_problem_memory(session, problem)
            for attempt_index in range(1, max_attempts + 1):
                memory_cards: Sequence[Mapping[str, Any]] = ()
                if session is not None:
                    retrieval_payload = retrieve_verified_memory(
                        session,
                        query=build_retrieval_query(problem, visible_feedback),
                    )
                    memory_cards = tuple(retrieval_payload["cards"])
                prompt = build_attempt_prompt(
                    problem,
                    attempt_index=attempt_index,
                    memory_cards=memory_cards,
                    visible_feedback=visible_feedback,
                )
                if method == "direct":
                    response_payload = client.create_chat_completion(
                        model=config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    response_text = client.extract_message_text(response_payload)
                    usage = _normalize_usage(response_payload.get("usage"))
                else:
                    response_payload = run_dspy_attempt(
                        prompt=prompt,
                        method=method,
                        session=session,
                        model_config=model_config,
                    )
                    response_text = response_payload["response_text"]
                    dspy_tool_calls += int(response_payload["tool_calls"])
                    usage = _normalize_usage(response_payload.get("usage"))
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
        finally:
            if session is not None:
                session.close()

    total_tool_calls = dspy_tool_calls + (
        int(retrieval_payload["tool_calls"])
        if isinstance(retrieval_payload, Mapping)
        else 0
    )
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
        "retrieval": retrieval_payload,
        "tool_calls": total_tool_calls,
    }


def run_dspy_attempt(
    *,
    prompt: str,
    method: PilotMethod,
    session: PilotMemorySession | None,
    model_config: DSPyOpenRouterConfig,
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
    else:
        agent = VTMReActCodingAgent(
            kernel=None,
            scopes=(),
            workspace_root=None,
            model_config=model_config,
        )
    result = agent.run(prompt)
    response_payload = result.get("response")
    return {
        "response_text": _coerce_response_text(response_payload),
        "tool_calls": _count_serialized_tool_calls(response_payload),
        "trajectory": result.get("trajectory"),
        "usage": None,
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
    feedback_summaries: list[str] = []
    if evaluation and evaluation.get("failure_feedback"):
        feedback_payload = json.dumps(
            evaluation.get("failure_feedback"),
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
        feedback_summaries.extend(str(item) for item in evaluation.get("failure_feedback", []))

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
    return {
        "used": bool(result.candidates),
        "query": query,
        "cards": [_serialize_candidate(candidate) for candidate in result.candidates],
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


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def _load_problem_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("items", "problems", "questions", "tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"unsupported problem payload layout: {path}")


def _problem_from_record(
    record: Mapping[str, Any],
    *,
    scenario: PilotScenario,
    source_path: Path,
    fallback_id: str,
) -> LiveCodeBenchProblem | None:
    prompt = _first_string(record, PROMPT_FIELD_CANDIDATES)
    if not prompt:
        return None
    problem_id = _first_string(record, ID_FIELD_CANDIDATES) or fallback_id
    starter_code = _first_string(record, STARTER_FIELD_CANDIDATES)
    prompt_metadata = {
        key: value
        for key, value in record.items()
        if key in PUBLIC_METADATA_KEYS and value not in (None, "", [], {})
    }
    for key in ("question_title", "platform", "contest_id"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            prompt_metadata[key] = value
    public_feedback = tuple(
        str(record[key]).strip()
        for key in VISIBLE_FEEDBACK_FIELD_CANDIDATES
        if key in record and str(record[key]).strip()
    )
    evaluator_payload = {}
    if "public_tests" in record and not _looks_hidden("public_tests"):
        evaluator_payload["public_tests"] = record.get("public_tests")
    if "sample_tests" in record:
        evaluator_payload["public_tests"] = record.get("sample_tests")
    public_test_cases = _normalize_public_test_cases(record.get("public_test_cases"))
    if public_test_cases:
        evaluator_payload["public_tests"] = public_test_cases
        if not public_feedback:
            public_feedback = tuple(
                _format_public_test_feedback(test_case)
                for test_case in public_test_cases[:3]
            )
    return LiveCodeBenchProblem(
        problem_id=problem_id,
        scenario=scenario,
        prompt=prompt,
        starter_code=starter_code,
        prompt_metadata=prompt_metadata,
        public_feedback=public_feedback,
        evaluator_payload=evaluator_payload,
        raw_record_path=str(source_path),
    )


def _first_string(payload: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_public_test_cases(raw_value: Any) -> list[dict[str, Any]]:
    if raw_value is None:
        return []
    value = raw_value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _format_public_test_feedback(test_case: Mapping[str, Any]) -> str:
    input_text = compact_text(str(test_case.get("input", "")), limit=100)
    output_text = compact_text(str(test_case.get("output", "")), limit=100)
    return f"Public sample: input={input_text!r} output={output_text!r}"


def _looks_hidden(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in HIDDEN_FIELD_FRAGMENTS)


def _reference_command(config: PilotRunConfig) -> str:
    scenario_token = (
        "codegeneration"
        if config.resolved_scenario == "code_generation"
        else "self_repair"
    )
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


def _normalize_usage(raw_usage: Any) -> dict[str, Any] | None:
    if raw_usage is None:
        return None
    if isinstance(raw_usage, Mapping):
        return {str(key): value for key, value in raw_usage.items()}
    return {"raw": raw_usage}


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
            if isinstance(value, int):
                return value
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
        "score": round(candidate.score, 6),
        "path": path,
        "symbol": symbol,
        "evidence_summary": evidence_summary,
    }


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
    "METHODS",
    "FilesystemProblemSource",
    "LiveCodeBenchProblem",
    "MethodRuntime",
    "PilotRunConfig",
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
