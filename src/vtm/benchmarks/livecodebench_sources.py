"""Explicit LiveCodeBench problem sources for the DSPy pilot."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, Literal, Protocol

PilotScenario = Literal["code_generation", "self_repair"]

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
CHECKOUT_LOADER_SCRIPT: Final[Path] = (
    PROJECT_ROOT / "scripts" / "livecodebench" / "checkout_problem_loader.py"
)
DEFAULT_DATASET_REPO: Final[str] = "livecodebench/code_generation_lite"
DEFAULT_DATASET_FILENAME: Final[str] = "test.jsonl"
DEFAULT_PUBLIC_TEST_TIMEOUT_SECONDS: Final[int] = 5
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
EXCEPTION_LINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][\w.]*?(?:Error|Exception|Exit|Interrupt)\b.*$"
)


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
class PublicTestExecutionResult:
    """Outcome for one public test case in the pilot evaluator."""

    passed: bool
    feedback: str
    syntax_error: bool = False


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
        problem_offset: int = 0,
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


class BaseProblemSource:
    """Shared problem parsing and public-test evaluation helpers."""

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
        candidate_code = (extracted_code or response_text).strip()
        if not candidate_code:
            return {
                "available": True,
                "scope": "public_tests",
                "passed": False,
                "pass_rate": 0.0,
                "passed_test_count": 0,
                "public_test_count": len(public_tests),
                "failure_feedback": ["No executable code was extracted from the response."],
                "syntax_error": False,
            }
        problem_metadata = _normalize_problem_metadata(
            problem.evaluator_payload.get("problem_metadata"),
        )
        test_results = [
            self._evaluate_public_test_case(
                candidate_code,
                raw_test,
                problem_metadata=problem_metadata,
            )
            for raw_test in public_tests
        ]
        passed_test_count = sum(1 for result in test_results if result.passed)
        failures = [result.feedback for result in test_results if not result.passed]
        passed = passed_test_count == len(test_results)
        return {
            "available": True,
            "scope": "public_tests",
            "passed": passed,
            "pass_rate": passed_test_count / len(test_results),
            "passed_test_count": passed_test_count,
            "public_test_count": len(test_results),
            "failure_feedback": failures[:3],
            "syntax_error": any(result.syntax_error for result in test_results),
        }

    def _evaluate_public_test_case(
        self,
        candidate_code: str,
        raw_test: Any,
        *,
        problem_metadata: Mapping[str, Any],
    ) -> PublicTestExecutionResult:
        if not isinstance(raw_test, Mapping):
            test_text = str(raw_test).strip()
            if not test_text:
                return PublicTestExecutionResult(
                    passed=False,
                    feedback="Encountered an empty public test case.",
                )
            if test_text in candidate_code:
                return PublicTestExecutionResult(passed=True, feedback="")
            return PublicTestExecutionResult(
                passed=False,
                feedback=f"Public test marker not found in the candidate code: {test_text}",
            )
        test_type = str(raw_test.get("testtype", "stdin")).strip().lower()
        if test_type == "functional":
            return _run_functional_public_test(
                candidate_code,
                raw_test,
                func_name=str(problem_metadata.get("func_name", "") or "").strip() or None,
            )
        return _run_stdin_public_test(candidate_code, raw_test)

    def _materialize_problems(
        self,
        raw_records: Sequence[Mapping[str, Any]],
        *,
        scenario: PilotScenario,
        source_path: Path,
        problem_offset: int,
        max_problems: int,
    ) -> list[LiveCodeBenchProblem]:
        problems: list[LiveCodeBenchProblem] = []
        window = raw_records[problem_offset : problem_offset + max_problems]
        for index, record in enumerate(window, start=problem_offset):
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


class ProblemFileSource(BaseProblemSource):
    """Explicit source for a user-provided JSON or JSONL file."""

    def __init__(self, problem_file: Path) -> None:
        self._problem_file = problem_file

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "problem_file",
            "problem_file": str(self._problem_file),
            "problem_file_exists": self._problem_file.exists(),
            "supported_scenarios": sorted(self.supported_scenarios()),
        }

    def supported_scenarios(self) -> set[PilotScenario]:
        return {"code_generation", "self_repair"}

    def load_problems(
        self,
        scenario: PilotScenario,
        *,
        problem_offset: int = 0,
        max_problems: int,
    ) -> list[LiveCodeBenchProblem]:
        raw_records = _load_problem_records(self._problem_file)
        return self._materialize_problems(
            raw_records,
            scenario=scenario,
            source_path=self._problem_file,
            problem_offset=max(0, int(problem_offset)),
            max_problems=max_problems,
        )


class LiveCodeBenchCheckoutSource(BaseProblemSource):
    """Source that uses the external LiveCodeBench checkout plus its local venv."""

    def __init__(
        self,
        *,
        benchmark_root: Path,
        dataset_repo: str = DEFAULT_DATASET_REPO,
        dataset_filename: str = DEFAULT_DATASET_FILENAME,
    ) -> None:
        self._benchmark_root = benchmark_root
        self._dataset_repo = dataset_repo
        self._dataset_filename = dataset_filename

    def describe(self) -> dict[str, Any]:
        python_bin = self._checkout_python()
        return {
            "kind": "checkout",
            "benchmark_root": str(self._benchmark_root),
            "benchmark_root_exists": self._benchmark_root.exists(),
            "benchmark_python": str(python_bin) if python_bin is not None else None,
            "benchmark_python_exists": python_bin.exists() if python_bin is not None else False,
            "checkout_loader_script": str(CHECKOUT_LOADER_SCRIPT),
            "checkout_loader_script_exists": CHECKOUT_LOADER_SCRIPT.exists(),
            "dataset_repo": self._dataset_repo,
            "dataset_filename": self._dataset_filename,
            "supported_scenarios": sorted(self.supported_scenarios()),
        }

    def supported_scenarios(self) -> set[PilotScenario]:
        if self._checkout_python() is not None and CHECKOUT_LOADER_SCRIPT.exists():
            return {"code_generation", "self_repair"}
        return set()

    def load_problems(
        self,
        scenario: PilotScenario,
        *,
        problem_offset: int = 0,
        max_problems: int,
    ) -> list[LiveCodeBenchProblem]:
        checkout_python = self._checkout_python()
        if checkout_python is None:
            raise FileNotFoundError(
                f"unable to locate LiveCodeBench checkout under {self._benchmark_root}"
            )
        if not CHECKOUT_LOADER_SCRIPT.exists():
            raise FileNotFoundError(
                f"unable to locate checkout loader script at {CHECKOUT_LOADER_SCRIPT}"
            )
        raw_records = self._load_problems_from_checkout(
            checkout_python,
            problem_offset=max(0, int(problem_offset)),
            max_problems=max_problems,
        )
        return self._materialize_problems(
            raw_records,
            scenario=scenario,
            source_path=self._benchmark_root / ".venv",
            problem_offset=0,
            max_problems=max_problems,
        )

    def _checkout_python(self) -> Path | None:
        candidate = self._benchmark_root / ".venv" / "bin" / "python"
        if candidate.exists():
            return candidate
        return None

    def _load_problems_from_checkout(
        self,
        python_bin: Path,
        *,
        problem_offset: int,
        max_problems: int,
    ) -> list[dict[str, Any]]:
        completed = subprocess.run(
            [
                str(python_bin),
                str(CHECKOUT_LOADER_SCRIPT),
                "--dataset-repo",
                self._dataset_repo,
                "--dataset-filename",
                self._dataset_filename,
                "--problem-offset",
                str(problem_offset),
                "--max-problems",
                str(max_problems),
            ],
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


def discover_problem_source(
    *,
    benchmark_root: Path,
    problem_file: Path | None = None,
) -> ProblemSource:
    if problem_file is not None:
        return ProblemFileSource(problem_file)
    return LiveCodeBenchCheckoutSource(benchmark_root=benchmark_root)


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
    evaluator_payload: dict[str, Any] = {}
    if "public_tests" in record and not _looks_hidden("public_tests"):
        evaluator_payload["public_tests"] = record.get("public_tests")
    if "sample_tests" in record:
        evaluator_payload["public_tests"] = record.get("sample_tests")
    public_test_cases = _normalize_public_test_cases(record.get("public_test_cases"))
    if public_test_cases:
        evaluator_payload["public_tests"] = public_test_cases
        evaluator_payload["problem_metadata"] = _normalize_problem_metadata(record.get("metadata"))
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


def compact_text(raw: str | None, *, limit: int = 240) -> str:
    if raw is None:
        return ""
    normalized = " ".join(raw.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def summarize_execution_feedback(raw_feedback: str, *, limit: int = 180) -> str:
    """Prefer the actionable terminal exception line over a truncated traceback."""
    stripped = raw_feedback.strip()
    if not stripped:
        return compact_text(raw_feedback, limit=limit)
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        if EXCEPTION_LINE_PATTERN.match(line):
            return compact_text(line, limit=limit)
    return compact_text(stripped, limit=limit)


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


def _normalize_problem_metadata(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    value = raw_value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, Mapping):
        return {}
    return {str(key): entry for key, entry in value.items()}


def _format_public_test_feedback(test_case: Mapping[str, Any]) -> str:
    input_text = compact_text(str(test_case.get("input", "")), limit=100)
    output_text = compact_text(str(test_case.get("output", "")), limit=100)
    return f"Public sample: input={input_text!r} output={output_text!r}"


def _normalize_text_output(value: str) -> str:
    normalized = value.replace("\r\n", "\n").strip().split("\n")
    return "\n".join(line.rstrip() for line in normalized).strip()


def _run_stdin_public_test(
    candidate_code: str,
    test_case: Mapping[str, Any],
) -> PublicTestExecutionResult:
    test_input = str(test_case.get("input", ""))
    expected_output = str(test_case.get("output", ""))
    with TemporaryDirectory(prefix="vtm-lcb-public-stdin-") as temp_dir:
        temp_root = Path(temp_dir)
        program_path = temp_root / "candidate.py"
        program_path.write_text(candidate_code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(program_path)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=DEFAULT_PUBLIC_TEST_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PublicTestExecutionResult(
                passed=False,
                feedback="Public stdin test timed out.",
            )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        syntax_error = "SyntaxError" in stderr
        return PublicTestExecutionResult(
            passed=False,
            feedback=f"Public stdin test failed to execute: {compact_text(stderr, limit=180)}",
            syntax_error=syntax_error,
        )
    actual_output = _normalize_text_output(completed.stdout)
    expected = _normalize_text_output(expected_output)
    if actual_output == expected:
        return PublicTestExecutionResult(passed=True, feedback="")
    return PublicTestExecutionResult(
        passed=False,
        feedback=(
            "Public stdin test mismatch: "
            f"expected={compact_text(expected, limit=80)!r} "
            f"actual={compact_text(actual_output, limit=80)!r}"
        ),
    )


def _run_functional_public_test(
    candidate_code: str,
    test_case: Mapping[str, Any],
    *,
    func_name: str | None,
) -> PublicTestExecutionResult:
    if not func_name:
        return PublicTestExecutionResult(
            passed=False,
            feedback="Functional public test missing metadata.func_name.",
        )
    parsed_input = _parse_json_like_value(test_case.get("input"))
    parsed_output = _parse_json_like_value(test_case.get("output"))
    with TemporaryDirectory(prefix="vtm-lcb-public-functional-") as temp_dir:
        temp_root = Path(temp_dir)
        program_path = temp_root / "candidate.py"
        harness_path = temp_root / "harness.py"
        program_path.write_text(candidate_code, encoding="utf-8")
        harness_path.write_text(
            _functional_harness_source(
                func_name=func_name,
                candidate_path=program_path,
                parsed_input=parsed_input,
                parsed_output=parsed_output,
            ),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(harness_path)],
                capture_output=True,
                text=True,
                timeout=DEFAULT_PUBLIC_TEST_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PublicTestExecutionResult(
                passed=False,
                feedback="Functional public test timed out.",
            )
    stdout = completed.stdout.strip()
    if completed.returncode == 0:
        return PublicTestExecutionResult(passed=True, feedback="")
    syntax_error = "SyntaxError" in ((completed.stderr or "") + stdout)
    feedback = stdout or (completed.stderr or "").strip() or "Functional public test failed."
    return PublicTestExecutionResult(
        passed=False,
        feedback=summarize_execution_feedback(feedback, limit=180),
        syntax_error=syntax_error,
    )


def _parse_json_like_value(raw_value: Any) -> Any:
    if not isinstance(raw_value, str):
        return raw_value
    stripped = raw_value.strip()
    if not stripped:
        return stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _functional_harness_source(
    *,
    func_name: str,
    candidate_path: Path,
    parsed_input: Any,
    parsed_output: Any,
) -> str:
    return (
        "import importlib.util\n"
        "import json\n"
        "from pathlib import Path\n\n"
        f"candidate_path = Path({str(candidate_path)!r})\n"
        f"func_name = {func_name!r}\n"
        f"parsed_input = json.loads({json.dumps(json.dumps(parsed_input))})\n"
        f"expected_output = json.loads({json.dumps(json.dumps(parsed_output))})\n"
        "spec = importlib.util.spec_from_file_location('candidate', candidate_path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "assert spec is not None and spec.loader is not None\n"
        "spec.loader.exec_module(module)\n"
        "target = getattr(module, func_name, None)\n"
        "if target is None:\n"
        "    solution_cls = getattr(module, 'Solution', None)\n"
        "    if solution_cls is not None:\n"
        "        target = getattr(solution_cls(), func_name, None)\n"
        "if target is None:\n"
        "    raise AttributeError(\n"
        "        f\"candidate module does not define {func_name!r} or Solution.{func_name}\"\n"
        "    )\n"
        "args = parsed_input if isinstance(parsed_input, list) else [parsed_input]\n"
        "result = target(*args)\n"
        "if result != expected_output:\n"
        "    raise SystemExit(\n"
        "        'Functional public test mismatch: expected=' + repr(expected_output) + "
        "        ' actual=' + repr(result)\n"
        "    )\n"
    )


def _looks_hidden(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in HIDDEN_FIELD_FRAGMENTS)


__all__ = [
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_DATASET_REPO",
    "LiveCodeBenchCheckoutSource",
    "LiveCodeBenchProblem",
    "PilotScenario",
    "ProblemFileSource",
    "ProblemSource",
    "discover_problem_source",
]
