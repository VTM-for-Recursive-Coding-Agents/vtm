from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import vtm.benchmarks.livecodebench_dspy_pilot as livecodebench_dspy_pilot
import vtm.benchmarks.livecodebench_sources as livecodebench_sources
from vtm.benchmarks.livecodebench_dspy_pilot import (
    RepairContext,
    _request_direct_completion,
    aggregate_summary,
    build_attempt_prompt,
    method_run_dir,
)
from vtm.benchmarks.livecodebench_dspy_pilot import (
    main as livecodebench_dspy_pilot_main,
)
from vtm.benchmarks.livecodebench_sources import (
    LiveCodeBenchCheckoutSource,
    LiveCodeBenchProblem,
    ProblemFileSource,
    discover_problem_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_baseline_module():
    module_path = REPO_ROOT / "scripts" / "livecodebench" / "baseline.py"
    spec = importlib.util.spec_from_file_location("vtm_livecodebench_baseline", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_livecodebench_script_paths_exist() -> None:
    expected = [
        REPO_ROOT / "scripts" / "livecodebench" / "baseline.py",
        REPO_ROOT / "scripts" / "livecodebench" / "checkout_problem_loader.py",
        REPO_ROOT / "scripts" / "livecodebench" / "export_dspy_pilot_results.py",
        REPO_ROOT / "scripts" / "livecodebench" / "export_results.py",
        REPO_ROOT / "scripts" / "livecodebench" / "setup_livecodebench.sh",
        REPO_ROOT / "scripts" / "shared" / "run_livecodebench.sh",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench.sh",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench_baseline.sh",
        REPO_ROOT / "scripts" / "local" / "preflight_checks.sh",
        REPO_ROOT / "scripts" / "run_livecodebench_baseline.sh",
        REPO_ROOT / "scripts" / "run_livecodebench_dspy_pilot.py",
        REPO_ROOT / "scripts" / "run_livecodebench_dspy_pilot_batch.sh",
        REPO_ROOT / "docs" / "livecodebench-baselines.md",
        REPO_ROOT / "results" / "README.md",
    ]

    for path in expected:
        assert path.exists(), path


def test_livecodebench_command_uses_openrouter_defaults() -> None:
    module = _load_baseline_module()
    config = module.normalize_config(
        module.BaselineConfig(
            model="google/gemma-test",
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
            run_id="lcb_baseline_smoke",
            output_root=Path(".benchmarks/livecodebench"),
            summary_root=Path(".benchmarks/paper-tables/livecodebench-baselines"),
            benchmark_root=Path("benchmarks/LiveCodeBench"),
            scenario="codegeneration",
            release_version="release_v1",
            n=10,
            temperature=0.2,
            evaluate=True,
            start_date=None,
            end_date=None,
            smoke=True,
            execute=False,
        )
    )

    command = module.build_livecodebench_command("python3", config)
    env = module.build_openrouter_env(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
    )

    assert command[:4] == ["python3", "-m", "lcb_runner.runner.main", "--model"]
    assert "google/gemma-test" in command
    assert "--evaluate" not in command
    assert "--start_date" in command
    assert "--end_date" in command
    assert env["VTM_EXECUTION_MODEL"] == "google/gemma-test"
    assert env["VTM_OPENROUTER_BASE_URL"] == "https://openrouter.example/api/v1"
    assert env["OPENROUTER_API_KEY"] == "openrouter-test-key"
    assert env["OPENAI_API_KEY"] == "openrouter-test-key"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.example/api/v1"
    assert env["OPENAI_API_BASE"] == "https://openrouter.example/api/v1"
    assert module.normalized_run_dir(config) == Path(
        ".benchmarks/livecodebench/google-gemma-test/lcb_baseline_smoke"
    )
    assert module.normalized_summary_path(config) == Path(
        ".benchmarks/paper-tables/livecodebench-baselines/lcb_baseline_smoke__google-gemma-test.json"
    )


def test_smoke_mode_defaults_to_small_window() -> None:
    module = _load_baseline_module()
    config = module.normalize_config(
        module.BaselineConfig(
            model="model",
            base_url="https://openrouter.example/api/v1",
            api_key=None,
            run_id="run",
            output_root=Path(".benchmarks/livecodebench"),
            summary_root=Path(".benchmarks/paper-tables/livecodebench-baselines"),
            benchmark_root=Path("benchmarks/LiveCodeBench"),
            scenario="codegeneration",
            release_version="release_v1",
            n=10,
            temperature=0.2,
            evaluate=True,
            start_date=None,
            end_date=None,
            smoke=True,
            execute=False,
        )
    )

    assert config.n == 1
    assert config.evaluate is False
    assert config.start_date == module.SMOKE_START_DATE
    assert config.end_date == module.SMOKE_END_DATE


def test_resolve_models_supports_openrouter_matrix() -> None:
    module = _load_baseline_module()

    matrix = module.resolve_models(explicit_model="", model_matrix="openrouter-baselines")

    assert matrix == (
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    )


def test_dry_run_does_not_execute_model_call(tmp_path: Path) -> None:
    module = _load_baseline_module()
    calls: list[object] = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("dry-run should not invoke subprocess.run")

    config = module.BaselineConfig(
        model="google/gemma-test",
        base_url="https://openrouter.example/api/v1",
        api_key=None,
        run_id="dry_run_case",
        output_root=tmp_path / ".benchmarks" / "livecodebench",
        summary_root=tmp_path / ".benchmarks" / "paper-tables" / "livecodebench-baselines",
        benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
        scenario="codegeneration",
        release_version="release_v1",
        n=1,
        temperature=0.2,
        evaluate=False,
        start_date=module.SMOKE_START_DATE,
        end_date=module.SMOKE_END_DATE,
        smoke=True,
        execute=False,
    )

    exit_code = module.run(config, command_runner=fake_runner)

    assert exit_code == 0
    assert calls == []
    metadata_path = module.normalized_run_dir(config) / "metadata.txt"
    assert metadata_path.exists()
    metadata = metadata_path.read_text(encoding="utf-8")
    assert "status=planned" in metadata
    assert "command=" in metadata
    assert module.normalized_summary_path(config).exists()


def test_livecodebench_docs_mark_baseline_only_scope() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    recipes = (REPO_ROOT / "docs" / "benchmark-recipes.md").read_text(encoding="utf-8")
    baselines = (REPO_ROOT / "docs" / "livecodebench-baselines.md").read_text(encoding="utf-8")
    dspy_doc = (REPO_ROOT / "docs" / "dspy-integration.md").read_text(encoding="utf-8")

    assert "LiveCodeBench support is available for baseline model coding ability checks" in readme
    assert (
        "main VTM evidence remains retrieval, drift verification, drifted retrieval, "
        "and controlled coding-drift"
    ) in readme
    assert "LiveCodeBench is available here as an external baseline model benchmark only" in recipes
    assert "bash scripts/run_livecodebench_baseline.sh --smoke --execute" in recipes
    assert "scripts/run_livecodebench_dspy_pilot.py" in recipes
    assert "qwen/qwen3-coder-next" in recipes
    assert "It is not the main VTM memory benchmark" in baselines
    assert (
        "Main VTM evidence remains retrieval, drift verification, drifted retrieval, "
        "and controlled coding-drift"
    ) in baselines
    assert "The DSPy plus VTM LiveCodeBench path is a scaffolded pilot only" in baselines
    assert "SWE-bench Lite remains demoted after empty-patch pilot failures" in baselines
    assert "VTM_OPENROUTER_BASE_URL" in baselines
    assert "OPENROUTER_API_KEY" in baselines
    assert "VTM_EXECUTION_MODEL" in baselines
    assert "scaffolded pilot" in dspy_doc


def test_shell_wrapper_defaults_to_dry_run() -> None:
    completed = subprocess.run(
        ["bash", "scripts/local/run_livecodebench.sh", "--smoke"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "[livecodebench] mode=dry-run" in completed.stdout
    assert "command=" in completed.stdout


def test_new_baseline_surface_has_no_swebench_cli_references() -> None:
    checked_paths = [
        REPO_ROOT / "docs" / "livecodebench-baselines.md",
        REPO_ROOT / "results" / "README.md",
        REPO_ROOT / "scripts" / "run_livecodebench_dspy_pilot.py",
        REPO_ROOT / "scripts" / "livecodebench" / "setup_livecodebench.sh",
        REPO_ROOT / "scripts" / "livecodebench" / "export_dspy_pilot_results.py",
        REPO_ROOT / "scripts" / "livecodebench" / "export_results.py",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench.sh",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench_baseline.sh",
        REPO_ROOT / "scripts" / "shared" / "run_livecodebench.sh",
    ]

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert "run_swebench" not in text
        assert "scripts/local/run_swebench.sh" not in text


def test_livecodebench_dspy_pilot_dry_run_supports_all_methods(
    tmp_path: Path,
    capsys,
) -> None:
    problem_file = tmp_path / "problems.jsonl"
    problem_file.write_text(
        json.dumps(
            {
                "question_id": "lcb-1",
                "prompt": "Write solve() that returns 42.",
                "starter_code": "def solve():\n    pass\n",
                "hidden_tests": ["assert solve() == 42"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = livecodebench_dspy_pilot_main(
        [
            "--method",
            "all",
            "--scenario",
            "code_generation",
            "--problem-offset",
            "0",
            "--max-problems",
            "1",
            "--output-root",
            str(tmp_path / ".benchmarks" / "livecodebench-dspy"),
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "LiveCodeBench"),
            "--problem-file",
            str(problem_file),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["methods"] == ["direct", "dspy_baseline", "dspy_vtm", "dspy_rlm_vtm"]
    assert payload["problem_offset"] == 0
    assert payload["problem_count"] == 1
    assert len(payload["runs"]) == 4
    assert all(
        str(tmp_path / ".benchmarks" / "livecodebench-dspy") in run["run_dir"]
        for run in payload["runs"]
    )


def test_livecodebench_dspy_summary_aggregation_handles_missing_retrieval_metrics() -> None:
    summary = aggregate_summary(
        [
            {
                "problem_id": "lcb-1",
                "method": "direct",
                "evaluation": {"available": True, "passed": True, "syntax_error": False},
                "tool_calls": 0,
            },
            {
                "problem_id": "lcb-2",
                "method": "direct",
                "evaluation": {"available": True, "passed": False, "syntax_error": False},
                "tool_calls": 0,
            },
        ],
        method="direct",
        scenario="code_generation",
        model="qwen/qwen3-coder-next",
    )

    assert summary["pass_count"] == 1
    assert summary["evaluation_available_count"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["retrieval_usage_rate"] == 0.0
    assert summary["retrieval_hit_rate"] == 0.0
    assert summary["mean_verified_count"] == 0.0
    assert summary["mean_stale_filtered_count"] == 0.0
    assert summary["mean_tool_calls"] == 0.0
    assert summary["pilot_limitations"]


def test_livecodebench_dspy_discovers_explicit_source_backends(tmp_path: Path) -> None:
    problem_file = tmp_path / "problems.jsonl"
    problem_file.write_text("", encoding="utf-8")

    assert isinstance(
        discover_problem_source(
            benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
            problem_file=problem_file,
        ),
        ProblemFileSource,
    )
    assert isinstance(
        discover_problem_source(
            benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
            problem_file=None,
        ),
        LiveCodeBenchCheckoutSource,
    )


def test_problem_file_source_honors_problem_offset(tmp_path: Path) -> None:
    problem_file = tmp_path / "problems.jsonl"
    problem_file.write_text(
        json.dumps({"question_id": "lcb-1", "prompt": "First problem"}) + "\n" +
        json.dumps({"question_id": "lcb-2", "prompt": "Second problem"}) + "\n",
        encoding="utf-8",
    )
    source = ProblemFileSource(problem_file)

    problems = source.load_problems("code_generation", problem_offset=1, max_problems=1)

    assert [problem.problem_id for problem in problems] == ["lcb-2"]


def test_livecodebench_dspy_evaluate_public_stdin_tests() -> None:
    source = ProblemFileSource(Path("/tmp/public-problems.jsonl"))
    problem = LiveCodeBenchProblem(
        problem_id="lcb-stdin",
        scenario="code_generation",
        prompt="Read a number and print it back.",
        evaluator_payload={
            "public_tests": [
                {"input": "5\n", "output": "5\n", "testtype": "stdin"},
                {"input": "9\n", "output": "9\n", "testtype": "stdin"},
            ]
        },
    )

    evaluation = source.evaluate(
        problem,
        response_text="print('wrong')",
        extracted_code="value = input().strip()\nprint(value)\n",
    )

    assert evaluation is not None
    assert evaluation["available"] is True
    assert evaluation["scope"] == "public_tests"
    assert evaluation["passed"] is True
    assert evaluation["pass_rate"] == 1.0
    assert evaluation["passed_test_count"] == 2
    assert evaluation["public_test_count"] == 2


def test_livecodebench_dspy_evaluate_functional_public_tests() -> None:
    source = ProblemFileSource(Path("/tmp/public-problems.jsonl"))
    problem = LiveCodeBenchProblem(
        problem_id="lcb-functional",
        scenario="code_generation",
        prompt="Implement add(a, b).",
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )

    evaluation = source.evaluate(
        problem,
        response_text="def add(a, b):\n    return a + b\n",
        extracted_code="def add(a, b):\n    return a + b\n",
    )

    assert evaluation is not None
    assert evaluation["available"] is True
    assert evaluation["passed"] is True
    assert evaluation["pass_rate"] == 1.0
    assert evaluation["passed_test_count"] == 1


def test_livecodebench_dspy_evaluate_functional_public_tests_accepts_solution_wrapper() -> None:
    source = ProblemFileSource(Path("/tmp/public-problems.jsonl"))
    problem = LiveCodeBenchProblem(
        problem_id="lcb-functional-solution-wrapper",
        scenario="code_generation",
        prompt="Implement add(a, b).",
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )

    evaluation = source.evaluate(
        problem,
        response_text="class Solution:\n    def add(self, a, b):\n        return a + b\n",
        extracted_code="class Solution:\n    def add(self, a, b):\n        return a + b\n",
    )

    assert evaluation is not None
    assert evaluation["available"] is True
    assert evaluation["passed"] is True
    assert evaluation["pass_rate"] == 1.0
    assert evaluation["passed_test_count"] == 1


def test_livecodebench_dspy_prompt_omits_gold_and_hidden_fields(tmp_path: Path) -> None:
    problem_file = tmp_path / "problems.jsonl"
    hidden_solution = "def solve():\n    return 42\n"
    hidden_test = "assert solve() == 42"
    problem_file.write_text(
        json.dumps(
            {
                "question_id": "lcb-1",
                "prompt": "Write solve() that returns 42.",
                "starter_code": "def solve():\n    pass\n",
                "public_feedback": "Sample case expects 42.",
                "solution": hidden_solution,
                "hidden_tests": [hidden_test],
                "private_notes": "do not leak",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = ProblemFileSource(problem_file)
    problem = source.load_problems("code_generation", max_problems=1)[0]

    prompt = build_attempt_prompt(problem, attempt_index=1)

    assert hidden_solution not in prompt
    assert hidden_test not in prompt
    assert "do not leak" not in prompt
    assert "Write solve() that returns 42." in prompt
    assert "Sample case expects 42." in prompt


def test_livecodebench_dspy_prompt_uses_agent_contract_for_dspy() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-dspy-contract",
        scenario="code_generation",
        prompt="Implement add(a, b).",
    )

    prompt = build_attempt_prompt(problem, attempt_index=1, agent_mode="dspy")

    assert "Return the final answer as a single ```python fenced code block and nothing else." not in prompt
    assert "Use tools only if needed." in prompt
    assert "final `response` field as a single ```python fenced code block" in prompt


def test_livecodebench_dspy_self_repair_prompt_includes_previous_code_and_feedback() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-repair",
        scenario="self_repair",
        prompt="Implement add(a, b).",
        public_feedback=("Public sample: input='[2, 3]' output='5'",),
    )

    prompt = build_attempt_prompt(
        problem,
        attempt_index=2,
        visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        repair_context=RepairContext(
            previous_response="def add(a, b):\n    return a - b\n",
            previous_code="def add(a, b):\n    return a - b\n",
            visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        ),
    )

    assert "Previous Attempt:" in prompt
    assert "return a - b" in prompt
    assert "Functional public test mismatch: expected=5 actual=4" in prompt
    assert "This is repair attempt 2. Fix the previous attempt." in prompt


def test_livecodebench_dspy_prompt_includes_function_contract_for_functional_problems() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-functional-contract",
        scenario="code_generation",
        prompt="Implement add(a, b).",
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )

    prompt = build_attempt_prompt(problem, attempt_index=1)

    assert "Implementation Contract:" in prompt
    assert "Define a top-level function named `add`" in prompt
    assert "Do not rely on a `class Solution` wrapper" in prompt


def test_livecodebench_dspy_counts_tool_calls_from_trajectory_mapping() -> None:
    payload = {
        "trajectory": {
            "thought_0": "inspect",
            "tool_name_0": "search_verified_memory",
            "tool_args_0": {"query": "parser"},
            "observation_0": [],
            "tool_name_1": "finish",
            "tool_args_1": {},
        }
    }

    assert livecodebench_dspy_pilot._count_serialized_tool_calls(payload) == 2


def test_livecodebench_dspy_counts_tool_calls_from_outer_result_when_response_has_none() -> None:
    result = {
        "trajectory": {
            "tool_name_0": "search_verified_memory",
            "tool_name_1": "expand_memory_evidence",
        }
    }

    assert livecodebench_dspy_pilot._count_result_tool_calls(result, {}) == 2


def test_livecodebench_dspy_summary_uses_retrieval_invocation_rate_not_hit_rate() -> None:
    summary = aggregate_summary(
        [
            {
                "problem_id": "lcb-1",
                "method": "dspy_vtm",
                "evaluation": {"available": True, "passed": True, "syntax_error": False},
                "retrieval": {
                    "invoked": True,
                    "used": False,
                    "verified_count": 0,
                    "stale_filtered_count": 0,
                },
                "tool_calls": 1,
            }
        ],
        method="dspy_vtm",
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
    )

    assert summary["retrieval_usage_rate"] == 1.0
    assert summary["retrieval_hit_rate"] == 0.0


def test_livecodebench_dspy_rlm_attempt_uses_rlm_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def run(self, prompt: str, *, query: str | None = None) -> dict[str, object]:
            captured["prompt"] = prompt
            captured["query"] = query
            return {
                "response": {
                    "response": "```python\nprint('ok')\n```",
                    "trajectory": [{"reasoning": "inspect", "code": "print(1)"}],
                },
                "trajectory": {"execution_mode": "rlm"},
            }

    monkeypatch.setattr(livecodebench_dspy_pilot, "VTMRLMCodingAgent", FakeAgent)
    session = SimpleNamespace(
        kernel=object(),
        scope=object(),
        dependency=object(),
        metadata_store=SimpleNamespace(get_memory_item=lambda _memory_id: None),
    )

    payload = livecodebench_dspy_pilot.run_dspy_attempt(
        prompt="Solve it",
        method="dspy_rlm_vtm",
        session=session,
        model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
            base_url_value="https://openrouter.example/api/v1",
            api_key_value="openrouter-test-key",
            execution_model_name="qwen/qwen3-coder-next",
            dspy_model_name="qwen/qwen3-coder-next",
        ),
        memory_query="lcb-123 | public feedback",
    )

    assert payload["response_text"] == "```python\nprint('ok')\n```"
    assert payload["tool_calls"] == 1
    assert captured["prompt"] == "Solve it"
    assert captured["query"] == "lcb-123 | public feedback"


def test_livecodebench_dspy_output_paths_stay_under_pilot_root(tmp_path: Path) -> None:
    output_root = tmp_path / ".benchmarks" / "livecodebench-dspy"

    run_dir = method_run_dir(
        output_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        run_id="pilot_run",
        method="dspy_vtm",
    )

    assert run_dir.is_relative_to(output_root)


def test_livecodebench_dspy_reference_command_uses_selfrepair_token() -> None:
    config = livecodebench_dspy_pilot.PilotRunConfig(
        methods=("direct",),
        requested_scenario="self_repair",
        resolved_scenario="self_repair",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
        temperature=0.0,
        max_tokens=8192,
        problem_offset=0,
        max_problems=3,
        execute=False,
        output_root=Path(".benchmarks/livecodebench-dspy"),
        benchmark_root=Path("benchmarks/LiveCodeBench"),
        problem_file=None,
        run_id="pilot_run",
    )

    command = livecodebench_dspy_pilot._reference_command(config)

    assert "--scenario selfrepair" in command
    assert "--scenario self_repair" not in command


def test_livecodebench_dspy_source_loads_checkout_problems_via_benchmark_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_root = tmp_path / "benchmarks" / "LiveCodeBench"
    python_bin = benchmark_root / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")

    payload = [
        {
            "question_id": "lcb-1",
            "question_title": "Return 42",
            "question_content": "Write solve() that returns 42.",
            "starter_code": "def solve():\n    pass\n",
            "difficulty": "easy",
            "platform": "leetcode",
            "contest_date": "2024-01-01T00:00:00",
            "metadata": {"func_name": "solve"},
            "public_test_cases": [{"input": "", "output": "42", "testtype": "functional"}],
        }
    ]

    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(livecodebench_sources.subprocess, "run", fake_run)
    source = LiveCodeBenchCheckoutSource(benchmark_root=benchmark_root)

    problems = source.load_problems("code_generation", problem_offset=7, max_problems=1)

    assert len(problems) == 1
    assert problems[0].problem_id == "lcb-1"
    assert problems[0].prompt == "Write solve() that returns 42."
    assert "--problem-offset" in calls[0]
    assert "7" in calls[0]


def test_livecodebench_dspy_execute_missing_checkout_exits_cleanly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="setup_livecodebench.sh"):
        livecodebench_dspy_pilot_main(
            [
                "--method",
                "all",
                "--scenario",
                "code_generation",
                "--max-problems",
                "1",
                "--benchmark-root",
                str(tmp_path / "benchmarks" / "LiveCodeBench"),
                "--execute",
            ]
        )


def test_livecodebench_dspy_dry_run_reports_self_repair_semantics(
    tmp_path: Path,
    capsys,
) -> None:
    problem_file = tmp_path / "problems.jsonl"
    problem_file.write_text(
        json.dumps(
            {
                "question_id": "lcb-1",
                "prompt": "Write solve() that returns 42.",
                "starter_code": "def solve():\n    pass\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = livecodebench_dspy_pilot_main(
        [
            "--method",
            "all",
            "--scenario",
            "self_repair",
            "--problem-offset",
            "1",
            "--max-problems",
            "1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "LiveCodeBench"),
            "--problem-file",
            str(problem_file),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["problem_offset"] == 1
    assert payload["resolved_scenario"] == "self_repair"
    assert "previous candidate code" in payload["scenario_semantics"].lower()


def test_livecodebench_dspy_direct_completion_retries_empty_provider_response() -> None:
    client = livecodebench_dspy_pilot.OpenAICompatibleChatClient(
        livecodebench_dspy_pilot.OpenAICompatibleChatConfig(
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
        )
    )
    responses = [
        {
            "choices": [
                {
                    "finish_reason": None,
                    "message": {"role": "assistant", "content": None, "refusal": None},
                }
            ],
            "usage": {"completion_tokens": 0},
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "```python\nprint(42)\n```"},
                }
            ],
            "usage": {"completion_tokens": 6},
        },
    ]

    def fake_create_chat_completion(**kwargs):
        return responses.pop(0)

    client.create_chat_completion = fake_create_chat_completion  # type: ignore[method-assign]

    payload, text, retry_count, error_text = _request_direct_completion(
        client,
        model="qwen/qwen3-coder-next",
        prompt="print 42",
        temperature=0.0,
        max_tokens=512,
    )

    assert text == "```python\nprint(42)\n```"
    assert retry_count == 1
    assert error_text is not None
    assert payload["usage"]["completion_tokens"] == 6
