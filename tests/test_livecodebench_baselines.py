from __future__ import annotations

import importlib
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
    extract_code,
    method_run_dir,
    open_memory_session,
    record_attempt_memory,
    retrieve_verified_memory,
    seed_problem_memory,
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
LCB_CHECKOUT_ROOT = REPO_ROOT / "benchmarks" / "LiveCodeBench"


def _load_baseline_module():
    module_path = REPO_ROOT / "scripts" / "livecodebench" / "baseline.py"
    spec = importlib.util.spec_from_file_location("vtm_livecodebench_baseline", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_export_results_module():
    module_path = REPO_ROOT / "scripts" / "livecodebench" / "export_results.py"
    spec = importlib.util.spec_from_file_location("vtm_livecodebench_export_results", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_module_from_path(module_name: str, module_path: Path):
    if not module_path.exists():
        pytest.skip(f"Missing module path: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_lcb_checkout_module(module_name: str):
    if not LCB_CHECKOUT_ROOT.exists():
        pytest.skip("LiveCodeBench checkout is not present")
    checkout_path = str(LCB_CHECKOUT_ROOT)
    sys.path.insert(0, checkout_path)
    try:
        for name in tuple(sys.modules):
            if name == "lcb_runner" or name.startswith("lcb_runner."):
                sys.modules.pop(name, None)
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(checkout_path)


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
    assert env["OPENAI_KEY"] == "openrouter-test-key"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.example/api/v1"
    assert env["OPENAI_API_BASE"] == "https://openrouter.example/api/v1"
    assert module.normalized_run_dir(config) == Path(
        ".benchmarks/livecodebench/google-gemma-test/lcb_baseline_smoke"
    )
    assert module.normalized_summary_path(config) == Path(
        ".benchmarks/paper-tables/livecodebench-baselines/lcb_baseline_smoke__google-gemma-test.json"
    )


def test_livecodebench_command_supports_debug_passthrough() -> None:
    module = _load_baseline_module()
    config = module.BaselineConfig(
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
        run_id="lcb_baseline_debug",
        output_root=Path(".benchmarks/livecodebench"),
        summary_root=Path(".benchmarks/paper-tables/livecodebench-baselines"),
        benchmark_root=Path("benchmarks/LiveCodeBench"),
        scenario="codegeneration",
        release_version="release_v1",
        n=1,
        temperature=0.2,
        evaluate=True,
        start_date="2023-05-07",
        end_date="2023-05-07",
        smoke=False,
        execute=False,
        debug=True,
    )

    command = module.build_livecodebench_command("python3", config)

    assert "--evaluate" in command
    assert "--debug" in command


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


def test_baseline_run_records_official_eval_metrics(tmp_path: Path) -> None:
    module = _load_baseline_module()
    benchmark_root = tmp_path / "benchmarks" / "LiveCodeBench"
    benchmark_root.mkdir(parents=True)
    output_dir = benchmark_root / "output" / module.model_slug("google/gemma-test")
    config = module.BaselineConfig(
        model="google/gemma-test",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
        run_id="executed_case",
        output_root=tmp_path / ".benchmarks" / "livecodebench",
        summary_root=tmp_path / ".benchmarks" / "paper-tables" / "livecodebench-baselines",
        benchmark_root=benchmark_root,
        scenario="codegeneration",
        release_version="release_v1",
        n=10,
        temperature=0.2,
        evaluate=True,
        start_date=None,
        end_date=None,
        smoke=False,
        execute=True,
    )

    def fake_runner(command, cwd, env, check):
        assert cwd == benchmark_root
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "codegeneration_10_0.2.json").write_text("[]\n", encoding="utf-8")
        (output_dir / "codegeneration_10_0.2_eval.json").write_text(
            json.dumps(
                [
                    {
                        "pass@1": 0.25,
                        "pass@5": 0.5,
                        "detail": {"pass@1": {"q1": 1.0, "q2": 0.0}},
                    },
                    {},
                    [],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "codegeneration_10_0.2_eval_all.json").write_text(
            json.dumps([{"question_id": "q1"}, {"question_id": "q2"}]) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=command, returncode=0)

    exit_code = module.run(config, command_runner=fake_runner)

    assert exit_code == 0
    summary = json.loads(module.normalized_summary_path(config).read_text(encoding="utf-8"))
    assert summary["official_metrics_available"] is True
    assert summary["official_pass_at_1"] == 0.25
    assert summary["official_pass_at_5"] == 0.5
    assert summary["official_problem_count"] == 2
    assert summary["eval_file"].endswith("codegeneration_10_0.2_eval.json")
    assert summary["wrapper_output_dir"].endswith("executed_case")
    assert summary["benchmark_output_dir"].endswith(module.model_slug("google/gemma-test"))
    metadata_text = (module.normalized_run_dir(config) / "metadata.txt").read_text(encoding="utf-8")
    assert "official_pass_at_1=0.25" in metadata_text
    assert "official_pass_at_5=0.5" in metadata_text
    assert "wrapper_output_dir=" in metadata_text
    assert "benchmark_output_dir=" in metadata_text


def test_baseline_run_does_not_reuse_stale_metrics_from_other_model(tmp_path: Path) -> None:
    module = _load_baseline_module()
    benchmark_root = tmp_path / "benchmarks" / "LiveCodeBench"
    stale_dir = benchmark_root / "output" / "stale-model"
    stale_dir.mkdir(parents=True)
    (stale_dir / "codegeneration_10_0.2.json").write_text("[]\n", encoding="utf-8")
    (stale_dir / "codegeneration_10_0.2_eval.json").write_text(
        json.dumps([{"pass@1": 1.0, "pass@5": 1.0, "detail": {"pass@1": {"q1": 1.0}}}, {}, []]) + "\n",
        encoding="utf-8",
    )
    (stale_dir / "codegeneration_10_0.2_eval_all.json").write_text(
        json.dumps([{"question_id": "q1"}]) + "\n",
        encoding="utf-8",
    )
    config = module.BaselineConfig(
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
        run_id="stale_metrics_case",
        output_root=tmp_path / ".benchmarks" / "livecodebench",
        summary_root=tmp_path / ".benchmarks" / "paper-tables" / "livecodebench-baselines",
        benchmark_root=benchmark_root,
        scenario="codegeneration",
        release_version="release_v1",
        n=10,
        temperature=0.2,
        evaluate=True,
        start_date=None,
        end_date=None,
        smoke=False,
        execute=True,
    )

    def fake_runner(command, cwd, env, check):
        return subprocess.CompletedProcess(args=command, returncode=0)

    exit_code = module.run(config, command_runner=fake_runner)

    assert exit_code == 0
    summary = json.loads(module.normalized_summary_path(config).read_text(encoding="utf-8"))
    assert summary["official_metrics_available"] is False
    assert summary["official_pass_at_1"] is None
    assert summary["eval_file"] is None


def test_export_results_merges_summary_json_metrics(tmp_path: Path) -> None:
    module = _load_export_results_module()
    input_root = tmp_path / ".benchmarks" / "livecodebench"
    run_dir = input_root / "fake-model" / "run"
    run_dir.mkdir(parents=True)
    summary_path = tmp_path / ".benchmarks" / "paper-tables" / "livecodebench-baselines" / "run__fake-model.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "official_metrics_available": True,
                "official_pass_at_1": 0.33,
                "official_pass_at_5": 0.66,
                "wrapper_output_dir": "/tmp/fake-wrapper-output",
                "benchmark_output_dir": "/tmp/fake-benchmark-output",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.txt").write_text(
        "\n".join(
            [
                "run_id=run",
                "model=fake-model",
                "scenario=codegeneration",
                "release_version=release_v1",
                "smoke=false",
                "evaluate=true",
                f"summary_path={summary_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module.collect_rows(input_root)

    assert len(rows) == 1
    assert rows[0]["status"] == "passed"
    assert rows[0]["official_pass_at_1"] == 0.33
    assert rows[0]["official_pass_at_5"] == 0.66
    assert rows[0]["wrapper_output_dir"] == "/tmp/fake-wrapper-output"
    assert rows[0]["benchmark_output_dir"] == "/tmp/fake-benchmark-output"

    markdown_path = tmp_path / "summary.md"
    module.write_markdown(markdown_path, rows)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "/tmp/fake-benchmark-output" in markdown


def test_livecodebench_checkout_resolves_custom_openrouter_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.example/api/v1")
    module = _load_module_from_path(
        "vtm_lcb_lm_styles",
        LCB_CHECKOUT_ROOT / "lcb_runner" / "lm_styles.py",
    )

    model = module.resolve_language_model("qwen/qwen3-coder-next")

    assert model.model_name == "qwen/qwen3-coder-next"
    assert model.model_repr == "qwen-qwen3-coder-next"
    assert model.model_style == module.LMStyle.OpenAIChat
    assert model.to_dict()["release_date"] is None


def test_livecodebench_checkout_oai_runner_uses_openrouter_env_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    pytest.importorskip("pebble")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.example/api/v1")
    module = _load_lcb_checkout_module("lcb_runner.runner.oai_runner")

    assert module._openai_api_key() == "openrouter-test-key"
    assert module._openai_base_url() == "https://openrouter.example/api/v1"


def test_livecodebench_checkout_code_generation_loader_reads_hub_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_hub_module = SimpleNamespace(hf_hub_download=lambda **_: str(tmp_path / "unused.jsonl"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub_module)
    module = _load_module_from_path(
        "vtm_lcb_code_generation",
        LCB_CHECKOUT_ROOT / "lcb_runner" / "benchmarks" / "code_generation.py",
    )
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_title": "Add",
                        "question_content": "Implement add(a, b).",
                        "platform": "leetcode",
                        "question_id": "q1",
                        "contest_id": "contest-1",
                        "contest_date": "2023-05-07T00:00:00",
                        "starter_code": "def add(a, b):\n    pass\n",
                        "difficulty": "easy",
                        "public_test_cases": json.dumps(
                            [{"input": "[2, 3]", "output": "5", "testtype": "functional"}]
                        ),
                        "private_test_cases": json.dumps(
                            [{"input": "[4, 5]", "output": "9", "testtype": "functional"}]
                        ),
                        "metadata": json.dumps({"func_name": "add"}),
                    }
                ),
                json.dumps(
                    {
                        "question_title": "Mul",
                        "question_content": "Implement mul(a, b).",
                        "platform": "leetcode",
                        "question_id": "q2",
                        "contest_id": "contest-2",
                        "contest_date": "2023-05-13T00:00:00",
                        "starter_code": "def mul(a, b):\n    pass\n",
                        "difficulty": "medium",
                        "public_test_cases": json.dumps(
                            [{"input": "[2, 3]", "output": "6", "testtype": "functional"}]
                        ),
                        "private_test_cases": json.dumps(
                            [{"input": "[4, 5]", "output": "20", "testtype": "functional"}]
                        ),
                        "metadata": json.dumps({"func_name": "mul"}),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_hf_hub_download(*, repo_id, repo_type, filename, revision=None):
        assert repo_id == module.CODE_GENERATION_LITE_DATASET_REPO
        assert repo_type == "dataset"
        assert filename == module.CODE_GENERATION_DATASET_FILENAME
        assert revision is None
        return str(dataset_path)

    monkeypatch.setattr(module, "hf_hub_download", fake_hf_hub_download)

    dataset = module.load_code_generation_dataset(
        release_version="release_v1",
        start_date="2023-05-07",
        end_date="2023-05-07",
    )

    assert len(dataset) == 1
    assert dataset[0].question_id == "q1"
    assert dataset[0].metadata["func_name"] == "add"


def test_livecodebench_checkout_code_generation_release_version_maps_expected_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_hub_module = SimpleNamespace(hf_hub_download=lambda **_: str(tmp_path / "unused.jsonl"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub_module)
    module = _load_module_from_path(
        "vtm_lcb_code_generation_release_map",
        LCB_CHECKOUT_ROOT / "lcb_runner" / "benchmarks" / "code_generation.py",
    )

    assert module._dataset_filenames_for_release(
        dataset_repo=module.CODE_GENERATION_LITE_DATASET_REPO,
        release_version="release_v2",
    ) == ("test.jsonl", "test2.jsonl")
    assert module._dataset_filenames_for_release(
        dataset_repo=module.CODE_GENERATION_LITE_DATASET_REPO,
        release_version="v2",
    ) == ("test2.jsonl",)
    assert module._dataset_filenames_for_release(
        dataset_repo=module.CODE_GENERATION_LITE_DATASET_REPO,
        release_version="v1_v3",
    ) == ("test.jsonl", "test2.jsonl", "test3.jsonl")


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
    assert "--start-date 2023-05-07" in recipes
    assert "--debug" in baselines
    assert "official_pass_at_1" in baselines
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
    assert payload["methods"] == [
        "direct",
        "dspy_baseline",
        "dspy_vtm",
        "dspy_rlm_baseline",
        "dspy_rlm_vtm",
    ]
    assert payload["problem_offset"] == 0
    assert payload["problem_count"] == 1
    assert len(payload["runs"]) == 5
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
    assert summary["public_test_pass_rate"] == 0.5
    assert "accuracy" not in summary
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


def test_livecodebench_dspy_extract_code_prefers_final_fenced_block() -> None:
    response = """
Here is the bug in the previous attempt:
```python
tmp = broken_call()
```

Final answer:
```python
def solve():
    print("ok")
```
""".strip()

    assert extract_code(response) == 'def solve():\n    print("ok")'


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


def test_livecodebench_dspy_retrieval_prioritizes_repair_feedback_cards(
    tmp_path: Path,
) -> None:
    session = open_memory_session(
        state_root=tmp_path / "pilot-state",
        problem_id="lcb-feedback-priority",
        workspace_root=tmp_path,
    )
    problem = LiveCodeBenchProblem(
        problem_id="lcb-feedback-priority",
        scenario="self_repair",
        prompt="Implement solve() for the provided stdin task.",
        evaluator_payload={
            "public_tests": [
                {"input": "2\n", "output": "4\n", "testtype": "stdin"},
            ],
        },
    )
    try:
        seed_problem_memory(session, problem)
        record_attempt_memory(
            session,
            problem=problem,
            attempt_index=1,
            response_text="```python\nprint(3)\n```",
            extracted_code="print(3)\n",
            evaluation={
                "failure_feedback": [
                    "Public stdin test mismatch: expected='4' actual='3'",
                ]
            },
        )

        payload = retrieve_verified_memory(
            session,
            query="lcb-feedback-priority | Public stdin test mismatch: expected='4' actual='3'",
            attempt_index=2,
        )
    finally:
        session.close()

    roles = [card.get("role") for card in payload["cards"]]

    assert payload["used"] is True
    assert roles[0] == "feedback_item"
    assert "problem_summary" not in roles


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


def test_livecodebench_dspy_attempt_propagates_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, prompt: str, *, query: str | None = None) -> dict[str, object]:
            return {
                "response": {
                    "response": "",
                    "error": "DSPy RLM emitted an action with code=None before final extraction.",
                },
                "trajectory": {
                    "execution_mode": "rlm",
                    "execution_error": "DSPy RLM emitted an action with code=None before final extraction.",
                },
            }

    monkeypatch.setattr(livecodebench_dspy_pilot, "VTMRLMCodingAgent", FakeAgent)

    payload = livecodebench_dspy_pilot.run_dspy_attempt(
        prompt="Solve it",
        method="dspy_rlm_baseline",
        session=None,
        model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
            base_url_value="https://openrouter.example/api/v1",
            api_key_value="openrouter-test-key",
            execution_model_name="qwen/qwen3-coder-next",
            dspy_model_name="qwen/qwen3-coder-next",
        ),
    )

    assert payload["response_text"] == ""
    assert payload["response_error"] is not None
    assert "code=None" in payload["response_error"]


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


def test_livecodebench_dspy_execute_skips_rlm_methods_without_deno(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem_file = tmp_path / "problems.jsonl"
    problem_file.write_text(
        json.dumps(
            {
                "question_id": "lcb-1",
                "prompt": "Write solve() that returns 42.",
                "starter_code": "def solve():\n    pass\n",
                "public_test_cases": [{"input": "", "output": "42", "testtype": "functional"}],
                "metadata": {"func_name": "solve"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = ProblemFileSource(problem_file)
    config = livecodebench_dspy_pilot.PilotRunConfig(
        methods=(
            "direct",
            "dspy_baseline",
            "dspy_vtm",
            "dspy_rlm_baseline",
            "dspy_rlm_vtm",
        ),
        requested_scenario="self_repair",
        resolved_scenario="self_repair",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
        temperature=0.0,
        max_tokens=4096,
        problem_offset=0,
        max_problems=1,
        execute=True,
        output_root=tmp_path / ".benchmarks" / "livecodebench-dspy",
        benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
        problem_file=problem_file,
        run_id="skip_rlm_missing_deno",
    )

    called_methods: list[str] = []

    def fake_execute_method(
        method: str,
        *,
        config,
        problems,
        source,
    ) -> list[dict[str, object]]:
        del config, problems, source
        called_methods.append(method)
        return [
            {
                "problem_id": "lcb-1",
                "method": method,
                "evaluation": {"available": True, "passed": True, "syntax_error": False},
                "tool_calls": 0,
            }
        ]

    monkeypatch.setattr(
        livecodebench_dspy_pilot,
        "rlm_interpreter_availability",
        lambda: (False, "Deno required for DSPy RLM."),
    )
    monkeypatch.setattr(livecodebench_dspy_pilot, "execute_method", fake_execute_method)

    payload = livecodebench_dspy_pilot.run_pilot(config, source=source)

    assert called_methods == ["direct", "dspy_baseline", "dspy_vtm"]
    skipped_runs = {
        run["method"]: run
        for run in payload["runs"]
        if run["method"] in {"dspy_rlm_baseline", "dspy_rlm_vtm"}
    }
    assert skipped_runs["dspy_rlm_baseline"]["skipped"] is True
    assert skipped_runs["dspy_rlm_vtm"]["skipped"] is True
    assert skipped_runs["dspy_rlm_baseline"]["summary"]["status"] == "skipped"
    assert skipped_runs["dspy_rlm_vtm"]["summary"]["status"] == "skipped"
    assert "Deno required" in skipped_runs["dspy_rlm_baseline"]["skip_reason"]


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
