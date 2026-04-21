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


def _load_export_dspy_pilot_results_module():
    module_path = REPO_ROOT / "scripts" / "livecodebench" / "export_dspy_pilot_results.py"
    spec = importlib.util.spec_from_file_location("vtm_livecodebench_export_dspy_pilot", module_path)
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


def test_export_dspy_pilot_results_surfaces_memory_attribution_and_pass_curves(
    tmp_path: Path,
) -> None:
    module = _load_export_dspy_pilot_results_module()
    input_root = tmp_path / ".benchmarks" / "livecodebench-dspy"
    summary_path = (
        input_root
        / "self_repair"
        / "qwen-qwen3-coder-next"
        / "run"
        / "dspy_vtm"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "kind": "dspy_pilot",
                "scenario": "self_repair",
                "method": "dspy_vtm",
                "model": "qwen/qwen3-coder-next",
                "total": 3,
                "public_test_pass_rate": 0.5,
                "attempt1_public_test_pass_at_1": 0.25,
                "attempt1_public_test_pass_at_k": 0.5,
                "attempt1_public_test_pass_curve": {"1": 0.25, "2": 0.5},
                "attempt2_public_test_pass_at_1": 0.4,
                "attempt2_public_test_pass_at_k": 0.6,
                "attempt2_public_test_pass_curve": {"1": 0.4, "2": 0.6},
                "attempt2_delta_over_attempt1": 0.15,
                "candidate_selection_mode": "best_of_k_public_tests",
                "candidates_per_attempt": 2,
                "retrieval_usage_rate": 1.0,
                "canonical_memory_hit_rate": 0.5,
                "agent_memory_write_rate": 0.25,
                "mean_agent_memory_write_count": 0.25,
                "consolidated_memory_card_rate": 0.25,
                "mean_verified_count": 2.0,
                "mean_stale_filtered_count": 0.0,
                "mean_tool_calls": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module.collect_rows(input_root)
    markdown_path = tmp_path / "paper_tables.md"
    module.write_markdown(markdown_path, rows)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert rows[0]["attempt2_pass_at_k"] == 0.6
    assert rows[0]["attempt1_pass_curve"] == {"1": 0.25, "2": 0.5}
    assert rows[0]["canonical_memory_hit_rate"] == 0.5
    assert "Pass Curve (A1)" in markdown
    assert "Pass Curve (A2)" in markdown
    assert "Canonical Hit Rate" in markdown
    assert "Agent Write Rate" in markdown
    assert "1:0.250, 2:0.500" in markdown
    assert "1:0.400, 2:0.600" in markdown


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
        "dspy_vtm_local_only",
        "dspy_vtm_persistent_only",
        "dspy_vtm",
    ]
    assert payload["problem_offset"] == 0
    assert payload["problem_count"] == 1
    assert payload["runs"][0]["runtime"]["method"] == "direct"


def test_livecodebench_dspy_default_max_tokens_is_high_for_long_repairs() -> None:
    parser = livecodebench_dspy_pilot.build_parser()
    args = parser.parse_args([])

    assert args.max_tokens == 65536


def test_livecodebench_dspy_summary_aggregation_handles_missing_retrieval_metrics() -> None:
    summary = aggregate_summary(
        [
            {
                "problem_id": "lcb-1",
                "method": "direct",
                "evaluation": {"available": True, "passed": True, "syntax_error": False},
                "tool_calls": 0,
                "candidate_batches": [
                    {
                        "attempt_index": 1,
                        "candidate_count": 1,
                        "selected_candidate_index": 1,
                        "selection_mode": "single_sample",
                        "candidates": [{"candidate_index": 1, "passed": True}],
                    }
                ],
            },
            {
                "problem_id": "lcb-2",
                "method": "direct",
                "evaluation": {"available": True, "passed": False, "syntax_error": False},
                "tool_calls": 0,
                "candidate_batches": [
                    {
                        "attempt_index": 1,
                        "candidate_count": 1,
                        "selected_candidate_index": 1,
                        "selection_mode": "single_sample",
                        "candidates": [{"candidate_index": 1, "passed": False}],
                    }
                ],
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
    assert summary["mean_candidates_per_attempt"] == 1.0
    assert summary["candidate_selection_mode"] == "single_sample"
    assert summary["attempt1_public_test_pass_at_1_count"] == 1
    assert summary["attempt1_public_test_pass_at_1"] == 0.5
    assert summary["attempt1_public_test_pass_at_k_count"] == 1
    assert summary["attempt1_public_test_pass_at_k"] == 0.5
    assert summary["pilot_limitations"]


def test_livecodebench_dspy_summary_aggregation_reports_best_of_k_metadata() -> None:
    summary = aggregate_summary(
        [
            {
                "problem_id": "lcb-1",
                "method": "direct",
                "evaluation": {"available": True, "passed": True, "syntax_error": False},
                "tool_calls": 0,
                "candidates_per_attempt": 3,
                "candidate_batches": [
                    {
                        "attempt_index": 1,
                        "candidate_count": 3,
                        "selected_candidate_index": 2,
                        "selection_mode": "best_of_k_public_tests",
                        "candidates": [
                            {"candidate_index": 1, "passed": False},
                            {"candidate_index": 2, "passed": True},
                            {"candidate_index": 3, "passed": False},
                        ],
                    }
                ],
            },
        ],
        method="direct",
        scenario="code_generation",
        model="qwen/qwen3-coder-next",
    )

    assert summary["mean_candidates_per_attempt"] == 3.0
    assert summary["candidate_selection_mode"] == "best_of_k_public_tests"
    assert summary["attempt1_public_test_pass_at_1"] == 0.0
    assert summary["attempt1_public_test_pass_at_k"] == 1.0
    assert summary["attempt1_public_test_pass_curve"] == {
        "1": 0.0,
        "2": 1.0,
        "3": 1.0,
    }


def test_livecodebench_dspy_summary_aggregation_reports_memory_attribution_metrics() -> None:
    summary = aggregate_summary(
        [
            {
                "problem_id": "lcb-1",
                "method": "dspy_vtm",
                "evaluation": {"available": True, "passed": True, "syntax_error": False},
                "canonical_memory_hit_count": 1,
                "agent_memory_write_count": 2,
                "consolidated_memory_card_count": 1,
                "candidate_batches": [
                    {
                        "attempt_index": 1,
                        "candidates": [{"candidate_index": 1, "passed": False}],
                    },
                    {
                        "attempt_index": 2,
                        "candidates": [{"candidate_index": 1, "passed": True}],
                    },
                ],
            },
            {
                "problem_id": "lcb-2",
                "method": "dspy_vtm",
                "evaluation": {"available": True, "passed": False, "syntax_error": False},
                "canonical_memory_hit_count": 0,
                "agent_memory_write_count": 0,
                "consolidated_memory_card_count": 0,
                "candidate_batches": [
                    {
                        "attempt_index": 1,
                        "candidates": [
                            {"candidate_index": 1, "passed": False},
                            {"candidate_index": 2, "passed": True},
                        ],
                    }
                ],
            },
        ],
        method="dspy_vtm",
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
    )

    assert summary["canonical_memory_hit_rate"] == 0.5
    assert summary["mean_canonical_memory_hit_count"] == 0.5
    assert summary["agent_memory_write_rate"] == 0.5
    assert summary["mean_agent_memory_write_count"] == 1.0
    assert summary["total_agent_memory_write_count"] == 2
    assert summary["consolidated_memory_card_rate"] == 0.5
    assert summary["attempt1_public_test_pass_curve"] == {"1": 0.0, "2": 0.5}
    assert summary["attempt2_public_test_pass_curve"] == {"1": 0.5}


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


def test_livecodebench_dspy_vtm_repair_prompt_requires_memory_workflow() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-dspy-memory-workflow",
        scenario="self_repair",
        prompt="Implement add(a, b).",
    )

    prompt = build_attempt_prompt(
        problem,
        attempt_index=2,
        agent_mode="dspy",
        require_memory_tooling=True,
        suggested_memory_query="self_repair | top_level_function | function add | expected 5 actual 4",
        visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        repair_context=RepairContext(
            previous_response="def add(a, b):\n    return a - b\n",
            previous_code="def add(a, b):\n    return a - b\n",
            visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        ),
    )

    assert "Memory Workflow:" in prompt
    assert "search_verified_memory" in prompt
    assert "expand_memory_evidence" in prompt
    assert "Do not skip memory lookup on repair attempts" in prompt
    assert "Suggested memory query:" in prompt


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

    assert "Previous Candidate:" in prompt
    assert "return a - b" in prompt
    assert "Functional public test mismatch: expected=5 actual=4" in prompt
    assert "This is repair attempt 2. Fix the previous attempt." in prompt


def test_livecodebench_dspy_prompt_groups_memory_by_role() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-role-sections",
        scenario="self_repair",
        prompt="Implement add(a, b).",
        public_feedback=("Public sample: input='[2, 3]' output='5'",),
        evaluator_payload={"problem_metadata": {"func_name": "add"}},
    )

    prompt = build_attempt_prompt(
        problem,
        attempt_index=2,
        memory_cards=(
            {
                "id": "contract",
                "role": "function_contract",
                "title": "Interface contract",
                "summary": "Expose top-level add(a, b).",
                "score": 1.0,
            },
            {
                "id": "tests",
                "role": "public_tests",
                "title": "Public tests",
                "summary": "Functional example 1: input='[2, 3]' output='5'",
                "score": 0.9,
            },
            {
                "id": "repair",
                "role": "repair_lesson",
                "title": "Repair lesson",
                "summary": "Fix sign errors before returning the sum.",
                "score": 0.8,
            },
        ),
        visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        repair_context=RepairContext(
            previous_response="def add(a, b):\n    return a - b\n",
            previous_code="def add(a, b):\n    return a - b\n",
            visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        ),
    )

    assert "Verified Contract Hints:" in prompt
    assert "Public-Test Signals:" in prompt
    assert "Visible Failure:" in prompt
    assert "Verified Repair Lessons:" in prompt
    assert "Verified Memory Cards:" not in prompt


def test_livecodebench_dspy_compact_repair_prompt_focuses_on_failure_and_one_lesson() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-compact-repair",
        scenario="self_repair",
        prompt="Implement add(a, b).",
    )

    prompt = build_attempt_prompt(
        problem,
        attempt_index=3,
        compact_repair=True,
        memory_cards=(
            {
                "id": "repair-1",
                "role": "repair_lesson",
                "title": "Lesson one",
                "summary": "Return the sum, not the difference.",
                "score": 1.0,
            },
            {
                "id": "repair-2",
                "role": "repair_lesson",
                "title": "Lesson two",
                "summary": "Do not mutate inputs.",
                "score": 0.9,
            },
        ),
        visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        repair_context=RepairContext(
            previous_response="def add(a, b):\n    return a - b\n",
            previous_code="def add(a, b):\n    return a - b\n",
            visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        ),
    )

    assert "Problem Statement:" not in prompt
    assert "Previous Candidate:" in prompt
    assert "Visible Failure:" in prompt
    assert "Verified Repair Lessons:" in prompt
    assert "Lesson one" in prompt
    assert "Lesson two" not in prompt
    assert "final short repair attempt 3" in prompt


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


def test_livecodebench_dspy_merge_memory_cards_budgets_and_dedupes_repair_cards() -> None:
    cards = livecodebench_dspy_pilot.merge_memory_cards(
        (
            {
                "id": "repair-1",
                "role": "repair_lesson",
                "title": "Repair lesson 1",
                "summary": "Fix NameError and return the right sum.",
                "score": 0.9,
                "problem_id": "lcb-1",
                "function_name": "add",
                "feedback_signature": "NameError List not defined | expected 5 actual 4",
            },
            {
                "id": "repair-2",
                "role": "repair_lesson",
                "title": "Repair lesson 2",
                "summary": "Same lesson phrased differently.",
                "score": 0.8,
                "problem_id": "lcb-1",
                "function_name": "add",
                "feedback_signature": "NameError List not defined | expected 5 actual 4",
            },
            {
                "id": "solution-1",
                "role": "successful_solution",
                "title": "Successful solution",
                "summary": "Full solved program.",
                "score": 0.95,
                "problem_id": "lcb-1",
                "function_name": "add",
            },
            {
                "id": "feedback-1",
                "role": "feedback_item",
                "title": "Visible failure",
                "summary": "expected 5 actual 4",
                "score": 1.0,
            },
            {
                "id": "feedback-2",
                "role": "feedback_item",
                "title": "Another visible failure",
                "summary": "NameError: List not defined",
                "score": 0.7,
            },
        ),
        attempt_index=2,
    )

    assert [card["id"] for card in cards] == ["feedback-1", "repair-1"]


def test_livecodebench_dspy_merge_memory_cards_attempt_one_stays_compact() -> None:
    cards = livecodebench_dspy_pilot.merge_memory_cards(
        (
            {
                "id": "contract-1",
                "role": "function_contract",
                "title": "Contract 1",
                "summary": "Expose add(a, b).",
                "score": 0.9,
            },
            {
                "id": "contract-2",
                "role": "function_contract",
                "title": "Contract 2",
                "summary": "Expose add(a, b) at module scope.",
                "score": 0.8,
            },
            {
                "id": "tests-1",
                "role": "public_tests",
                "title": "Tests",
                "summary": "input='[2,3]' output='5'",
                "score": 0.85,
            },
            {
                "id": "problem-1",
                "role": "problem_summary",
                "title": "Problem",
                "summary": "Add two numbers.",
                "score": 1.0,
            },
        ),
        attempt_index=1,
    )

    assert [card["id"] for card in cards] == ["tests-1", "contract-1"]


def test_livecodebench_dspy_attempt_one_retrieval_uses_contract_cards_only(
    tmp_path: Path,
) -> None:
    session = open_memory_session(
        state_root=tmp_path / "pilot-state",
        problem_id="lcb-attempt-one",
        workspace_root=tmp_path,
    )
    problem = LiveCodeBenchProblem(
        problem_id="lcb-attempt-one",
        scenario="code_generation",
        prompt="Implement add(a, b).",
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )
    try:
        seed_problem_memory(session, problem)
        payload = retrieve_verified_memory(
            session,
            query="lcb-attempt-one | implement add",
            attempt_index=1,
            allowed_roles=livecodebench_dspy_pilot.ATTEMPT_ONE_MEMORY_ROLES,
            limit=2,
        )
    finally:
        session.close()

    roles = {card.get("role") for card in payload["cards"]}

    assert payload["used"] is True
    assert roles <= {"function_contract", "public_tests"}


def test_livecodebench_dspy_persistent_success_memory_is_retrievable_across_sessions(
    tmp_path: Path,
) -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-persistent-success",
        scenario="self_repair",
        prompt="Implement add(a, b) and return the sum.",
        prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )
    persistent_root = tmp_path / "persistent-memory"
    session = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        write_result = livecodebench_dspy_pilot.write_persistent_success_memory(
            session,
            problem=problem,
            attempt_index=2,
            response_text="```python\ndef add(a, b):\n    return a + b\n```",
            extracted_code="def add(a, b):\n    return a + b\n",
            evaluation={"available": True, "passed": True, "failure_feedback": []},
            visible_feedback=("Functional public test mismatch: expected=5 actual=4",),
        )
    finally:
        session.close()

    reopened = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        payload = retrieve_verified_memory(
            reopened,
            query="add implement add return sum expected 5 actual 4",
            attempt_index=1,
            allowed_roles=frozenset({"successful_solution"}),
            limit=2,
        )
    finally:
        reopened.close()

    assert write_result is not None
    assert write_result["success_memory_id"] == livecodebench_dspy_pilot.success_memory_id(problem.problem_id)
    assert payload["used"] is True
    assert payload["cards"][0]["role"] == "successful_solution"
    assert payload["cards"][0]["title"] == "Successful top_level_function pattern for add"
    assert "reusable successful solution pattern" in payload["cards"][0]["summary"].lower()
    assert "leetcode" in payload["cards"][0]["transfer_terms"]
    assert "top_level_function" in payload["cards"][0]["transfer_terms"]
    assert "add" in payload["cards"][0]["summary"].lower()


def test_livecodebench_dspy_persistent_repair_lesson_is_retrievable_across_sessions(
    tmp_path: Path,
) -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-persistent-repair",
        scenario="self_repair",
        prompt="Implement add(a, b) and return the sum.",
        prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )
    persistent_root = tmp_path / "persistent-memory"
    visible_feedback = (
        "Functional public test mismatch: expected=5 actual=4",
        "NameError: name 'List' is not defined",
    )
    session = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        livecodebench_dspy_pilot.write_persistent_success_memory(
            session,
            problem=problem,
            attempt_index=2,
            response_text="```python\ndef add(a, b):\n    return a + b\n```",
            extracted_code="def add(a, b):\n    return a + b\n",
            evaluation={"available": True, "passed": True, "failure_feedback": []},
            visible_feedback=visible_feedback,
        )
    finally:
        session.close()

    reopened = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        payload = retrieve_verified_memory(
            reopened,
            query="add expected 5 actual 4 NameError List not defined",
            attempt_index=2,
            allowed_roles=frozenset({"repair_lesson"}),
            limit=2,
            expand_top_k=1,
        )
    finally:
        reopened.close()

    assert payload["used"] is True
    assert payload["cards"][0]["role"] == "repair_lesson"
    assert payload["cards"][0]["title"] == "Repair lesson: runtime_nameerror on top_level_function"
    assert "resolved visible feedback" in str(payload["cards"][0]["summary"]).lower()
    assert payload["cards"][0]["repair_kind"] == "runtime_nameerror"
    assert payload["cards"][0]["interface_mode"] == "top_level_function"
    assert payload["cards"][0]["platform"] == "leetcode"
    assert payload["cards"][0]["difficulty"] == "easy"
    assert "nameerror" in payload["cards"][0]["transfer_terms"]
    assert "Resolved Feedback:" in str(payload["cards"][0]["rationale_preview"])


def test_livecodebench_dspy_persistent_repair_consolidation_creates_canonical_summary_card(
    tmp_path: Path,
) -> None:
    persistent_root = tmp_path / "persistent-memory"
    problems = [
        LiveCodeBenchProblem(
            problem_id="lcb-canonical-repair-a",
            scenario="self_repair",
            prompt="Implement add(a, b) and return the sum.",
            prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
            evaluator_payload={
                "public_tests": [
                    {"input": "[2, 3]", "output": "5", "testtype": "functional"},
                ],
                "problem_metadata": {"func_name": "add"},
            },
        ),
        LiveCodeBenchProblem(
            problem_id="lcb-canonical-repair-b",
            scenario="self_repair",
            prompt="Implement add(a, b) for another task.",
            prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
            evaluator_payload={
                "public_tests": [
                    {"input": "[9, 2]", "output": "11", "testtype": "functional"},
                ],
                "problem_metadata": {"func_name": "add"},
            },
        ),
    ]
    visible_feedback = ("Functional public test mismatch: expected=5 actual=4",)

    session = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        for problem in problems:
            livecodebench_dspy_pilot.write_persistent_success_memory(
                session,
                problem=problem,
                attempt_index=2,
                response_text="```python\ndef add(a, b):\n    return a + b\n```",
                extracted_code="def add(a, b):\n    return a + b\n",
                evaluation={"available": True, "passed": True, "failure_feedback": []},
                visible_feedback=visible_feedback,
            )
        payload = retrieve_verified_memory(
            session,
            query=livecodebench_dspy_pilot.build_retrieval_query(
                problems[-1],
                visible_feedback,
                store_kind="persistent",
            ),
            attempt_index=2,
            allowed_roles=frozenset({"canonical_repair_lesson"}),
            limit=3,
            expand_top_k=1,
        )
    finally:
        session.close()

    assert payload["used"] is True
    assert payload["cards"][0]["role"] == "canonical_repair_lesson"
    assert payload["cards"][0]["title"] == (
        "Canonical repair lesson: public_test_logic_mismatch on top_level_function for add"
    )
    assert payload["cards"][0]["canonical_support_count"] == 2
    assert "canonical repair lesson distilled" in str(payload["cards"][0]["summary"]).lower()


def test_livecodebench_dspy_persistent_retrieval_prefers_canonical_repair_cards(
    tmp_path: Path,
) -> None:
    persistent_root = tmp_path / "persistent-memory"
    problems = [
        LiveCodeBenchProblem(
            problem_id="lcb-canonical-rank-a",
            scenario="self_repair",
            prompt="Implement add(a, b) and return the sum.",
            prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
            evaluator_payload={
                "public_tests": [
                    {"input": "[2, 3]", "output": "5", "testtype": "functional"},
                ],
                "problem_metadata": {"func_name": "add"},
            },
        ),
        LiveCodeBenchProblem(
            problem_id="lcb-canonical-rank-b",
            scenario="self_repair",
            prompt="Implement add(a, b) again.",
            prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
            evaluator_payload={
                "public_tests": [
                    {"input": "[6, 1]", "output": "7", "testtype": "functional"},
                ],
                "problem_metadata": {"func_name": "add"},
            },
        ),
    ]
    visible_feedback = ("Functional public test mismatch: expected=5 actual=4",)

    session = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        for problem in problems:
            livecodebench_dspy_pilot.write_persistent_success_memory(
                session,
                problem=problem,
                attempt_index=2,
                response_text="```python\ndef add(a, b):\n    return a + b\n```",
                extracted_code="def add(a, b):\n    return a + b\n",
                evaluation={"available": True, "passed": True, "failure_feedback": []},
                visible_feedback=visible_feedback,
            )
        payload = retrieve_verified_memory(
            session,
            query=livecodebench_dspy_pilot.build_retrieval_query(
                problems[-1],
                visible_feedback,
                store_kind="persistent",
            ),
            attempt_index=2,
            allowed_roles=livecodebench_dspy_pilot.REPAIR_PERSISTENT_MEMORY_ROLES,
            limit=4,
            expand_top_k=1,
        )
    finally:
        session.close()

    assert payload["used"] is True
    assert payload["cards"][0]["role"] == "canonical_repair_lesson"
    assert any(card["role"] == "repair_lesson" for card in payload["cards"][1:])


def test_livecodebench_dspy_attempt_one_success_does_not_create_repair_lesson(
    tmp_path: Path,
) -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-no-repair-lesson",
        scenario="code_generation",
        prompt="Implement add(a, b) and return the sum.",
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )
    persistent_root = tmp_path / "persistent-memory"
    session = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="code_generation",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        livecodebench_dspy_pilot.write_persistent_success_memory(
            session,
            problem=problem,
            attempt_index=1,
            response_text="```python\ndef add(a, b):\n    return a + b\n```",
            extracted_code="def add(a, b):\n    return a + b\n",
            evaluation={"available": True, "passed": True, "failure_feedback": []},
            visible_feedback=(),
        )
    finally:
        session.close()

    reopened = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=persistent_root,
        scenario="code_generation",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        payload = retrieve_verified_memory(
            reopened,
            query="add expected 5 actual 4",
            attempt_index=2,
            allowed_roles=frozenset({"repair_lesson"}),
            limit=2,
        )
    finally:
        reopened.close()

    assert payload["used"] is False


def test_livecodebench_dspy_retrieval_plan_skips_persistent_attempt_one() -> None:
    assert livecodebench_dspy_pilot.retrieval_plan(
        attempt_index=1,
        store_kind="persistent",
    ) is None
    plan = livecodebench_dspy_pilot.retrieval_plan(
        attempt_index=2,
        store_kind="persistent",
    )
    assert plan is not None
    assert plan.allowed_roles == livecodebench_dspy_pilot.REPAIR_PERSISTENT_MEMORY_ROLES
    assert "repair_lesson" in plan.allowed_roles
    compact_plan = livecodebench_dspy_pilot.retrieval_plan(
        attempt_index=3,
        store_kind="persistent",
    )
    assert compact_plan is not None
    assert compact_plan.allowed_roles == livecodebench_dspy_pilot.CHEAP_REPAIR_PERSISTENT_MEMORY_ROLES
    assert compact_plan.limit == 1


def test_livecodebench_dspy_retrieval_query_includes_failure_signature_fields() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-query",
        scenario="self_repair",
        prompt="Implement add(a, b).",
        evaluator_payload={"problem_metadata": {"func_name": "add"}},
    )

    query = livecodebench_dspy_pilot.build_retrieval_query(
        problem,
        (
            "Functional public test mismatch: expected=5 actual=4",
            "NameError: name 'List' is not defined",
        ),
    )

    assert "function add" in query
    assert "NameError" in query
    assert "expected 5 actual 4" in query


def test_livecodebench_dspy_persistent_retrieval_query_avoids_problem_specific_tokens() -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-query-persistent",
        scenario="self_repair",
        prompt="Implement add(a, b) and return the sum of two integers.",
        prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )

    query = livecodebench_dspy_pilot.build_retrieval_query(
        problem,
        (
            "Functional public test mismatch: expected=5 actual=4",
            "NameError: name 'List' is not defined",
        ),
        store_kind="persistent",
    )

    assert "lcb-query-persistent" not in query
    assert "return the sum of two integers" not in query
    assert "self_repair" in query
    assert "top_level_function" in query
    assert "platform leetcode" in query
    assert "difficulty easy" in query
    assert "repair_kind runtime_nameerror" in query
    assert "integers" in query


def test_livecodebench_dspy_self_repair_uses_short_third_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    class FakeSource:
        def evaluate(self, problem, *, response_text, extracted_code):
            del problem, extracted_code
            if "return a + b" in response_text:
                return {"available": True, "passed": True, "failure_feedback": []}
            return {
                "available": True,
                "passed": False,
                "failure_feedback": ["Functional public test mismatch: expected=5 actual=4"],
            }

    def fake_run_dspy_attempt(**kwargs):
        prompts.append(kwargs["prompt"])
        if len(prompts) < 3:
            return {
                "response_text": "```python\ndef add(a, b):\n    return a - b\n```",
                "tool_calls": 0,
                "trajectory": {},
                "usage": None,
                "response_error": None,
            }
        return {
            "response_text": "```python\ndef add(a, b):\n    return a + b\n```",
            "tool_calls": 0,
            "trajectory": {},
            "usage": None,
            "response_error": None,
        }

    monkeypatch.setattr(livecodebench_dspy_pilot, "run_dspy_attempt", fake_run_dspy_attempt)
    problem = LiveCodeBenchProblem(
        problem_id="lcb-third-attempt",
        scenario="self_repair",
        prompt="Implement add(a, b).",
        evaluator_payload={
            "public_tests": [{"input": "[2, 3]", "output": "5", "testtype": "functional"}],
            "problem_metadata": {"func_name": "add"},
        },
    )

    row = livecodebench_dspy_pilot.execute_problem(
        problem,
        method="dspy_baseline",
        config=livecodebench_dspy_pilot.PilotRunConfig(
            methods=("dspy_baseline",),
            requested_scenario="self_repair",
            resolved_scenario="self_repair",
            model="qwen/qwen3-coder-next",
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
            temperature=0.0,
            max_tokens=4096,
            candidates_per_attempt=1,
            problem_offset=0,
            max_problems=1,
            execute=True,
            output_root=Path(".benchmarks/livecodebench-dspy"),
            persistent_memory_root=Path(".benchmarks/livecodebench-dspy/_persistent_vtm_memory"),
            benchmark_root=Path("benchmarks/LiveCodeBench"),
            problem_file=None,
            run_id="third_attempt_test",
        ),
        source=FakeSource(),
        client=livecodebench_dspy_pilot.OpenAICompatibleChatClient(
            livecodebench_dspy_pilot.OpenAICompatibleChatConfig(
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
            )
        ),
        model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
            base_url_value="https://openrouter.example/api/v1",
            api_key_value="openrouter-test-key",
            execution_model_name="qwen/qwen3-coder-next",
            dspy_model_name="qwen/qwen3-coder-next",
        ),
    )

    assert row["attempt_count"] == 3
    assert row["cheap_repair_used"] is True
    assert len(prompts) == 3
    assert "Problem Statement:" not in prompts[-1]
    assert "final short repair attempt 3" in prompts[-1]


def test_livecodebench_dspy_best_of_k_selects_better_public_test_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    completions = iter(
        [
            {
                "response_text": "```python\ndef add(a, b):\n    return a - b\n```",
                "tool_calls": 0,
                "trajectory": {},
                "usage": {"completion_tokens": 5, "prompt_tokens": 10},
                "response_error": None,
            },
            {
                "response_text": "```python\ndef add(a, b):\n    return a + b\n```",
                "tool_calls": 0,
                "trajectory": {},
                "usage": {"completion_tokens": 7, "prompt_tokens": 10},
                "response_error": None,
            },
        ]
    )

    class FakeSource:
        def evaluate(self, problem, *, response_text, extracted_code):
            del problem, extracted_code
            if "return a + b" in response_text:
                return {
                    "available": True,
                    "passed": True,
                    "pass_rate": 1.0,
                    "passed_test_count": 1,
                    "public_test_count": 1,
                    "failure_feedback": [],
                    "syntax_error": False,
                }
            return {
                "available": True,
                "passed": False,
                "pass_rate": 0.0,
                "passed_test_count": 0,
                "public_test_count": 1,
                "failure_feedback": ["Functional public test mismatch: expected=5 actual=4"],
                "syntax_error": False,
            }

    def fake_run_dspy_attempt(**kwargs):
        prompts.append(kwargs["prompt"])
        return next(completions)

    monkeypatch.setattr(livecodebench_dspy_pilot, "run_dspy_attempt", fake_run_dspy_attempt)
    problem = LiveCodeBenchProblem(
        problem_id="lcb-best-of-k",
        scenario="code_generation",
        prompt="Implement add(a, b).",
        evaluator_payload={
            "public_tests": [{"input": "[2, 3]", "output": "5", "testtype": "functional"}],
            "problem_metadata": {"func_name": "add"},
        },
    )

    row = livecodebench_dspy_pilot.execute_problem(
        problem,
        method="dspy_baseline",
        config=livecodebench_dspy_pilot.PilotRunConfig(
            methods=("dspy_baseline",),
            requested_scenario="code_generation",
            resolved_scenario="code_generation",
            model="qwen/qwen3-coder-next",
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
            temperature=0.7,
            max_tokens=4096,
            candidates_per_attempt=2,
            problem_offset=0,
            max_problems=1,
            execute=True,
            output_root=Path(".benchmarks/livecodebench-dspy"),
            persistent_memory_root=Path(".benchmarks/livecodebench-dspy/_persistent_vtm_memory"),
            benchmark_root=Path("benchmarks/LiveCodeBench"),
            problem_file=None,
            run_id="best_of_k_test",
        ),
        source=FakeSource(),
        client=livecodebench_dspy_pilot.OpenAICompatibleChatClient(
            livecodebench_dspy_pilot.OpenAICompatibleChatConfig(
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
            )
        ),
        model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
            base_url_value="https://openrouter.example/api/v1",
            api_key_value="openrouter-test-key",
            execution_model_name="qwen/qwen3-coder-next",
            dspy_model_name="qwen/qwen3-coder-next",
        ),
    )

    assert row["evaluation"]["passed"] is True
    assert "return a + b" in row["response"]
    assert row["candidate_selection_mode"] == "best_of_k_public_tests"
    assert row["selected_candidate_indices"] == [2]
    assert row["candidate_batches"][0]["selected_candidate_index"] == 2
    assert row["usage"]["completion_tokens"] == 12
    assert row["usage"]["prompt_tokens"] == 20
    assert len(prompts) == 2


@pytest.mark.parametrize(
    ("method", "expected_retrieval_kinds", "expected_agent_session_kind"),
    [
        ("dspy_vtm_local_only", ["local", "local"], "local"),
        ("dspy_vtm_persistent_only", ["persistent"], "persistent"),
        ("dspy_vtm", ["local", "local", "persistent"], "local"),
    ],
)
def test_livecodebench_dspy_memory_ablation_routes_correct_memory_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_retrieval_kinds: list[str],
    expected_agent_session_kind: str,
) -> None:
    retrieval_kinds: list[str] = []
    agent_session_kinds: list[str] = []
    prompts: list[str] = []

    problem = LiveCodeBenchProblem(
        problem_id="lcb-ablation-routing",
        scenario="self_repair",
        prompt="Implement add(a, b).",
        prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )

    class FakeSource:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, problem, *, response_text, extracted_code):
            del problem, extracted_code
            self.calls += 1
            if self.calls == 1:
                return {
                    "available": True,
                    "passed": False,
                    "pass_rate": 0.0,
                    "passed_test_count": 0,
                    "public_test_count": 1,
                    "failure_feedback": ["Functional public test mismatch: expected=5 actual=4"],
                    "syntax_error": False,
                }
            return {
                "available": True,
                "passed": True,
                "pass_rate": 1.0,
                "passed_test_count": 1,
                "public_test_count": 1,
                "failure_feedback": [],
                "syntax_error": False,
            }

    def fake_retrieve_verified_memory(session, **kwargs):
        del kwargs
        scope_id = str(session.scope.scope_id)
        retrieval_kinds.append("persistent" if "persistent" in scope_id else "local")
        return {
            "used": True,
            "query": "fake-query",
            "cards": [],
            "verified_count": 1,
            "stale_filtered_count": 0,
            "tool_calls": 1,
        }

    def fake_run_dspy_attempt(**kwargs):
        prompts.append(kwargs["prompt"])
        session = kwargs["session"]
        assert session is not None
        scope_id = str(session.scope.scope_id)
        agent_session_kinds.append("persistent" if "persistent" in scope_id else "local")
        return {
            "response_text": "```python\ndef add(a, b):\n    return a + b\n```",
            "tool_calls": 0,
            "trajectory": {},
            "usage": None,
            "response_error": None,
        }

    monkeypatch.setattr(
        livecodebench_dspy_pilot,
        "retrieve_verified_memory",
        fake_retrieve_verified_memory,
    )
    monkeypatch.setattr(livecodebench_dspy_pilot, "run_dspy_attempt", fake_run_dspy_attempt)

    persistent_session = (
        livecodebench_dspy_pilot.open_persistent_memory_session(
            state_root=tmp_path / "persistent-memory",
            scenario="self_repair",
            model="qwen/qwen3-coder-next",
            workspace_root=tmp_path,
        )
        if method in {"dspy_vtm_persistent_only", "dspy_vtm"}
        else None
    )
    try:
        row = livecodebench_dspy_pilot.execute_problem(
            problem,
            method=method,  # type: ignore[arg-type]
            config=livecodebench_dspy_pilot.PilotRunConfig(
                methods=(method,),  # type: ignore[arg-type]
                requested_scenario="self_repair",
                resolved_scenario="self_repair",
                model="qwen/qwen3-coder-next",
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
                temperature=0.0,
                max_tokens=4096,
                candidates_per_attempt=1,
                problem_offset=0,
                max_problems=1,
                execute=True,
                output_root=tmp_path / ".benchmarks" / "livecodebench-dspy",
                persistent_memory_root=tmp_path / ".benchmarks" / "livecodebench-dspy" / "_persistent_vtm_memory",
                benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
                problem_file=None,
                run_id="ablation_routing",
            ),
            source=FakeSource(),
            client=livecodebench_dspy_pilot.OpenAICompatibleChatClient(
                livecodebench_dspy_pilot.OpenAICompatibleChatConfig(
                    base_url="https://openrouter.example/api/v1",
                    api_key="openrouter-test-key",
                )
            ),
            model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
                base_url_value="https://openrouter.example/api/v1",
                api_key_value="openrouter-test-key",
                execution_model_name="qwen/qwen3-coder-next",
                dspy_model_name="qwen/qwen3-coder-next",
            ),
            persistent_session=persistent_session,
        )
    finally:
        if persistent_session is not None:
            persistent_session.close()

    assert row["attempt_count"] == 2
    assert retrieval_kinds == expected_retrieval_kinds
    assert all(kind == expected_agent_session_kind for kind in agent_session_kinds)
    assert len(prompts) == 2


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


def test_livecodebench_dspy_summary_reports_attempt2_repair_metrics() -> None:
    summary = aggregate_summary(
        [
            {
                "problem_id": "lcb-1",
                "method": "dspy_vtm",
                "evaluation": {"available": True, "passed": True, "syntax_error": False},
                "candidate_batches": [
                    {
                        "attempt_index": 1,
                        "candidates": [{"candidate_index": 1, "passed": False}],
                    },
                    {
                        "attempt_index": 2,
                        "candidates": [{"candidate_index": 1, "passed": True}],
                    },
                ],
            }
        ],
        method="dspy_vtm",
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
    )

    assert summary["attempt1_public_test_pass_at_1"] == 0.0
    assert summary["attempt2_public_test_pass_at_1"] == 1.0
    assert summary["attempt2_delta_over_attempt1"] == 1.0


def test_livecodebench_dspy_vtm_attempt_exposes_dynamic_memory_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, prompt: str) -> dict[str, object]:  # noqa: ANN001
        captured["prompt"] = prompt
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        return {
            "response": {"response": "```python\nprint('ok')\n```"},
            "memory_write_proposals": [
                {
                    "proposal_kind": "memory_lesson",
                    "memory_role": "repair_lesson",
                    "title": "Guard public mismatch repairs",
                    "summary": "Use the public mismatch to isolate the faulty branch before rewriting the whole function.",
                }
            ],
            "trajectory": {"execution_mode": "predict"},
        }

    monkeypatch.setattr(
        livecodebench_dspy_pilot.VTMReActCodingAgent,
        "run",
        fake_run,
    )
    session = SimpleNamespace(
        kernel=object(),
        scope=object(),
        dependency=object(),
        metadata_store=SimpleNamespace(get_memory_item=lambda _memory_id: None),
    )

    payload = livecodebench_dspy_pilot.run_dspy_attempt(
        prompt="Solve it",
        method="dspy_vtm",
        session=session,
        model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
            base_url_value="https://openrouter.example/api/v1",
            api_key_value="openrouter-test-key",
            execution_model_name="qwen/qwen3-coder-next",
            dspy_model_name="qwen/qwen3-coder-next",
        ),
        attempt_index=2,
    )

    assert payload["response_text"] == "```python\nprint('ok')\n```"
    assert captured["prompt"] == "Solve it"
    assert captured["memory_tools_enabled"] is True
    assert "search_verified_memory" in captured["tool_names"]
    assert "search_naive_memory" in captured["tool_names"]
    assert "expand_memory_evidence" in captured["tool_names"]
    assert "verify_memory" in captured["tool_names"]
    assert "propose_memory_lesson" in captured["tool_names"]
    assert "propose_failure_pattern" in captured["tool_names"]
    assert "propose_solution_pattern" in captured["tool_names"]
    assert payload["memory_write_proposals"][0]["title"] == "Guard public mismatch repairs"


def test_livecodebench_dspy_vtm_attempt_one_keeps_write_tools_but_hides_lookup_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, prompt: str) -> dict[str, object]:  # noqa: ANN001
        captured["prompt"] = prompt
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        return {
            "response": {"response": "```python\nprint('ok')\n```"},
            "memory_write_proposals": [
                {
                    "proposal_kind": "memory_lesson",
                    "memory_role": "repair_lesson",
                    "title": "Capture attempt-one state",
                    "summary": "Record the current implementation state before repair.",
                }
            ],
            "trajectory": {"execution_mode": "predict"},
        }

    monkeypatch.setattr(
        livecodebench_dspy_pilot.VTMReActCodingAgent,
        "run",
        fake_run,
    )
    session = SimpleNamespace(
        kernel=object(),
        scope=object(),
        dependency=object(),
        metadata_store=SimpleNamespace(get_memory_item=lambda _memory_id: None),
    )

    payload = livecodebench_dspy_pilot.run_dspy_attempt(
        prompt="Solve it",
        method="dspy_vtm",
        session=session,
        model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
            base_url_value="https://openrouter.example/api/v1",
            api_key_value="openrouter-test-key",
            execution_model_name="qwen/qwen3-coder-next",
            dspy_model_name="qwen/qwen3-coder-next",
        ),
        attempt_index=1,
    )

    assert payload["response_text"] == "```python\nprint('ok')\n```"
    assert captured["prompt"] == "Solve it"
    assert captured["memory_tools_enabled"] is True
    assert "search_verified_memory" not in captured["tool_names"]
    assert "search_naive_memory" not in captured["tool_names"]
    assert "expand_memory_evidence" not in captured["tool_names"]
    assert "verify_memory" not in captured["tool_names"]
    assert "propose_memory_lesson" in captured["tool_names"]
    assert "propose_failure_pattern" in captured["tool_names"]
    assert "propose_solution_pattern" in captured["tool_names"]
    assert payload["memory_write_proposals"][0]["title"] == "Capture attempt-one state"


def test_livecodebench_dspy_vtm_attempt_falls_back_when_dependency_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, prompt: str) -> dict[str, object]:  # noqa: ANN001
        captured["prompt"] = prompt
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        return {
            "response": {"response": "```python\nprint('ok')\n```"},
            "trajectory": {"execution_mode": "predict"},
        }

    monkeypatch.setattr(
        livecodebench_dspy_pilot.VTMReActCodingAgent,
        "run",
        fake_run,
    )
    session = SimpleNamespace(
        kernel=object(),
        scope=object(),
        dependency=None,
        metadata_store=SimpleNamespace(get_memory_item=lambda _memory_id: None),
    )

    payload = livecodebench_dspy_pilot.run_dspy_attempt(
        prompt="Solve it",
        method="dspy_vtm",
        session=session,
        model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
            base_url_value="https://openrouter.example/api/v1",
            api_key_value="openrouter-test-key",
            execution_model_name="qwen/qwen3-coder-next",
            dspy_model_name="qwen/qwen3-coder-next",
        ),
    )

    assert payload["response_text"] == "```python\nprint('ok')\n```"
    assert captured["prompt"] == "Solve it"
    assert captured["memory_tools_enabled"] is False
    assert "search_verified_memory" not in captured["tool_names"]
    assert "verify_memory" not in captured["tool_names"]


def test_livecodebench_dspy_agent_proposal_promotes_transferable_repair_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = LiveCodeBenchProblem(
        problem_id="lcb-agent-write-source",
        scenario="self_repair",
        prompt="Implement add(a, b) and return the sum.",
        prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
        evaluator_payload={
            "public_tests": [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )
    later_problem = LiveCodeBenchProblem(
        problem_id="lcb-agent-write-target",
        scenario="self_repair",
        prompt="Implement add(a, b) for a later benchmark task.",
        prompt_metadata={"platform": "leetcode", "difficulty": "easy"},
        evaluator_payload={
            "public_tests": [
                {"input": "[7, 4]", "output": "11", "testtype": "functional"},
            ],
            "problem_metadata": {"func_name": "add"},
        },
    )

    class FakeSource:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, problem, *, response_text, extracted_code):
            del problem, response_text, extracted_code
            self.calls += 1
            if self.calls == 1:
                return {
                    "available": True,
                    "passed": False,
                    "pass_rate": 0.0,
                    "passed_test_count": 0,
                    "public_test_count": 1,
                    "failure_feedback": ["Functional public test mismatch: expected=5 actual=4"],
                    "syntax_error": False,
                }
            return {
                "available": True,
                "passed": True,
                "pass_rate": 1.0,
                "passed_test_count": 1,
                "public_test_count": 1,
                "failure_feedback": [],
                "syntax_error": False,
            }

    attempts = {"count": 0}

    def fake_run_dspy_attempt(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {
                "response_text": "```python\ndef add(a, b):\n    return a - b\n```",
                "tool_calls": 1,
                "trajectory": {},
                "usage": None,
                "response_error": None,
                "memory_write_proposals": [],
            }
        return {
            "response_text": "```python\ndef add(a, b):\n    return a + b\n```",
            "tool_calls": 2,
            "trajectory": {},
            "usage": None,
            "response_error": None,
            "memory_write_proposals": [
                {
                    "proposal_kind": "failure_pattern",
                    "memory_role": "repair_lesson",
                    "title": "Avoid subtraction on sum tasks",
                    "summary": (
                        "When public tests for a top-level add function show a smaller actual "
                        "total than expected, check that the final branch returns the sum instead "
                        "of the difference."
                    ),
                    "rationale": "The repaired candidate passed after replacing subtraction with addition.",
                    "function_name": "add",
                    "repair_kind": "public_test_logic_mismatch",
                    "interface_mode": "top_level_function",
                    "failure_signature": "Functional public test mismatch: expected=5 actual=4",
                    "transfer_terms": ["logic_mismatch", "sum", "addition", "top_level_function"],
                }
            ],
        }

    monkeypatch.setattr(livecodebench_dspy_pilot, "run_dspy_attempt", fake_run_dspy_attempt)
    persistent_session = livecodebench_dspy_pilot.open_persistent_memory_session(
        state_root=tmp_path / "persistent-memory",
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        row = livecodebench_dspy_pilot.execute_problem(
            problem,
            method="dspy_vtm_persistent_only",
            config=livecodebench_dspy_pilot.PilotRunConfig(
                methods=("dspy_vtm_persistent_only",),
                requested_scenario="self_repair",
                resolved_scenario="self_repair",
                model="qwen/qwen3-coder-next",
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
                temperature=0.0,
                max_tokens=4096,
                candidates_per_attempt=1,
                problem_offset=0,
                max_problems=1,
                execute=True,
                output_root=tmp_path / ".benchmarks" / "livecodebench-dspy",
                persistent_memory_root=tmp_path / ".benchmarks" / "livecodebench-dspy" / "_persistent_vtm_memory",
                benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
                problem_file=None,
                run_id="agent_write_transfer",
            ),
            source=FakeSource(),
            client=livecodebench_dspy_pilot.OpenAICompatibleChatClient(
                livecodebench_dspy_pilot.OpenAICompatibleChatConfig(
                    base_url="https://openrouter.example/api/v1",
                    api_key="openrouter-test-key",
                )
            ),
            model_config=livecodebench_dspy_pilot.DSPyOpenRouterConfig.from_env(
                base_url_value="https://openrouter.example/api/v1",
                api_key_value="openrouter-test-key",
                execution_model_name="qwen/qwen3-coder-next",
                dspy_model_name="qwen/qwen3-coder-next",
            ),
            persistent_session=persistent_session,
        )

        retrieval = retrieve_verified_memory(
            persistent_session,
            query=livecodebench_dspy_pilot.build_retrieval_query(
                later_problem,
                ("Functional public test mismatch: expected=11 actual=10",),
                store_kind="persistent",
            ),
            attempt_index=2,
            allowed_roles=frozenset({"repair_lesson"}),
            limit=5,
            expand_top_k=1,
        )
    finally:
        persistent_session.close()

    assert row["evaluation"]["passed"] is True
    assert row["agent_memory_write_count"] == 1
    assert len(row["agent_memory_write_ids"]) == 1
    assert retrieval["used"] is True
    assert any(
        card["title"] == "Avoid subtraction on sum tasks"
        for card in retrieval["cards"]
    )


def test_livecodebench_dspy_attempt_propagates_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, prompt: str) -> dict[str, object]:
            return {
                "response": {
                    "response": "",
                    "error": "DSPy ReAct failed before final extraction.",
                },
                "trajectory": {
                    "execution_mode": "react",
                    "execution_error": "DSPy ReAct failed before final extraction.",
                },
            }

    monkeypatch.setattr(livecodebench_dspy_pilot, "VTMReActCodingAgent", FakeAgent)

    payload = livecodebench_dspy_pilot.run_dspy_attempt(
        prompt="Solve it",
        method="dspy_baseline",
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
    assert "final extraction" in payload["response_error"]


def test_livecodebench_functional_feedback_surfaces_exception_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traceback_text = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "/tmp/harness.py", line 12, in <module>',
            "    spec.loader.exec_module(module)",
            "  File \"/tmp/candidate.py\", line 1, in <module>",
            "    class Solution:",
            "NameError: name 'List' is not defined",
        ]
    )

    def fake_run(*args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["python3", "harness.py"],
            returncode=1,
            stdout=traceback_text,
            stderr="",
        )

    monkeypatch.setattr(livecodebench_sources.subprocess, "run", fake_run)

    result = livecodebench_sources._run_functional_public_test(
        "class Solution:\n    pass\n",
        {"input": "10", "output": "[]", "testtype": "functional"},
        func_name="findPrimePairs",
    )

    assert result.passed is False
    assert result.feedback == "NameError: name 'List' is not defined"


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


def test_livecodebench_dspy_default_persistent_memory_root_stays_under_output_root(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / ".benchmarks" / "livecodebench-dspy"

    root = livecodebench_dspy_pilot.default_persistent_memory_root(
        output_root,
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
    )

    assert root.is_relative_to(output_root)


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
        candidates_per_attempt=1,
        problem_offset=0,
        max_problems=3,
        execute=False,
        output_root=Path(".benchmarks/livecodebench-dspy"),
        persistent_memory_root=Path(".benchmarks/livecodebench-dspy/_persistent_vtm_memory"),
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
    assert payload["candidates_per_attempt"] == 1
    expected_root = livecodebench_dspy_pilot.default_persistent_memory_root(
        Path(payload["output_root"]),
        scenario="self_repair",
        model=payload["model"],
    )
    assert Path(payload["persistent_memory_root"]) == expected_root
    assert "previous candidate code" in payload["scenario_semantics"].lower()


def test_livecodebench_dspy_execute_runs_all_supported_methods(
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
            "dspy_vtm_local_only",
            "dspy_vtm_persistent_only",
            "dspy_vtm",
        ),
        requested_scenario="self_repair",
        resolved_scenario="self_repair",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
        temperature=0.0,
        max_tokens=4096,
        candidates_per_attempt=1,
        problem_offset=0,
        max_problems=1,
        execute=True,
        output_root=tmp_path / ".benchmarks" / "livecodebench-dspy",
        persistent_memory_root=tmp_path / ".benchmarks" / "livecodebench-dspy" / "_persistent_vtm_memory",
        benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
        problem_file=problem_file,
        run_id="execute_all_methods",
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

    monkeypatch.setattr(livecodebench_dspy_pilot, "execute_method", fake_execute_method)

    payload = livecodebench_dspy_pilot.run_pilot(config, source=source)

    assert called_methods == [
        "direct",
        "dspy_baseline",
        "dspy_vtm_local_only",
        "dspy_vtm_persistent_only",
        "dspy_vtm",
    ]
    assert [run["method"] for run in payload["runs"]] == called_methods


def test_livecodebench_dspy_execute_method_writes_incremental_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = [
        LiveCodeBenchProblem(problem_id="lcb-1", scenario="self_repair", prompt="Problem one."),
        LiveCodeBenchProblem(problem_id="lcb-2", scenario="self_repair", prompt="Problem two."),
    ]

    class FakeSource:
        def describe(self) -> dict[str, object]:
            return {"kind": "fake"}

    config = livecodebench_dspy_pilot.PilotRunConfig(
        methods=("direct",),
        requested_scenario="self_repair",
        resolved_scenario="self_repair",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
        temperature=0.0,
        max_tokens=4096,
        candidates_per_attempt=1,
        problem_offset=0,
        max_problems=2,
        execute=True,
        output_root=tmp_path / ".benchmarks" / "livecodebench-dspy",
        persistent_memory_root=tmp_path / ".benchmarks" / "livecodebench-dspy" / "_persistent_vtm_memory",
        benchmark_root=tmp_path / "benchmarks" / "LiveCodeBench",
        problem_file=None,
        run_id="incremental_outputs",
    )

    emitted_problem_ids: list[str] = []

    def fake_execute_problem(
        problem,
        *,
        method,
        config,
        source,
        client,
        model_config,
        persistent_session=None,
    ):
        del method, config, source, client, model_config, persistent_session
        emitted_problem_ids.append(problem.problem_id)
        return {
            "problem_id": problem.problem_id,
            "method": "direct",
            "evaluation": {"available": True, "passed": True, "syntax_error": False},
            "candidate_batches": [
                {
                    "attempt_index": 1,
                    "candidates": [{"candidate_index": 1, "passed": True}],
                }
            ],
            "tool_calls": 0,
        }

    monkeypatch.setattr(livecodebench_dspy_pilot, "execute_problem", fake_execute_problem)

    rows = livecodebench_dspy_pilot.execute_method(
        "direct",
        config=config,
        problems=problems,
        source=FakeSource(),
    )

    paths = livecodebench_dspy_pilot.method_run_paths(
        config.output_root,
        scenario=config.resolved_scenario,
        model=config.model,
        run_id=config.run_id,
        method="direct",
    )
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    problem_lines = paths.problems_jsonl.read_text(encoding="utf-8").strip().splitlines()

    assert [row["problem_id"] for row in rows] == ["lcb-1", "lcb-2"]
    assert emitted_problem_ids == ["lcb-1", "lcb-2"]
    assert len(problem_lines) == 2
    assert summary["status"] == "running"
    assert summary["completed_problem_count"] == 2
    assert summary["planned_problem_count"] == 2
    assert summary["remaining_problem_count"] == 0


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
