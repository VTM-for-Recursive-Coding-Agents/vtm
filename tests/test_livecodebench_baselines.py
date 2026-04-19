from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

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
        REPO_ROOT / "scripts" / "livecodebench" / "export_results.py",
        REPO_ROOT / "scripts" / "livecodebench" / "setup_livecodebench.sh",
        REPO_ROOT / "scripts" / "shared" / "run_livecodebench.sh",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench.sh",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench_baseline.sh",
        REPO_ROOT / "scripts" / "local" / "preflight_checks.sh",
        REPO_ROOT / "scripts" / "run_livecodebench_baseline.sh",
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
    assert module.normalized_summary_path(config).exists()


def test_livecodebench_docs_mark_baseline_only_scope() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    recipes = (REPO_ROOT / "docs" / "benchmark-recipes.md").read_text(encoding="utf-8")
    baselines = (REPO_ROOT / "docs" / "livecodebench-baselines.md").read_text(encoding="utf-8")

    assert "LiveCodeBench support is available for baseline model coding ability checks" in readme
    assert "main VTM evidence remains retrieval, drift, and drifted retrieval" in readme
    assert "LiveCodeBench is available here as an external baseline model benchmark only" in recipes
    assert "It is not the main VTM memory benchmark" in baselines
    assert "SWE-bench Lite remains demoted after empty-patch pilot failures" in baselines
    assert "VTM_OPENROUTER_BASE_URL" in baselines
    assert "OPENROUTER_API_KEY" in baselines
    assert "VTM_EXECUTION_MODEL" in baselines


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
        REPO_ROOT / "scripts" / "livecodebench" / "setup_livecodebench.sh",
        REPO_ROOT / "scripts" / "livecodebench" / "export_results.py",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench.sh",
        REPO_ROOT / "scripts" / "local" / "run_livecodebench_baseline.sh",
        REPO_ROOT / "scripts" / "shared" / "run_livecodebench.sh",
    ]

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert "run_swebench" not in text
        assert "scripts/local/run_swebench.sh" not in text
