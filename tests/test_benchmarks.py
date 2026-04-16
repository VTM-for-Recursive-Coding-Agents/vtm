from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from vtm.adapters.rlm import RLMRankRequest, RLMRankResponse
from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks.models import CommitPair, RepoSpec
from vtm.benchmarks.symbol_index import SymbolIndexer
from vtm.benchmarks.synthetic import SyntheticPythonSmokeCorpus
from vtm.harness.models import HarnessTaskPack


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_local_repo(repo: Path) -> tuple[str, str]:
    repo.mkdir(parents=True, exist_ok=True)
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.name", "VTM Tests")
    _run(repo, "git", "config", "user.email", "vtm@example.com")
    (repo / "sample.py").write_text(
        "def alpha():\n"
        '    """Alpha symbol."""\n'
        "    return 1\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "sample.py")
    _run(repo, "git", "commit", "-m", "base")
    base = _run(repo, "git", "rev-parse", "HEAD")
    (repo / "sample.py").write_text(
        "\n"
        "def alpha():\n"
        '    """Alpha symbol."""\n'
        "    return 1\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "sample.py")
    _run(repo, "git", "commit", "-m", "head")
    head = _run(repo, "git", "rev-parse", "HEAD")
    return base, head


def _build_duplicate_symbol_repo(repo: Path) -> tuple[str, str]:
    repo.mkdir(parents=True, exist_ok=True)
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.name", "VTM Tests")
    _run(repo, "git", "config", "user.email", "vtm@example.com")
    (repo / "alpha.py").write_text(
        "def cli():\n"
        '    """Return the alpha CLI value."""\n'
        "    return 'alpha'\n",
        encoding="utf-8",
    )
    (repo / "beta.py").write_text(
        "def cli():\n"
        '    """Return the beta CLI value."""\n'
        "    return 'beta'\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "alpha.py", "beta.py")
    _run(repo, "git", "commit", "-m", "base")
    base = _run(repo, "git", "rev-parse", "HEAD")
    (repo / "alpha.py").write_text(
        "\n"
        "def cli():\n"
        '    """Return the alpha CLI value."""\n'
        "    return 'alpha'\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "alpha.py")
    _run(repo, "git", "commit", "-m", "head")
    head = _run(repo, "git", "rev-parse", "HEAD")
    return base, head


class FakeBenchmarkRLMAdapter:
    def rank_candidates(self, request: RLMRankRequest) -> RLMRankResponse:
        reranked = tuple(
            candidate.model_copy(
                update={
                    "rlm_score": float(len(request.candidates) - index + 1),
                    "final_score": float(len(request.candidates) - index + 1),
                    "reason": f"benchmark-rerank-{candidate.candidate_id}",
                }
            )
            for index, candidate in enumerate(request.candidates, start=1)
        )
        return RLMRankResponse(candidates=reranked, model_name="fake-benchmark-model")


class FailingBenchmarkRLMAdapter:
    def rank_candidates(self, request: RLMRankRequest) -> RLMRankResponse:
        raise RuntimeError("benchmark rerank failure")


def test_manifest_lock_is_deterministic_for_identical_runs(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    config_one = BenchmarkRunConfig(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="retrieval",
        mode="lexical",
        output_dir=str(tmp_path / "run-one"),
        top_k=3,
        max_cases=3,
        seed=7,
    )
    config_two = BenchmarkRunConfig(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="retrieval",
        mode="lexical",
        output_dir=str(tmp_path / "run-two"),
        top_k=3,
        max_cases=3,
        seed=7,
    )

    first = BenchmarkRunner(manifest, config_one).run()
    second = BenchmarkRunner(manifest, config_two).run()

    assert first.run_id == second.run_id
    assert first.manifest_digest == second.manifest_digest


def test_synthetic_benchmark_retrieval_and_drift_runs(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    retrieval = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            mode="lexical",
            output_dir=str(tmp_path / "retrieval"),
            top_k=3,
            max_cases=4,
        ),
    ).run()
    drift = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="drift",
            mode="lexical",
            output_dir=str(tmp_path / "drift"),
            top_k=3,
            max_cases=4,
        ),
    ).run()

    assert retrieval.case_count == 4
    assert "recall_at_1" in retrieval.metrics
    assert set(retrieval.metrics["slice_metrics"]) == {"smoke_identity", "taskish_behavior"}
    assert Path(retrieval.artifacts["summary_json"]).exists()
    assert "status_confusion" in drift.metrics
    assert Path(drift.artifacts["cases_jsonl"]).exists()
    retrieval_cases = Path(retrieval.artifacts["cases_jsonl"]).read_text(encoding="utf-8")
    retrieval_results = Path(retrieval.artifacts["results_jsonl"]).read_text(encoding="utf-8")
    assert len(retrieval_cases.splitlines()) == 4
    assert len(retrieval_results.splitlines()) == 4


def test_synthetic_benchmark_embedding_run(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            mode="embedding",
            output_dir=str(tmp_path / "embedding"),
            top_k=3,
            max_cases=2,
        ),
    ).run()

    assert result.case_count == 2
    assert "slice_metrics" in result.metrics
    assert Path(result.artifacts["results_jsonl"]).exists()


def test_git_repo_checkout_supports_local_remote(tmp_path: Path) -> None:
    remote_repo = tmp_path / "remote"
    base, head = _build_local_repo(remote_repo)
    manifest_path = tmp_path / "local-manifest.json"
    manifest_path.write_text(
        BenchmarkManifest(
            manifest_id="local_git_manifest",
            repos=(
                RepoSpec(
                    repo_name="local_git_repo",
                    source_kind="git",
                    remote_url=str(remote_repo),
                    branch="main",
                    commit_pairs=(CommitPair(pair_id="local_pair", base_ref=base, head_ref=head),),
                ),
            ),
        ).to_json(),
        encoding="utf-8",
    )
    manifest = BenchmarkManifest.from_path(manifest_path)
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path=str(manifest_path),
            suite="retrieval",
            mode="no_memory",
            output_dir=str(tmp_path / "git-run"),
            top_k=1,
            max_cases=1,
        ),
    ).run()

    assert result.case_count == 1
    assert (tmp_path / "git-run" / "corpus" / "local_git_repo").exists()


def test_coding_suite_writes_task_pack_and_executes_rlm(
    tmp_path: Path,
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm()
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="lexical",
            output_dir=str(tmp_path / "coding"),
            top_k=3,
            max_cases=1,
            rlm_model_id="fake-model",
        ),
    ).run()

    assert result.case_count == 1
    assert result.metrics["executed_count"] == 1
    task_pack = HarnessTaskPack.from_json(
        (tmp_path / "coding" / "task-packs" / "synthetic_bugfix_unittest.json").read_text(
            encoding="utf-8"
        )
    )
    assert task_pack.base_ref == "smoke-bug"
    assert task_pack.head_ref == "smoke-bugfix"
    assert task_pack.expected_changed_paths == ("bugfix_module.py",)
    assert task_pack.memory_mode == "lexical"
    assert task_pack.top_k == 3
    assert task_pack.memory_context
    assert task_pack.memory_context[0].raw_anchor_path is not None


def test_coding_suite_executor_writes_benchmark_local_artifacts(
    tmp_path: Path,
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm()
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="lexical",
            output_dir=str(tmp_path / "coding-exec"),
            top_k=3,
            max_cases=1,
            pair_filters=("bugfix",),
            rlm_model_id="fake-model",
        ),
    ).run()

    assert result.case_count == 1
    assert result.metrics["executed_count"] == 1
    assert result.metrics["pass_rate"] == 1.0
    assert result.metrics["mean_changed_path_f1"] == 1.0

    results_path = Path(result.artifacts["results_jsonl"])
    first_result = json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_result["metrics"]["executed"] is True
    assert first_result["metrics"]["executor_succeeded"] is True
    assert first_result["metrics"]["produced_patch_nonempty"] is True
    assert first_result["metrics"]["changed_path_precision"] == 1.0
    assert first_result["metrics"]["changed_path_recall"] == 1.0
    assert first_result["metrics"]["changed_path_f1"] == 1.0
    assert first_result["metrics"]["final_verification_passed"] is True
    assert first_result["metrics"]["infra_failure"] is False
    assert first_result["metadata"]["test_exit_code"] is not None
    assert Path(first_result["metadata"]["executor_stdout_path"]).exists()
    assert first_result["metadata"]["executor_stderr_path"] is None
    assert Path(first_result["metadata"]["produced_patch_path"]).exists()
    assert Path(first_result["metadata"]["command_events_path"]).exists()
    assert Path(first_result["metadata"]["final_git_status_path"]).exists()
    assert Path(first_result["metadata"]["final_verification_stdout_path"]).exists()
    assert Path(first_result["metadata"]["final_verification_stderr_path"]).exists()
    assert (tmp_path / "coding-exec" / "executor-artifacts").exists()


def test_synthetic_coding_tasks_fail_on_base_and_pass_on_head(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    repo_root = tmp_path / "synthetic-coding-corpus"
    SyntheticPythonSmokeCorpus().materialize(repo_root)
    pair_map = {
        pair.pair_id: pair
        for repo in manifest.repos
        if repo.repo_name == "synthetic_python_smoke"
        for pair in repo.commit_pairs
    }

    assert len(manifest.coding_tasks) >= 5
    for task in manifest.coding_tasks:
        pair = pair_map[task.commit_pair_id]
        _run(repo_root, "git", "checkout", "--quiet", pair.base_ref)
        base_result = subprocess.run(
            list(task.test_command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        _run(repo_root, "git", "checkout", "--quiet", pair.head_ref)
        head_result = subprocess.run(
            list(task.test_command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

        assert base_result.returncode != 0, task.case_id
        assert head_result.returncode == 0, task.case_id


def test_coding_suite_reports_multiple_tasks_and_filtering(
    tmp_path: Path,
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm()
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="no_memory",
            output_dir=str(tmp_path / "coding-many"),
            pair_filters=("bugfix", "sentinel"),
            rlm_model_id="fake-model",
        ),
    ).run()

    cases = [
        json.loads(line)
        for line in Path(result.artifacts["cases_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]

    assert result.case_count == 2
    assert result.metrics["total_tasks"] == 2
    assert result.metrics["testable_task_count"] == 2
    assert {case["commit_pair_id"] for case in cases} == {"bugfix", "sentinel"}


def test_coding_summary_compares_modes(tmp_path: Path, install_fake_vendored_rlm) -> None:
    install_fake_vendored_rlm()
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    no_memory = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="no_memory",
            output_dir=str(tmp_path / "coding-no-memory"),
            pair_filters=("bugfix",),
            max_cases=1,
            rlm_model_id="fake-model",
        ),
    ).run()
    lexical = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="lexical",
            output_dir=str(tmp_path / "coding-lexical"),
            pair_filters=("bugfix",),
            max_cases=1,
            rlm_model_id="fake-model",
        ),
    ).run()
    embedding = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="embedding",
            output_dir=str(tmp_path / "coding-embedding"),
            pair_filters=("bugfix",),
            max_cases=1,
            rlm_model_id="fake-model",
        ),
    ).run()

    assert no_memory.case_count == 1
    assert lexical.case_count == 1
    assert embedding.case_count == 1
    assert no_memory.metrics["retrieval_usage_rate"] == 0.0
    assert lexical.metrics["retrieval_usage_rate"] == 1.0
    assert embedding.metrics["retrieval_usage_rate"] == 1.0
    assert lexical.metrics["testable_task_count"] == 1
    assert embedding.metrics["testable_task_count"] == 1


def test_coding_runner_rejects_unknown_pair_filters(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    try:
        BenchmarkRunner(
            manifest,
            BenchmarkRunConfig(
                manifest_path="benchmarks/manifests/synthetic-smoke.json",
                suite="coding",
                mode="lexical",
                output_dir=str(tmp_path / "coding-bad-pair"),
                pair_filters=("missing_pair",),
                rlm_model_id="fake-model",
            ),
        ).run()
    except ValueError as exc:
        assert "unknown benchmark pairs" in str(exc)
    else:
        raise AssertionError("expected unknown pair filter to fail for coding suite")


def test_coding_changed_path_metrics_detect_extra_paths(
    tmp_path: Path,
    install_fake_vendored_rlm,
) -> None:
    def apply_extra_change(task_pack, workspace_root: Path, artifact_root: Path) -> None:
        del task_pack, artifact_root
        (workspace_root / "bugfix_module.py").write_text(
            "def buggy_increment(value: int) -> int:\n"
            '    """Return value plus one."""\n'
            "    return value + 1\n",
            encoding="utf-8",
        )
        (workspace_root / "tests" / "test_bugfix_module.py").write_text(
            "import unittest\n\n"
            "from bugfix_module import buggy_increment\n\n\n"
            "class BugfixModuleTests(unittest.TestCase):\n"
            "    def test_buggy_increment(self) -> None:\n"
            "        self.assertEqual(buggy_increment(3), 4)\n\n\n"
            "# rewritten during benchmark test\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

    install_fake_vendored_rlm(apply_workspace_update=apply_extra_change)
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="lexical",
            output_dir=str(tmp_path / "coding-extra-path"),
            pair_filters=("bugfix",),
            max_cases=1,
            rlm_model_id="fake-model",
        ),
    ).run()

    result_rows = Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()
    row = json.loads(result_rows[0])
    assert row["metrics"]["changed_path_precision"] == 0.5
    assert row["metrics"]["changed_path_recall"] == 1.0
    assert math.isclose(row["metrics"]["changed_path_f1"], 2.0 / 3.0)


def test_coding_changed_path_metrics_detect_missing_patch(
    tmp_path: Path,
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm(
        apply_workspace_update=lambda task_pack, workspace_root, artifact_root: None
    )
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="lexical",
            output_dir=str(tmp_path / "coding-empty-patch"),
            pair_filters=("bugfix",),
            max_cases=1,
            rlm_model_id="fake-model",
        ),
    ).run()

    result_rows = Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()
    row = json.loads(result_rows[0])
    assert row["metrics"]["produced_patch_nonempty"] is False
    assert row["metrics"]["changed_path_precision"] == 0.0
    assert row["metrics"]["changed_path_recall"] == 0.0
    assert row["metrics"]["changed_path_f1"] == 0.0


def test_duplicate_symbol_case_ids_stay_unique_and_counts_align(tmp_path: Path) -> None:
    remote_repo = tmp_path / "duplicate-remote"
    base, head = _build_duplicate_symbol_repo(remote_repo)
    manifest = BenchmarkManifest(
        manifest_id="duplicate_symbol_manifest",
        repos=(
            RepoSpec(
                repo_name="duplicate_symbols",
                source_kind="git",
                remote_url=str(remote_repo),
                branch="main",
                commit_pairs=(CommitPair(pair_id="duplicate_pair", base_ref=base, head_ref=head),),
            ),
        ),
    )

    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="duplicate-symbol.json",
            suite="retrieval",
            mode="lexical",
            output_dir=str(tmp_path / "duplicate-run"),
            max_cases=3,
        ),
    ).run()

    cases = [
        json.loads(line)
        for line in Path(result.artifacts["cases_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    outputs = [
        json.loads(line)
        for line in Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]

    assert result.case_count == 3
    assert len(cases) == 3
    assert len(outputs) == 3
    assert len({case["case_id"] for case in cases}) == 3
    assert all(case["relative_path"] in {"alpha.py", "beta.py"} for case in cases)


def test_retrieval_case_generation_adds_taskish_and_smoke_slices(tmp_path: Path) -> None:
    repo_root = tmp_path / "synthetic-corpus"
    SyntheticPythonSmokeCorpus().materialize(repo_root)
    indexer = SymbolIndexer()
    stable_pair = CommitPair(pair_id="stable", base_ref="smoke-initial", head_ref="smoke-stable")

    cases = indexer.build_retrieval_cases(
        "synthetic_python_smoke",
        stable_pair,
        indexer.extract_symbols(repo_root),
    )
    taskish_cases = [case for case in cases if case.slice_name == "taskish_behavior"]
    smoke_cases = [case for case in cases if case.slice_name == "smoke_identity"]

    assert taskish_cases
    assert smoke_cases
    assert {case.memory_id for case in taskish_cases} == {case.memory_id for case in smoke_cases}
    assert all(case.symbol not in case.query for case in taskish_cases)


def test_repo_and_pair_filters_apply_before_case_limiting(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            mode="no_memory",
            output_dir=str(tmp_path / "filtered-run"),
            repo_filters=("synthetic_python_smoke",),
            pair_filters=("stable",),
            max_cases=3,
        ),
    ).run()

    cases = [
        json.loads(line)
        for line in Path(result.artifacts["cases_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]

    assert result.case_count == 3
    assert {case["repo_name"] for case in cases} == {"synthetic_python_smoke"}
    assert {case["commit_pair_id"] for case in cases} == {"stable"}


def test_runner_rejects_unknown_repo_or_pair_filters(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    try:
        BenchmarkRunner(
            manifest,
            BenchmarkRunConfig(
                manifest_path="benchmarks/manifests/synthetic-smoke.json",
                suite="retrieval",
                mode="no_memory",
                output_dir=str(tmp_path / "bad-filter"),
                repo_filters=("missing_repo",),
            ),
        ).run()
    except ValueError as exc:
        assert "unknown benchmark repos" in str(exc)
    else:
        raise AssertionError("expected unknown repo filter to fail")

    try:
        BenchmarkRunner(
            manifest,
            BenchmarkRunConfig(
                manifest_path="benchmarks/manifests/synthetic-smoke.json",
                suite="retrieval",
                mode="no_memory",
                output_dir=str(tmp_path / "bad-pair"),
                pair_filters=("missing_pair",),
            ),
        ).run()
    except ValueError as exc:
        assert "unknown benchmark pairs" in str(exc)
    else:
        raise AssertionError("expected unknown pair filter to fail")


def test_synthetic_reranking_benchmark_run_works_with_fake_adapter(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            mode="lexical_rlm_rerank",
            output_dir=str(tmp_path / "rerank"),
            max_cases=2,
        ),
        rlm_adapter=FakeBenchmarkRLMAdapter(),
    ).run()

    rows = [
        json.loads(line)
        for line in Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]

    assert result.case_count == 2
    assert rows[0]["metadata"]["slice_name"] in {"smoke_identity", "taskish_behavior"}
    assert rows[0]["metrics"]["rank"] is not None


def test_synthetic_reranking_benchmark_run_falls_back_on_adapter_failure(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            mode="lexical_rlm_rerank",
            output_dir=str(tmp_path / "rerank-failure"),
            max_cases=1,
        ),
        rlm_adapter=FailingBenchmarkRLMAdapter(),
    ).run()

    result_rows = Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()
    row = json.loads(result_rows[0])

    assert result.case_count == 1
    assert row["metrics"]["rank"] is not None
