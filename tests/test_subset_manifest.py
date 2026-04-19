from __future__ import annotations

import pytest

from vtm.benchmarks.models import BenchmarkManifest, CodingTaskCase, CommitPair, RepoSpec
from vtm.benchmarks.subset_manifest import create_subset_manifest


def _build_manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        manifest_id="swebench_lite_generated",
        description="Generated SWE-bench Lite coding benchmark manifest.",
        repos=(
            RepoSpec(
                repo_name="astropy__astropy",
                source_kind="git",
                remote_url="/tmp/astropy",
                branch="main",
                commit_pairs=(
                    CommitPair(
                        pair_id="astropy__astropy-14365",
                        base_ref="base-1",
                        head_ref="head-1",
                    ),
                    CommitPair(
                        pair_id="astropy__astropy-14995",
                        base_ref="base-2",
                        head_ref="head-2",
                    ),
                ),
            ),
            RepoSpec(
                repo_name="django__django",
                source_kind="git",
                remote_url="/tmp/django",
                branch="main",
                commit_pairs=(
                    CommitPair(
                        pair_id="django__django-11848",
                        base_ref="base-3",
                        head_ref="head-3",
                    ),
                ),
            ),
        ),
        coding_tasks=(
            CodingTaskCase(
                case_id="astropy__astropy-14365",
                repo_name="astropy__astropy",
                commit_pair_id="astropy__astropy-14365",
                evaluation_backend="swebench_harness",
                task_statement="Fix astropy case 14365.",
            ),
            CodingTaskCase(
                case_id="astropy__astropy-14995",
                repo_name="astropy__astropy",
                commit_pair_id="astropy__astropy-14995",
                evaluation_backend="swebench_harness",
                task_statement="Fix astropy case 14995.",
            ),
            CodingTaskCase(
                case_id="django__django-11848",
                repo_name="django__django",
                commit_pair_id="django__django-11848",
                evaluation_backend="swebench_harness",
                task_statement="Fix django case 11848.",
            ),
        ),
        seed=17,
    )


def test_subset_manifest_keeps_exactly_requested_tasks() -> None:
    subset = create_subset_manifest(
        _build_manifest(),
        case_ids=("astropy__astropy-14995",),
    )

    assert subset.manifest_id == "swebench_lite_generated_subset_1"
    assert (
        subset.description
        == "Generated SWE-bench Lite coding benchmark manifest. [subset: 1 case]"
    )
    assert subset.seed == 17
    assert tuple(task.case_id for task in subset.coding_tasks) == ("astropy__astropy-14995",)


def test_subset_manifest_keeps_matching_commit_pairs() -> None:
    subset = create_subset_manifest(
        _build_manifest(),
        case_ids=("astropy__astropy-14365", "astropy__astropy-14995"),
    )

    assert tuple(repo.repo_name for repo in subset.repos) == ("astropy__astropy",)
    assert tuple(pair.pair_id for pair in subset.repos[0].commit_pairs) == (
        "astropy__astropy-14365",
        "astropy__astropy-14995",
    )


def test_subset_manifest_missing_case_id_fails_clearly() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_subset_manifest(
            _build_manifest(),
            case_ids=("astropy__astropy-14365", "missing-case"),
        )

    assert str(exc_info.value) == "requested case_id values not found in manifest: missing-case"
