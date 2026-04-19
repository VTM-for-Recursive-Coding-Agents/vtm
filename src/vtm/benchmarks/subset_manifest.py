"""CLI utility for slicing a benchmark manifest down to selected coding tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtm.benchmarks.models import BenchmarkManifest, RepoSpec


def build_parser() -> argparse.ArgumentParser:
    """Build the subset-manifest CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create a benchmark manifest subset containing selected coding tasks."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an existing generated manifest JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the subset manifest JSON file.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        required=True,
        default=[],
        help="Coding task case_id to keep. Repeat to select multiple cases.",
    )
    return parser


def create_subset_manifest(
    manifest: BenchmarkManifest,
    *,
    case_ids: tuple[str, ...],
) -> BenchmarkManifest:
    """Return a manifest containing only the selected coding tasks and commit pairs."""
    requested_case_ids = tuple(dict.fromkeys(case_ids))
    selected_tasks = tuple(
        task for task in manifest.coding_tasks if task.case_id in requested_case_ids
    )
    selected_case_ids = {task.case_id for task in selected_tasks}
    missing_case_ids = [
        case_id for case_id in requested_case_ids if case_id not in selected_case_ids
    ]
    if missing_case_ids:
        missing = ", ".join(missing_case_ids)
        raise ValueError(f"requested case_id values not found in manifest: {missing}")

    required_pair_ids_by_repo: dict[str, set[str]] = {}
    for task in selected_tasks:
        required_pair_ids_by_repo.setdefault(task.repo_name, set()).add(task.commit_pair_id)

    subset_repos: list[RepoSpec] = []
    for repo in manifest.repos:
        required_pair_ids = required_pair_ids_by_repo.get(repo.repo_name)
        if not required_pair_ids:
            continue
        subset_repos.append(
            RepoSpec(
                schema_version=repo.schema_version,
                repo_name=repo.repo_name,
                source_kind=repo.source_kind,
                remote_url=repo.remote_url,
                branch=repo.branch,
                commit_pairs=tuple(
                    pair for pair in repo.commit_pairs if pair.pair_id in required_pair_ids
                ),
            )
        )

    subset_count = len(selected_tasks)
    case_label = "case" if subset_count == 1 else "cases"
    description_prefix = manifest.description or f"Benchmark manifest {manifest.manifest_id}"
    return BenchmarkManifest(
        schema_version=manifest.schema_version,
        manifest_id=f"{manifest.manifest_id}_subset_{subset_count}",
        description=f"{description_prefix} [subset: {subset_count} {case_label}]",
        repos=tuple(subset_repos),
        coding_tasks=selected_tasks,
        seed=manifest.seed,
    )


def write_subset_manifest(
    *,
    input_path: str | Path,
    output_path: str | Path,
    case_ids: tuple[str, ...],
) -> BenchmarkManifest:
    """Load, filter, and write a subset manifest."""
    manifest = BenchmarkManifest.from_path(input_path)
    subset = create_subset_manifest(manifest, case_ids=case_ids)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(subset.to_json(), encoding="utf-8")
    return subset


def main() -> int:
    """CLI entrypoint for writing a subset benchmark manifest."""
    args = build_parser().parse_args()
    try:
        subset = write_subset_manifest(
            input_path=args.input,
            output_path=args.output,
            case_ids=tuple(args.case_id),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(subset.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
