"""CLI entrypoint for executing maintained benchmark mode matrices."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from vtm.benchmarks.compare import compare_completed_runs
from vtm.benchmarks.models import (
    BenchmarkComparisonResult,
    BenchmarkManifest,
    BenchmarkMatrixResult,
    BenchmarkMode,
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkSuite,
)
from vtm.benchmarks.run import execute_benchmark_run

DEFAULT_MATRIX_MODES: tuple[BenchmarkMode, ...] = (
    "no_memory",
    "verified_lexical",
)


@dataclass(frozen=True)
class MatrixPreset:
    """Maintained matrix preset that pins a manifest and suite."""

    manifest_path: str
    suite: BenchmarkSuite
    description: str
    modes: tuple[BenchmarkMode, ...]
    baseline_mode: BenchmarkMode
    seed_on_base_query_on_head: bool = False


PRESETS: dict[str, MatrixPreset] = {
    "synthetic_retrieval": MatrixPreset(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="retrieval",
        description=(
            "Synthetic retrieval matrix for no_memory, naive_lexical, "
            "and verified_lexical."
        ),
        modes=("no_memory", "naive_lexical", "verified_lexical"),
        baseline_mode="no_memory",
    ),
    "synthetic_retrieval_drifted": MatrixPreset(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="retrieval",
        description=(
            "Synthetic retrieval matrix that seeds memories on base_ref and "
            "queries on head_ref for no_memory, naive_lexical, and verified_lexical."
        ),
        modes=("no_memory", "naive_lexical", "verified_lexical"),
        baseline_mode="no_memory",
        seed_on_base_query_on_head=True,
    ),
    "synthetic_drift": MatrixPreset(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="drift",
        description="Synthetic drift matrix for verified lexical verification only.",
        modes=("verified_lexical",),
        baseline_mode="verified_lexical",
    ),
    "synthetic_coding": MatrixPreset(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="coding",
        description="Synthetic coding matrix for no_memory, naive_lexical, and verified_lexical.",
        modes=("no_memory", "naive_lexical", "verified_lexical"),
        baseline_mode="no_memory",
    ),
    "controlled_coding_drift": MatrixPreset(
        manifest_path="benchmarks/manifests/controlled-coding-drift.json",
        suite="coding",
        description=(
            "Controlled coding-drift matrix for no_memory, naive_lexical, "
            "and verified_lexical."
        ),
        modes=("no_memory", "naive_lexical", "verified_lexical"),
        baseline_mode="no_memory",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark matrix CLI parser."""
    parser = argparse.ArgumentParser(description="Run a maintained VTM benchmark matrix.")
    parser.add_argument(
        "--preset",
        choices=tuple(sorted(PRESETS)),
        default="synthetic_retrieval",
        help="Maintained matrix preset to execute.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest override. Overrides the selected preset when provided.",
    )
    parser.add_argument(
        "--suite",
        choices=("retrieval", "drift", "coding"),
        default="",
        help="Optional suite override. Required when --manifest is used without a preset manifest.",
    )
    parser.add_argument("--output", required=True, help="Directory for matrix outputs.")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help=(
            "Maintained study mode to include in the matrix. Repeat to select multiple "
            "modes. Defaults to the selected maintained preset."
        ),
    )
    parser.add_argument(
        "--baseline-mode",
        choices=(
            "no_memory",
            "naive_lexical",
            "verified_lexical",
        ),
        default="",
        help="Mode used as the comparison baseline. Defaults to the preset baseline.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K memories to evaluate.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional case limit.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic run seed.")
    parser.add_argument(
        "--seed-on-base-query-on-head",
        action="store_true",
        help=(
            "Retrieval-only mode that seeds memory from base_ref symbols and "
            "queries after checking out head_ref."
        ),
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Optional repo name filter. Repeat to select multiple repos.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Optional commit-pair filter. Repeat to select multiple pairs.",
    )
    parser.add_argument(
        "--coding-engine",
        choices=("dspy_react",),
        default="dspy_react",
        help="Maintained coding execution engine.",
    )
    parser.add_argument(
        "--workspace-backend",
        choices=("local_workspace", "docker_workspace"),
        default="local_workspace",
        help=(
            "Workspace backend for coding execution. Maintained: local_workspace. "
            "docker_workspace is legacy/non-maintained."
        ),
    )
    parser.add_argument(
        "--docker-image",
        default="",
        help="Legacy docker_workspace image override. Non-maintained.",
    )
    parser.add_argument(
        "--docker-binary",
        default="docker",
        help="Legacy docker_workspace Docker CLI override. Non-maintained.",
    )
    parser.add_argument(
        "--docker-network",
        choices=("none", "bridge"),
        default="none",
        help="Legacy docker_workspace network mode. Non-maintained.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of attempts to execute per coding task.",
    )
    parser.add_argument(
        "--pass-k",
        action="append",
        default=[],
        help="Additional pass@k checkpoints to report for coding tasks. Repeat to add more.",
    )
    parser.add_argument(
        "--execution-model",
        dest="execution_model_id",
        default="",
        help=(
            "Execution model id for coding runs. Falls back to VTM_EXECUTION_MODEL "
            "or the maintained OpenRouter default."
        ),
    )
    parser.add_argument(
        "--agent-max-iterations",
        type=int,
        default=12,
        help="Maximum number of DSPy ReAct iterations.",
    )
    parser.add_argument(
        "--workspace-command-timeout-seconds",
        type=int,
        default=120,
        help="Per-command timeout for workspace operations.",
    )
    parser.add_argument(
        "--workspace-max-output-chars",
        type=int,
        default=20000,
        help="Maximum characters captured from a single workspace command.",
    )
    parser.add_argument(
        "--comparison-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples used for paired numeric confidence intervals.",
    )
    parser.add_argument(
        "--comparison-bootstrap-seed",
        type=int,
        default=0,
        help="Deterministic seed used for bootstrap comparison resampling.",
    )
    return parser


def main() -> int:
    """Execute the configured matrix and print the durable JSON result."""
    args = build_parser().parse_args()
    try:
        result = run_matrix_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(result.to_json())
    return 0


def run_matrix_from_args(args: argparse.Namespace) -> BenchmarkMatrixResult:
    """Execute one benchmark matrix from parsed CLI arguments."""
    manifest_path, suite, preset_name = _resolve_manifest_and_suite(args)
    manifest = BenchmarkManifest.from_path(manifest_path)
    baseline_mode = _resolve_baseline_mode(args.baseline_mode, preset_name=preset_name)
    seed_on_base_query_on_head = _resolve_seed_on_base_query_on_head(
        args.seed_on_base_query_on_head,
        preset_name=preset_name,
    )
    modes = _resolve_modes(
        args.mode,
        baseline_mode=baseline_mode,
        preset_name=preset_name,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    comparisons_root = output_dir / "comparisons"
    run_results: dict[str, BenchmarkRunResult] = {}
    comparison_results: dict[str, BenchmarkComparisonResult] = {}

    for mode in modes:
        run_output_dir = runs_root / mode
        config = BenchmarkRunConfig(
            manifest_path=manifest_path,
            suite=suite,
            mode=mode,
            output_dir=str(run_output_dir),
            top_k=args.top_k,
            max_cases=args.max_cases,
            seed=args.seed,
            repo_filters=tuple(args.repo),
            pair_filters=tuple(args.pair),
            seed_on_base_query_on_head=seed_on_base_query_on_head,
            coding_engine=args.coding_engine,
            workspace_backend=args.workspace_backend,
            docker_image=args.docker_image or None,
            docker_binary=args.docker_binary,
            docker_network=args.docker_network,
            attempt_count=args.attempts,
            pass_k_values=_resolve_pass_k_values(args.pass_k, attempts=args.attempts, suite=suite),
            execution_model_id=args.execution_model_id or None,
            agent_max_iterations=args.agent_max_iterations,
            workspace_command_timeout_seconds=args.workspace_command_timeout_seconds,
            workspace_max_output_chars=args.workspace_max_output_chars,
        )
        run_results[mode] = execute_benchmark_run(
            manifest,
            config,
            execution_model_name=args.execution_model_id or None,
        )

    baseline_dir = runs_root / baseline_mode
    for mode in modes:
        if mode == baseline_mode:
            continue
        comparison_output_dir = comparisons_root / f"{baseline_mode}-vs-{mode}"
        comparison_results[mode] = compare_completed_runs(
            baseline_location=baseline_dir,
            candidate_location=runs_root / mode,
            output_dir=comparison_output_dir,
            bootstrap_samples=args.comparison_bootstrap_samples,
            bootstrap_seed=args.comparison_bootstrap_seed,
        )

    matrix_id = hashlib.sha256(
        "|".join(
            [
                preset_name or "custom",
                manifest_path,
                suite,
                baseline_mode,
                *modes,
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    matrix_json_path = output_dir / "matrix.json"
    matrix_md_path = output_dir / "matrix.md"
    result = BenchmarkMatrixResult(
        matrix_id=matrix_id,
        preset_name=preset_name,
        manifest_path=manifest_path,
        suite=suite,
        output_dir=str(output_dir),
        baseline_mode=baseline_mode,
        modes=modes,
        run_results=run_results,
        comparison_results=comparison_results,
        artifacts={
            "matrix_json": str(matrix_json_path),
            "matrix_md": str(matrix_md_path),
            "runs_dir": str(runs_root),
            "comparisons_dir": str(comparisons_root),
        },
    )
    matrix_json_path.write_text(result.to_json(), encoding="utf-8")
    matrix_md_path.write_text(render_matrix_summary(result), encoding="utf-8")
    return result


def render_matrix_summary(result: BenchmarkMatrixResult) -> str:
    """Render a human-readable Markdown summary for a matrix run."""
    lines = [
        "# VTM Benchmark Matrix",
        "",
        f"- Matrix ID: `{result.matrix_id}`",
        f"- Preset: `{result.preset_name or 'custom'}`",
        f"- Manifest: `{result.manifest_path}`",
        f"- Suite: `{result.suite}`",
        f"- Baseline mode: `{result.baseline_mode}`",
        f"- Modes: `{', '.join(result.modes)}`",
        "",
        "## Runs",
    ]
    for mode in result.modes:
        run = result.run_results[mode]
        lines.append(
            "- "
            f"{mode}: `run_id={run.run_id} cases={run.case_count} "
            f"summary={run.artifacts.get('summary_json', '')}`"
        )
    lines.extend(["", "## Comparisons"])
    if not result.comparison_results:
        lines.append("- none")
    else:
        for comparison_mode, comparison in sorted(result.comparison_results.items()):
            lines.append(
                "- "
                f"{result.baseline_mode} vs {comparison_mode}: "
                f"`common_cases={comparison.common_case_count} "
                f"comparison={comparison.artifacts.get('comparison_json', '')}`"
            )
    return "\n".join(lines) + "\n"


def _resolve_manifest_and_suite(
    args: argparse.Namespace,
) -> tuple[str, BenchmarkSuite, str | None]:
    if args.manifest:
        if not args.suite:
            raise ValueError("--suite is required when --manifest is provided")
        return args.manifest, args.suite, None
    preset = PRESETS[args.preset]
    return preset.manifest_path, preset.suite, args.preset


def _resolve_baseline_mode(
    requested_baseline_mode: str,
    *,
    preset_name: str | None,
) -> BenchmarkMode:
    if requested_baseline_mode:
        return requested_baseline_mode  # type: ignore[return-value]
    if preset_name is None:
        return "no_memory"
    return PRESETS[preset_name].baseline_mode


def _resolve_seed_on_base_query_on_head(
    requested_seed_on_base_query_on_head: bool,
    *,
    preset_name: str | None,
) -> bool:
    if requested_seed_on_base_query_on_head:
        return True
    if preset_name is None:
        return False
    return PRESETS[preset_name].seed_on_base_query_on_head


def _resolve_modes(
    requested_modes: list[str],
    *,
    baseline_mode: str,
    preset_name: str | None,
) -> tuple[BenchmarkMode, ...]:
    if not requested_modes:
        if preset_name is None:
            modes = DEFAULT_MATRIX_MODES
        else:
            modes = PRESETS[preset_name].modes
    else:
        deduped: list[BenchmarkMode] = []
        seen: set[str] = set()
        for mode in requested_modes:
            if mode not in seen:
                deduped.append(mode)  # type: ignore[arg-type]
                seen.add(mode)
        modes = tuple(deduped)
    if baseline_mode not in modes:
        raise ValueError("--baseline-mode must be included in the selected --mode set")
    return modes


def _resolve_pass_k_values(
    raw_values: list[str],
    *,
    attempts: int,
    suite: BenchmarkSuite,
) -> tuple[int, ...]:
    if suite != "coding":
        return (1,)
    if not raw_values:
        return (1, attempts) if attempts > 1 else (1,)
    return tuple(int(value) for value in raw_values)


if __name__ == "__main__":
    raise SystemExit(main())
