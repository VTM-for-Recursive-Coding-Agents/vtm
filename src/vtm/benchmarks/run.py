"""CLI entrypoint for running benchmark suites."""

from __future__ import annotations

import argparse
import os

from vtm.adapters import (
    DeterministicHashEmbeddingAdapter,
    EmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    OpenAIRLMAdapter,
)
from vtm.benchmarks.models import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunResult
from vtm.benchmarks.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark runner CLI parser."""
    parser = argparse.ArgumentParser(description="Run VTM benchmark suites.")
    parser.add_argument("--manifest", required=True, help="Path to a benchmark manifest JSON file.")
    parser.add_argument(
        "--suite",
        required=True,
        choices=("retrieval", "drift", "coding"),
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--mode",
        choices=("no_memory", "lexical", "lexical_rlm_rerank", "embedding"),
        default="lexical",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument("--output", required=True, help="Directory for benchmark outputs.")
    parser.add_argument("--top-k", type=int, default=5, help="Top K memories to evaluate.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional case limit.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic run seed.")
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
        "--workspace-backend",
        choices=("local_workspace", "docker_workspace"),
        default="local_workspace",
        help="Workspace backend used for coding-task execution.",
    )
    parser.add_argument(
        "--docker-image",
        default="",
        help="Docker image for docker_workspace coding runs.",
    )
    parser.add_argument(
        "--docker-binary",
        default="docker",
        help="Docker CLI binary to use for docker_workspace runs.",
    )
    parser.add_argument(
        "--docker-network",
        choices=("none", "bridge"),
        default="none",
        help="Docker network mode for docker_workspace runs.",
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
        "--rlm-model-id",
        default="",
        help=(
            "Model id for vendored-RLM coding runs. Falls back to "
            "VTM_AGENT_MODEL or VTM_LOCAL_LLM_MODEL."
        ),
    )
    parser.add_argument(
        "--rlm-max-iterations",
        type=int,
        default=12,
        help="Maximum number of vendored-RLM iterations.",
    )
    parser.add_argument(
        "--rlm-max-runtime-seconds",
        type=int,
        default=600,
        help="Maximum runtime budget for vendored RLM.",
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
        "--rlm-model",
        default="",
        help="Model name for lexical_rlm_rerank mode. Falls back to VTM_OPENAI_MODEL.",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Optional model name for embedding mode. Falls back to VTM_OPENAI_EMBEDDING_MODEL.",
    )
    parser.add_argument(
        "--swebench-dataset-name",
        default="",
        help="Optional SWE-bench dataset identifier for harness-backed coding tasks.",
    )
    parser.add_argument(
        "--swebench-harness-workers",
        type=int,
        default=4,
        help="Worker count for official SWE-bench harness evaluation.",
    )
    parser.add_argument(
        "--swebench-cache-level",
        choices=("none", "base", "env", "instance"),
        default="env",
        help="Cache level for official SWE-bench harness evaluation.",
    )
    parser.add_argument(
        "--swebench-run-id",
        default="",
        help="Optional explicit run identifier for official SWE-bench harness evaluation.",
    )
    return parser


def main() -> int:
    """Parse CLI args, configure adapters, and execute a benchmark run."""
    args = build_parser().parse_args()
    pass_k_values: tuple[int, ...]
    if args.suite != "coding" and args.attempts > 1:
        raise SystemExit("--attempts > 1 is only supported for coding suites")
    if args.suite != "coding" and args.pass_k:
        raise SystemExit("--pass-k is only supported for coding suites")
    if args.workspace_backend == "docker_workspace" and args.suite != "coding":
        raise SystemExit("--workspace-backend docker_workspace is only supported for coding suites")
    if args.workspace_backend == "docker_workspace" and not args.docker_image:
        raise SystemExit("--docker-image is required when --workspace-backend docker_workspace")
    if args.workspace_backend == "local_workspace":
        if args.docker_image:
            raise SystemExit(
                "--docker-image is only supported with --workspace-backend "
                "docker_workspace"
            )
        if args.docker_binary != "docker":
            raise SystemExit(
                "--docker-binary is only supported with --workspace-backend "
                "docker_workspace"
            )
        if args.docker_network != "none":
            raise SystemExit(
                "--docker-network is only supported with --workspace-backend "
                "docker_workspace"
            )
    if not args.pass_k:
        pass_k_values = (1, args.attempts) if args.attempts > 1 else (1,)
    else:
        pass_k_values = tuple(int(value) for value in args.pass_k)
    manifest = BenchmarkManifest.from_path(args.manifest)
    config = BenchmarkRunConfig(
        manifest_path=args.manifest,
        suite=args.suite,
        mode=args.mode,
        output_dir=args.output,
        top_k=args.top_k,
        max_cases=args.max_cases,
        seed=args.seed,
        repo_filters=tuple(args.repo),
        pair_filters=tuple(args.pair),
        workspace_backend=args.workspace_backend,
        docker_image=args.docker_image or None,
        docker_binary=args.docker_binary,
        docker_network=args.docker_network,
        attempt_count=args.attempts,
        pass_k_values=pass_k_values,
        rlm_model_id=args.rlm_model_id or None,
        rlm_max_iterations=args.rlm_max_iterations,
        rlm_max_runtime_seconds=args.rlm_max_runtime_seconds,
        workspace_command_timeout_seconds=args.workspace_command_timeout_seconds,
        workspace_max_output_chars=args.workspace_max_output_chars,
        swebench_dataset_name=args.swebench_dataset_name or None,
        swebench_harness_workers=args.swebench_harness_workers,
        swebench_harness_cache_level=args.swebench_cache_level,
        swebench_harness_run_id=args.swebench_run_id or None,
    )
    try:
        result = execute_benchmark_run(
            manifest,
            config,
            rlm_model_name=args.rlm_model or None,
            embedding_model_name=args.embedding_model or None,
            execution_model_name=args.rlm_model_id or None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(result.to_json())
    return 0


def execute_benchmark_run(
    manifest: BenchmarkManifest,
    config: BenchmarkRunConfig,
    *,
    rlm_model_name: str | None = None,
    embedding_model_name: str | None = None,
    execution_model_name: str | None = None,
) -> BenchmarkRunResult:
    """Execute one benchmark run with environment-aware optional adapters."""
    rlm_adapter = None
    embedding_adapter: EmbeddingAdapter | None = None
    resolved_config = config

    if config.mode == "lexical_rlm_rerank":
        model_name = rlm_model_name or os.getenv("VTM_OPENAI_MODEL")
        if not model_name:
            raise ValueError("lexical_rlm_rerank mode requires --rlm-model or VTM_OPENAI_MODEL")
        rlm_adapter = OpenAIRLMAdapter(model=model_name)
    if config.mode == "embedding":
        resolved_embedding_model = embedding_model_name or os.getenv(
            "VTM_OPENAI_EMBEDDING_MODEL"
        )
        if resolved_embedding_model:
            embedding_adapter = OpenAIEmbeddingAdapter(model=resolved_embedding_model)
        else:
            embedding_adapter = DeterministicHashEmbeddingAdapter()
    if config.suite == "coding":
        resolved_rlm_model = (
            execution_model_name
            or config.rlm_model_id
            or os.getenv("VTM_AGENT_MODEL")
            or os.getenv("VTM_LOCAL_LLM_MODEL")
        )
        if not resolved_rlm_model:
            raise ValueError(
                "coding runs require --rlm-model-id, "
                "VTM_AGENT_MODEL, or "
                "VTM_LOCAL_LLM_MODEL"
            )
        resolved_config = resolved_config.model_copy(
            update={"rlm_model_id": resolved_rlm_model}
        )

    return BenchmarkRunner(
        manifest,
        resolved_config,
        rlm_adapter=rlm_adapter,
        embedding_adapter=embedding_adapter,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
