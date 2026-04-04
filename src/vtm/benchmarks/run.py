from __future__ import annotations

import argparse
import os
import shlex

from vtm.adapters import (
    DeterministicHashEmbeddingAdapter,
    EmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    OpenAIRLMAdapter,
)
from vtm.benchmarks.models import BenchmarkManifest, BenchmarkRunConfig
from vtm.benchmarks.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
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
        "--executor-command",
        default="",
        help="Optional command template for coding tasks. Supports {task_file} and {workspace}.",
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
    args = build_parser().parse_args()
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
        executor_command=tuple(shlex.split(args.executor_command)),
        swebench_dataset_name=args.swebench_dataset_name or None,
        swebench_harness_workers=args.swebench_harness_workers,
        swebench_harness_cache_level=args.swebench_cache_level,
        swebench_harness_run_id=args.swebench_run_id or None,
    )

    rlm_adapter = None
    embedding_adapter: EmbeddingAdapter | None = None
    if args.mode == "lexical_rlm_rerank":
        model_name = args.rlm_model or os.getenv("VTM_OPENAI_MODEL")
        if not model_name:
            raise SystemExit(
                "lexical_rlm_rerank mode requires --rlm-model or VTM_OPENAI_MODEL"
            )
        rlm_adapter = OpenAIRLMAdapter(model=model_name)
    if args.mode == "embedding":
        embedding_model = args.embedding_model or os.getenv("VTM_OPENAI_EMBEDDING_MODEL")
        if embedding_model:
            embedding_adapter = OpenAIEmbeddingAdapter(model=embedding_model)
        else:
            embedding_adapter = DeterministicHashEmbeddingAdapter()

    result = BenchmarkRunner(
        manifest,
        config,
        rlm_adapter=rlm_adapter,
        embedding_adapter=embedding_adapter,
    ).run()
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
