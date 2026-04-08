"""CLI entrypoint for running benchmark suites."""

from __future__ import annotations

import argparse
import os
import shlex

from vtm.adapters import (
    DeterministicHashEmbeddingAdapter,
    EmbeddingAdapter,
    OpenAICompatibleAgentModelAdapter,
    OpenAIEmbeddingAdapter,
    OpenAIRLMAdapter,
)
from vtm.benchmarks.models import BenchmarkManifest, BenchmarkRunConfig
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
        "--coding-executor",
        choices=("external_command", "native_agent"),
        default="external_command",
        help="Coding-task execution path to use.",
    )
    parser.add_argument(
        "--executor-command",
        default="",
        help=(
            "Optional command template for coding tasks. Supports {task_file}, {workspace}, "
            "{attempt}, and {artifact_root}."
        ),
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
        "--agent-model",
        default="",
        help=(
            "Model id for native-agent coding runs. Falls back to VTM_AGENT_MODEL "
            "or VTM_LOCAL_LLM_MODEL."
        ),
    )
    parser.add_argument(
        "--agent-mode",
        choices=("interactive_guarded", "benchmark_autonomous"),
        default="benchmark_autonomous",
        help="Permission mode for native-agent coding runs.",
    )
    parser.add_argument(
        "--agent-max-turns",
        type=int,
        default=12,
        help="Maximum number of turns for the native agent.",
    )
    parser.add_argument(
        "--agent-max-tool-failures",
        type=int,
        default=8,
        help="Maximum number of failed tool calls before the native agent stops.",
    )
    parser.add_argument(
        "--agent-max-runtime-seconds",
        type=int,
        default=600,
        help="Maximum runtime budget for the native agent.",
    )
    parser.add_argument(
        "--agent-compaction-window",
        type=int,
        default=10,
        help="Conversation message window for deterministic native-agent compaction.",
    )
    parser.add_argument(
        "--agent-command-timeout-seconds",
        type=int,
        default=120,
        help="Per-command timeout for native-agent workspace operations.",
    )
    parser.add_argument(
        "--agent-max-output-chars",
        type=int,
        default=20000,
        help="Maximum characters captured from a single native-agent command.",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for native-agent turns.",
    )
    parser.add_argument(
        "--agent-seed-base",
        type=int,
        default=None,
        help="Optional base seed used to derive one sampling seed per attempt.",
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
        coding_executor=args.coding_executor,
        executor_command=tuple(shlex.split(args.executor_command)),
        attempt_count=args.attempts,
        pass_k_values=pass_k_values,
        agent_model_id=args.agent_model or None,
        agent_mode=args.agent_mode,
        agent_max_turns=args.agent_max_turns,
        agent_max_tool_failures=args.agent_max_tool_failures,
        agent_max_runtime_seconds=args.agent_max_runtime_seconds,
        agent_compaction_window=args.agent_compaction_window,
        agent_command_timeout_seconds=args.agent_command_timeout_seconds,
        agent_max_output_chars=args.agent_max_output_chars,
        agent_temperature=args.agent_temperature,
        agent_seed_base=args.agent_seed_base,
        swebench_dataset_name=args.swebench_dataset_name or None,
        swebench_harness_workers=args.swebench_harness_workers,
        swebench_harness_cache_level=args.swebench_cache_level,
        swebench_harness_run_id=args.swebench_run_id or None,
    )

    rlm_adapter = None
    embedding_adapter: EmbeddingAdapter | None = None
    agent_model_adapter = None
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
    if args.suite == "coding" and args.coding_executor == "native_agent":
        agent_model = (
            args.agent_model
            or os.getenv("VTM_AGENT_MODEL")
            or os.getenv("VTM_LOCAL_LLM_MODEL")
        )
        if not agent_model:
            raise SystemExit(
                "native_agent coding runs require --agent-model, VTM_AGENT_MODEL, or "
                "VTM_LOCAL_LLM_MODEL"
            )
        agent_model_adapter = OpenAICompatibleAgentModelAdapter(
            model=agent_model,
            base_url=(
                os.getenv("VTM_AGENT_BASE_URL")
                or os.getenv("VTM_LOCAL_LLM_BASE_URL")
                or ""
            ),
            api_key=(
                os.getenv("VTM_AGENT_API_KEY")
                or os.getenv("VTM_LOCAL_LLM_API_KEY")
                or "vtm-agent"
            ),
            temperature=config.agent_temperature,
        )
        config = config.model_copy(
            update={
                "agent_model_id": agent_model,
            }
        )

    result = BenchmarkRunner(
        manifest,
        config,
        rlm_adapter=rlm_adapter,
        embedding_adapter=embedding_adapter,
        agent_model_adapter=agent_model_adapter,
    ).run()
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
