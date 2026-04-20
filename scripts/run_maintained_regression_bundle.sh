#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-$PROJECT_ROOT/.benchmarks}"
PAPER_ROOT="${PAPER_ROOT:-$PROJECT_ROOT/.benchmarks/paper-tables}"
EXECUTION_MODEL="${VTM_EXECUTION_MODEL:-}"
RUN_REPORT="${RUN_REPORT:-true}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_maintained_regression_bundle.sh [--benchmark-root DIR] [--paper-root DIR] [--execution-model MODEL] [--skip-report]

Runs the maintained VTM regression bundle:
  1. synthetic retrieval matrix
  2. synthetic drifted retrieval matrix
  3. synthetic drift matrix
  4. controlled coding-drift matrix

By default it also exports suite-specific paper tables plus a combined paper-table bundle.

Environment:
  VTM_EXECUTION_MODEL   Execution model for the maintained coding benchmark.
  BENCHMARK_ROOT        Optional override for benchmark output directories.
  PAPER_ROOT            Optional override for exported paper-table directories.
  RUN_REPORT            Set to false to skip report export.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-root)
      BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --paper-root)
      PAPER_ROOT="$2"
      shift 2
      ;;
    --execution-model)
      EXECUTION_MODEL="$2"
      shift 2
      ;;
    --skip-report)
      RUN_REPORT="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$EXECUTION_MODEL" ]]; then
  echo "Missing execution model. Set VTM_EXECUTION_MODEL or pass --execution-model." >&2
  exit 2
fi

cd "$PROJECT_ROOT"

RETRIEVAL_DIR="$BENCHMARK_ROOT/matrix-retrieval"
DRIFTED_RETRIEVAL_DIR="$BENCHMARK_ROOT/matrix-retrieval-drifted"
DRIFT_DIR="$BENCHMARK_ROOT/matrix-drift"
CODING_DIR="$BENCHMARK_ROOT/controlled-coding-drift"

uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval \
  --output "$RETRIEVAL_DIR"

uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval_drifted \
  --output "$DRIFTED_RETRIEVAL_DIR"

uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_drift \
  --output "$DRIFT_DIR"

uv run --extra rlm python -m vtm.benchmarks.matrix \
  --preset controlled_coding_drift \
  --output "$CODING_DIR" \
  --execution-model "$EXECUTION_MODEL"

if [[ "$RUN_REPORT" == "true" ]]; then
  uv run python -m vtm.benchmarks.report \
    --retrieval-run "$RETRIEVAL_DIR/runs/no_memory" \
    --retrieval-run "$RETRIEVAL_DIR/runs/naive_lexical" \
    --retrieval-run "$RETRIEVAL_DIR/runs/verified_lexical" \
    --output "$PAPER_ROOT/retrieval"

  uv run python -m vtm.benchmarks.report \
    --retrieval-run "$DRIFTED_RETRIEVAL_DIR/runs/no_memory" \
    --retrieval-run "$DRIFTED_RETRIEVAL_DIR/runs/naive_lexical" \
    --retrieval-run "$DRIFTED_RETRIEVAL_DIR/runs/verified_lexical" \
    --output "$PAPER_ROOT/retrieval-drifted"

  uv run python -m vtm.benchmarks.report \
    --drift-run "$DRIFT_DIR/runs/verified_lexical" \
    --output "$PAPER_ROOT/drift"

  uv run python -m vtm.benchmarks.report \
    --coding-run "$CODING_DIR/runs/no_memory" \
    --coding-run "$CODING_DIR/runs/naive_lexical" \
    --coding-run "$CODING_DIR/runs/verified_lexical" \
    --output "$PAPER_ROOT/controlled-coding-drift"

  uv run python -m vtm.benchmarks.report \
    --retrieval-run "$RETRIEVAL_DIR/runs/no_memory" \
    --retrieval-run "$RETRIEVAL_DIR/runs/naive_lexical" \
    --retrieval-run "$RETRIEVAL_DIR/runs/verified_lexical" \
    --drift-run "$DRIFT_DIR/runs/verified_lexical" \
    --coding-run "$CODING_DIR/runs/no_memory" \
    --coding-run "$CODING_DIR/runs/naive_lexical" \
    --coding-run "$CODING_DIR/runs/verified_lexical" \
    --output "$PAPER_ROOT/combined"
fi
