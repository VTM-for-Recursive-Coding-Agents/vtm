#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BATCH_INDEX=0
BATCH_SIZE=25
RUN_PREFIX="lcb_dspy_pilot_batch"
OUTPUT_ROOT=".benchmarks/livecodebench-dspy"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_livecodebench_dspy_pilot_batch.sh [options] [-- extra pilot args]

Options:
  --batch-index N   Zero-based batch index. Default: 0
  --batch-size N    Problems per batch. Default: 25
  --run-prefix STR  Run-id prefix. Default: lcb_dspy_pilot_batch
  --output-root DIR Output root passed to the pilot runner.
  -h, --help        Show this help text.

Examples:
  bash scripts/run_livecodebench_dspy_pilot_batch.sh \
    --batch-index 1 \
    -- --method all --scenario self_repair --model qwen/qwen3-coder-next --execute
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-index)
      BATCH_INDEX="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if ! [[ "$BATCH_INDEX" =~ ^[0-9]+$ ]]; then
  echo "batch-index must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -lt 1 ]]; then
  echo "batch-size must be a positive integer" >&2
  exit 2
fi

PROBLEM_OFFSET=$((BATCH_INDEX * BATCH_SIZE))
RUN_ID="${RUN_PREFIX}_b$(printf '%03d' "$BATCH_INDEX")_o$(printf '%04d' "$PROBLEM_OFFSET")_n$(printf '%03d' "$BATCH_SIZE")"

cd "$PROJECT_ROOT"
exec uv run --extra dspy python scripts/run_livecodebench_dspy_pilot.py \
  --problem-offset "$PROBLEM_OFFSET" \
  --max-problems "$BATCH_SIZE" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  "$@"
