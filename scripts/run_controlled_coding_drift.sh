#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/.benchmarks/controlled-coding-drift}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_ROOT/.benchmarks/paper-tables/controlled-coding-drift}"
EXECUTION_MODEL="${VTM_EXECUTION_MODEL:-}"
RUN_REPORT="${RUN_REPORT:-true}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_controlled_coding_drift.sh [--output DIR] [--report DIR] [--execution-model MODEL] [--skip-report]

Runs the maintained controlled_coding_drift matrix and, by default, exports the coding comparison table.

Environment:
  VTM_EXECUTION_MODEL   Execution model for the maintained coding benchmark.
  OUTPUT_DIR            Optional override for the matrix output directory.
  REPORT_DIR            Optional override for the exported report directory.
  RUN_REPORT            Set to false to skip report export.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT_DIR="$2"
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

uv run --extra rlm python -m vtm.benchmarks.matrix \
  --preset controlled_coding_drift \
  --output "$OUTPUT_DIR" \
  --execution-model "$EXECUTION_MODEL"

if [[ "$RUN_REPORT" == "true" ]]; then
  uv run python -m vtm.benchmarks.report \
    --coding-run "$OUTPUT_DIR/runs/no_memory" \
    --coding-run "$OUTPUT_DIR/runs/naive_lexical" \
    --coding-run "$OUTPUT_DIR/runs/verified_lexical" \
    --output "$REPORT_DIR"
fi
