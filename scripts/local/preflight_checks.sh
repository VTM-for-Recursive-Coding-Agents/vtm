#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK="lcb"

usage() {
  cat <<'EOF'
Usage: bash scripts/local/preflight_checks.sh [--benchmark lcb]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$BENCHMARK" != "lcb" ]]; then
  echo "Only LiveCodeBench preflight is supported in this baseline-only helper." >&2
  exit 1
fi

for command_name in git python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

if [[ ! -d "$PROJECT_ROOT/benchmarks/LiveCodeBench" ]]; then
  echo "Missing LiveCodeBench checkout at $PROJECT_ROOT/benchmarks/LiveCodeBench" >&2
  echo "Run bash scripts/livecodebench/setup_livecodebench.sh first." >&2
  exit 1
fi

echo "[preflight] LiveCodeBench baseline checks passed"
