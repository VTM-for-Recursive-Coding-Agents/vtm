#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

should_run_preflight="true"
should_require_checkout="false"
for arg in "$@"; do
  if [[ "$arg" == "--execute" ]]; then
    should_require_checkout="true"
  fi
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    should_run_preflight="false"
    break
  fi
done

if [[ "$should_run_preflight" == "true" && "$should_require_checkout" == "true" && -n "${VTM_PREFLIGHT_SCRIPT:-}" ]]; then
  "$VTM_PREFLIGHT_SCRIPT" --benchmark lcb
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python "$PROJECT_ROOT/scripts/livecodebench/baseline.py" "$@"
fi

exec python3 "$PROJECT_ROOT/scripts/livecodebench/baseline.py" "$@"
