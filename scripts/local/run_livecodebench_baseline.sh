#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VTM_PREFLIGHT_SCRIPT="$SCRIPT_DIR/preflight_checks.sh"

exec "$SCRIPT_DIR/../shared/run_livecodebench.sh" "$@"
