#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BENCHMARK="all"

usage() {
    cat <<'EOF'
Usage: scripts/local/preflight_checks.sh [--benchmark all|lcb|swe]

Checks local workstation requirements before running benchmark pipelines.
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

if [[ "$BENCHMARK" != "all" && "$BENCHMARK" != "lcb" && "$BENCHMARK" != "swe" ]]; then
    echo "Invalid --benchmark value: $BENCHMARK" >&2
    exit 1
fi

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

check_path_exists() {
    local path="$1"
    local label="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing $label at: $path" >&2
        exit 1
    fi
}

echo "[preflight] Checking common local requirements"
check_command git
check_command python3
check_path_exists "$PROJECT_ROOT/benchmarks" "benchmarks directory"

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "lcb" ]]; then
    echo "[preflight] Checking LiveCodeBench requirements"
    check_path_exists "$PROJECT_ROOT/benchmarks/LiveCodeBench" "LiveCodeBench repository"
fi

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "swe" ]]; then
    echo "[preflight] Checking SWE-bench requirements"
    check_path_exists "$PROJECT_ROOT/benchmarks/SWE-bench" "SWE-bench repository"
    check_command docker
    if ! docker info >/dev/null 2>&1; then
        echo "Docker is installed but daemon is not reachable. Start Docker and retry." >&2
        exit 1
    fi
fi

echo "[preflight] All checks passed for benchmark=$BENCHMARK"