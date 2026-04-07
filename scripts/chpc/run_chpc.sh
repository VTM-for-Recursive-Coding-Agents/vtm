#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/chpc_env.sh"

MODEL=""
PROVIDER="baseline"
MAX_INSTANCES="1"
STORAGE_ROOT=""
PYTHON_MODULE=""
CUDA_MODULE=""
SKIP_MODULES="false"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/chpc/run_chpc.sh --model <model_name> [options]

Options:
    --model <name>            LiveCodeBench model key (required)
    --provider <name>         Provider label for the shared driver (default: baseline)
    --max-instances <int>     Limit dataset size for smoke runs (default: 1)
    --storage-root <path>     Scratch-backed CHPC root
    --python-module <name>    Module to load before resolving Python
    --cuda-module <name>      CUDA module to load before the smoke run
    --skip-modules            Do not attempt to load environment modules
    --extra-arg <arg>         Additional argument forwarded to the driver (repeatable)
    -h, --help                Show help

This script is an interactive smoke runner for validating CHPC setup on a GPU node.
For production execution, prefer scripts/chpc/submit_livecodebench_local_model.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --storage-root)
            STORAGE_ROOT="$2"
            shift 2
            ;;
        --python-module)
            PYTHON_MODULE="$2"
            shift 2
            ;;
        --cuda-module)
            CUDA_MODULE="$2"
            shift 2
            ;;
        --skip-modules)
            SKIP_MODULES="true"
            shift
            ;;
        --extra-arg)
            EXTRA_ARGS+=("$2")
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

if [[ -z "$MODEL" ]]; then
    echo "--model is required" >&2
    usage
    exit 1
fi

if [[ -n "$PYTHON_MODULE" ]]; then
    export VTM_CHPC_PYTHON_MODULE="$PYTHON_MODULE"
fi
if [[ -n "$CUDA_MODULE" ]]; then
    export VTM_CHPC_CUDA_MODULE="$CUDA_MODULE"
fi
if [[ "$SKIP_MODULES" == "true" ]]; then
    export VTM_CHPC_SKIP_MODULES=true
fi

vtm_chpc_setup_environment "$STORAGE_ROOT"
vtm_chpc_load_requested_modules

PYTHON_BIN="$(vtm_chpc_resolve_python)" || {
    echo "[run] Unable to resolve a Python interpreter. Run scripts/chpc/setup_chpc.sh first." >&2
    exit 1
}

echo "=================================================="
echo "VTM CHPC Smoke Runner"
echo "=================================================="
vtm_chpc_print_summary
echo "[run] python_bin=$PYTHON_BIN"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[run] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[run] WARNING: nvidia-smi not found; GPU visibility not confirmed." >&2
fi

RUN_ID="lcb_smoke_${PROVIDER}_$(date +%Y%m%d_%H%M%S)"
DRIVER_SCRIPT="$PROJECT_ROOT/scripts/livecodebench_${PROVIDER}_driver.py"

if [[ ! -f "$DRIVER_SCRIPT" ]]; then
    echo "[run] Driver script not found: $DRIVER_SCRIPT" >&2
    exit 1
fi

CMD=(
    "$PYTHON_BIN"
    "$DRIVER_SCRIPT"
    --provider "$PROVIDER"
    --run-id "$RUN_ID"
    --model "$MODEL"
    --scenario codegeneration
    --n 1
    --temperature 0.2
    --evaluate false
    --max-instances "$MAX_INSTANCES"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[run] Executing smoke command: ${CMD[*]}"
cd "$PROJECT_ROOT"
"${CMD[@]}"
