#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LCB_DIR="$PROJECT_ROOT/benchmarks/LiveCodeBench"
RESULTS_ROOT="$PROJECT_ROOT/results"

select_python() {
    local benchmark_dir="$1"
    if [[ -x "$benchmark_dir/.venv/bin/python" ]]; then
        echo "$benchmark_dir/.venv/bin/python"
        return 0
    fi
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        echo "$VIRTUAL_ENV/bin/python"
        return 0
    fi
    command -v python3
}

MODEL=""
PROVIDER="baseline"
SCENARIO="codegeneration"
RELEASE_VERSION="release_latest"
N="10"
TEMPERATURE="0.2"
EVALUATE="true"
RUN_ID=""
LM_STUDIO_MODEL_ID=""
CONTINUE_EXISTING="false"
CONTINUE_EXISTING_WITH_EVAL="false"
NUM_PROCESS_EVALUATE=""
TIMEOUT_SECONDS=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/local/run_livecodebench.sh --model <model_name> [options]

Options:
  --model <name>                 Model key from lcb_runner/lm_styles.py (required)
  --provider <label>             Experiment label for normalization (default: baseline)
  --scenario <name>              Scenario to run (default: codegeneration)
  --release-version <name>       Benchmark release (default: release_latest)
  --n <int>                      Number of generations (default: 10)
  --temperature <float>          Generation temperature (default: 0.2)
  --evaluate <true|false>        Run evaluation pass (default: true)
  --run-id <id>                  Custom run id (default: timestamp-based)
  --lm-studio-model-id <id>      LM Studio API model name (optional, overrides --model for API calls)
  --continue-existing            Reuse existing generations and continue missing ones
  --continue-existing-with-eval  Reuse existing generations and run evaluation on them
  --num-process-evaluate <int>   Parallel evaluation workers forwarded to lcb runner
  --timeout-seconds <int>        Per-problem evaluation timeout forwarded to lcb runner
  --extra-arg <arg>              Additional argument forwarded to lcb runner (repeatable)
  -h, --help                     Show help
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
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --release-version)
            RELEASE_VERSION="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --evaluate)
            EVALUATE="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --lm-studio-model-id)
            LM_STUDIO_MODEL_ID="$2"
            shift 2
            ;;
        --continue-existing)
            CONTINUE_EXISTING="true"
            shift
            ;;
        --continue-existing-with-eval)
            CONTINUE_EXISTING_WITH_EVAL="true"
            shift
            ;;
        --num-process-evaluate)
            NUM_PROCESS_EVALUATE="$2"
            shift 2
            ;;
        --timeout-seconds)
            TIMEOUT_SECONDS="$2"
            shift 2
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

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="lcb_$(date +%Y%m%d_%H%M%S)"
fi

RUN_DIR="$RESULTS_ROOT/raw/livecodebench/$RUN_ID"
mkdir -p "$RUN_DIR"

if [[ -n "${VTM_PREFLIGHT_SCRIPT:-}" ]]; then
    "$VTM_PREFLIGHT_SCRIPT" --benchmark lcb
fi

PYTHON_BIN="$(select_python "$LCB_DIR")"
if [[ "$PYTHON_BIN" == "$LCB_DIR/.venv/bin/python" ]]; then
    echo "[lcb] Using benchmark-local Python: $PYTHON_BIN"
else
    echo "[lcb] Using active Python: $PYTHON_BIN"
    echo "[lcb] Tip: create $LCB_DIR/.venv with 'cd benchmarks/LiveCodeBench && uv venv --python 3.11 && uv pip install -e .'"
fi

CMD=(
    "$PYTHON_BIN" -m lcb_runner.runner.main
    --model "$MODEL"
    --scenario "$SCENARIO"
    --release_version "$RELEASE_VERSION"
    --n "$N"
    --temperature "$TEMPERATURE"
)

if [[ "$CONTINUE_EXISTING" == "true" ]]; then
    CMD+=(--continue_existing)
fi

if [[ "$CONTINUE_EXISTING_WITH_EVAL" == "true" ]]; then
    CMD+=(--continue_existing_with_eval)
fi

if [[ "$EVALUATE" == "true" ]]; then
    CMD+=(--evaluate)
fi

if [[ -n "$NUM_PROCESS_EVALUATE" ]]; then
    CMD+=(--num_process_evaluate "$NUM_PROCESS_EVALUATE")
fi

if [[ -n "$TIMEOUT_SECONDS" ]]; then
    CMD+=(--timeout "$TIMEOUT_SECONDS")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

LOG_FILE="$RUN_DIR/command.log"
METADATA_FILE="$RUN_DIR/metadata.txt"

{
    echo "run_id=$RUN_ID"
    echo "benchmark=livecodebench"
    echo "model=$MODEL"
    echo "provider=$PROVIDER"
    echo "scenario=$SCENARIO"
    echo "release_version=$RELEASE_VERSION"
    echo "n=$N"
    echo "temperature=$TEMPERATURE"
    echo "evaluate=$EVALUATE"
    echo "continue_existing=$CONTINUE_EXISTING"
    echo "continue_existing_with_eval=$CONTINUE_EXISTING_WITH_EVAL"
    if [[ -n "$NUM_PROCESS_EVALUATE" ]]; then
        echo "num_process_evaluate=$NUM_PROCESS_EVALUATE"
    fi
    if [[ -n "$TIMEOUT_SECONDS" ]]; then
        echo "timeout_seconds=$TIMEOUT_SECONDS"
    fi
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$METADATA_FILE"

pushd "$LCB_DIR" >/dev/null

if [[ "$MODEL" == "qwen3.5-35b-a3b" || "$MODEL" == "deepseek-r1-0528-qwen3-8b" ]]; then
    export LMSTUDIO_BASE_URL="${LMSTUDIO_BASE_URL:-http://localhost:1234/v1}"
    if [[ -n "$LM_STUDIO_MODEL_ID" ]]; then
        export LMSTUDIO_MODEL="$LM_STUDIO_MODEL_ID"
    fi
    echo "[lcb] Using LM Studio API at: $LMSTUDIO_BASE_URL"
    if [[ -n "${LMSTUDIO_MODEL:-}" ]]; then
        echo "[lcb] Original model key '$MODEL' will use LM Studio model: $LMSTUDIO_MODEL"
    fi
fi

echo "[lcb] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

if [[ -d "$LCB_DIR/output" ]]; then
    find "$LCB_DIR/output" -type f \( -name '*.json' -o -name '*.jsonl' \) -print > "$RUN_DIR/output_files.txt"
fi

popd >/dev/null

echo "[lcb] Completed run_id=$RUN_ID"
echo "[lcb] Results folder: $RUN_DIR"