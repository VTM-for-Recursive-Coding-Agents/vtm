#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

select_python() {
  if [[ -x "$PROJECT_ROOT/benchmarks/LiveCodeBench/.venv/bin/python" ]]; then
    echo "$PROJECT_ROOT/benchmarks/LiveCodeBench/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}

RUN_ID="lcb_qwen3_5_35b_a3b_rlm_rag_implsmoke_a"
PROVIDER="rlm_rag"
MODEL="qwen3.5-35b-a3b"
LM_STUDIO_MODEL_ID="qwen/qwen3.5-35b-a3b"
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"
DEFAULT_DRIVER="$PROJECT_ROOT/scripts/livecodebench_rlm_rag_driver.py"
DEFAULT_PYTHON="$(select_python)"

DEFAULT_METHOD_COMMAND=(
  "$DEFAULT_PYTHON" "$DEFAULT_DRIVER"
  --provider "rlm_rag"
  --run-id "lcb_qwen3_5_35b_a3b_rlm_rag_implsmoke_a"
  --model "qwen3.5-35b-a3b"
  --lm-studio-model-id "qwen/qwen3.5-35b-a3b"
  --scenario "codegeneration"
  --n "1"
  --temperature "0.2"
)

if [[ -z "${METHOD_COMMAND:-}" ]]; then
  if [[ ! -f "${DEFAULT_DRIVER}" ]]; then
    echo "Default driver not found: ${DEFAULT_DRIVER}" >&2
    echo "Provide a custom METHOD_COMMAND to continue." >&2
    exit 2
  fi

  cd "$PROJECT_ROOT"
  echo "[queue] Launching ${RUN_ID} (${PROVIDER}) with default driver"
  echo "[queue] command=${DEFAULT_METHOD_COMMAND[*]}"
  "${DEFAULT_METHOD_COMMAND[@]}"
  exit 0
fi

cd "$PROJECT_ROOT"
echo "[queue] Launching ${RUN_ID} (${PROVIDER})"
echo "[queue] METHOD_COMMAND=${METHOD_COMMAND}"
eval "${METHOD_COMMAND}"
