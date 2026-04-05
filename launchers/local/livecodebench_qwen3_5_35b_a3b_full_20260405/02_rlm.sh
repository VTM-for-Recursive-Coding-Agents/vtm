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

RUN_ID="lcb_qwen3_5_35b_a3b_rlm_full_20260405_a"
PROVIDER="rlm"
MODEL="qwen3.5-35b-a3b"
LM_STUDIO_MODEL_ID="qwen/qwen3.5-35b-a3b"
SCENARIO="codegeneration"
N="10"
TEMPERATURE="0.2"
EVALUATE="true"
MAX_TOKENS="512"
MAX_INSTANCES=""
RAG_TOP_K=""
RAG_MAX_CHARS_PER_CHUNK=""
RLM_MAX_DEPTH=""
RLM_MAX_ITERATIONS=""
RLM_MAX_TIMEOUT=""
DEFAULT_DRIVER="$PROJECT_ROOT/scripts/livecodebench_rlm_driver.py"
DEFAULT_PYTHON="$(select_python)"

DEFAULT_METHOD_COMMAND=(
  "$DEFAULT_PYTHON" "$DEFAULT_DRIVER"
  --provider "rlm"
  --run-id "lcb_qwen3_5_35b_a3b_rlm_full_20260405_a"
  --model "qwen3.5-35b-a3b"
  --lm-studio-model-id "qwen/qwen3.5-35b-a3b"
  --scenario "codegeneration"
  --n "10"
  --temperature "0.2"
  --evaluate "true"
)

if [[ -n "512" ]]; then
  DEFAULT_METHOD_COMMAND+=(--max-tokens "512")
fi

if [[ -n "" ]]; then
  DEFAULT_METHOD_COMMAND+=(--max-instances "")
fi

if [[ "$PROVIDER" == "rag" || "$PROVIDER" == "rlm_rag" ]]; then
  if [[ -n "" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rag-top-k "")
  fi
  if [[ -n "" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rag-max-chars-per-chunk "")
  fi
fi

if [[ "$PROVIDER" == "rlm" || "$PROVIDER" == "rlm_rag" ]]; then
  if [[ -n "" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rlm-max-depth "")
  fi
  if [[ -n "" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rlm-max-iterations "")
  fi
  if [[ -n "" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rlm-max-timeout "")
  fi
fi

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
