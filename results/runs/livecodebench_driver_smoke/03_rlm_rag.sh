#!/usr/bin/env bash
set -euo pipefail

RUN_ID="lcb_qwen3_5_35b_a3b_rlm_rag_driver_smoke_a"
PROVIDER="rlm_rag"
MODEL="qwen3.5-35b-a3b"
LM_STUDIO_MODEL_ID="qwen/qwen3.5-35b-a3b"
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"

PROJECT_ROOT="/Users/alexanderfraser/Projects /vtm"
DEFAULT_DRIVER="/Users/alexanderfraser/Projects /vtm/scripts/livecodebench_rlm_rag_driver.py"

DEFAULT_METHOD_COMMAND=(
  "/Users/alexanderfraser/Projects /vtm/benchmarks/LiveCodeBench/.venv/bin/python" "/Users/alexanderfraser/Projects /vtm/scripts/livecodebench_rlm_rag_driver.py"
  --provider "rlm_rag"
  --run-id "lcb_qwen3_5_35b_a3b_rlm_rag_driver_smoke_a"
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

  echo "[queue] Launching ${RUN_ID} (${PROVIDER}) with default driver"
  echo "[queue] command=${DEFAULT_METHOD_COMMAND[*]}"
  "${DEFAULT_METHOD_COMMAND[@]}"
  exit 0
fi

echo "[queue] Launching ${RUN_ID} (${PROVIDER})"
echo "[queue] METHOD_COMMAND=${METHOD_COMMAND}"
eval "${METHOD_COMMAND}"
