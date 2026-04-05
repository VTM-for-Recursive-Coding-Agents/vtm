#!/usr/bin/env bash
set -euo pipefail

RUN_ID="lcb_qwen3_5_35b_a3b_rag_20260404_a"
PROVIDER="rag"
MODEL="qwen3.5-35b-a3b"
LM_STUDIO_MODEL_ID="qwen/qwen3.5-35b-a3b"
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"

if [[ -z "${METHOD_COMMAND:-}" ]]; then
  echo "Queued job is ready, but METHOD_COMMAND is not attached yet." >&2
  echo "Expected metadata:" >&2
  echo "  benchmark=livecodebench" >&2
  echo "  provider=${PROVIDER}" >&2
  echo "  run_id=${RUN_ID}" >&2
  echo "  model=${MODEL}" >&2
  echo "  scenario=${SCENARIO}" >&2
  echo >&2
  echo "Example:" >&2
  echo "  METHOD_COMMAND='python path/to/your_rag_driver.py ...' \"/Users/alexanderfraser/Projects /vtm/results/runs/livecodebench_qwen3_5_35b_a3b_20260404/01_rag.sh\"" >&2
  exit 2
fi

echo "[queue] Launching ${RUN_ID} (${PROVIDER})"
echo "[queue] METHOD_COMMAND=${METHOD_COMMAND}"
eval "${METHOD_COMMAND}"
