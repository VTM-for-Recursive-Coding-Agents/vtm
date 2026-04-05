#!/usr/bin/env bash
#SBATCH --job-name=qwen3_5_35b_a3b_rag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

RUN_ID="lcb_qwen3_5_35b_a3b_rag_implsmoke_a"
PROVIDER="rag"
MODEL="qwen3.5-35b-a3b"
LM_STUDIO_MODEL_ID="qwen/qwen3.5-35b-a3b"
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"

if [[ -z "${METHOD_COMMAND:-}" ]]; then
  echo "Set METHOD_COMMAND before submitting this job." >&2
  echo "Example:" >&2
  echo "  METHOD_COMMAND='benchmarks/LiveCodeBench/.venv/bin/python scripts/livecodebench_rag_driver.py --provider rag --run-id lcb_qwen3_5_35b_a3b_rag_implsmoke_a --model qwen3.5-35b-a3b --scenario codegeneration --n 1 --temperature 0.2' sbatch \"$SCRIPT_DIR/01_rag.sbatch\"" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
echo "[chpc] Running ${RUN_ID} (${PROVIDER})"
echo "[chpc] METHOD_COMMAND=${METHOD_COMMAND}"
eval "${METHOD_COMMAND}"
