#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to submit CHPC jobs." >&2
  exit 1
fi

if [[ -z "${BASELINE_COMMAND:-}" || -z "${RAG_COMMAND:-}" || -z "${RLM_COMMAND:-}" || -z "${RLM_RAG_COMMAND:-}" ]]; then
  echo "Set BASELINE_COMMAND, RAG_COMMAND, RLM_COMMAND, and RLM_RAG_COMMAND before running submit_all.sh" >&2
  exit 2
fi

METHOD_COMMAND="${BASELINE_COMMAND}" sbatch "$SCRIPT_DIR/00_baseline.sbatch"
METHOD_COMMAND="${RAG_COMMAND}" sbatch "$SCRIPT_DIR/01_rag.sbatch"
METHOD_COMMAND="${RLM_COMMAND}" sbatch "$SCRIPT_DIR/02_rlm.sbatch"
METHOD_COMMAND="${RLM_RAG_COMMAND}" sbatch "$SCRIPT_DIR/03_rlm_rag.sbatch"
