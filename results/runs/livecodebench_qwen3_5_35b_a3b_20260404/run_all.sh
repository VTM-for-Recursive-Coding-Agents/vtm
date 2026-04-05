#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="/Users/alexanderfraser/Projects /vtm/results/runs/livecodebench_qwen3_5_35b_a3b_20260404"

if [[ -z "${RAG_COMMAND:-}" || -z "${RLM_COMMAND:-}" || -z "${RLM_RAG_COMMAND:-}" ]]; then
  echo "Set RAG_COMMAND, RLM_COMMAND, and RLM_RAG_COMMAND before running run_all.sh" >&2
  exit 2
fi

METHOD_COMMAND="${RAG_COMMAND}" "/Users/alexanderfraser/Projects /vtm/results/runs/livecodebench_qwen3_5_35b_a3b_20260404/01_rag.sh"
METHOD_COMMAND="${RLM_COMMAND}" "/Users/alexanderfraser/Projects /vtm/results/runs/livecodebench_qwen3_5_35b_a3b_20260404/02_rlm.sh"
METHOD_COMMAND="${RLM_RAG_COMMAND}" "/Users/alexanderfraser/Projects /vtm/results/runs/livecodebench_qwen3_5_35b_a3b_20260404/03_rlm_rag.sh"
