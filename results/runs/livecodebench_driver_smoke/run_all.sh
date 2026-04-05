#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="results/runs/livecodebench_driver_smoke"

if [[ -n "${RAG_COMMAND:-}" ]]; then
  METHOD_COMMAND="${RAG_COMMAND}" "results/runs/livecodebench_driver_smoke/01_rag.sh"
else
  "results/runs/livecodebench_driver_smoke/01_rag.sh"
fi

if [[ -n "${RLM_COMMAND:-}" ]]; then
  METHOD_COMMAND="${RLM_COMMAND}" "results/runs/livecodebench_driver_smoke/02_rlm.sh"
else
  "results/runs/livecodebench_driver_smoke/02_rlm.sh"
fi

if [[ -n "${RLM_RAG_COMMAND:-}" ]]; then
  METHOD_COMMAND="${RLM_RAG_COMMAND}" "results/runs/livecodebench_driver_smoke/03_rlm_rag.sh"
else
  "results/runs/livecodebench_driver_smoke/03_rlm_rag.sh"
fi
