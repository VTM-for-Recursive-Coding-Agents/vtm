#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${RAG_COMMAND:-}" ]]; then
  METHOD_COMMAND="${RAG_COMMAND}" "$SCRIPT_DIR/01_rag.sh"
else
  "$SCRIPT_DIR/01_rag.sh"
fi

if [[ -n "${RLM_COMMAND:-}" ]]; then
  METHOD_COMMAND="${RLM_COMMAND}" "$SCRIPT_DIR/02_rlm.sh"
else
  "$SCRIPT_DIR/02_rlm.sh"
fi

if [[ -n "${RLM_RAG_COMMAND:-}" ]]; then
  METHOD_COMMAND="${RLM_RAG_COMMAND}" "$SCRIPT_DIR/03_rlm_rag.sh"
else
  "$SCRIPT_DIR/03_rlm_rag.sh"
fi
