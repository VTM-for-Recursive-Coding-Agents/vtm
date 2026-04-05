#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCHERS_ROOT="$PROJECT_ROOT/launchers/chpc"

MODEL=""
MODEL_TAG=""
LM_STUDIO_MODEL_ID=""
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"
EVALUATE="true"
MAX_TOKENS=""
MAX_INSTANCES=""
RAG_TOP_K=""
RAG_MAX_CHARS_PER_CHUNK=""
RLM_MAX_DEPTH=""
RLM_MAX_ITERATIONS=""
RLM_MAX_TIMEOUT=""
QUEUE_TAG="$(date +%Y%m%d)"
QUEUE_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/chpc/queue_livecodebench_methods.sh --model <model_name> [options]

Options:
  --model <name>                 Model key for LiveCodeBench (required)
  --model-tag <tag>              Short slug for run ids (default: derived from model)
  --lm-studio-model-id <id>      Model id forwarded to method commands (optional)
  --scenario <name>              Scenario to queue (default: codegeneration)
  --n <int>                      Samples per problem (default: 1)
  --temperature <float>          Temperature (default: 0.2)
  --evaluate <true|false>        Run evaluation after generation (default: true)
  --max-tokens <int>             Max output tokens per response (optional)
  --max-instances <int>          Limit dataset instances for smoke runs (optional)
  --rag-top-k <int>              Retrieval chunk count for rag/rlm_rag (optional)
  --rag-max-chars-per-chunk <n>  Max chars per retrieved chunk for rag/rlm_rag (optional)
  --rlm-max-depth <int>          Max recursion depth for rlm/rlm_rag (optional)
  --rlm-max-iterations <int>     Max iterations for rlm/rlm_rag (optional)
  --rlm-max-timeout <seconds>    Max wall-clock time for rlm/rlm_rag (optional)
  --queue-tag <tag>              Queue label/date suffix (default: YYYYMMDD)
  --queue-dir <path>             Output bundle directory (default: launchers/chpc/<derived>)
  -h, --help                     Show help

This creates CHPC submission bundles only. It does not duplicate benchmark logic.
You supply provider-specific METHOD_COMMAND values at submit time.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model-tag)
      MODEL_TAG="$2"
      shift 2
      ;;
    --lm-studio-model-id)
      LM_STUDIO_MODEL_ID="$2"
      shift 2
      ;;
    --scenario)
      SCENARIO="$2"
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
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --max-instances)
      MAX_INSTANCES="$2"
      shift 2
      ;;
    --rag-top-k)
      RAG_TOP_K="$2"
      shift 2
      ;;
    --rag-max-chars-per-chunk)
      RAG_MAX_CHARS_PER_CHUNK="$2"
      shift 2
      ;;
    --rlm-max-depth)
      RLM_MAX_DEPTH="$2"
      shift 2
      ;;
    --rlm-max-iterations)
      RLM_MAX_ITERATIONS="$2"
      shift 2
      ;;
    --rlm-max-timeout)
      RLM_MAX_TIMEOUT="$2"
      shift 2
      ;;
    --queue-tag)
      QUEUE_TAG="$2"
      shift 2
      ;;
    --queue-dir)
      QUEUE_DIR="$2"
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

if [[ -z "$MODEL_TAG" ]]; then
  MODEL_TAG="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_//; s/_$//')"
fi

if [[ -z "$QUEUE_DIR" ]]; then
  QUEUE_DIR="$LAUNCHERS_ROOT/livecodebench_${MODEL_TAG}_${QUEUE_TAG}"
fi

mkdir -p "$QUEUE_DIR"
QUEUE_NAME="$(basename "$QUEUE_DIR")"
QUEUE_RELATIVE="launchers/chpc/$QUEUE_NAME"

if [[ "$EVALUATE" != "true" && "$EVALUATE" != "false" ]]; then
  echo "--evaluate must be true or false" >&2
  exit 1
fi

json_optional_line() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    printf '    "%s": "%s",\n' "$key" "$value"
  fi
}

write_manifest_entry() {
  local provider="$1"
  local launch_script="$2"
  local run_id="lcb_${MODEL_TAG}_${provider}_${QUEUE_TAG}_a"

  printf '  {\n'
  printf '    "benchmark": "livecodebench",\n'
  printf '    "execution_env": "chpc",\n'
  printf '    "provider": "%s",\n' "$provider"
  printf '    "run_id": "%s",\n' "$run_id"
  printf '    "model": "%s",\n' "$MODEL"
  printf '    "lm_studio_model_id": "%s",\n' "$LM_STUDIO_MODEL_ID"
  printf '    "scenario": "%s",\n' "$SCENARIO"
  printf '    "n": "%s",\n' "$N"
  printf '    "temperature": "%s",\n' "$TEMPERATURE"
  printf '    "evaluate": "%s",\n' "$EVALUATE"
  json_optional_line "max_tokens" "$MAX_TOKENS"
  json_optional_line "max_instances" "$MAX_INSTANCES"
  if [[ "$provider" == "rag" || "$provider" == "rlm_rag" ]]; then
    json_optional_line "rag_top_k" "$RAG_TOP_K"
    json_optional_line "rag_max_chars_per_chunk" "$RAG_MAX_CHARS_PER_CHUNK"
  fi
  if [[ "$provider" == "rlm" || "$provider" == "rlm_rag" ]]; then
    json_optional_line "rlm_max_depth" "$RLM_MAX_DEPTH"
    json_optional_line "rlm_max_iterations" "$RLM_MAX_ITERATIONS"
    json_optional_line "rlm_max_timeout" "$RLM_MAX_TIMEOUT"
  fi
  printf '    "status": "queued",\n'
  printf '    "launch_script": "%s"\n' "$launch_script"
  printf '  }'
}

write_job_script() {
  local order="$1"
  local provider="$2"
  local run_id="lcb_${MODEL_TAG}_${provider}_${QUEUE_TAG}_a"
  local job_script="$QUEUE_DIR/${order}_${provider}.sbatch"
  local method_example="benchmarks/LiveCodeBench/.venv/bin/python scripts/livecodebench_${provider}_driver.py --provider ${provider} --run-id ${run_id} --model ${MODEL} --scenario ${SCENARIO} --n ${N} --temperature ${TEMPERATURE} --evaluate ${EVALUATE}"

  if [[ -n "$LM_STUDIO_MODEL_ID" ]]; then
    method_example+=" --lm-studio-model-id ${LM_STUDIO_MODEL_ID}"
  fi
  if [[ -n "$MAX_TOKENS" ]]; then
    method_example+=" --max-tokens ${MAX_TOKENS}"
  fi
  if [[ -n "$MAX_INSTANCES" ]]; then
    method_example+=" --max-instances ${MAX_INSTANCES}"
  fi
  if [[ "$provider" == "rag" || "$provider" == "rlm_rag" ]]; then
    if [[ -n "$RAG_TOP_K" ]]; then
      method_example+=" --rag-top-k ${RAG_TOP_K}"
    fi
    if [[ -n "$RAG_MAX_CHARS_PER_CHUNK" ]]; then
      method_example+=" --rag-max-chars-per-chunk ${RAG_MAX_CHARS_PER_CHUNK}"
    fi
  fi
  if [[ "$provider" == "rlm" || "$provider" == "rlm_rag" ]]; then
    if [[ -n "$RLM_MAX_DEPTH" ]]; then
      method_example+=" --rlm-max-depth ${RLM_MAX_DEPTH}"
    fi
    if [[ -n "$RLM_MAX_ITERATIONS" ]]; then
      method_example+=" --rlm-max-iterations ${RLM_MAX_ITERATIONS}"
    fi
    if [[ -n "$RLM_MAX_TIMEOUT" ]]; then
      method_example+=" --rlm-max-timeout ${RLM_MAX_TIMEOUT}"
    fi
  fi

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${MODEL_TAG}_${provider}
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="\$(cd "\$SCRIPT_DIR/../../.." && pwd)"

RUN_ID="$run_id"
PROVIDER="$provider"
MODEL="$MODEL"
LM_STUDIO_MODEL_ID="$LM_STUDIO_MODEL_ID"
SCENARIO="$SCENARIO"
N="$N"
TEMPERATURE="$TEMPERATURE"
EVALUATE="$EVALUATE"
MAX_TOKENS="$MAX_TOKENS"
MAX_INSTANCES="$MAX_INSTANCES"
RAG_TOP_K="$RAG_TOP_K"
RAG_MAX_CHARS_PER_CHUNK="$RAG_MAX_CHARS_PER_CHUNK"
RLM_MAX_DEPTH="$RLM_MAX_DEPTH"
RLM_MAX_ITERATIONS="$RLM_MAX_ITERATIONS"
RLM_MAX_TIMEOUT="$RLM_MAX_TIMEOUT"

if [[ -z "\${METHOD_COMMAND:-}" ]]; then
  echo "Set METHOD_COMMAND before submitting this job." >&2
  echo "Example:" >&2
  echo "  METHOD_COMMAND='${method_example}' sbatch \"\$SCRIPT_DIR/${order}_${provider}.sbatch\"" >&2
  exit 2
fi

cd "\$PROJECT_ROOT"
echo "[chpc] Running \${RUN_ID} (\${PROVIDER})"
echo "[chpc] METHOD_COMMAND=\${METHOD_COMMAND}"
eval "\${METHOD_COMMAND}"
EOF

  chmod +x "$job_script"
}

write_job_script "01" "rag"
write_job_script "02" "rlm"
write_job_script "03" "rlm_rag"

cat > "$QUEUE_DIR/submit_all.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to submit CHPC jobs." >&2
  exit 1
fi

if [[ -z "\${RAG_COMMAND:-}" || -z "\${RLM_COMMAND:-}" || -z "\${RLM_RAG_COMMAND:-}" ]]; then
  echo "Set RAG_COMMAND, RLM_COMMAND, and RLM_RAG_COMMAND before running submit_all.sh" >&2
  exit 2
fi

METHOD_COMMAND="\${RAG_COMMAND}" sbatch "\$SCRIPT_DIR/01_rag.sbatch"
METHOD_COMMAND="\${RLM_COMMAND}" sbatch "\$SCRIPT_DIR/02_rlm.sbatch"
METHOD_COMMAND="\${RLM_RAG_COMMAND}" sbatch "\$SCRIPT_DIR/03_rlm_rag.sbatch"
EOF

chmod +x "$QUEUE_DIR/submit_all.sh"

{
  printf '[\n'
  write_manifest_entry "rag" "$QUEUE_RELATIVE/01_rag.sbatch"
  printf ',\n'
  write_manifest_entry "rlm" "$QUEUE_RELATIVE/02_rlm.sbatch"
  printf ',\n'
  write_manifest_entry "rlm_rag" "$QUEUE_RELATIVE/03_rlm_rag.sbatch"
  printf '\n]\n'
} > "$QUEUE_DIR/manifest.json"

echo "[queue] Created CHPC launcher bundle at: $QUEUE_DIR"
echo "[queue] Submission files: 01_rag.sbatch, 02_rlm.sbatch, 03_rlm_rag.sbatch"
echo "[queue] Manifest: $QUEUE_DIR/manifest.json"