#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCHERS_ROOT="$PROJECT_ROOT/launchers/local"

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
Usage: scripts/local/queue_livecodebench_methods.sh --model <model_name> [options]

Options:
  --model <name>                 Model key for LiveCodeBench (required)
  --model-tag <tag>              Short slug for run ids (default: derived from model)
  --lm-studio-model-id <id>      LM Studio API model id (optional)
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
  --queue-dir <path>             Output queue directory (default: launchers/local/<derived>)
  -h, --help                     Show help

This generates local launcher bundles only:
  launchers/local/<queue>/01_rag.sh
  launchers/local/<queue>/02_rlm.sh
  launchers/local/<queue>/03_rlm_rag.sh

Actual benchmark outputs remain under results/raw/.
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
QUEUE_RELATIVE="launchers/local/$QUEUE_NAME"

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
  printf '    "execution_env": "local",\n'
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
  local job_script="$QUEUE_DIR/${order}_${provider}.sh"

  cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="\$(cd "\$SCRIPT_DIR/../../.." && pwd)"

select_python() {
  if [[ -x "\$PROJECT_ROOT/benchmarks/LiveCodeBench/.venv/bin/python" ]]; then
    echo "\$PROJECT_ROOT/benchmarks/LiveCodeBench/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}

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
DEFAULT_DRIVER="\$PROJECT_ROOT/scripts/livecodebench_${provider}_driver.py"
DEFAULT_PYTHON="\$(select_python)"

DEFAULT_METHOD_COMMAND=(
  "\$DEFAULT_PYTHON" "\$DEFAULT_DRIVER"
  --provider "$provider"
  --run-id "$run_id"
  --model "$MODEL"
  --lm-studio-model-id "$LM_STUDIO_MODEL_ID"
  --scenario "$SCENARIO"
  --n "$N"
  --temperature "$TEMPERATURE"
  --evaluate "$EVALUATE"
)

if [[ -n "$MAX_TOKENS" ]]; then
  DEFAULT_METHOD_COMMAND+=(--max-tokens "$MAX_TOKENS")
fi

if [[ -n "$MAX_INSTANCES" ]]; then
  DEFAULT_METHOD_COMMAND+=(--max-instances "$MAX_INSTANCES")
fi

if [[ "\$PROVIDER" == "rag" || "\$PROVIDER" == "rlm_rag" ]]; then
  if [[ -n "$RAG_TOP_K" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rag-top-k "$RAG_TOP_K")
  fi
  if [[ -n "$RAG_MAX_CHARS_PER_CHUNK" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rag-max-chars-per-chunk "$RAG_MAX_CHARS_PER_CHUNK")
  fi
fi

if [[ "\$PROVIDER" == "rlm" || "\$PROVIDER" == "rlm_rag" ]]; then
  if [[ -n "$RLM_MAX_DEPTH" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rlm-max-depth "$RLM_MAX_DEPTH")
  fi
  if [[ -n "$RLM_MAX_ITERATIONS" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rlm-max-iterations "$RLM_MAX_ITERATIONS")
  fi
  if [[ -n "$RLM_MAX_TIMEOUT" ]]; then
    DEFAULT_METHOD_COMMAND+=(--rlm-max-timeout "$RLM_MAX_TIMEOUT")
  fi
fi

if [[ -z "\${METHOD_COMMAND:-}" ]]; then
  if [[ ! -f "\${DEFAULT_DRIVER}" ]]; then
    echo "Default driver not found: \${DEFAULT_DRIVER}" >&2
    echo "Provide a custom METHOD_COMMAND to continue." >&2
    exit 2
  fi

  cd "\$PROJECT_ROOT"
  echo "[queue] Launching \${RUN_ID} (\${PROVIDER}) with default driver"
  echo "[queue] command=\${DEFAULT_METHOD_COMMAND[*]}"
  "\${DEFAULT_METHOD_COMMAND[@]}"
  exit 0
fi

cd "\$PROJECT_ROOT"
echo "[queue] Launching \${RUN_ID} (\${PROVIDER})"
echo "[queue] METHOD_COMMAND=\${METHOD_COMMAND}"
eval "\${METHOD_COMMAND}"
EOF

  chmod +x "$job_script"
}

write_job_script "01" "rag"
write_job_script "02" "rlm"
write_job_script "03" "rlm_rag"

cat > "$QUEUE_DIR/run_all.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "\${RAG_COMMAND:-}" ]]; then
  METHOD_COMMAND="\${RAG_COMMAND}" "\$SCRIPT_DIR/01_rag.sh"
else
  "\$SCRIPT_DIR/01_rag.sh"
fi

if [[ -n "\${RLM_COMMAND:-}" ]]; then
  METHOD_COMMAND="\${RLM_COMMAND}" "\$SCRIPT_DIR/02_rlm.sh"
else
  "\$SCRIPT_DIR/02_rlm.sh"
fi

if [[ -n "\${RLM_RAG_COMMAND:-}" ]]; then
  METHOD_COMMAND="\${RLM_RAG_COMMAND}" "\$SCRIPT_DIR/03_rlm_rag.sh"
else
  "\$SCRIPT_DIR/03_rlm_rag.sh"
fi
EOF

chmod +x "$QUEUE_DIR/run_all.sh"

{
  printf '[\n'
  write_manifest_entry "rag" "$QUEUE_RELATIVE/01_rag.sh"
  printf ',\n'
  write_manifest_entry "rlm" "$QUEUE_RELATIVE/02_rlm.sh"
  printf ',\n'
  write_manifest_entry "rlm_rag" "$QUEUE_RELATIVE/03_rlm_rag.sh"
  printf '\n]\n'
} > "$QUEUE_DIR/manifest.json"

echo "[queue] Created local launcher bundle at: $QUEUE_DIR"
echo "[queue] Jobs: 01_rag.sh, 02_rlm.sh, 03_rlm_rag.sh"
echo "[queue] Manifest: $QUEUE_DIR/manifest.json"