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

write_job_script() {
  local order="$1"
  local provider="$2"
  local run_id="lcb_${MODEL_TAG}_${provider}_${QUEUE_TAG}_a"
  local job_script="$QUEUE_DIR/${order}_${provider}.sbatch"

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

if [[ -z "\${METHOD_COMMAND:-}" ]]; then
  echo "Set METHOD_COMMAND before submitting this job." >&2
  echo "Example:" >&2
  echo "  METHOD_COMMAND='benchmarks/LiveCodeBench/.venv/bin/python scripts/livecodebench_${provider}_driver.py --provider ${provider} --run-id ${run_id} --model ${MODEL} --scenario ${SCENARIO} --n ${N} --temperature ${TEMPERATURE}' sbatch \"\$SCRIPT_DIR/${order}_${provider}.sbatch\"" >&2
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

cat > "$QUEUE_DIR/manifest.json" <<EOF
[
  {
    "benchmark": "livecodebench",
    "execution_env": "chpc",
    "provider": "rag",
    "run_id": "lcb_${MODEL_TAG}_rag_${QUEUE_TAG}_a",
    "model": "$MODEL",
    "lm_studio_model_id": "$LM_STUDIO_MODEL_ID",
    "scenario": "$SCENARIO",
    "n": "$N",
    "temperature": "$TEMPERATURE",
    "status": "queued",
    "launch_script": "$QUEUE_RELATIVE/01_rag.sbatch"
  },
  {
    "benchmark": "livecodebench",
    "execution_env": "chpc",
    "provider": "rlm",
    "run_id": "lcb_${MODEL_TAG}_rlm_${QUEUE_TAG}_a",
    "model": "$MODEL",
    "lm_studio_model_id": "$LM_STUDIO_MODEL_ID",
    "scenario": "$SCENARIO",
    "n": "$N",
    "temperature": "$TEMPERATURE",
    "status": "queued",
    "launch_script": "$QUEUE_RELATIVE/02_rlm.sbatch"
  },
  {
    "benchmark": "livecodebench",
    "execution_env": "chpc",
    "provider": "rlm_rag",
    "run_id": "lcb_${MODEL_TAG}_rlm_rag_${QUEUE_TAG}_a",
    "model": "$MODEL",
    "lm_studio_model_id": "$LM_STUDIO_MODEL_ID",
    "scenario": "$SCENARIO",
    "n": "$N",
    "temperature": "$TEMPERATURE",
    "status": "queued",
    "launch_script": "$QUEUE_RELATIVE/03_rlm_rag.sbatch"
  }
]
EOF

echo "[queue] Created CHPC launcher bundle at: $QUEUE_DIR"
echo "[queue] Submission files: 01_rag.sbatch, 02_rlm.sbatch, 03_rlm_rag.sbatch"
echo "[queue] Manifest: $QUEUE_DIR/manifest.json"