#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/chpc/chpc_env.sh" ]]; then
  PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/chpc/chpc_env.sh"

STORAGE_ROOT="${VTM_CHPC_ROOT:-${STORAGE_ROOT:-}}"
PYTHON_BIN="${VTM_CHPC_PYTHON_BIN:-${PYTHON_BIN:-}}"
STATUS_FILE="${VTM_SERVE_STATUS_FILE:-$PROJECT_ROOT/logs/slurm/serve_status.txt}"
ENDPOINT_FILE="${VTM_SERVE_ENDPOINT_FILE:-$PROJECT_ROOT/logs/slurm/serve_node.txt}"

MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME:-Qwen/Qwen2.5-Coder-32B-Instruct}}"
SERVED_MODEL_ALIAS="${SERVED_MODEL_ALIAS:-${MODEL_ALIAS:-$MODEL_NAME}}"
HOST_BIND="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

mkdir -p "$(dirname "$STATUS_FILE")" "$(dirname "$ENDPOINT_FILE")"

write_atomic_file() {
  local target_path="$1"
  local tmp_path="${target_path}.tmp.$$"
  cat >"$tmp_path"
  mv "$tmp_path" "$target_path"
}

record_status() {
  local state="$1"
  local advertise_host="$2"
  write_atomic_file "$STATUS_FILE" <<EOF
state=$state
job_id=${SLURM_JOB_ID:-}
job_name=${SLURM_JOB_NAME:-vtm_serve}
endpoint_file=$ENDPOINT_FILE
model=$MODEL_NAME
served_model_alias=$SERVED_MODEL_ALIAS
python_bin=$PYTHON_BIN
tensor_parallel_size=$TENSOR_PARALLEL_SIZE
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
max_model_len=$MAX_MODEL_LEN
started_at=$STARTED_AT
updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
slurm_nodelist=${SLURM_JOB_NODELIST:-}
EOF

  if [[ "$state" != "starting" ]]; then
    cat >>"$STATUS_FILE" <<EOF
host=$advertise_host
port=$PORT
endpoint=http://$advertise_host:$PORT/v1
EOF
  fi
}

resolve_python() {
  if [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  PYTHON_BIN="$(vtm_chpc_resolve_python)"
  printf '%s\n' "$PYTHON_BIN"
}

infer_tensor_parallel_size() {
  if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
    printf '%s\n' "$VLLM_TENSOR_PARALLEL_SIZE"
    return 0
  fi

  if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$SLURM_GPUS_ON_NODE" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$SLURM_GPUS_ON_NODE"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [[ "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s\n' "$gpu_count"
      return 0
    fi
  fi

  printf '1\n'
}

await_readiness() {
  local url="$1"
  local attempts="${VLLM_READY_ATTEMPTS:-750}"
  local delay="${VLLM_READY_DELAY_SECONDS:-2}"
  local attempt

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --silent --show-error --fail "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      return 1
    fi
    sleep "$delay"
  done

  return 1
}

cleanup() {
  local rc=$?
  local final_state="exited"

  if (( rc != 0 )); then
    final_state="failed"
  fi

  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" || true
  fi

  record_status "$final_state" "$ADVERTISE_HOST"
}

trap cleanup EXIT

vtm_chpc_setup_environment "$STORAGE_ROOT"
vtm_chpc_load_requested_modules
PYTHON_BIN="$(resolve_python)"
TENSOR_PARALLEL_SIZE="$(infer_tensor_parallel_size)"
ADVERTISE_HOST="${VLLM_ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

record_status "starting" "$ADVERTISE_HOST"
rm -f "$ENDPOINT_FILE"

CMD=(
  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
  --model "$MODEL_NAME"
  --served-model-name "$SERVED_MODEL_ALIAS"
  --host "$HOST_BIND"
  --port "$PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "$MAX_NUM_SEQS" ]]; then
  CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_parts=($EXTRA_ARGS)
  CMD+=("${extra_parts[@]}")
fi

vtm_chpc_print_summary
echo "[serve] job_id=${SLURM_JOB_ID:-} model=$MODEL_NAME alias=$SERVED_MODEL_ALIAS"
echo "[serve] endpoint=http://$ADVERTISE_HOST:$PORT/v1"
echo "[serve] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION max_model_len=${MAX_MODEL_LEN:-<model-default>}"
echo "[serve] command=${CMD[*]}"

"${CMD[@]}" &
SERVER_PID=$!

if ! await_readiness "http://127.0.0.1:$PORT/v1/models"; then
  echo "[serve] vLLM server did not become ready at http://127.0.0.1:$PORT/v1/models" >&2
  exit 1
fi

printf 'http://%s:%s/v1\n' "$ADVERTISE_HOST" "$PORT" | write_atomic_file "$ENDPOINT_FILE"
record_status "running" "$ADVERTISE_HOST"
echo "[serve] Ready: http://$ADVERTISE_HOST:$PORT/v1"

wait "$SERVER_PID"