#!/usr/bin/env bash
#SBATCH --job-name=qwen_qwen2_5_coder_32b_instruct_rag
set -euo pipefail

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm"
STORAGE_ROOT="/scratch/general/vast/u1406806/vtm"
PYTHON_MODULE=""
CUDA_MODULE=""
EXTRA_MODULES=""
SKIP_MODULES="false"
LOCAL_VLLM_MAX_MODEL_LEN="28672"
LOCAL_VLLM_GPU_MEMORY_UTILIZATION="0.85"

# shellcheck disable=SC1091
source "/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/scripts/chpc/chpc_env.sh"

if [[ -n "" ]]; then
  export VTM_CHPC_PYTHON_MODULE=""
fi
if [[ -n "" ]]; then
  export VTM_CHPC_CUDA_MODULE=""
fi
if [[ -n "" ]]; then
  export VTM_CHPC_EXTRA_MODULES=""
fi
if [[ "false" == "true" ]]; then
  export VTM_CHPC_SKIP_MODULES=true
fi
if [[ -n "$LOCAL_VLLM_MAX_MODEL_LEN" ]]; then
  export VLLM_MAX_MODEL_LEN="$LOCAL_VLLM_MAX_MODEL_LEN"
fi
if [[ -n "$LOCAL_VLLM_GPU_MEMORY_UTILIZATION" ]]; then
  export VLLM_GPU_MEMORY_UTILIZATION="$LOCAL_VLLM_GPU_MEMORY_UTILIZATION"
fi

vtm_chpc_setup_environment "/scratch/general/vast/u1406806/vtm"
vtm_chpc_load_requested_modules

wait_for_rlm_serve() {
  local status_file=""
  local deadline
  local wait_seconds
  local poll_seconds
  local state
  local endpoint_file
  local endpoint_value

  read_status_value() {
    local key="$1"
    local file_path="$2"
    sed -n "s/^${key}=//p" "$file_path" | head -n 1 | tr -d '\r'
  }

  status_file="${VTM_RLM_SERVE_STATUS_FILE:-/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/logs/slurm/serve_status.txt}"
  wait_seconds="${VTM_RLM_SERVE_WAIT_SECONDS:-900}"
  poll_seconds="${VTM_RLM_SERVE_WAIT_POLL_SECONDS:-2}"
  deadline=$((SECONDS + wait_seconds))

  while (( SECONDS <= deadline )); do
    if [[ -f "$status_file" ]]; then
      state="$(read_status_value "state" "$status_file")"
      endpoint_file="$(read_status_value "endpoint_file" "$status_file")"

      if [[ "$state" == "failed" || "$state" == "error" || "$state" == "cancelled" || "$state" == "exited" ]]; then
        echo "[chpc] Serve status reports state=$state; aborting ${PROVIDER} launch." >&2
        return 1
      fi

      if [[ -n "$endpoint_file" && -f "$endpoint_file" ]]; then
        endpoint_value="$(sed -n '1p' "$endpoint_file" | tr -d '\r')"
        if [[ -n "$endpoint_value" ]]; then
          echo "[chpc] Found ready RLM endpoint: $endpoint_value"
          return 0
        fi
      fi
    fi

    if [[ ! -f "$status_file" && -f "/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/logs/slurm/serve_node.txt" ]]; then
      endpoint_value="$(sed -n '1p' "/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/logs/slurm/serve_node.txt" | tr -d '\r')"
      if [[ -n "$endpoint_value" ]]; then
        echo "[chpc] Found ready RLM endpoint from default endpoint file: $endpoint_value"
        return 0
      fi
    fi

    sleep "$poll_seconds"
  done

  echo "[chpc] Timed out waiting for ready RLM serve status at $status_file" >&2
  return 1
}

RUN_ID="lcb_qwen_qwen2_5_coder_32b_instruct_rag_rlmrag_stabletest_20260418c12h_a"
PROVIDER="rag"
MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
LM_STUDIO_MODEL_ID=""
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"
EVALUATE="false"
MAX_TOKENS=""
MAX_INSTANCES=""
TENSOR_PARALLEL_SIZE="2"
RAG_TOP_K="3"
RAG_MAX_CHARS_PER_CHUNK="3000"
RLM_MAX_DEPTH=""
RLM_MAX_ITERATIONS="4"
RLM_MAX_TIMEOUT=""

if [[ -z "${METHOD_COMMAND:-}" ]]; then
  echo "Set METHOD_COMMAND before submitting this job." >&2
  echo "Example:" >&2
  echo "  METHOD_COMMAND='/scratch/general/vast/u1406806/vtm/venvs/livecodebench/bin/python /uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/scripts/livecodebench_rag_driver.py --provider rag --run-id lcb_qwen_qwen2_5_coder_32b_instruct_rag_rlmrag_stabletest_20260418c12h_a --model Qwen/Qwen2.5-Coder-32B-Instruct --scenario codegeneration --n 1 --temperature 0.2 --evaluate false --tensor-parallel-size 2 --rag-top-k 3 --rag-max-chars-per-chunk 3000' sbatch \"/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_rlmrag_stabletest_20260418c12h/01_rag.sbatch\"" >&2
  exit 2
fi

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root does not exist on this node: $PROJECT_ROOT" >&2
  exit 1
fi

vtm_chpc_print_summary

if [[ "false" == "true" ]]; then
  wait_for_rlm_serve
fi

cd "$PROJECT_ROOT"
echo "[chpc] Running ${RUN_ID} (${PROVIDER})"
echo "[chpc] METHOD_COMMAND=${METHOD_COMMAND}"
eval "${METHOD_COMMAND}"
