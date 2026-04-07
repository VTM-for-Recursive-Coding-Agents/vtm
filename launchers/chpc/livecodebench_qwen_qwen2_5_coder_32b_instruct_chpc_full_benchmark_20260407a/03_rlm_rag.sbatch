#!/usr/bin/env bash
#SBATCH --job-name=qwen_qwen2_5_coder_32b_instruct_rlm_rag
set -euo pipefail

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm"
STORAGE_ROOT="/scratch/general/vast/u1406806/vtm"
PYTHON_MODULE=""
CUDA_MODULE=""
EXTRA_MODULES=""
SKIP_MODULES="false"

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

vtm_chpc_setup_environment "/scratch/general/vast/u1406806/vtm"
vtm_chpc_load_requested_modules

RUN_ID="lcb_qwen_qwen2_5_coder_32b_instruct_rlm_rag_chpc_full_benchmark_20260407a_a"
PROVIDER="rlm_rag"
MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
LM_STUDIO_MODEL_ID=""
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"
EVALUATE="false"
MAX_TOKENS=""
MAX_INSTANCES="1"
RAG_TOP_K=""
RAG_MAX_CHARS_PER_CHUNK=""
RLM_MAX_DEPTH=""
RLM_MAX_ITERATIONS="2"
RLM_MAX_TIMEOUT=""

if [[ -z "${METHOD_COMMAND:-}" ]]; then
  echo "Set METHOD_COMMAND before submitting this job." >&2
  echo "Example:" >&2
  echo "  METHOD_COMMAND='/scratch/general/vast/u1406806/vtm/venvs/livecodebench/bin/python /uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/scripts/livecodebench_rlm_rag_driver.py --provider rlm_rag --run-id lcb_qwen_qwen2_5_coder_32b_instruct_rlm_rag_chpc_full_benchmark_20260407a_a --model Qwen/Qwen2.5-Coder-32B-Instruct --scenario codegeneration --n 1 --temperature 0.2 --evaluate false --max-instances 1 --rlm-max-iterations 2' sbatch \"/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_chpc_full_benchmark_20260407a/03_rlm_rag.sbatch\"" >&2
  exit 2
fi

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root does not exist on this node: $PROJECT_ROOT" >&2
  exit 1
fi

vtm_chpc_print_summary

cd "$PROJECT_ROOT"
echo "[chpc] Running ${RUN_ID} (${PROVIDER})"
echo "[chpc] METHOD_COMMAND=${METHOD_COMMAND}"
eval "${METHOD_COMMAND}"
