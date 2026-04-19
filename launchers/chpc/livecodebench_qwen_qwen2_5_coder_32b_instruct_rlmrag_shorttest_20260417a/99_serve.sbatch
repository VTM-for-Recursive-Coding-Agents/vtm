#!/usr/bin/env bash
#SBATCH --job-name=qwen_qwen2_5_coder_32b_instruct_serve
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

export PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm"
export STORAGE_ROOT="/scratch/general/vast/u1406806/vtm"
export VTM_SERVE_STATUS_FILE="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_rlmrag_shorttest_20260417a/serve_status.txt"
export VTM_SERVE_ENDPOINT_FILE="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_rlmrag_shorttest_20260417a/serve_endpoint.txt"
export VTM_RLM_SERVE_STATUS_FILE="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_rlmrag_shorttest_20260417a/serve_status.txt"
export VTM_RLM_SERVE_ENDPOINT_FILE="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_rlmrag_shorttest_20260417a/serve_endpoint.txt"
export VTM_CHPC_PYTHON_BIN="/scratch/general/vast/u1406806/vtm/venvs/livecodebench/bin/python"
export SERVED_MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export SERVED_MODEL_ALIAS="Qwen/Qwen2.5-Coder-32B-Instruct"
export VLLM_TENSOR_PARALLEL_SIZE="2"
if [[ -n "28672" ]]; then
export VLLM_MAX_MODEL_LEN="28672"
fi
export VLLM_GPU_MEMORY_UTILIZATION="0.9"

exec "/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm/scripts/slurm/serve_model.sh"
