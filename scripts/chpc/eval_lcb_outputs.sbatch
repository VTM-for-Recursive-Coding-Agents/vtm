#!/usr/bin/env bash
#SBATCH --job-name=eval_lcb_outputs
set -euo pipefail

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm"
source "$PROJECT_ROOT/scripts/chpc/chpc_env.sh"

vtm_chpc_setup_environment "/scratch/general/vast/u1406806/vtm"
vtm_chpc_load_requested_modules

cd "$PROJECT_ROOT/benchmarks/LiveCodeBench"

"$PROJECT_ROOT/benchmarks/LiveCodeBench/.venv-chpc/bin/python" -m lcb_runner.runner.custom_evaluator \
  --scenario codegeneration \
  --release_version release_latest \
  --n 1 --temperature 0.2 \
  --num_process_evaluate 12 --timeout 6 \
  --custom_output_file "${LCB_CUSTOM_OUTPUT_FILE:?LCB_CUSTOM_OUTPUT_FILE is required}"
