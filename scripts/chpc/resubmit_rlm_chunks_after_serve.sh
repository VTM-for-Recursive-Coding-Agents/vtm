#!/usr/bin/env bash
# Resubmit all 10 rlm + 10 rlm_rag _d chunk jobs after a serve restart.
# Usage: bash resubmit_rlm_chunks_after_serve.sh
# Waits for the serve_status.txt to show state=running, then submits all 20 jobs.
set -euo pipefail

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm"
PYTHON="/scratch/general/vast/u1406806/vtm/venvs/livecodebench/bin/python"
SERVE_STATUS_FILE="${PROJECT_ROOT}/logs/slurm/serve_status.txt"
SERVE_ENDPOINT_FILE="${PROJECT_ROOT}/logs/slurm/serve_node.txt"
SERVE_LAUNCHER="${PROJECT_ROOT}/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_20260413_rlmfix_serve"
SERVE_STATUS_FALLBACK_FILE="${SERVE_LAUNCHER}/serve_status.txt"
SERVE_ENDPOINT_FALLBACK_FILE="${SERVE_LAUNCHER}/serve_endpoint.txt"
LOG_DIR="${PROJECT_ROOT}/logs/slurm"
RESUBMIT_LOCK_FILE="${LOG_DIR}/resubmit_rlm_chunks.lock"
RESUBMIT_PID_FILE="${LOG_DIR}/resubmit_rlm_chunks.pid"
LAUNCHER_BASE="${PROJECT_ROOT}/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_20260413_rlmfix_c"

CHUNKS=(0 106 212 318 424 530 636 742 848 954)

mkdir -p "$LOG_DIR"
exec 9>"$RESUBMIT_LOCK_FILE"
if ! flock -n 9; then
  echo "[resubmit] Another resubmit process is already running. Exiting."
  exit 0
fi
echo "$$" > "$RESUBMIT_PID_FILE"

cleanup_resubmit_state() {
  rm -f "$RESUBMIT_PID_FILE"
}

trap cleanup_resubmit_state EXIT

# ---------------------------------------------------------------------------
# Wait for serve to be running
# ---------------------------------------------------------------------------
echo "[resubmit] Waiting for serve to report state=running in:"
echo "           $SERVE_STATUS_FILE"

deadline=$((SECONDS + 7200))  # wait up to 2 hours
while (( SECONDS <= deadline )); do
  status_source="$SERVE_STATUS_FILE"
  if [[ ! -f "$status_source" && -f "$SERVE_STATUS_FALLBACK_FILE" ]]; then
    status_source="$SERVE_STATUS_FALLBACK_FILE"
  elif [[ -f "$status_source" ]]; then
    state_hint=$(sed -n 's/^state=//p' "$status_source" | head -1 | tr -d '\r')
    if [[ "$state_hint" == "pending" && -f "$SERVE_STATUS_FALLBACK_FILE" ]]; then
      fb_state_hint=$(sed -n 's/^state=//p' "$SERVE_STATUS_FALLBACK_FILE" | head -1 | tr -d '\r')
      if [[ "$fb_state_hint" == "running" ]]; then
        status_source="$SERVE_STATUS_FALLBACK_FILE"
      fi
    fi
  fi

  if [[ -f "$status_source" ]]; then
    state=$(sed -n 's/^state=//p' "$status_source" | head -1 | tr -d '\r')
    if [[ "$state" == "running" ]]; then
      endpoint_file=$(sed -n 's/^endpoint_file=//p' "$status_source" | head -1 | tr -d '\r')
      if [[ -z "${endpoint_file:-}" ]]; then
        if [[ "$status_source" == "$SERVE_STATUS_FALLBACK_FILE" ]]; then
          endpoint_file="$SERVE_ENDPOINT_FALLBACK_FILE"
        else
          endpoint_file="$SERVE_ENDPOINT_FILE"
        fi
      fi

      endpoint=""
      if [[ -f "$endpoint_file" ]]; then
        endpoint=$(sed -n '1p' "$endpoint_file" | tr -d '\r')
      fi

      if [[ -z "$endpoint" ]]; then
        endpoint=$(sed -n 's/^endpoint=//p' "$status_source" | head -1 | tr -d '\r')
      fi

      echo "[resubmit] Serve is running at: $endpoint"
      break
    elif [[ "$state" == "failed" || "$state" == "error" || "$state" == "exited" || "$state" == "cancelled" ]]; then
      echo "[resubmit] ERROR: serve_status.txt reports state=$state. Aborting." >&2
      exit 1
    fi
    echo "[resubmit] Serve state=$state — waiting..."
  else
    echo "[resubmit] serve_status.txt not yet present — waiting..."
  fi
  sleep 10
done

if (( SECONDS > deadline )); then
  echo "[resubmit] ERROR: Timed out waiting for serve." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build SBATCH --export spec (mirrors submit_livecodebench_local_model.sh)
# ---------------------------------------------------------------------------
EXPORT_SPEC="ALL,VTM_RLM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_RLM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE},VTM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE}"

# Common args (URL filled in from serve_status.txt)
COMMON_ARGS=(
  "--model" "Qwen/Qwen2.5-Coder-32B-Instruct"
  "--scenario" "codegeneration"
  "--n" "1"
  "--temperature" "0.2"
  "--evaluate" "false"
  "--tensor-parallel-size" "2"
  "--rlm-backend" "vllm"
  "--rlm-backend-url" "${endpoint}"
  "--rlm-max-iterations" "12"
  "--rlm-max-timeout" "21300"
)

# ---------------------------------------------------------------------------
# Submit all 20 chunk jobs
# ---------------------------------------------------------------------------
submitted=0
for start in "${CHUNKS[@]}"; do
  end=$((start + 106))
  launcher_dir="${LAUNCHER_BASE}${start}_d"

  # --- rlm ---
  run_id="lcb_qwen_qwen2_5_coder_32b_instruct_rlm_20260413_rlmfix_c${start}_d_a"
  method_cmd="${PYTHON} ${PROJECT_ROOT}/scripts/livecodebench_rlm_driver.py \
--provider rlm --run-id ${run_id} \
${COMMON_ARGS[*]} \
--start-index ${start} --end-index ${end}"

  jid=$(METHOD_COMMAND="${method_cmd}" sbatch \
    -A soc-gpu-np -p soc-gpu-np \
    --time 360 --nodes 1 --cpus-per-task 8 --mem 64G \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export="${EXPORT_SPEC}" \
    "${launcher_dir}/02_rlm.sbatch" \
    | grep -oP '\d+')
  echo "[resubmit] Submitted rlm   c${start}: ${run_id} → job ${jid}"
  (( submitted++ ))

  # --- rlm_rag ---
  run_id="lcb_qwen_qwen2_5_coder_32b_instruct_rlm_rag_20260413_rlmfix_c${start}_d_a"
  method_cmd="${PYTHON} ${PROJECT_ROOT}/scripts/livecodebench_rlm_rag_driver.py \
--provider rlm_rag --run-id ${run_id} \
${COMMON_ARGS[*]} \
--start-index ${start} --end-index ${end}"

  jid=$(METHOD_COMMAND="${method_cmd}" sbatch \
    -A soc-gpu-np -p soc-gpu-np \
    --time 360 --nodes 1 --cpus-per-task 8 --mem 64G \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export="${EXPORT_SPEC}" \
    "${launcher_dir}/03_rlm_rag.sbatch" \
    | grep -oP '\d+')
  echo "[resubmit] Submitted rlm_rag c${start}: ${run_id} → job ${jid}"
  (( submitted++ ))
done

echo "[resubmit] Done. ${submitted}/20 jobs submitted."
