#!/usr/bin/env bash
# Watches the serve job. When it dies, submits a new serve and resubmits all
# incomplete rlm/rlm_rag _d chunks (they resume from rlm_progress.jsonl).
# Run in background: nohup bash watch_and_cycle_serve.sh >> logs/slurm/serve_cycle.log 2>&1 &
set -euo pipefail

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm"
PYTHON="/scratch/general/vast/u1406806/vtm/venvs/livecodebench/bin/python"
SERVE_LAUNCHER="${PROJECT_ROOT}/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_20260413_rlmfix_serve"
SERVE_STATUS_FILE="${SERVE_LAUNCHER}/serve_status.txt"
SERVE_ENDPOINT_FILE="${SERVE_LAUNCHER}/serve_endpoint.txt"
SERVE_SBATCH="${SERVE_LAUNCHER}/99_serve.sbatch"
LOG_DIR="${PROJECT_ROOT}/logs/slurm"
RAW_DIR="${PROJECT_ROOT}/results/raw/livecodebench"
LAUNCHER_BASE="${PROJECT_ROOT}/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_20260413_rlmfix_c"

CHUNKS=(0 106 212 318 424 530 636 742 848 954)
SERVE_SBATCH_ARGS=(-A soc-gpu-np -p soc-gpu-np --time 720 --nodes 1 --cpus-per-task 8 --mem 64G --gres gpu:2)
WORKER_SBATCH_ARGS=(-A soc-gpu-np -p soc-gpu-np --time 360 --nodes 1 --cpus-per-task 8 --mem 64G)

EXPORT_SPEC="ALL,VTM_RLM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_RLM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE},VTM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE}"

log() { echo "[cycle $(date '+%H:%M:%S')] $*" >&2; }

chunk_is_complete() {
  local start="$1" prov="$2"
  local run_dir="${RAW_DIR}/lcb_qwen_qwen2_5_coder_32b_instruct_${prov}_20260413_rlmfix_c${start}_d_a"
  # Complete = output_files.txt exists AND the first file it references exists on disk
  # AND the output JSON has >= 80% non-empty code_list entries (guard against cascade-empty runs).
  local manifest="${run_dir}/output_files.txt"
  if [[ -f "$manifest" ]]; then
    local first_output
    first_output=$(head -1 "$manifest" | tr -d '\r')
    if [[ -n "$first_output" && -f "$first_output" ]]; then
      # Quality check: count total vs non-empty code_list entries
      if python3 - "$first_output" <<'PYEOF'
import json, sys
path = sys.argv[1]
try:
    data = json.loads(open(path).read())
except Exception:
    sys.exit(1)  # can't parse → not complete
if not isinstance(data, list) or len(data) == 0:
    sys.exit(1)
total = len(data)
non_empty = sum(
    1 for entry in data
    if any(c.strip() for c in (entry.get("code_list") or []))
)
# Accept if >= 80% have code
sys.exit(0 if non_empty >= max(1, total * 0.8) else 1)
PYEOF
      then
        return 0  # genuinely complete with good outputs
      else
        return 1  # output file exists but mostly empty — needs re-run
      fi
    fi
  fi
  # Or all 106 lines checkpointed (job finished but manifest not yet written)
  local cnt=0
  if [[ -f "${run_dir}/rlm_progress.jsonl" ]]; then
    cnt=$(wc -l < "${run_dir}/rlm_progress.jsonl" 2>/dev/null || true)
    cnt=${cnt:-0}
  fi
  [[ "${cnt}" -ge 106 ]]
}

count_incomplete() {
  local n=0
  for start in "${CHUNKS[@]}"; do
    for prov in rlm rlm_rag; do
      if ! chunk_is_complete "$start" "$prov" 2>/dev/null; then
        n=$((n + 1))
      fi
    done
  done
  echo "$n"
}

all_complete() {
  local n
  n=$(count_incomplete)
  [[ "$n" -eq 0 ]]
}

wait_for_serve() {
  local deadline=$((SECONDS + 7200))
  while (( SECONDS <= deadline )); do
    if [[ -f "$SERVE_STATUS_FILE" ]]; then
      local state endpoint
      state=$(sed -n 's/^state=//p' "$SERVE_STATUS_FILE" | head -1 | tr -d '\r')
      if [[ "$state" == "running" && -f "$SERVE_ENDPOINT_FILE" ]]; then
        endpoint=$(cat "$SERVE_ENDPOINT_FILE" | tr -d '\r')
        if [[ -n "$endpoint" ]]; then
          log "Serve ready at: $endpoint"
          echo "$endpoint"
          return 0
        fi
      elif [[ "$state" == "failed" || "$state" == "error" || "$state" == "cancelled" || "$state" == "exited" ]]; then
        log "ERROR: serve_status.txt shows state=$state"
        return 1
      fi
    fi
    sleep 15
  done
  log "ERROR: Timed out waiting for serve."
  return 1
}

submit_serve() {
  log "Submitting new serve job..."
  # Reset status file so the new serve can write cleanly
  printf 'state=pending\njob_id=pending\n' > "$SERVE_STATUS_FILE"
  > "$SERVE_ENDPOINT_FILE"
  local jid
  jid=$(sbatch "${SERVE_SBATCH_ARGS[@]}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    "$SERVE_SBATCH" 2>/dev/null | grep -oP '\d+')
  log "New serve job: $jid"
  echo "$jid"
}

submit_incomplete_workers() {
  local endpoint="$1"
  local COMMON_ARGS="--model Qwen/Qwen2.5-Coder-32B-Instruct --scenario codegeneration --n 1 --temperature 0.2 --evaluate false --tensor-parallel-size 2 --rlm-backend vllm --rlm-backend-url ${endpoint} --rlm-max-iterations 12 --rlm-max-timeout 21300"
  local submitted=0 skipped=0
  for start in "${CHUNKS[@]}"; do
    local end=$((start + 106))
    local launcher_dir="${LAUNCHER_BASE}${start}_d"
    for prov in rlm rlm_rag; do
      if chunk_is_complete "$start" "$prov"; then
        log "SKIP ${prov} c${start} (already complete)"
        skipped=$((skipped + 1))
        continue
      fi
      local script run_id driver
      if [[ "$prov" == "rlm" ]]; then
        script="${launcher_dir}/02_rlm.sbatch"
        driver="${PROJECT_ROOT}/scripts/livecodebench_rlm_driver.py"
      else
        script="${launcher_dir}/03_rlm_rag.sbatch"
        driver="${PROJECT_ROOT}/scripts/livecodebench_rlm_rag_driver.py"
      fi
      run_id="lcb_qwen_qwen2_5_coder_32b_instruct_${prov}_20260413_rlmfix_c${start}_d_a"
      local method_cmd="${PYTHON} ${driver} --provider ${prov} --run-id ${run_id} ${COMMON_ARGS} --start-index ${start} --end-index ${end}"
      local jid
      jid=$(METHOD_COMMAND="${method_cmd}" sbatch "${WORKER_SBATCH_ARGS[@]}" \
        --output="${LOG_DIR}/%x-%j.out" \
        --error="${LOG_DIR}/%x-%j.err" \
        --export="${EXPORT_SPEC}" \
        "$script" 2>/dev/null | grep -oP '\d+')
      log "Submitted ${prov} c${start}: ${run_id} → job ${jid}"
      submitted=$((submitted + 1))
    done
  done
  log "Workers: ${submitted} submitted, ${skipped} skipped (complete)"
}

# ---------------------------------------------------------------------------
# Main loop — keep cycling until all chunks are done
# ---------------------------------------------------------------------------
CYCLE=0
while true; do
  CYCLE=$((CYCLE + 1))
  log "=== Cycle ${CYCLE} ==="

  if all_complete; then
    log "All chunks complete! Done."
    break
  fi

  # Wait for current serve job to finish (or detect it's gone)
  CURRENT_SERVE_JID=$(sed -n 's/^job_id=//p' "$SERVE_STATUS_FILE" 2>/dev/null | head -1 | tr -d '\r')
  if [[ -n "$CURRENT_SERVE_JID" && "$CURRENT_SERVE_JID" != "pending" ]]; then
    log "Waiting for serve job ${CURRENT_SERVE_JID} to finish..."
    # CHPC Slurm returns exit 0 even for completed/gone jobs, so check output presence
    while [[ -n "$(squeue -j "$CURRENT_SERVE_JID" -h 2>/dev/null)" ]]; do
      sleep 60
    done
    log "Serve job ${CURRENT_SERVE_JID} is gone."
    sleep 10  # allow shutdown handlers to write the status file
  fi

  if all_complete; then
    log "All chunks complete after serve ended! Done."
    break
  fi

  # Cancel any stale workers (they will have failed when serve died)
  log "Cancelling any remaining rlm/rlm_rag jobs..."
  scancel --user u1406806 --name qwen_qwen2_5_coder_32b_instruct_rlm 2>/dev/null || true
  scancel --user u1406806 --name qwen_qwen2_5_coder_32b_instruct_rlm_rag 2>/dev/null || true
  sleep 5

  # Start new serve (retry up to 3 times on failure)
  ENDPOINT=""
  for _retry in 1 2 3; do
    submit_serve
    ENDPOINT=$(wait_for_serve) && break
    log "WARNING: serve failed to come up (attempt ${_retry}/3). Retrying..."
    sleep 30
  done
  if [[ -z "$ENDPOINT" ]]; then
    log "ERROR: serve failed 3 times. Exiting."; exit 1
  fi

  # Submit all incomplete workers
  submit_incomplete_workers "$ENDPOINT"

  log "Cycle ${CYCLE} complete. Sleeping before monitoring next serve..."
  sleep 30
done
