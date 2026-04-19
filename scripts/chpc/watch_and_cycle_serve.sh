#!/usr/bin/env bash
# Watches the serve job. When it dies, submits a new serve and resubmits all
# incomplete rlm/rlm_rag _d chunks (they resume from rlm_progress.jsonl).
# Run in background: nohup bash watch_and_cycle_serve.sh >> logs/slurm/serve_cycle.log 2>&1 &
set -euo pipefail

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1406806/Multimodal-Class/vtm"
PYTHON="/scratch/general/vast/u1406806/vtm/venvs/livecodebench/bin/python"
SERVE_LAUNCHER="${PROJECT_ROOT}/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_20260413_rlmfix_serve"
# Canonical serve state is written under logs/slurm for serve/watcher/worker consistency.
SERVE_STATUS_FILE="${PROJECT_ROOT}/logs/slurm/serve_status.txt"
SERVE_ENDPOINT_FILE="${PROJECT_ROOT}/logs/slurm/serve_node.txt"
SERVE_STATUS_FALLBACK_FILE="${SERVE_LAUNCHER}/serve_status.txt"
SERVE_ENDPOINT_FALLBACK_FILE="${SERVE_LAUNCHER}/serve_endpoint.txt"
SERVE_SBATCH="${SERVE_LAUNCHER}/99_serve.sbatch"
LOG_DIR="${PROJECT_ROOT}/logs/slurm"
RAW_DIR="${PROJECT_ROOT}/results/raw/livecodebench"
LAUNCHER_BASE="${PROJECT_ROOT}/launchers/chpc/livecodebench_qwen_qwen2_5_coder_32b_instruct_20260413_rlmfix_c"
WATCH_LOCK_FILE="${LOG_DIR}/watch_and_cycle_serve.lock"
WATCH_PID_FILE="${LOG_DIR}/watch_and_cycle_serve.pid"
ACTIVE_WORKER_JOBS_FILE="${LOG_DIR}/serve_cycle_active_worker_jobs.txt"

CHUNKS=(0 106 212 318 424 530 636 742 848 954)
SERVE_SBATCH_ARGS=(-A soc-gpu-np -p soc-gpu-np --time 720 --nodes 1 --cpus-per-task 8 --mem 64G --gres gpu:2)
WORKER_SBATCH_ARGS=(-A soc-gpu-np -p soc-gpu-np --time 360 --nodes 1 --cpus-per-task 8 --mem 64G)

EXPORT_SPEC="ALL,VTM_RLM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_RLM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE},VTM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE}"

log() { echo "[cycle $(date '+%H:%M:%S')] $*" >&2; }

mkdir -p "$LOG_DIR"
exec 9>"$WATCH_LOCK_FILE"
if ! flock -n 9; then
  log "Another watcher is already running. Exiting."
  exit 0
fi
echo "$$" > "$WATCH_PID_FILE"

cleanup_watcher_state() {
  rm -f "$WATCH_PID_FILE"
}

trap cleanup_watcher_state EXIT

write_serve_status() {
  local state="$1"
  local job_id="${2:-}"
  local endpoint="${3:-}"
  local tmp="${SERVE_STATUS_FILE}.tmp.$$"

  {
    printf 'state=%s\n' "$state"
    if [[ -n "$job_id" ]]; then
      printf 'job_id=%s\n' "$job_id"
    fi
    printf 'endpoint_file=%s\n' "$SERVE_ENDPOINT_FILE"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ -n "$endpoint" ]]; then
      printf 'endpoint=%s\n' "$endpoint"
    fi
  } > "$tmp"

  mv "$tmp" "$SERVE_STATUS_FILE"

  if [[ -n "$endpoint" ]]; then
    printf '%s\n' "$endpoint" > "${SERVE_ENDPOINT_FILE}.tmp.$$"
    mv "${SERVE_ENDPOINT_FILE}.tmp.$$" "$SERVE_ENDPOINT_FILE"
  fi
}

endpoint_is_healthy() {
  local endpoint="$1"
  python3 - "$endpoint" <<'PYEOF'
import sys
import urllib.request

endpoint = sys.argv[1].rstrip('/')
url = f"{endpoint}/models"
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

sync_status_from_fallback() {
  if [[ ! -f "$SERVE_STATUS_FALLBACK_FILE" ]]; then
    return
  fi

  local fb_state fb_job fb_endpoint_file fb_endpoint
  fb_state=$(sed -n 's/^state=//p' "$SERVE_STATUS_FALLBACK_FILE" | head -1 | tr -d '\r')
  fb_job=$(sed -n 's/^job_id=//p' "$SERVE_STATUS_FALLBACK_FILE" | head -1 | tr -d '\r')
  fb_endpoint_file=$(sed -n 's/^endpoint_file=//p' "$SERVE_STATUS_FALLBACK_FILE" | head -1 | tr -d '\r')

  if [[ -z "$fb_endpoint_file" ]]; then
    fb_endpoint_file="$SERVE_ENDPOINT_FALLBACK_FILE"
  fi

  fb_endpoint=""
  if [[ -f "$fb_endpoint_file" ]]; then
    fb_endpoint=$(sed -n '1p' "$fb_endpoint_file" | tr -d '\r')
  fi
  if [[ -z "$fb_endpoint" ]]; then
    fb_endpoint=$(sed -n 's/^endpoint=//p' "$SERVE_STATUS_FALLBACK_FILE" | head -1 | tr -d '\r')
  fi

  if [[ "$fb_state" == "running" && -n "$fb_endpoint" && -n "$fb_job" ]]; then
    write_serve_status "running" "$fb_job" "$fb_endpoint"
    return
  fi

  if [[ "$fb_state" == "failed" || "$fb_state" == "error" || "$fb_state" == "cancelled" || "$fb_state" == "exited" ]]; then
    write_serve_status "$fb_state" "$fb_job" ""
  fi
}

cancel_active_workers() {
  if [[ ! -f "$ACTIVE_WORKER_JOBS_FILE" ]]; then
    log "No tracked worker jobs to cancel."
    return
  fi

  mapfile -t tracked_ids < <(grep -E '^[0-9]+$' "$ACTIVE_WORKER_JOBS_FILE" || true)
  if [[ "${#tracked_ids[@]}" -eq 0 ]]; then
    log "Tracked worker job list is empty."
    return
  fi

  log "Cancelling ${#tracked_ids[@]} tracked worker job(s)."
  scancel "${tracked_ids[@]}" 2>/dev/null || true
  : > "$ACTIVE_WORKER_JOBS_FILE"
}

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
    sync_status_from_fallback

    if [[ -f "$SERVE_STATUS_FILE" ]]; then
      local state endpoint_file endpoint
      state=$(sed -n 's/^state=//p' "$SERVE_STATUS_FILE" | head -1 | tr -d '\r')

      endpoint_file=$(sed -n 's/^endpoint_file=//p' "$SERVE_STATUS_FILE" | head -1 | tr -d '\r')
      if [[ -z "$endpoint_file" ]]; then
        endpoint_file="$SERVE_ENDPOINT_FILE"
      fi

      if [[ "$state" == "running" && -f "$endpoint_file" ]]; then
        endpoint=$(cat "$endpoint_file" | tr -d '\r')
        if [[ -n "$endpoint" ]]; then
          if endpoint_is_healthy "$endpoint"; then
            log "Serve ready at: $endpoint"
            echo "$endpoint"
            return 0
          fi
          log "Serve endpoint present but unhealthy: $endpoint"
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
  # Mark pending but keep the previous endpoint file until the new serve is healthy.
  # This avoids a race where workers observe a truncated endpoint file.
  write_serve_status "pending" "pending" ""
  local jid
  local serve_export_spec
  serve_export_spec="ALL,VTM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE},VTM_RLM_SERVE_STATUS_FILE=${SERVE_STATUS_FILE},VTM_RLM_SERVE_ENDPOINT_FILE=${SERVE_ENDPOINT_FILE}"
  jid=$(sbatch "${SERVE_SBATCH_ARGS[@]}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export="${serve_export_spec}" \
    "$SERVE_SBATCH" 2>/dev/null | grep -oP '\d+')
  write_serve_status "pending" "$jid" ""
  log "New serve job: $jid"
  echo "$jid"
}

submit_incomplete_workers() {
  local endpoint="$1"
  local COMMON_ARGS="--model Qwen/Qwen2.5-Coder-32B-Instruct --scenario codegeneration --n 1 --temperature 0.2 --evaluate false --tensor-parallel-size 2 --rlm-backend vllm --rlm-backend-url ${endpoint} --rlm-max-iterations 12 --rlm-max-timeout 21300"
  local submitted=0 skipped=0
  local submitted_ids=()
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
      submitted_ids+=("$jid")
      submitted=$((submitted + 1))
    done
  done

  if [[ "${#submitted_ids[@]}" -gt 0 ]]; then
    printf '%s\n' "${submitted_ids[@]}" > "$ACTIVE_WORKER_JOBS_FILE"
  else
    : > "$ACTIVE_WORKER_JOBS_FILE"
  fi

  log "Workers: ${submitted} submitted, ${skipped} skipped (complete)"
}

# ---------------------------------------------------------------------------
# Main loop — keep cycling until all chunks are done
# ---------------------------------------------------------------------------
CYCLE=0
while true; do
  CYCLE=$((CYCLE + 1))
  log "=== Cycle ${CYCLE} ==="

  sync_status_from_fallback

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

  # Cancel only workers tracked by this watcher to avoid killing unrelated runs.
  cancel_active_workers
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
