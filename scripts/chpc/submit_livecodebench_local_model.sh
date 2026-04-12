#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/chpc_env.sh"
QUEUE_SCRIPT="$SCRIPT_DIR/queue_livecodebench_methods.sh"
SLURM_LOG_DIR="$PROJECT_ROOT/logs/slurm"
RLM_DIR="$PROJECT_ROOT/rlm"

MODEL=""
MODEL_TAG=""
PYTHON_BIN=""
PYTHON_BIN_EXPLICIT="false"
SCENARIO="codegeneration"
N="1"
TEMPERATURE="0.2"
EVALUATE="false"
MAX_TOKENS=""
MAX_INSTANCES=""
RAG_TOP_K=""
RAG_MAX_CHARS_PER_CHUNK=""
RLM_BACKEND=""
RLM_BACKEND_URL=""
RLM_MAX_DEPTH=""
RLM_MAX_ITERATIONS="12"
RLM_MAX_TIMEOUT=""
TENSOR_PARALLEL_SIZE=""
VLLM_MAX_MODEL_LEN=""
VLLM_GPU_MEMORY_UTILIZATION=""
QUEUE_TAG="$(date +%Y%m%d)"
QUEUE_DIR=""

ACCOUNT=""
PARTITION=""
QOS=""
CLUSTER=""
GRES='gpu:2'
GRES_SERVE=""
GRES_BASELINE=""
GRES_RAG=""
GRES_RLM=""
GRES_RLM_RAG=""
# ="gpu:rtxpr6000bl:1"
TIME_LIMIT="06:00:00"
TIME_LIMIT_EXPLICIT="false"
TIME_LIMIT_SERVE=""
TIME_LIMIT_BASELINE=""
TIME_LIMIT_RAG=""
TIME_LIMIT_RLM=""
TIME_LIMIT_RLM_RAG=""
CPUS_PER_TASK="8"
MEMORY="64G"
SBATCH_ARGS=()
DRY_RUN="false"
STORAGE_ROOT=""
PYTHON_MODULE=""
CUDA_MODULE=""
EXTRA_MODULES=""
SKIP_MODULES="false"
SKIP_PROVIDERS=""
REQUIRED_MODULES=(datasets openai torch vllm)
AUTO_SUBMIT_RLM_SERVE="true"
SERVE_STATUS_FILE=""
SERVE_ENDPOINT_FILE=""

usage() {
  cat <<'EOF'
Usage: scripts/chpc/submit_livecodebench_local_model.sh --model <model_name> --account <account> --partition <partition> [options]

Submission options:
  --account <name>               Slurm account for sbatch (required)
  --partition <name>             Slurm partition for sbatch (required)
  --qos <name>                   Slurm qos value (optional)
  --cluster <name>               Slurm cluster name for remote submission (optional)
  --gres <value>                 Default Slurm gres request for direct-inference jobs (default: gpu:2)
  --gres-serve <value>           Override gres for the vLLM serve job
  --gres-baseline <value>        Override gres for baseline only
  --gres-rag <value>             Override gres for rag only
  --gres-rlm <value>             Override gres for rlm only; use 'none' for CPU-only client jobs
  --gres-rlm-rag <value>         Override gres for rlm_rag only; use 'none' for CPU-only client jobs
  --time <HH:MM:SS>              Slurm walltime (default: 06:00:00)
  --time-serve <HH:MM:SS>        Override walltime for the vLLM serve job
  --time-baseline <HH:MM:SS>     Override walltime for baseline only
  --time-rag <HH:MM:SS>          Override walltime for rag only
  --time-rlm <HH:MM:SS>          Override walltime for rlm only
  --time-rlm-rag <HH:MM:SS>      Override walltime for rlm_rag only
  --cpus-per-task <int>          Slurm cpus-per-task (default: 8)
  --mem <value>                  Slurm memory request (default: 64G)
  --sbatch-arg <arg>             Extra sbatch argument (repeatable)
  --storage-root <path>          Scratch-backed root for CHPC envs and caches
  --python-module <name>         Module to load before resolving Python
  --cuda-module <name>           CUDA module to load inside submitted jobs
  --extra-modules <list>         Space-separated extra modules to load in CHPC env
  --skip-modules                 Do not attempt to load environment modules
  --dry-run                      Print derived commands without submitting jobs

Benchmark options:
  --model <name>                 LiveCodeBench model key (required)
  --model-tag <tag>              Short slug for run ids (default: derived from model)
  --python-bin <path>            Python executable relative to repo root or absolute path
  --scenario <name>              Scenario to run (default: codegeneration)
  --n <int>                      Samples per problem (default: 1)
  --temperature <float>          Temperature (default: 0.2)
  --evaluate <true|false>        Run evaluation after generation (default: false)
  --max-tokens <int>             Max output tokens per response (optional)
  --max-instances <int>          Limit instances for smoke runs (default: full dataset)
  --tensor-parallel-size <int>   vLLM tensor parallel size (default: inferred from --gres)
  --vllm-max-model-len <int>     Override local vLLM max model length for CHPC jobs (optional)
  --vllm-gpu-memory-utilization <float>
                                 Override local vLLM gpu_memory_utilization for CHPC jobs (optional)
  --rag-top-k <int>              Retrieval chunk count for rag/rlm_rag (optional)
  --rag-max-chars-per-chunk <n>  Max chars per retrieved chunk for rag/rlm_rag (optional)
  --rlm-backend <name>           RLM backend override (optional)
  --rlm-backend-url <url>        RLM backend base URL (optional; inferred from serve status when available)
  --rlm-max-depth <int>          Max recursion depth for rlm/rlm_rag (optional)
  --rlm-max-iterations <int>     Max iterations for rlm/rlm_rag (default: 12)
  --rlm-max-timeout <seconds>    Max wall-clock time for rlm/rlm_rag (default: inferred from --time)
  --no-auto-rlm-serve            Do not submit a dedicated local vLLM serve job for rlm providers
  --skip-providers <list>        Comma-separated providers to skip e.g. 'baseline,rag'
  --queue-tag <tag>              Queue label/date suffix (default: YYYYMMDD)
  --queue-dir <path>             Output launcher bundle directory
  -h, --help                     Show help

This helper:
  1. Generates a four-provider CHPC launcher bundle.
  2. Builds baseline, rag, rlm, and rlm_rag method commands for a local model.
  3. Submits each job with the requested sbatch resource flags.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --cluster)
      CLUSTER="$2"
      shift 2
      ;;
    --gres)
      GRES="$2"
      shift 2
      ;;
    --gres-serve)
      GRES_SERVE="$2"
      shift 2
      ;;
    --gres-baseline)
      GRES_BASELINE="$2"
      shift 2
      ;;
    --gres-rag)
      GRES_RAG="$2"
      shift 2
      ;;
    --gres-rlm)
      GRES_RLM="$2"
      shift 2
      ;;
    --gres-rlm-rag)
      GRES_RLM_RAG="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      TIME_LIMIT_EXPLICIT="true"
      shift 2
      ;;
    --time-serve)
      TIME_LIMIT_SERVE="$2"
      shift 2
      ;;
    --time-baseline)
      TIME_LIMIT_BASELINE="$2"
      shift 2
      ;;
    --time-rag)
      TIME_LIMIT_RAG="$2"
      shift 2
      ;;
    --time-rlm)
      TIME_LIMIT_RLM="$2"
      shift 2
      ;;
    --time-rlm-rag)
      TIME_LIMIT_RLM_RAG="$2"
      shift 2
      ;;
    --cpus-per-task)
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --sbatch-arg)
      SBATCH_ARGS+=("$2")
      shift 2
      ;;
    --storage-root)
      STORAGE_ROOT="$2"
      shift 2
      ;;
    --python-module)
      PYTHON_MODULE="$2"
      shift 2
      ;;
    --cuda-module)
      CUDA_MODULE="$2"
      shift 2
      ;;
    --extra-modules)
      EXTRA_MODULES="$2"
      shift 2
      ;;
    --skip-modules)
      SKIP_MODULES="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model-tag)
      MODEL_TAG="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      PYTHON_BIN_EXPLICIT="true"
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
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --vllm-max-model-len)
      VLLM_MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --vllm-gpu-memory-utilization)
      VLLM_GPU_MEMORY_UTILIZATION="$2"
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
    --rlm-backend)
      RLM_BACKEND="$2"
      shift 2
      ;;
    --rlm-backend-url)
      RLM_BACKEND_URL="$2"
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
    --no-auto-rlm-serve)
      AUTO_SUBMIT_RLM_SERVE="false"
      shift
      ;;
    --skip-providers)
      SKIP_PROVIDERS="$2"
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

if [[ -z "$MODEL" || -z "$ACCOUNT" || -z "$PARTITION" ]]; then
  echo "--model, --account, and --partition are required" >&2
  usage
  exit 1
fi

if [[ -n "$PYTHON_MODULE" ]]; then
  export VTM_CHPC_PYTHON_MODULE="$PYTHON_MODULE"
fi
if [[ -n "$CUDA_MODULE" ]]; then
  export VTM_CHPC_CUDA_MODULE="$CUDA_MODULE"
fi
if [[ -n "$EXTRA_MODULES" ]]; then
  export VTM_CHPC_EXTRA_MODULES="$EXTRA_MODULES"
fi
if [[ "$SKIP_MODULES" == "true" ]]; then
  export VTM_CHPC_SKIP_MODULES=true
fi

vtm_chpc_setup_environment "$STORAGE_ROOT"
vtm_chpc_load_requested_modules

if [[ -z "$CLUSTER" && "$PARTITION" == *-grn ]]; then
  CLUSTER="granite"
fi

if [[ "$EVALUATE" != "true" && "$EVALUATE" != "false" ]]; then
  echo "--evaluate must be true or false" >&2
  exit 1
fi

if [[ -z "$MODEL_TAG" ]]; then
  MODEL_TAG="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_//; s/_$//')"
fi

if [[ -z "$QUEUE_DIR" ]]; then
  QUEUE_DIR="$PROJECT_ROOT/launchers/chpc/livecodebench_${MODEL_TAG}_${QUEUE_TAG}"
fi

SERVE_STATUS_FILE="$QUEUE_DIR/serve_status.txt"
SERVE_ENDPOINT_FILE="$QUEUE_DIR/serve_endpoint.txt"

resolve_default_python_bin() {
  local candidate
  local resolved_candidate
  local default_python_candidates=(
    "$VTM_CHPC_LCB_VENV_DIR/bin/python"
    "benchmarks/LiveCodeBench/.venv-granite/bin/python"
    "benchmarks/LiveCodeBench/.venv-chpc/bin/python"
    "benchmarks/LiveCodeBench/.venv/bin/python"
  )

  for candidate in "${default_python_candidates[@]}"; do
    resolved_candidate="$candidate"
    if [[ "$resolved_candidate" != /* ]]; then
      resolved_candidate="$PROJECT_ROOT/$resolved_candidate"
    fi
    if [[ -x "$resolved_candidate" ]]; then
      printf '%s\n' "$resolved_candidate"
      return 0
    fi
  done

  return 1
}

resolve_python_bin() {
  if [[ "$PYTHON_BIN_EXPLICIT" == "true" ]]; then
    if [[ "$PYTHON_BIN" != /* ]]; then
      PYTHON_BIN="$PROJECT_ROOT/$PYTHON_BIN"
    fi
    return 0
  fi

  if PYTHON_BIN="$(resolve_default_python_bin)"; then
    return 0
  fi

  PYTHON_BIN="$VTM_CHPC_LCB_VENV_DIR/bin/python"
  return 0
}

resolve_python_bin

parse_gpu_count_from_gres() {
  local gres_value="$1"
  local suffix

  suffix="${gres_value##*:}"
  if [[ "$suffix" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$suffix"
    return 0
  fi

  if [[ "$gres_value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$gres_value"
    return 0
  fi

  return 1
}

normalize_gres_request() {
  local gres_value="${1:-}"
  local lowered="${gres_value,,}"

  case "$lowered" in
    ""|none|cpu|0)
      return 0
      ;;
  esac

  printf '%s\n' "$gres_value"
}

serve_gres() {
  local gres_value="${GRES_SERVE:-$GRES}"
  normalize_gres_request "$gres_value"
}

provider_gres() {
  local provider="$1"
  local gres_value=""

  case "$provider" in
    baseline)
      gres_value="$GRES_BASELINE"
      if [[ -z "$gres_value" ]]; then
        gres_value="$GRES"
      fi
      ;;
    rag)
      gres_value="$GRES_RAG"
      if [[ -z "$gres_value" ]]; then
        gres_value="$GRES"
      fi
      ;;
    rlm)
      gres_value="$GRES_RLM"
      ;;
    rlm_rag)
      gres_value="$GRES_RLM_RAG"
      ;;
    *)
      gres_value="$GRES"
      ;;
  esac

  normalize_gres_request "$gres_value"
}

slurm_time_to_seconds() {
  local raw="$1"
  local days=0
  local clock="$raw"
  local hours minutes seconds

  if [[ "$raw" == *-* ]]; then
    days="${raw%%-*}"
    clock="${raw#*-}"
  fi

  IFS=: read -r hours minutes seconds <<< "$clock"
  if [[ -z "$hours" || -z "$minutes" || -z "$seconds" ]]; then
    return 1
  fi
  if [[ ! "$days" =~ ^[0-9]+$ || ! "$hours" =~ ^[0-9]+$ || ! "$minutes" =~ ^[0-9]+$ || ! "$seconds" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  printf '%s\n' "$((days * 86400 + hours * 3600 + minutes * 60 + seconds))"
}

normalize_openai_base_url() {
  local base_url="$1"
  base_url="${base_url%/}"
  if [[ "$base_url" != */v1 ]]; then
    base_url="$base_url/v1"
  fi
  printf '%s\n' "$base_url"
}

status_file_value() {
  local file_path="$1"
  local key="$2"

  awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, "", $0); print $0; exit }' "$file_path"
}

resolve_default_rlm_backend_url() {
  local status_file="$PROJECT_ROOT/logs/slurm/serve_status.txt"
  local endpoint_file=""
  local endpoint_value=""
  local state=""
  local host=""
  local port=""

  if [[ ! -f "$status_file" ]]; then
    return 1
  fi

  endpoint_file="$(status_file_value "$status_file" endpoint_file)"
  if [[ -n "$endpoint_file" && -f "$endpoint_file" ]]; then
    endpoint_value="$(sed -n '1p' "$endpoint_file" | tr -d '\r')"
    if [[ -n "$endpoint_value" ]]; then
      normalize_openai_base_url "$endpoint_value"
      return 0
    fi
  fi

  state="$(status_file_value "$status_file" state)"
  if [[ "$state" == "failed" || "$state" == "error" || "$state" == "cancelled" ]]; then
    return 1
  fi

  host="$(status_file_value "$status_file" host)"
  port="$(status_file_value "$status_file" port)"
  if [[ -z "$host" || -z "$port" ]]; then
    return 1
  fi

  normalize_openai_base_url "http://$host:$port"
}

provider_time_limit() {
  local provider="$1"

  case "$provider" in
    baseline)
      if [[ -n "$TIME_LIMIT_BASELINE" ]]; then
        printf '%s\n' "$TIME_LIMIT_BASELINE"
        return 0
      fi
      ;;
    rag)
      if [[ -n "$TIME_LIMIT_RAG" ]]; then
        printf '%s\n' "$TIME_LIMIT_RAG"
        return 0
      fi
      ;;
    rlm)
      if [[ -n "$TIME_LIMIT_RLM" ]]; then
        printf '%s\n' "$TIME_LIMIT_RLM"
        return 0
      fi
      ;;
    rlm_rag)
      if [[ -n "$TIME_LIMIT_RLM_RAG" ]]; then
        printf '%s\n' "$TIME_LIMIT_RLM_RAG"
        return 0
      fi
      ;;
  esac

  printf '%s\n' "$TIME_LIMIT"
}

infer_rlm_timeout_from_time() {
  local raw_time="$1"
  local walltime_seconds
  local timeout_seconds

  if ! walltime_seconds="$(slurm_time_to_seconds "$raw_time" 2>/dev/null)"; then
    return 1
  fi

  timeout_seconds="$((walltime_seconds - 300))"
  if (( timeout_seconds < 60 )); then
    timeout_seconds=60
  fi

  printf '%s\n' "$timeout_seconds"
}

max_time_limit() {
  local best_time="$1"
  local candidate="$2"
  local best_seconds
  local candidate_seconds

  if [[ -z "$best_time" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if [[ -z "$candidate" ]]; then
    printf '%s\n' "$best_time"
    return 0
  fi

  best_seconds="$(slurm_time_to_seconds "$best_time")"
  candidate_seconds="$(slurm_time_to_seconds "$candidate")"
  if (( candidate_seconds > best_seconds )); then
    printf '%s\n' "$candidate"
    return 0
  fi

  printf '%s\n' "$best_time"
}

serve_time_limit() {
  local resolved_time="${TIME_LIMIT_SERVE:-}"

  if [[ -n "$resolved_time" ]]; then
    printf '%s\n' "$resolved_time"
    return 0
  fi

  resolved_time="$(provider_time_limit rlm)"
  resolved_time="$(max_time_limit "$resolved_time" "$(provider_time_limit rlm_rag)")"
  printf '%s\n' "$resolved_time"
}

should_submit_provider() {
  local provider="$1"
  # Returns 0 (true) if the provider should be submitted, 1 if it should be skipped.
  if [[ -z "$SKIP_PROVIDERS" ]]; then
    return 0
  fi
  local skip
  IFS=',' read -ra skip <<< "$SKIP_PROVIDERS"
  local p
  for p in "${skip[@]}"; do
    if [[ "${p// /}" == "$provider" ]]; then
      return 1
    fi
  done
  return 0
}

should_auto_submit_rlm_serve() {
  if [[ "$AUTO_SUBMIT_RLM_SERVE" != "true" ]]; then
    return 1
  fi

  if [[ -n "$RLM_BACKEND_URL" ]]; then
    return 1
  fi

  if [[ -z "$RLM_BACKEND" || "$RLM_BACKEND" == "vllm" ]]; then
    return 0
  fi

  return 1
}

effective_tensor_parallel_size() {
  if [[ -n "$TENSOR_PARALLEL_SIZE" ]]; then
    printf '%s\n' "$TENSOR_PARALLEL_SIZE"
    return 0
  fi

  if tensor_parallel_size="$(parse_gpu_count_from_gres "$GRES" 2>/dev/null)"; then
    printf '%s\n' "$tensor_parallel_size"
    return 0
  fi

  printf '1\n'
}

default_vllm_max_model_len() {
  local normalized_model="${MODEL,,}"
  local effective_tensor_parallel="$(effective_tensor_parallel_size)"

  if [[ "$normalized_model" == "qwen/qwen2.5-coder-32b-instruct" && "$effective_tensor_parallel" == "2" ]]; then
    printf '28672\n'
    return 0
  fi

  return 1
}

effective_vllm_max_model_len() {
  if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
    printf '%s\n' "$VLLM_MAX_MODEL_LEN"
    return 0
  fi

  default_vllm_max_model_len
}

effective_vllm_gpu_memory_utilization() {
  if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    printf '%s\n' "$VLLM_GPU_MEMORY_UTILIZATION"
    return 0
  fi

  printf '0.9\n'
}
normalize_submit_configuration() {
  local inferred_tensor_parallel_size=""
  local gpu_count
  local provider
  local provider_time
  local resolved_rlm_backend_url
  local effective_max_model_len=""
  local effective_gpu_memory_utilization=""

  if [[ -n "$MAX_INSTANCES" && ! "$MAX_INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-instances must be a positive integer" >&2
    exit 1
  fi
  if [[ -n "$RLM_MAX_ITERATIONS" && ! "$RLM_MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--rlm-max-iterations must be a positive integer" >&2
    exit 1
  fi
  if [[ -n "$TENSOR_PARALLEL_SIZE" && ! "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "--tensor-parallel-size must be a positive integer" >&2
    exit 1
  fi
  if [[ -n "$RLM_MAX_TIMEOUT" && ! "$RLM_MAX_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--rlm-max-timeout must be a positive integer number of seconds" >&2
    exit 1
  fi
  if [[ -n "$VLLM_MAX_MODEL_LEN" && ! "$VLLM_MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo "--vllm-max-model-len must be a positive integer" >&2
    exit 1
  fi
  if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    if ! [[ "$VLLM_GPU_MEMORY_UTILIZATION" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "--vllm-gpu-memory-utilization must be a positive decimal" >&2
      exit 1
    fi
    if ! awk -v value="$VLLM_GPU_MEMORY_UTILIZATION" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
      echo "--vllm-gpu-memory-utilization must be in the range (0, 1]" >&2
      exit 1
    fi
  fi
  if [[ -n "$RLM_BACKEND" ]]; then
    case "$RLM_BACKEND" in
      openai|portkey|openrouter|vercel|vllm|anthropic|azure_openai|gemini)
        ;;
      *)
        echo "Unsupported --rlm-backend: $RLM_BACKEND" >&2
        exit 1
        ;;
    esac
  fi

  if should_auto_submit_rlm_serve && [[ -z "$RLM_BACKEND" ]]; then
    RLM_BACKEND="vllm"
  fi

  if should_auto_submit_rlm_serve; then
    :
  elif [[ -n "$RLM_BACKEND_URL" ]]; then
    RLM_BACKEND_URL="$(normalize_openai_base_url "$RLM_BACKEND_URL")"
    if [[ -z "$RLM_BACKEND" ]]; then
      RLM_BACKEND="vllm"
    fi
  elif resolved_rlm_backend_url="$(resolve_default_rlm_backend_url 2>/dev/null)"; then
    RLM_BACKEND_URL="$resolved_rlm_backend_url"
    if [[ -z "$RLM_BACKEND" ]]; then
      RLM_BACKEND="vllm"
    fi
    echo "[submit] Inferred RLM backend URL: $RLM_BACKEND_URL" >&2
  fi

  if [[ "$RLM_BACKEND" == "vllm" && -z "$RLM_BACKEND_URL" ]] && ! should_auto_submit_rlm_serve; then
    echo "RLM backend 'vllm' requires a base URL. Pass --rlm-backend-url or restore a healthy serve status file." >&2
    exit 1
  fi

  for provider_time in "$TIME_LIMIT" "$TIME_LIMIT_SERVE" "$TIME_LIMIT_BASELINE" "$TIME_LIMIT_RAG" "$TIME_LIMIT_RLM" "$TIME_LIMIT_RLM_RAG"; do
    if [[ -z "$provider_time" ]]; then
      continue
    fi
    if ! slurm_time_to_seconds "$provider_time" >/dev/null 2>&1; then
      echo "Invalid Slurm time value: $provider_time" >&2
      exit 1
    fi
  done

  if [[ -z "$TENSOR_PARALLEL_SIZE" ]]; then
    if gpu_count="$(parse_gpu_count_from_gres "$GRES" 2>/dev/null)"; then
      TENSOR_PARALLEL_SIZE="$gpu_count"
      inferred_tensor_parallel_size="$gpu_count"
    else
      TENSOR_PARALLEL_SIZE="1"
      inferred_tensor_parallel_size="1"
    fi
  fi

  if [[ "$QUEUE_TAG" == *full_benchmark* && -n "$MAX_INSTANCES" ]]; then
    echo "[submit] Warning: queue tag '$QUEUE_TAG' suggests a full benchmark, but --max-instances=$MAX_INSTANCES limits the run." >&2
  fi

  if [[ "$EVALUATE" == "false" && -n "$MAX_INSTANCES" ]]; then
    echo "[submit] Note: evaluate=false with --max-instances=$MAX_INSTANCES is a smoke-style run, not a full benchmark." >&2
  fi

  if gpu_count="$(parse_gpu_count_from_gres "$GRES" 2>/dev/null)"; then
    if (( TENSOR_PARALLEL_SIZE > gpu_count )); then
      echo "Requested --tensor-parallel-size=$TENSOR_PARALLEL_SIZE exceeds GPU count inferred from --gres=$GRES" >&2
      exit 1
    fi
    if (( gpu_count > 1 && TENSOR_PARALLEL_SIZE == 1 )); then
      echo "[submit] Warning: --gres=$GRES allocates $gpu_count GPUs but tensor parallelism is 1; large models may still OOM on a single GPU." >&2
    fi
  fi

  if [[ -z "$TIME_LIMIT_RLM" && -z "$TIME_LIMIT_RLM_RAG" && "$TIME_LIMIT_EXPLICIT" == "false" ]]; then
    echo "[submit] Note: using the generic default --time=$TIME_LIMIT for all providers; pass --time-rlm and --time-rlm-rag for longer RLM runs." >&2
  fi

  if [[ -z "$RLM_MAX_TIMEOUT" ]]; then
    for provider in rlm rlm_rag; do
      provider_time="$(provider_time_limit "$provider")"
      if provider_timeout="$(infer_rlm_timeout_from_time "$provider_time" 2>/dev/null)"; then
        echo "[submit] Inferred ${provider} timeout: ${provider_timeout}s from --time=$(provider_time_limit "$provider")" >&2
      fi
    done
  fi

  if [[ -n "$inferred_tensor_parallel_size" ]]; then
    echo "[submit] Inferred tensor parallel size: $inferred_tensor_parallel_size" >&2
  fi
  effective_max_model_len="$(effective_vllm_max_model_len 2>/dev/null || true)"
  effective_gpu_memory_utilization="$(effective_vllm_gpu_memory_utilization)"
  if [[ -n "$effective_max_model_len" ]]; then
    echo "[submit] Local vLLM max model len: $effective_max_model_len" >&2
    if [[ -z "$VLLM_MAX_MODEL_LEN" ]]; then
      echo "[submit] Defaulted local vLLM max model len for this model/resource profile." >&2
    fi
  fi
  echo "[submit] Local vLLM gpu_memory_utilization: $effective_gpu_memory_utilization" >&2
  if should_auto_submit_rlm_serve; then
    echo "[submit] Local vLLM serve will be submitted first with queue-scoped status files:" >&2
    echo "[submit]   status: $SERVE_STATUS_FILE" >&2
    echo "[submit]   endpoint: $SERVE_ENDPOINT_FILE" >&2
  fi
  if [[ -z "$GRES_RLM" && -z "$GRES_RLM_RAG" ]]; then
    echo "[submit] Note: rlm and rlm_rag default to CPU-only client jobs; pass --gres-rlm/--gres-rlm-rag to override." >&2
  fi
}

resolve_symlink_target() {
  local symlink_path="$1"
  local target

  target="$(readlink "$symlink_path")"
  if [[ -z "$target" ]]; then
    return 1
  fi

  if [[ "$target" == /* ]]; then
    printf '%s\n' "$target"
    return 0
  fi

  printf '%s/%s\n' "$(cd "$(dirname "$symlink_path")" && pwd)" "$target"
}

validate_python_portability() {
  local python_path="$1"
  local python_dir
  local venv_dir
  local pyvenv_cfg
  local interpreter_path
  local symlink_target

  if [[ -L "$python_path" ]]; then
    symlink_target="$(resolve_symlink_target "$python_path" || true)"
    if [[ -n "$symlink_target" && ! -e "$symlink_target" ]]; then
      echo "Python symlink target does not exist on this machine: $symlink_target" >&2
      return 1
    fi
  fi

  python_dir="$(cd "$(dirname "$python_path")" && pwd)"
  venv_dir="$(cd "$python_dir/.." && pwd)"
  pyvenv_cfg="$venv_dir/pyvenv.cfg"

  if [[ ! -f "$pyvenv_cfg" ]]; then
    return 0
  fi

  interpreter_path="$(awk -F' = ' '/^executable = / { print $2; exit }' "$pyvenv_cfg")"
  if [[ -n "$interpreter_path" && ! -x "$interpreter_path" ]]; then
    echo "Virtualenv base interpreter does not exist on this machine: $interpreter_path" >&2
    echo "Virtualenv config: $pyvenv_cfg" >&2
    return 1
  fi

  return 0
}

normalize_submit_configuration

driver_script_path() {
  local provider="$1"
  printf '%s/scripts/livecodebench_%s_driver.py\n' "$PROJECT_ROOT" "$provider"
}

validate_rlm_import() {
  if [[ ! -f "$RLM_DIR/rlm/__init__.py" ]]; then
    echo "RLM source checkout not found: $RLM_DIR" >&2
    echo "Run scripts/setup_rlm.sh or scripts/chpc/setup_chpc.sh before submitting rlm providers." >&2
    return 1
  fi

  "$PYTHON_BIN" - "$PROJECT_ROOT" <<'PY'
import pathlib
import sys

project_root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(project_root / "rlm"))

from rlm import RLM  # noqa: F401
PY
}

validate_submit_environment() {
  local missing=0
  local provider
  local driver_script
  local missing_modules

  if ! vtm_chpc_require_writable_dir "$VTM_CHPC_ROOT"; then
    missing=1
  fi
  if ! vtm_chpc_require_writable_dir "$VTM_CHPC_CACHE_ROOT"; then
    missing=1
  fi
  if ! vtm_chpc_require_writable_dir "$VTM_CHPC_TMP_ROOT"; then
    missing=1
  fi

  if [[ ! -x "$QUEUE_SCRIPT" ]]; then
    echo "Queue helper not found or not executable: $QUEUE_SCRIPT" >&2
    missing=1
  fi

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    missing=1
  elif ! validate_python_portability "$PYTHON_BIN"; then
    missing=1
  elif ! vtm_chpc_python_version_ok "$PYTHON_BIN" 3 10; then
    echo "Python executable must be >= 3.10 for LiveCodeBench: $PYTHON_BIN" >&2
    missing=1
  else
    missing_modules="$(vtm_chpc_missing_python_modules "$PYTHON_BIN" "${REQUIRED_MODULES[@]}")"
    if [[ -n "$missing_modules" ]]; then
      echo "Python environment is missing required runtime modules: $PYTHON_BIN" >&2
      while IFS= read -r module_name; do
        [[ -n "$module_name" ]] && echo "  - $module_name" >&2
      done <<< "$missing_modules"
      echo "Run scripts/chpc/setup_chpc.sh --force-install or point --python-bin at a fully provisioned environment." >&2
      missing=1
    fi
  fi

  for provider in baseline rag rlm rlm_rag; do
    driver_script="$(driver_script_path "$provider")"
    if [[ ! -f "$driver_script" ]]; then
      echo "Driver script not found: $driver_script" >&2
      missing=1
    fi
  done

  if ! validate_rlm_import; then
    missing=1
  fi

  return "$missing"
}

build_queue_args() {
  local effective_tensor_parallel
  local args=(
    --model "$MODEL"
    --model-tag "$MODEL_TAG"
    --scenario "$SCENARIO"
    --n "$N"
    --temperature "$TEMPERATURE"
    --evaluate "$EVALUATE"
    --queue-tag "$QUEUE_TAG"
    --queue-dir "$QUEUE_DIR"
  )

  if [[ -n "$STORAGE_ROOT" ]]; then
    args+=(--storage-root "$STORAGE_ROOT")
  fi
  if [[ -n "$PYTHON_MODULE" ]]; then
    args+=(--python-module "$PYTHON_MODULE")
  fi
  if [[ -n "$CUDA_MODULE" ]]; then
    args+=(--cuda-module "$CUDA_MODULE")
  fi
  if [[ -n "$EXTRA_MODULES" ]]; then
    args+=(--extra-modules "$EXTRA_MODULES")
  fi
  if [[ "$SKIP_MODULES" == "true" ]]; then
    args+=(--skip-modules)
  fi

  if [[ -n "$MAX_TOKENS" ]]; then
    args+=(--max-tokens "$MAX_TOKENS")
  fi
  if [[ -n "$MAX_INSTANCES" ]]; then
    args+=(--max-instances "$MAX_INSTANCES")
  fi
  effective_tensor_parallel="$(effective_tensor_parallel_size)"
  if [[ -n "$effective_tensor_parallel" ]]; then
    args+=(--tensor-parallel-size "$effective_tensor_parallel")
  fi
  if [[ -n "$(effective_vllm_max_model_len 2>/dev/null || true)" ]]; then
    args+=(--vllm-max-model-len "$(effective_vllm_max_model_len)")
  fi
  if [[ -n "$(effective_vllm_gpu_memory_utilization)" ]]; then
    args+=(--vllm-gpu-memory-utilization "$(effective_vllm_gpu_memory_utilization)")
  fi
  if [[ -n "$RAG_TOP_K" ]]; then
    args+=(--rag-top-k "$RAG_TOP_K")
  fi
  if [[ -n "$RAG_MAX_CHARS_PER_CHUNK" ]]; then
    args+=(--rag-max-chars-per-chunk "$RAG_MAX_CHARS_PER_CHUNK")
  fi
  if [[ -n "$RLM_MAX_DEPTH" ]]; then
    args+=(--rlm-max-depth "$RLM_MAX_DEPTH")
  fi
  if [[ -n "$RLM_BACKEND" ]]; then
    args+=(--rlm-backend "$RLM_BACKEND")
  fi
  if [[ -n "$RLM_BACKEND_URL" ]]; then
    args+=(--rlm-backend-url "$RLM_BACKEND_URL")
  fi
  if [[ -n "$RLM_MAX_ITERATIONS" ]]; then
    args+=(--rlm-max-iterations "$RLM_MAX_ITERATIONS")
  fi
  if [[ -n "$RLM_MAX_TIMEOUT" ]]; then
    args+=(--rlm-max-timeout "$RLM_MAX_TIMEOUT")
  fi

  printf '%s\n' "${args[@]}"
}

build_method_command() {
  local provider="$1"
  local run_id="lcb_${MODEL_TAG}_${provider}_${QUEUE_TAG}_a"
  local driver_script
  local effective_rlm_timeout="${RLM_MAX_TIMEOUT:-}"
  local effective_tensor_parallel
  driver_script="$(driver_script_path "$provider")"
  local cmd=(
    "$PYTHON_BIN"
    "$driver_script"
    --provider "$provider"
    --run-id "$run_id"
    --model "$MODEL"
    --scenario "$SCENARIO"
    --n "$N"
    --temperature "$TEMPERATURE"
    --evaluate "$EVALUATE"
  )

  if [[ -n "$MAX_TOKENS" ]]; then
    cmd+=(--max-tokens "$MAX_TOKENS")
  fi
  if [[ -n "$MAX_INSTANCES" ]]; then
    cmd+=(--max-instances "$MAX_INSTANCES")
  fi
  effective_tensor_parallel="$(effective_tensor_parallel_size)"
  if [[ -n "$effective_tensor_parallel" ]]; then
    cmd+=(--tensor-parallel-size "$effective_tensor_parallel")
  fi
  if [[ "$provider" == "rag" || "$provider" == "rlm_rag" ]]; then
    if [[ -n "$RAG_TOP_K" ]]; then
      cmd+=(--rag-top-k "$RAG_TOP_K")
    fi
    if [[ -n "$RAG_MAX_CHARS_PER_CHUNK" ]]; then
      cmd+=(--rag-max-chars-per-chunk "$RAG_MAX_CHARS_PER_CHUNK")
    fi
  fi
  if [[ "$provider" == "rlm" || "$provider" == "rlm_rag" ]]; then
    if [[ -z "$effective_rlm_timeout" ]]; then
      effective_rlm_timeout="$(infer_rlm_timeout_from_time "$(provider_time_limit "$provider")")"
    fi
    if [[ -n "$RLM_BACKEND" ]]; then
      cmd+=(--rlm-backend "$RLM_BACKEND")
    fi
    if [[ -n "$RLM_BACKEND_URL" ]]; then
      cmd+=(--rlm-backend-url "$RLM_BACKEND_URL")
    fi
    if [[ -n "$RLM_MAX_DEPTH" ]]; then
      cmd+=(--rlm-max-depth "$RLM_MAX_DEPTH")
    fi
    # For RLM smoke runs (with --max-instances), reduce iterations from 12 to 4 to avoid timeout at problem 2
    local effective_rlm_iterations="${RLM_MAX_ITERATIONS:-}"
    if [[ -n "$MAX_INSTANCES" ]]; then
      effective_rlm_iterations="4"
    fi
    if [[ -n "$effective_rlm_iterations" ]]; then
      cmd+=(--rlm-max-iterations "$effective_rlm_iterations")
    fi
    if [[ -n "$effective_rlm_timeout" ]]; then
      cmd+=(--rlm-max-timeout "$effective_rlm_timeout")
    fi
  fi

  local quoted=""
  local part
  for part in "${cmd[@]}"; do
    printf -v part '%q' "$part"
    if [[ -n "$quoted" ]]; then
      quoted+=" "
    fi
    quoted+="$part"
  done
  printf '%s\n' "$quoted"
}

build_sbatch_args() {
  local provider="$1"
  local dependency_spec="${2:-}"
  local provider_gres_value=""
  local export_spec="ALL,VTM_RLM_SERVE_STATUS_FILE=$SERVE_STATUS_FILE,VTM_RLM_SERVE_ENDPOINT_FILE=$SERVE_ENDPOINT_FILE,VTM_SERVE_STATUS_FILE=$SERVE_STATUS_FILE,VTM_SERVE_ENDPOINT_FILE=$SERVE_ENDPOINT_FILE"
  local effective_max_model_len="$(effective_vllm_max_model_len 2>/dev/null || true)"
  local effective_gpu_memory_utilization="$(effective_vllm_gpu_memory_utilization)"
  local args=(
    -A "$ACCOUNT"
    -p "$PARTITION"
    --time "$(provider_time_limit "$provider")"
    --cpus-per-task "$CPUS_PER_TASK"
    --mem "$MEMORY"
    --output "$SLURM_LOG_DIR/%x-%j.out"
    --error "$SLURM_LOG_DIR/%x-%j.err"
    --export "$export_spec"
  )

  if [[ -n "$effective_max_model_len" ]]; then
    export_spec+=",VLLM_MAX_MODEL_LEN=$effective_max_model_len"
  fi
  if [[ -n "$effective_gpu_memory_utilization" ]]; then
    export_spec+=",VLLM_GPU_MEMORY_UTILIZATION=$effective_gpu_memory_utilization"
  fi
  args[${#args[@]}-1]="$export_spec"

  provider_gres_value="$(provider_gres "$provider")"
  if [[ -n "$provider_gres_value" ]]; then
    args+=(--gres "$provider_gres_value")
  fi

  if [[ -n "$CLUSTER" ]]; then
    args=(-M "$CLUSTER" "${args[@]}")
  fi
  if [[ -n "$QOS" ]]; then
    args+=(--qos "$QOS")
  fi
  if [[ -n "$dependency_spec" ]]; then
    args+=(--dependency "$dependency_spec")
  fi
  if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
    args+=("${SBATCH_ARGS[@]}")
  fi

  printf '%s\n' "${args[@]}"
}

build_serve_sbatch_args() {
  local export_spec="ALL,VTM_RLM_SERVE_STATUS_FILE=$SERVE_STATUS_FILE,VTM_RLM_SERVE_ENDPOINT_FILE=$SERVE_ENDPOINT_FILE,VTM_SERVE_STATUS_FILE=$SERVE_STATUS_FILE,VTM_SERVE_ENDPOINT_FILE=$SERVE_ENDPOINT_FILE,VTM_CHPC_PYTHON_BIN=$PYTHON_BIN,SERVED_MODEL_NAME=$MODEL,SERVED_MODEL_ALIAS=$MODEL,VLLM_TENSOR_PARALLEL_SIZE=$(effective_tensor_parallel_size)"
  local effective_max_model_len="$(effective_vllm_max_model_len 2>/dev/null || true)"
  local effective_gpu_memory_utilization="$(effective_vllm_gpu_memory_utilization)"
  local serve_gres_value=""
  local args=(
    -A "$ACCOUNT"
    -p "$PARTITION"
    --job-name "${MODEL_TAG}_serve"
    --time "$(serve_time_limit)"
    --cpus-per-task "$CPUS_PER_TASK"
    --mem "$MEMORY"
    --output "$SLURM_LOG_DIR/%x-%j.out"
    --error "$SLURM_LOG_DIR/%x-%j.err"
    --export "$export_spec"
  )

  if [[ -n "$effective_max_model_len" ]]; then
    export_spec+=",VLLM_MAX_MODEL_LEN=$effective_max_model_len"
  fi
  if [[ -n "$effective_gpu_memory_utilization" ]]; then
    export_spec+=",VLLM_GPU_MEMORY_UTILIZATION=$effective_gpu_memory_utilization"
  fi
  args[${#args[@]}-1]="$export_spec"

  serve_gres_value="$(serve_gres)"
  if [[ -n "$serve_gres_value" ]]; then
    args+=(--gres "$serve_gres_value")
  fi

  if [[ -n "$CLUSTER" ]]; then
    args=(-M "$CLUSTER" "${args[@]}")
  fi
  if [[ -n "$QOS" ]]; then
    args+=(--qos "$QOS")
  fi
  if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
    args+=("${SBATCH_ARGS[@]}")
  fi

  printf '%s\n' "${args[@]}"
}

write_serve_job_script() {
  local serve_job_script="$QUEUE_DIR/99_serve.sbatch"

  cat > "$serve_job_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${MODEL_TAG}_serve
set -euo pipefail

PROJECT_ROOT="$PROJECT_ROOT"
STORAGE_ROOT="$VTM_CHPC_ROOT"
PYTHON_MODULE="$PYTHON_MODULE"
CUDA_MODULE="$CUDA_MODULE"
EXTRA_MODULES="$EXTRA_MODULES"
SKIP_MODULES="$SKIP_MODULES"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/chpc/chpc_env.sh"

if [[ -n "$PYTHON_MODULE" ]]; then
  export VTM_CHPC_PYTHON_MODULE="$PYTHON_MODULE"
fi
if [[ -n "$CUDA_MODULE" ]]; then
  export VTM_CHPC_CUDA_MODULE="$CUDA_MODULE"
fi
if [[ -n "$EXTRA_MODULES" ]]; then
  export VTM_CHPC_EXTRA_MODULES="$EXTRA_MODULES"
fi
if [[ "$SKIP_MODULES" == "true" ]]; then
  export VTM_CHPC_SKIP_MODULES=true
fi

export PROJECT_ROOT="$PROJECT_ROOT"
export STORAGE_ROOT="$VTM_CHPC_ROOT"
export VTM_SERVE_STATUS_FILE="$SERVE_STATUS_FILE"
export VTM_SERVE_ENDPOINT_FILE="$SERVE_ENDPOINT_FILE"
export VTM_RLM_SERVE_STATUS_FILE="$SERVE_STATUS_FILE"
export VTM_RLM_SERVE_ENDPOINT_FILE="$SERVE_ENDPOINT_FILE"
export VTM_CHPC_PYTHON_BIN="$PYTHON_BIN"
export SERVED_MODEL_NAME="$MODEL"
export SERVED_MODEL_ALIAS="$MODEL"
export VLLM_TENSOR_PARALLEL_SIZE="$(effective_tensor_parallel_size)"
if [[ -n "$(effective_vllm_max_model_len 2>/dev/null || true)" ]]; then
export VLLM_MAX_MODEL_LEN="$(effective_vllm_max_model_len)"
fi
export VLLM_GPU_MEMORY_UTILIZATION="$(effective_vllm_gpu_memory_utilization)"

exec "$PROJECT_ROOT/scripts/slurm/serve_model.sh"
EOF

  chmod +x "$serve_job_script"
  printf '%s\n' "$serve_job_script"
}

extract_submitted_job_id() {
  local submit_output="$1"
  printf '%s\n' "$submit_output" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1
}

submit_sbatch_job() {
  local __job_id_var="$1"
  shift
  local submit_output
  local job_id

  submit_output="$("$@" 2>&1)"
  printf '%s\n' "$submit_output"
  job_id="$(extract_submitted_job_id "$submit_output")"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse submitted batch job ID from sbatch output." >&2
    return 1
  fi
  printf -v "$__job_id_var" '%s' "$job_id"
}

if ! validate_submit_environment; then
  exit 1
fi

mkdir -p "$SLURM_LOG_DIR"

mapfile -t QUEUE_ARGS < <(build_queue_args)
"$QUEUE_SCRIPT" "${QUEUE_ARGS[@]}"

if should_auto_submit_rlm_serve; then
  rm -f "$SERVE_STATUS_FILE" "$SERVE_ENDPOINT_FILE"
fi

BASELINE_COMMAND="$(build_method_command baseline)"
RAG_COMMAND="$(build_method_command rag)"
RLM_COMMAND="$(build_method_command rlm)"
RLM_RAG_COMMAND="$(build_method_command rlm_rag)"

echo "[submit] Queue directory: $QUEUE_DIR"
echo "[submit] Python executable: $PYTHON_BIN"
echo "[submit] Tensor parallel size: $(effective_tensor_parallel_size)"
if [[ -n "$(effective_vllm_max_model_len 2>/dev/null || true)" ]]; then
  echo "[submit] Local vLLM max model len: $(effective_vllm_max_model_len)"
fi
echo "[submit] Local vLLM gpu memory utilization: $(effective_vllm_gpu_memory_utilization)"
if [[ -n "$RLM_BACKEND" ]]; then
  echo "[submit] RLM backend: $RLM_BACKEND"
fi
if [[ -n "$RLM_BACKEND_URL" ]]; then
  echo "[submit] RLM backend URL: $RLM_BACKEND_URL"
fi
if [[ -n "$RLM_MAX_TIMEOUT" ]]; then
  echo "[submit] RLM timeout override: $RLM_MAX_TIMEOUT"
fi
if should_auto_submit_rlm_serve; then
  echo "[submit] serve time: $(serve_time_limit)"
  if [[ -n "$(serve_gres)" ]]; then
    echo "[submit] serve gres: $(serve_gres)"
  fi
fi
echo "[submit] baseline time: $(provider_time_limit baseline)"
echo "[submit] rag time: $(provider_time_limit rag)"
echo "[submit] rlm time: $(provider_time_limit rlm)"
echo "[submit] rlm_rag time: $(provider_time_limit rlm_rag)"
vtm_chpc_print_summary

if [[ "$DRY_RUN" == "true" ]]; then
  if should_auto_submit_rlm_serve; then
    mapfile -t SERVE_SBATCH_ARGS < <(build_serve_sbatch_args)
    SERVE_JOB_SCRIPT="$(write_serve_job_script)"
  fi
  mapfile -t BASELINE_SBATCH_ARGS < <(build_sbatch_args baseline "" "$BASELINE_COMMAND")
  mapfile -t RAG_SBATCH_ARGS < <(build_sbatch_args rag "" "$RAG_COMMAND")
  if should_auto_submit_rlm_serve; then
    mapfile -t RLM_SBATCH_ARGS < <(build_sbatch_args rlm 'after:<serve_job_id>' "$RLM_COMMAND")
    mapfile -t RLM_RAG_SBATCH_ARGS < <(build_sbatch_args rlm_rag 'after:<serve_job_id>' "$RLM_RAG_COMMAND")
  else
    mapfile -t RLM_SBATCH_ARGS < <(build_sbatch_args rlm "" "$RLM_COMMAND")
    mapfile -t RLM_RAG_SBATCH_ARGS < <(build_sbatch_args rlm_rag "" "$RLM_RAG_COMMAND")
  fi
  echo "[submit] Dry run only. Commands:"
  if should_auto_submit_rlm_serve; then
    echo "  serve sbatch:    ${SERVE_SBATCH_ARGS[*]}"
    echo "  serve script:    $SERVE_JOB_SCRIPT"
  fi
  if should_submit_provider baseline; then
    echo "  baseline sbatch: ${BASELINE_SBATCH_ARGS[*]}"
    echo "  baseline: $BASELINE_COMMAND"
  fi
  if should_submit_provider rag; then
    echo "  rag sbatch:      ${RAG_SBATCH_ARGS[*]}"
    echo "  rag:      $RAG_COMMAND"
  fi
  if should_submit_provider rlm; then
    echo "  rlm sbatch:      ${RLM_SBATCH_ARGS[*]}"
    echo "  rlm:      $RLM_COMMAND"
  fi
  if should_submit_provider rlm_rag; then
    echo "  rlm_rag sbatch:  ${RLM_RAG_SBATCH_ARGS[*]}"
    echo "  rlm_rag:  $RLM_RAG_COMMAND"
  fi
  exit 0
fi

SERVE_JOB_ID=""
if should_auto_submit_rlm_serve; then
  SERVE_JOB_SCRIPT="$(write_serve_job_script)"
  mapfile -t SERVE_SBATCH_ARGS < <(build_serve_sbatch_args)
  echo "[submit] Submitting serve"
  submit_sbatch_job SERVE_JOB_ID sbatch "${SERVE_SBATCH_ARGS[@]}" "$SERVE_JOB_SCRIPT"
  echo "[submit] Serve job id: $SERVE_JOB_ID"
fi

echo "[submit] Submitting baseline"
mapfile -t BASELINE_SBATCH_ARGS < <(build_sbatch_args baseline)
if should_submit_provider baseline; then
  submit_sbatch_job BASELINE_JOB_ID env METHOD_COMMAND="$BASELINE_COMMAND" sbatch "${BASELINE_SBATCH_ARGS[@]}" "$QUEUE_DIR/00_baseline.sbatch"
else
  echo "[submit] Skipping baseline (--skip-providers)"
fi
echo "[submit] Submitting rag"
mapfile -t RAG_SBATCH_ARGS < <(build_sbatch_args rag)
if should_submit_provider rag; then
  submit_sbatch_job RAG_JOB_ID env METHOD_COMMAND="$RAG_COMMAND" sbatch "${RAG_SBATCH_ARGS[@]}" "$QUEUE_DIR/01_rag.sbatch"
else
  echo "[submit] Skipping rag (--skip-providers)"
fi
echo "[submit] Submitting rlm"
if [[ -n "$SERVE_JOB_ID" ]]; then
  mapfile -t RLM_SBATCH_ARGS < <(build_sbatch_args rlm "after:$SERVE_JOB_ID")
else
  mapfile -t RLM_SBATCH_ARGS < <(build_sbatch_args rlm)
fi
if should_submit_provider rlm; then
  submit_sbatch_job RLM_JOB_ID env METHOD_COMMAND="$RLM_COMMAND" sbatch "${RLM_SBATCH_ARGS[@]}" "$QUEUE_DIR/02_rlm.sbatch"
else
  echo "[submit] Skipping rlm (--skip-providers)"
fi
echo "[submit] Submitting rlm_rag"
if [[ -n "$SERVE_JOB_ID" ]]; then
  mapfile -t RLM_RAG_SBATCH_ARGS < <(build_sbatch_args rlm_rag "after:$SERVE_JOB_ID")
else
  mapfile -t RLM_RAG_SBATCH_ARGS < <(build_sbatch_args rlm_rag)
fi
if should_submit_provider rlm_rag; then
  submit_sbatch_job RLM_RAG_JOB_ID env METHOD_COMMAND="$RLM_RAG_COMMAND" sbatch "${RLM_RAG_SBATCH_ARGS[@]}" "$QUEUE_DIR/03_rlm_rag.sbatch"
else
  echo "[submit] Skipping rlm_rag (--skip-providers)"
fi
