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
MAX_INSTANCES="1"
RAG_TOP_K=""
RAG_MAX_CHARS_PER_CHUNK=""
RLM_MAX_DEPTH=""
RLM_MAX_ITERATIONS="2"
RLM_MAX_TIMEOUT=""
QUEUE_TAG="$(date +%Y%m%d)"
QUEUE_DIR=""

ACCOUNT=""
PARTITION=""
QOS=""
CLUSTER=""
GRES="gpu:1"
TIME_LIMIT="02:00:00"
CPUS_PER_TASK="8"
MEMORY="64G"
SBATCH_ARGS=()
DRY_RUN="false"
STORAGE_ROOT=""
PYTHON_MODULE=""
CUDA_MODULE=""
EXTRA_MODULES=""
SKIP_MODULES="false"
REQUIRED_MODULES=(datasets openai torch vllm)

usage() {
  cat <<'EOF'
Usage: scripts/chpc/submit_livecodebench_local_model.sh --model <model_name> --account <account> --partition <partition> [options]

Submission options:
  --account <name>               Slurm account for sbatch (required)
  --partition <name>             Slurm partition for sbatch (required)
  --qos <name>                   Slurm qos value (optional)
  --cluster <name>               Slurm cluster name for remote submission (optional)
  --gres <value>                 Slurm gres request (default: gpu:1)
  --time <HH:MM:SS>              Slurm walltime (default: 02:00:00)
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
  --max-instances <int>          Limit instances for smoke runs (default: 1)
  --rag-top-k <int>              Retrieval chunk count for rag/rlm_rag (optional)
  --rag-max-chars-per-chunk <n>  Max chars per retrieved chunk for rag/rlm_rag (optional)
  --rlm-max-depth <int>          Max recursion depth for rlm/rlm_rag (optional)
  --rlm-max-iterations <int>     Max iterations for rlm/rlm_rag (default: 2)
  --rlm-max-timeout <seconds>    Max wall-clock time for rlm/rlm_rag (optional)
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
    --time)
      TIME_LIMIT="$2"
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
    --rag-top-k)
      RAG_TOP_K="$2"
      shift 2
      ;;
    --rag-max-chars-per-chunk)
      RAG_MAX_CHARS_PER_CHUNK="$2"
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
  if [[ -n "$RAG_TOP_K" ]]; then
    args+=(--rag-top-k "$RAG_TOP_K")
  fi
  if [[ -n "$RAG_MAX_CHARS_PER_CHUNK" ]]; then
    args+=(--rag-max-chars-per-chunk "$RAG_MAX_CHARS_PER_CHUNK")
  fi
  if [[ -n "$RLM_MAX_DEPTH" ]]; then
    args+=(--rlm-max-depth "$RLM_MAX_DEPTH")
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
  if [[ "$provider" == "rag" || "$provider" == "rlm_rag" ]]; then
    if [[ -n "$RAG_TOP_K" ]]; then
      cmd+=(--rag-top-k "$RAG_TOP_K")
    fi
    if [[ -n "$RAG_MAX_CHARS_PER_CHUNK" ]]; then
      cmd+=(--rag-max-chars-per-chunk "$RAG_MAX_CHARS_PER_CHUNK")
    fi
  fi
  if [[ "$provider" == "rlm" || "$provider" == "rlm_rag" ]]; then
    if [[ -n "$RLM_MAX_DEPTH" ]]; then
      cmd+=(--rlm-max-depth "$RLM_MAX_DEPTH")
    fi
    if [[ -n "$RLM_MAX_ITERATIONS" ]]; then
      cmd+=(--rlm-max-iterations "$RLM_MAX_ITERATIONS")
    fi
    if [[ -n "$RLM_MAX_TIMEOUT" ]]; then
      cmd+=(--rlm-max-timeout "$RLM_MAX_TIMEOUT")
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
  local args=(
    -A "$ACCOUNT"
    -p "$PARTITION"
    --gres "$GRES"
    --time "$TIME_LIMIT"
    --cpus-per-task "$CPUS_PER_TASK"
    --mem "$MEMORY"
    --output "$SLURM_LOG_DIR/%x-%j.out"
    --error "$SLURM_LOG_DIR/%x-%j.err"
  )

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

if ! validate_submit_environment; then
  exit 1
fi

mkdir -p "$SLURM_LOG_DIR"

mapfile -t QUEUE_ARGS < <(build_queue_args)
"$QUEUE_SCRIPT" "${QUEUE_ARGS[@]}"

mapfile -t SBATCH_SUBMIT_ARGS < <(build_sbatch_args)
BASELINE_COMMAND="$(build_method_command baseline)"
RAG_COMMAND="$(build_method_command rag)"
RLM_COMMAND="$(build_method_command rlm)"
RLM_RAG_COMMAND="$(build_method_command rlm_rag)"

echo "[submit] Queue directory: $QUEUE_DIR"
echo "[submit] Python executable: $PYTHON_BIN"
echo "[submit] sbatch args: ${SBATCH_SUBMIT_ARGS[*]}"
vtm_chpc_print_summary

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[submit] Dry run only. Commands:"
  echo "  baseline: $BASELINE_COMMAND"
  echo "  rag:      $RAG_COMMAND"
  echo "  rlm:      $RLM_COMMAND"
  echo "  rlm_rag:  $RLM_RAG_COMMAND"
  exit 0
fi

echo "[submit] Submitting baseline"
METHOD_COMMAND="$BASELINE_COMMAND" sbatch "${SBATCH_SUBMIT_ARGS[@]}" "$QUEUE_DIR/00_baseline.sbatch"
echo "[submit] Submitting rag"
METHOD_COMMAND="$RAG_COMMAND" sbatch "${SBATCH_SUBMIT_ARGS[@]}" "$QUEUE_DIR/01_rag.sbatch"
echo "[submit] Submitting rlm"
METHOD_COMMAND="$RLM_COMMAND" sbatch "${SBATCH_SUBMIT_ARGS[@]}" "$QUEUE_DIR/02_rlm.sbatch"
echo "[submit] Submitting rlm_rag"
METHOD_COMMAND="$RLM_RAG_COMMAND" sbatch "${SBATCH_SUBMIT_ARGS[@]}" "$QUEUE_DIR/03_rlm_rag.sbatch"