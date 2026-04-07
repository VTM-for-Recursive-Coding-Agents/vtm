#!/usr/bin/env bash

if [[ -n "${VTM_CHPC_ENV_SH_LOADED:-}" ]]; then
  return 0
fi
VTM_CHPC_ENV_SH_LOADED=1

VTM_CHPC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VTM_PROJECT_ROOT="$(cd "$VTM_CHPC_SCRIPT_DIR/../.." && pwd)"
: "${VTM_CHPC_ROOT:=}"
: "${VTM_CHPC_CACHE_ROOT:=}"
: "${VTM_CHPC_TMP_ROOT:=}"
: "${VTM_CHPC_VENV_ROOT:=}"
: "${VTM_CHPC_LCB_VENV_DIR:=}"

vtm_chpc_enable_modules() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
  fi

  command -v module >/dev/null 2>&1
}

vtm_chpc_maybe_load_module() {
  local module_name="$1"

  if [[ -z "$module_name" || "$module_name" == "none" ]]; then
    return 0
  fi

  if ! vtm_chpc_enable_modules; then
    echo "[chpc-env] Environment modules are unavailable; cannot load '$module_name'." >&2
    return 1
  fi

  module load "$module_name"
}

vtm_chpc_default_root() {
  if [[ -n "${VTM_CHPC_ROOT:-}" ]]; then
    printf '%s\n' "$VTM_CHPC_ROOT"
    return 0
  fi

  if [[ -n "${VTM_CHPC_STORAGE_ROOT:-}" ]]; then
    printf '%s\n' "$VTM_CHPC_STORAGE_ROOT"
    return 0
  fi

  if [[ -d "/scratch/general/vast" ]]; then
    printf '/scratch/general/vast/%s/vtm\n' "$USER"
    return 0
  fi

  if [[ -d "/scratch/general/nfs1" ]]; then
    printf '/scratch/general/nfs1/%s/vtm\n' "$USER"
    return 0
  fi

  return 1
}

vtm_chpc_init_layout() {
  local requested_root="${1:-}"

  if [[ -n "$requested_root" ]]; then
    VTM_CHPC_ROOT="$requested_root"
  elif [[ -z "$VTM_CHPC_ROOT" ]]; then
    VTM_CHPC_ROOT="$(vtm_chpc_default_root)" || {
      echo "[chpc-env] Unable to determine scratch-backed storage. Set VTM_CHPC_ROOT explicitly." >&2
      return 1
    }
  fi

  VTM_CHPC_CACHE_ROOT="$VTM_CHPC_ROOT/cache"
  VTM_CHPC_TMP_ROOT="$VTM_CHPC_ROOT/tmp"
  VTM_CHPC_VENV_ROOT="$VTM_CHPC_ROOT/venvs"
  VTM_CHPC_LCB_VENV_DIR="${VTM_CHPC_LCB_VENV_DIR:-$VTM_CHPC_VENV_ROOT/livecodebench}"
}

vtm_chpc_ensure_layout() {
  local dir

  for dir in \
    "$VTM_CHPC_ROOT" \
    "$VTM_CHPC_CACHE_ROOT" \
    "$VTM_CHPC_CACHE_ROOT/huggingface" \
    "$VTM_CHPC_CACHE_ROOT/huggingface/hub" \
    "$VTM_CHPC_CACHE_ROOT/huggingface/datasets" \
    "$VTM_CHPC_CACHE_ROOT/transformers" \
    "$VTM_CHPC_CACHE_ROOT/torch" \
    "$VTM_CHPC_CACHE_ROOT/triton" \
    "$VTM_CHPC_CACHE_ROOT/matplotlib" \
    "$VTM_CHPC_CACHE_ROOT/pip" \
    "$VTM_CHPC_CACHE_ROOT/uv" \
    "$VTM_CHPC_CACHE_ROOT/xdg" \
    "$VTM_CHPC_CACHE_ROOT/wheels" \
    "$VTM_CHPC_TMP_ROOT" \
    "$VTM_CHPC_VENV_ROOT"; do
    mkdir -p "$dir"
  done
}

vtm_chpc_export_runtime_env() {
  export VTM_PROJECT_ROOT
  export VTM_CHPC_ROOT
  export VTM_CHPC_CACHE_ROOT
  export VTM_CHPC_TMP_ROOT
  export VTM_CHPC_VENV_ROOT
  export VTM_CHPC_LCB_VENV_DIR

  export XDG_CACHE_HOME="$VTM_CHPC_CACHE_ROOT/xdg"
  export HF_HOME="$VTM_CHPC_CACHE_ROOT/huggingface"
  export HUGGINGFACE_HUB_CACHE="$VTM_CHPC_CACHE_ROOT/huggingface/hub"
  export HF_DATASETS_CACHE="$VTM_CHPC_CACHE_ROOT/huggingface/datasets"
  export TRANSFORMERS_CACHE="$VTM_CHPC_CACHE_ROOT/transformers"
  export TORCH_HOME="$VTM_CHPC_CACHE_ROOT/torch"
  export TRITON_CACHE_DIR="$VTM_CHPC_CACHE_ROOT/triton"
  export MPLCONFIGDIR="$VTM_CHPC_CACHE_ROOT/matplotlib"
  export PIP_CACHE_DIR="$VTM_CHPC_CACHE_ROOT/pip"
  export UV_CACHE_DIR="$VTM_CHPC_CACHE_ROOT/uv"
  export TMPDIR="$VTM_CHPC_TMP_ROOT"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
}

vtm_chpc_setup_environment() {
  local requested_root="${1:-}"

  vtm_chpc_init_layout "$requested_root" || return 1
  vtm_chpc_ensure_layout || return 1
  vtm_chpc_export_runtime_env
}

vtm_chpc_load_requested_modules() {
  local module_name

  if [[ "${VTM_CHPC_SKIP_MODULES:-false}" == "true" ]]; then
    return 0
  fi

  if [[ -n "${VTM_CHPC_PYTHON_MODULE:-}" ]]; then
    vtm_chpc_maybe_load_module "$VTM_CHPC_PYTHON_MODULE" || return 1
  fi

  if [[ -n "${VTM_CHPC_CUDA_MODULE:-}" ]]; then
    vtm_chpc_maybe_load_module "$VTM_CHPC_CUDA_MODULE" || return 1
  fi

  if [[ -n "${VTM_CHPC_EXTRA_MODULES:-}" ]]; then
    for module_name in $VTM_CHPC_EXTRA_MODULES; do
      vtm_chpc_maybe_load_module "$module_name" || return 1
    done
  fi
}

vtm_chpc_python_candidates() {
  if [[ -n "${VTM_CHPC_PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$VTM_CHPC_PYTHON_BIN"
  fi
  printf '%s\n' "$VTM_CHPC_LCB_VENV_DIR/bin/python"
  printf '%s\n' "$VTM_PROJECT_ROOT/benchmarks/LiveCodeBench/.venv-granite/bin/python"
  printf '%s\n' "$VTM_PROJECT_ROOT/benchmarks/LiveCodeBench/.venv-chpc/bin/python"
  printf '%s\n' "$VTM_PROJECT_ROOT/benchmarks/LiveCodeBench/.venv/bin/python"
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    printf '%s\n' "$VIRTUAL_ENV/bin/python"
  fi
  command -v python3 2>/dev/null || true
}

vtm_chpc_resolve_python() {
  local candidate

  while IFS= read -r candidate; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(vtm_chpc_python_candidates)

  return 1
}

vtm_chpc_python_version_ok() {
  local python_bin="$1"
  local minimum_major="${2:-3}"
  local minimum_minor="${3:-10}"

  "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (int(sys.argv[1]), int(sys.argv[2])) else 1)' "$minimum_major" "$minimum_minor"
}

vtm_chpc_missing_python_modules() {
  local python_bin="$1"
  shift

  "$python_bin" - "$@" <<'EOF'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
for name in missing:
    print(name)
EOF
}

vtm_chpc_require_writable_dir() {
  local dir_path="$1"

  mkdir -p "$dir_path"
  if [[ ! -w "$dir_path" ]]; then
    echo "[chpc-env] Directory is not writable: $dir_path" >&2
    return 1
  fi
}

vtm_chpc_print_summary() {
  cat <<EOF
[chpc-env] project_root=$VTM_PROJECT_ROOT
[chpc-env] storage_root=$VTM_CHPC_ROOT
[chpc-env] livecodebench_venv=$VTM_CHPC_LCB_VENV_DIR
[chpc-env] xdg_cache=$XDG_CACHE_HOME
[chpc-env] hf_home=$HF_HOME
[chpc-env] transformers_cache=$TRANSFORMERS_CACHE
[chpc-env] torch_home=$TORCH_HOME
[chpc-env] triton_cache=$TRITON_CACHE_DIR
[chpc-env] tmpdir=$TMPDIR
EOF
}