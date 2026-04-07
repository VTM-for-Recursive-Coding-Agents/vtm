#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/chpc_env.sh"

LCB_DIR="$PROJECT_ROOT/benchmarks/LiveCodeBench"
RLM_DIR="$PROJECT_ROOT/rlm"
LCB_VENV_DIR=""
PYTHON_BIN=""
PYTHON_MODULE=""
CUDA_MODULE=""
EXTRA_MODULES=""
STORAGE_ROOT=""
RECREATE="false"
DRY_RUN="false"
SKIP_MODULES="false"
FORCE_INSTALL="false"
SETUP_STRATEGY="auto"
SOURCE_VENV_DIR=""
REQUIRED_MODULES=(datasets openai torch vllm)
INSTALL_RUNTIME_MODULES=(
    annotated-types
    anthropic
    cohere
    "datasets>=3.2.0,<4.0.0"
    google-genai
    "huggingface-hub<1.0"
    mistralai==0.4.2
    "numpy<2.3"
    openai
    pandas
    pebble
    together
    torch
    tqdm
    vllm
)

usage() {
    cat <<'EOF'
Usage: scripts/chpc/setup_chpc.sh [options]

Options:
    --storage-root <path>     Scratch-backed root for CHPC envs and caches
    --venv-dir <path>         Virtualenv location (default: <storage-root>/venvs/livecodebench)
    --python-bin <path>       Base Python executable to create the venv with
    --python-module <name>    Module to load before resolving Python
    --cuda-module <name>      CUDA module to load when preparing CHPC runtime
    --extra-modules <list>    Space-separated extra modules to load
    --skip-modules            Do not attempt to load any environment modules
    --recreate                Remove and recreate the target virtualenv
    --force-install           Build/install into the target venv even if a reusable env exists
    --dry-run                 Print the resolved environment without installing
    -h, --help                Show help

This command prepares a LiveCodeBench virtualenv and redirects large caches away
from HOME. By default it only reuses an existing compatible LiveCodeBench
environment and fails fast if none is available. Use --force-install to opt into
the slower full install path.
EOF
}

python_to_venv_dir() {
    local python_bin="$1"
    printf '%s\n' "$(cd -P "$(dirname "$python_bin")/.." && pwd)"
}

python_is_virtualenv() {
    local python_bin="$1"
    local venv_dir

    venv_dir="$(python_to_venv_dir "$python_bin")"
    [[ -f "$venv_dir/pyvenv.cfg" ]]
}

python_has_livecodebench() {
    local python_bin="$1"

    "$python_bin" -c 'import lcb_runner' >/dev/null 2>&1
}

python_has_required_runtime() {
    local python_bin="$1"
    local missing_modules

    missing_modules="$(vtm_chpc_missing_python_modules "$python_bin" "${REQUIRED_MODULES[@]}")"
    [[ -z "$missing_modules" ]]
}

print_missing_runtime_modules() {
    local python_bin="$1"
    local missing_modules

    missing_modules="$(vtm_chpc_missing_python_modules "$python_bin" "${REQUIRED_MODULES[@]}")"
    if [[ -n "$missing_modules" ]]; then
        echo "[setup] Missing required modules in $python_bin:" >&2
        while IFS= read -r module_name; do
            [[ -n "$module_name" ]] && echo "[setup]   - $module_name" >&2
        done <<< "$missing_modules"
    fi
}

python_is_benchmark_local_livecodebench_env() {
    local python_bin="$1"
    local venv_dir

    venv_dir="$(python_to_venv_dir "$python_bin")"
    [[ "$venv_dir" == "$LCB_DIR"/.venv* ]]
}

python_version_string() {
    local python_bin="$1"

    "$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

path_is_within_home() {
    local target_path="$1"
    local resolved_target
    local resolved_home

    resolved_target="$(readlink -f "$target_path" 2>/dev/null || true)"
    resolved_home="$(readlink -f "$HOME" 2>/dev/null || printf '%s\n' "$HOME")"

    [[ -n "$resolved_target" && "$resolved_target" == "$resolved_home"/* ]]
}

resolve_install_python_bin() {
    local preferred_candidates=(
        "$LCB_DIR/.venv-chpc/bin/python"
        "$LCB_DIR/.venv/bin/python"
        "$LCB_DIR/.venv-granite/bin/python"
    )
    local candidate
    local version_string

    for candidate in "${preferred_candidates[@]}"; do
        if [[ -x "$candidate" ]] && vtm_chpc_python_version_ok "$candidate" 3 10; then
            version_string="$(python_version_string "$candidate")"
            if [[ "$version_string" == 3.11 ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done

    for candidate in "${preferred_candidates[@]}"; do
        if [[ -x "$candidate" ]] && vtm_chpc_python_version_ok "$candidate" 3 10; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if command -v python3 >/dev/null 2>&1 && vtm_chpc_python_version_ok "$(command -v python3)" 3 10; then
        command -v python3
        return 0
    fi

    return 1
}

install_runtime_dependencies() {
    local python_bin="$1"

    "$python_bin" -m pip install --upgrade pip setuptools wheel
    "$python_bin" -m pip install --prefer-binary --only-binary=:all: "${INSTALL_RUNTIME_MODULES[@]}"
    "$python_bin" -m pip install --no-deps --no-build-isolation -e "$LCB_DIR"
}

ensure_rlm_repository() {
    if [[ -d "$RLM_DIR/.git" ]]; then
        return 0
    fi

    if [[ -d "$RLM_DIR" ]] && find "$RLM_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        echo "[setup] RLM directory exists but is not a git checkout: $RLM_DIR" >&2
        echo "[setup] Remove it or run scripts/setup_rlm.sh manually to repair it." >&2
        return 1
    fi

    echo "[setup] Cloning missing RLM repository into $RLM_DIR"
    git clone https://github.com/alexzhang13/rlm.git "$RLM_DIR"
}

target_venv_ready() {
    [[ -x "$VTM_CHPC_LCB_VENV_DIR/bin/python" ]] && \
    { python_has_livecodebench "$VTM_CHPC_LCB_VENV_DIR/bin/python" || \
    python_is_benchmark_local_livecodebench_env "$VTM_CHPC_LCB_VENV_DIR/bin/python"; } && \
    python_has_required_runtime "$VTM_CHPC_LCB_VENV_DIR/bin/python"
}

choose_setup_strategy() {
    local resolved_source_venv=""

    if [[ "$FORCE_INSTALL" == "true" ]]; then
        SETUP_STRATEGY="install"
        return 0
    fi

    if target_venv_ready; then
        SETUP_STRATEGY="reuse-target"
        return 0
    elif [[ -x "$VTM_CHPC_LCB_VENV_DIR/bin/python" ]]; then
        print_missing_runtime_modules "$VTM_CHPC_LCB_VENV_DIR/bin/python"
    fi

    if [[ -x "$PYTHON_BIN" ]] && python_is_virtualenv "$PYTHON_BIN" && \
        (python_has_livecodebench "$PYTHON_BIN" || python_is_benchmark_local_livecodebench_env "$PYTHON_BIN") && \
        python_has_required_runtime "$PYTHON_BIN"; then
        resolved_source_venv="$(python_to_venv_dir "$PYTHON_BIN")"
        if [[ "$resolved_source_venv" != "$VTM_CHPC_LCB_VENV_DIR" ]]; then
            if path_is_within_home "$resolved_source_venv"; then
                echo "[setup] Refusing to link scratch target to HOME-backed virtualenv: $resolved_source_venv" >&2
            else
                SOURCE_VENV_DIR="$resolved_source_venv"
                SETUP_STRATEGY="link-existing"
                return 0
            fi
        fi
    elif [[ -x "$PYTHON_BIN" ]] && python_is_virtualenv "$PYTHON_BIN"; then
        print_missing_runtime_modules "$PYTHON_BIN"
    fi

    SETUP_STRATEGY="missing-runtime"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --storage-root)
            STORAGE_ROOT="$2"
            shift 2
            ;;
        --venv-dir)
            LCB_VENV_DIR="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
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
        --recreate)
            RECREATE="true"
            shift
            ;;
        --force-install)
            FORCE_INSTALL="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
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

if [[ -n "$LCB_VENV_DIR" ]]; then
    VTM_CHPC_LCB_VENV_DIR="$LCB_VENV_DIR"
    export VTM_CHPC_LCB_VENV_DIR
fi

vtm_chpc_load_requested_modules

if [[ -n "$PYTHON_BIN" ]]; then
    if [[ "$PYTHON_BIN" != /* ]]; then
        PYTHON_BIN="$PROJECT_ROOT/$PYTHON_BIN"
    fi
elif [[ "$FORCE_INSTALL" == "true" ]]; then
    PYTHON_BIN="$(resolve_install_python_bin)" || {
        echo "[setup] Unable to resolve a Python >= 3.10 interpreter for force-install." >&2
        exit 1
    }
else
    PYTHON_BIN="$(vtm_chpc_resolve_python)" || {
        echo "[setup] Unable to resolve a Python interpreter. Use --python-bin or --python-module." >&2
        exit 1
    }
fi

if [[ "$PYTHON_BIN" == "$VTM_CHPC_LCB_VENV_DIR/bin/python" && "$RECREATE" != "true" && ! -x "$VTM_CHPC_LCB_VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "[setup] Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if ! vtm_chpc_python_version_ok "$PYTHON_BIN" 3 10; then
    echo "[setup] Python must be >= 3.10 for LiveCodeBench: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$RECREATE" == "true" && -d "$VTM_CHPC_LCB_VENV_DIR" ]]; then
    rm -rf "$VTM_CHPC_LCB_VENV_DIR"
fi

choose_setup_strategy

vtm_chpc_require_writable_dir "$VTM_CHPC_ROOT"
vtm_chpc_require_writable_dir "$VTM_CHPC_CACHE_ROOT"
vtm_chpc_require_writable_dir "$VTM_CHPC_TMP_ROOT"
vtm_chpc_require_writable_dir "$(dirname "$VTM_CHPC_LCB_VENV_DIR")"

echo "=================================================="
echo "VTM CHPC Setup"
echo "=================================================="
vtm_chpc_print_summary
echo "[setup] python_bin=$PYTHON_BIN"
echo "[setup] benchmark_dir=$LCB_DIR"
echo "[setup] rlm_dir=$RLM_DIR"
echo "[setup] target_venv=$VTM_CHPC_LCB_VENV_DIR"
echo "[setup] strategy=$SETUP_STRATEGY"
if [[ -n "$SOURCE_VENV_DIR" ]]; then
    echo "[setup] source_venv=$SOURCE_VENV_DIR"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[setup] Dry run only; no environment changes made."
    exit 0
fi

ensure_rlm_repository

case "$SETUP_STRATEGY" in
    reuse-target)
        echo "[setup] Reusing existing target virtualenv at $VTM_CHPC_LCB_VENV_DIR"
        ;;
    link-existing)
        rm -rf "$VTM_CHPC_LCB_VENV_DIR"
        echo "[setup] Linking target virtualenv to existing environment at $SOURCE_VENV_DIR"
        ln -s "$SOURCE_VENV_DIR" "$VTM_CHPC_LCB_VENV_DIR"
        ;;
    install)
        if [[ -L "$VTM_CHPC_LCB_VENV_DIR" ]]; then
            echo "[setup] Removing symlinked target virtualenv so installation stays on scratch: $VTM_CHPC_LCB_VENV_DIR"
            rm -rf "$VTM_CHPC_LCB_VENV_DIR"
        fi
        if [[ -d "$VTM_CHPC_LCB_VENV_DIR" ]] && path_is_within_home "$VTM_CHPC_LCB_VENV_DIR"; then
            echo "[setup] Removing HOME-backed target virtualenv so installation stays off HOME: $VTM_CHPC_LCB_VENV_DIR"
            rm -rf "$VTM_CHPC_LCB_VENV_DIR"
        fi
        if [[ ! -d "$VTM_CHPC_LCB_VENV_DIR" ]]; then
            echo "[setup] Creating virtualenv at $VTM_CHPC_LCB_VENV_DIR"
            "$PYTHON_BIN" -m venv "$VTM_CHPC_LCB_VENV_DIR"
        else
            echo "[setup] Reusing existing virtualenv at $VTM_CHPC_LCB_VENV_DIR"
        fi
        ;;
    missing-runtime)
        echo "[setup] No reusable LiveCodeBench environment with required runtime dependencies was found." >&2
        echo "[setup] Use --force-install to allow a full environment build, or point --python-bin at a fully provisioned venv." >&2
        exit 2
        ;;
    *)
        echo "[setup] Unknown setup strategy: $SETUP_STRATEGY" >&2
        exit 1
        ;;
esac

# shellcheck disable=SC1091
source "$VTM_CHPC_LCB_VENV_DIR/bin/activate"

python - <<'EOF'
import sys
if sys.version_info < (3, 10):
        raise SystemExit("LiveCodeBench requires Python >= 3.10")
print(f"[setup] active_python={sys.executable}")
print(f"[setup] active_version={sys.version.split()[0]}")
EOF

if [[ "$SETUP_STRATEGY" == "install" ]]; then
    echo "[setup] Installing runtime dependencies from prebuilt wheels when available."
    echo "[setup] This path avoids source builds and should fail fast if a required wheel is unavailable."
    install_runtime_dependencies python
else
    echo "[setup] Skipping package installation; using existing compatible LiveCodeBench environment."
fi

echo "[setup] LiveCodeBench environment is ready."
echo "[setup] Activate with: source $VTM_CHPC_LCB_VENV_DIR/bin/activate"
echo "[setup] Dry-run submissions with: scripts/chpc/submit_livecodebench_local_model.sh --dry-run ..."
