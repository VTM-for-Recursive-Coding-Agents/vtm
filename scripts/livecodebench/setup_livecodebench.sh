#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LCB_DIR="$PROJECT_ROOT/benchmarks/LiveCodeBench"
LCB_REPO_URL="${LCB_REPO_URL:-https://github.com/LiveCodeBench/LiveCodeBench.git}"
LCB_REF=""
SKIP_INSTALL="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/livecodebench/setup_livecodebench.sh [options]

Options:
  --ref <git-ref>     Optional git ref to checkout after clone/fetch.
  --skip-install      Skip uv venv / uv pip install -e .
  -h, --help          Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      LCB_REF="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL="true"
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

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/benchmarks"

if [[ -d "$LCB_DIR/.git" ]]; then
  echo "[setup-livecodebench] Existing checkout found at $LCB_DIR"
  git -C "$LCB_DIR" fetch --tags origin
else
  echo "[setup-livecodebench] Cloning $LCB_REPO_URL -> $LCB_DIR"
  git clone "$LCB_REPO_URL" "$LCB_DIR"
fi

if [[ -n "$LCB_REF" ]]; then
  echo "[setup-livecodebench] Checking out $LCB_REF"
  git -C "$LCB_DIR" checkout "$LCB_REF"
fi

if [[ "$SKIP_INSTALL" == "true" ]]; then
  echo "[setup-livecodebench] Skipping uv install"
  exit 0
fi

echo "[setup-livecodebench] Creating benchmark-local venv"
(
  cd "$LCB_DIR"
  uv venv --python 3.11
  uv pip install -e .
)

echo "[setup-livecodebench] Ready: $LCB_DIR"
