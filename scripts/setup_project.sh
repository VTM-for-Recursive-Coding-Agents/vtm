#!/usr/bin/env bash
#created by Alexander Fraser, this file sets up the project by downloading necessary repositories for benchmarks and the RLM. It ensures that the required directory structure is in place and that all necessary components are available before running benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


# Run the RLM setup script
bash "$SCRIPT_DIR/setup_rlm.sh"

# Run the benchmarks setup script
bash "$SCRIPT_DIR/download_setup_benchmarks.sh"


