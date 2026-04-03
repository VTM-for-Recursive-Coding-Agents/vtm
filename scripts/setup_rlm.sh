
#!/usr/bin/env bash
#created by Alexander Fraser, this file sets up the project by downloading necessary repositories for benchmarks and the RLM. It ensures that the required directory structure is in place and that all necessary components are available before running benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RLM_DIR="$PROJECT_ROOT/rlm"

echo "============================================================"
echo "Downloading the RLM Repository"
echo "-------------------------------------------------------------"
if [ -d "$RLM_DIR/.git" ]; then
	echo "RLM already exists in $RLM_DIR; skipping clone"
else
	git clone https://github.com/alexzhang13/rlm.git "$RLM_DIR"
fi
echo "-------------------------------------------------------------"
echo "Finished downloading the RLM Repository"
echo "============================================================"
