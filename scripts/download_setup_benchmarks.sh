#!/usr/bin/env bash
#created by Alexander Fraser, this file sets up the project by downloading necessary repositories for benchmarks and the RLM. It ensures that the required directory structure is in place and that all necessary components are available before running benchmarks.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARKS_DIR="$PROJECT_ROOT/benchmarks"
# Always run from this script's directory so clone targets are predictable.

echo "============================================================"
echo "Downloading the LiveCodeBench Repository"
echo "-------------------------------------------------------------"
if [ -d "$BENCHMARKS_DIR/LiveCodeBench/.git" ]; then
	echo "LiveCodeBench already exists in $BENCHMARKS_DIR/LiveCodeBench; skipping clone"
else
	git clone https://github.com/LiveCodeBench/LiveCodeBench.git "$BENCHMARKS_DIR/LiveCodeBench"
fi
echo "-------------------------------------------------------------"
echo "finished downloading the LiveCodeBench Repository"
echo "============================================================"



echo "============================================================"
echo "Downloading the SWE-bench Repository"
echo "-------------------------------------------------------------"
if [ -d "$BENCHMARKS_DIR/SWE-bench/.git" ]; then
	echo "SWE-bench already exists in $BENCHMARKS_DIR/SWE-bench; skipping clone"
else
	git clone https://github.com/SWE-bench/SWE-bench.git "$BENCHMARKS_DIR/SWE-bench"
fi
echo "-------------------------------------------------------------"
echo "finished downloading the SWE-bench Repository"
echo "============================================================"