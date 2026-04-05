# VTM Workspace

This repository is organized around benchmark execution and result analysis.

## Execution Split

- `scripts/shared/` contains environment-agnostic benchmark runners.
- `scripts/local/` contains workstation-specific wrappers and preflight checks.
- `scripts/chpc/` contains CHPC submission wrappers only.
- `launchers/` contains generated launcher bundles.
- `results/` contains benchmark outputs, normalized metrics, and visualizations.

## Local Runs

Primary local entrypoints:

- `scripts/local/run_livecodebench.sh`
- `scripts/local/run_swebench.sh`
- `scripts/local/run_benchmarks.sh`
- `scripts/local/queue_livecodebench_methods.sh`

Legacy top-level script names under `scripts/` remain as compatibility wrappers.

## CHPC Runs

Primary CHPC entrypoints:

- `scripts/chpc/queue_livecodebench_methods.sh`

CHPC scripts should only handle submission and scheduler concerns. Shared benchmark logic stays under `scripts/shared/`.

## Outputs vs Launchers

- Generated launcher bundles belong under `launchers/local/` or `launchers/chpc/`.
- Actual benchmark outputs belong under `results/raw/`.
- Normalized summaries and plots belong under `results/metrics/` and `results/visualizations/`.
