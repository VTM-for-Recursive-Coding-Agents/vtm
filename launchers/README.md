# Launchers Layout

This directory stores generated execution bundles, not benchmark outputs.

Structure:

- `launchers/local/` for generated local launcher bundles
- `launchers/chpc/` for generated CHPC submission bundles

Each bundle can contain:

- per-provider launch scripts
- a `run_all.sh` or `submit_all.sh` convenience entrypoint
- a `manifest.json` metadata file

Benchmark outputs remain under `results/raw/`.