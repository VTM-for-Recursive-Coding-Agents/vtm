# CHPC Launchers

This directory is reserved for CHPC-specific submission wrappers.

Rules:

- Keep benchmark logic shared. Do not copy LiveCodeBench or SWE-bench runner internals here.
- Put scheduler-specific files here only.
- Generated CHPC submission bundles belong under `launchers/chpc/`.
- Actual benchmark outputs still belong under `results/raw/`.

Current entrypoint:

- `scripts/chpc/queue_livecodebench_methods.sh`

Typical flow:

1. Generate a CHPC submission bundle.
2. Set provider-specific `RAG_COMMAND`, `RLM_COMMAND`, and `RLM_RAG_COMMAND` values.
3. Submit the generated `*.sbatch` files from `launchers/chpc/<queue>/`.