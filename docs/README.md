# docs

Purpose: source-of-truth architecture, contract, and maintainer guidance for VTM.

Entry points
- user-facing benchmark commands and runbooks live in `benchmark-recipes.md`
- execution-contract details live in `harness.md`
- code ownership and edit guidance live in `codebase-guide.md`

Contents
- `architecture.md`: Kernel, harness, benchmark, and vendored-RLM boundaries plus main flows.
- `api.md`: Stable kernel-facing API notes and package-root guidance.
- `codebase-guide.md`: Maintainer-oriented repository map, ownership guide, and test/doc change paths.
- `code-reference.md`: Generated inventory of every Python code file and its top-level symbols.
- `../flake.nix`: Nix flake providing the dev shell, package, and CLI apps.
- `harness.md`: Public harness contracts for task packs, workspaces, executors, and artifact layout.
- `benchmark-recipes.md`: Maintained benchmark commands for retrieval, coding, vendored-RLM, and SWE-bench workflows.
- `benchmark-results-template.md`: Template for recording run results outside the tracked docs.
- `current-state-audit.md`: Current guarantees, known gaps, and intentionally limited areas.
- `runtime-example.md`: Executable end-to-end kernel example.
- `swebench-lite-windows.md`: Windows and WSL2 runbook for the SWE-bench Lite path.
- `type-system.md`: Core record and enum reference for the kernel layer.
- `decisions/`: ADRs for stable architectural and compatibility choices.
