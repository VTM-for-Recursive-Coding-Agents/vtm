# docs

Purpose: source-of-truth architecture, contract, and maintainer guidance for VTM.

Contents
- `architecture.md`: Kernel, harness, agent, and benchmark boundaries plus main flows.
- `api.md`: Stable kernel-facing API notes and package-root guidance.
- `codebase-guide.md`: Maintainer-oriented repository map, ownership guide, and test/doc change paths.
- `code-reference.md`: Generated inventory of every Python code file and its top-level symbols.
- `../flake.nix`: Nix flake providing the dev shell, package, and CLI apps.
- `harness.md`: Public harness contracts for task packs, workspaces, executors, and traces.
- `benchmark-recipes.md`: Maintained benchmark commands for retrieval, coding, native-agent, and SWE-bench workflows.
- `benchmark-results-template.md`: Template for recording run results outside the tracked docs.
- `current-state-audit.md`: Current guarantees, known gaps, and intentionally limited areas.
- `runtime-example.md`: Executable end-to-end kernel example.
- `swebench-lite-windows.md`: Windows and WSL2 runbook for the SWE-bench Lite path.
- `type-system.md`: Core record and enum reference for the kernel layer.
- `decisions/`: ADRs for stable architectural and compatibility choices.
