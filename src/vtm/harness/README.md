# src/vtm/harness

Purpose: public harness boundary for task-pack contracts, workspace preparation, executor implementations, and coding-benchmark scoring helpers.

Use this README for the stable execution contract.
For runnable benchmark commands, use [`docs/benchmark-recipes.md`](../../../docs/benchmark-recipes.md).

Start here
- `models.py`: the public task-pack and executor/result contracts.
- `workspace.py`: local workspace preparation and the reference `WorkspaceDriver`.
- `executors.py`: maintained DSPy benchmark executor implementation.

Contents
- `__init__.py`: Re-exports the public harness contracts and local reference implementations.
- `executors.py`: DSPy ReAct benchmark executor implementation.
- `models.py`: Typed task-pack plus executor request/result records.
- `scoring.py`: Changed-path and patch-similarity helpers used by coding evaluation.
- `workspace.py`: Local workspace backend, persistent workspace driver, and command-result records.

Important contract details
- `HarnessTaskPack`: canonical per-case task file, including derived or explicit `retrieval_query`, retained `execution_style` compatibility, and visible task context such as `verifier_output` / `localization_notes`.
  When not provided directly, `retrieval_query` is derived from visible task text, tests, verifier output, and deterministic localization notes.
- `TaskMemoryContextItem`: normalized retrieved-memory entry stored in `memory_context`, including advisory match metadata such as `matched_terms`, `matched_fields`, and `relevance_reason`.
- `ExecutorRequest`: per-attempt execution request, including `attempt_index`, `artifact_root`, and `workspace_backend`.
- `ExecutorResult`: per-attempt execution output used for `attempts.jsonl`.
- Local layout:
  - `task-packs/<case-id>.json`
  - `workspaces/<mode>/<case-id>/attempt-01`
  - `executor-artifacts/<case-id>/attempt-01`
- DSPy benchmark runs write benchmark-local artifacts under `executor-artifacts/<case-id>/attempt-01/dspy-react/`.
