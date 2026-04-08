# src/vtm/harness

Purpose: public harness boundary for task-pack contracts, workspace preparation, executor implementations, and coding-benchmark scoring helpers.

Start here
- `models.py`: the public task-pack and executor/result contracts.
- `workspace.py`: local workspace preparation and the reference `WorkspaceDriver`.
- `executors.py`: subprocess and native-agent executor implementations.

Contents
- `__init__.py`: Re-exports the public harness contracts and local reference implementations.
- `executors.py`: Local subprocess and native-agent executor implementations.
- `models.py`: Typed task-pack, executor request/result, and trace-manifest records.
- `scoring.py`: Changed-path and patch-similarity helpers used by coding evaluation.
- `workspace.py`: Local workspace backend, persistent workspace driver, and command-result records.

Important contract details
- `HarnessTaskPack`: canonical per-case task file, including optional `retrieval_query`.
- `ExecutorRequest`: per-attempt execution request, including `attempt_index` and `artifact_root`.
- `ExecutorResult`: per-attempt execution output used for `attempts.jsonl`.
- Local layout:
  - `task-packs/<case-id>.json`
  - `workspaces/<mode>/<case-id>/attempt-01`
  - `executor-artifacts/<case-id>/attempt-01`
