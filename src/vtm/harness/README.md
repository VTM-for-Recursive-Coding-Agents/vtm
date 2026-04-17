# src/vtm/harness

Purpose: public harness boundary for task-pack contracts, workspace preparation, executor implementations, and coding-benchmark scoring helpers.

Use this README for the stable execution contract.
For runnable benchmark commands, use [`docs/benchmark-recipes.md`](../../../docs/benchmark-recipes.md).

Start here
- `models.py`: the public task-pack and executor/result contracts.
- `workspace.py`: local workspace preparation and the reference `WorkspaceDriver`.
- `workspace_docker.py`: Docker-backed workspace preparation and driver lifecycle.
- `executors.py`: subprocess and vendored-RLM executor implementations.

Contents
- `__init__.py`: Re-exports the public harness contracts and local reference implementations.
- `executors.py`: Local subprocess and vendored-RLM executor implementations.
- `models.py`: Typed task-pack plus executor request/result records.
- `scoring.py`: Changed-path and patch-similarity helpers used by coding evaluation.
- `workspace.py`: Local workspace backend, persistent workspace driver, and command-result records.
- `workspace_docker.py`: Docker-backed workspace backend and persistent in-container driver.

Important contract details
- `HarnessTaskPack`: canonical per-case task file, including derived or explicit `retrieval_query`, `execution_style`, and visible task context such as `verifier_output` / `localization_notes`.
- `ExecutorRequest`: per-attempt execution request, including `attempt_index`, `artifact_root`, and `workspace_backend`.
- `ExecutorResult`: per-attempt execution output used for `attempts.jsonl`, plus normalized Docker metadata when applicable.
- Local layout:
  - `task-packs/<case-id>.json`
  - `workspaces/<mode>/<case-id>/attempt-01`
  - `executor-artifacts/<case-id>/attempt-01`
- Docker-backed attempts default to `--network none` and keep one long-lived container per attempt.
- Docker-backed attempts also default to a read-only root filesystem, `pids-limit=256`, `memory=2g`, `cpus=2`, hardened tmpfs mounts, and persisted `docker-run.stdout` / `docker-run.stderr` startup logs.
- Vendored-RLM runs reuse the same workspace contract and write benchmark-local artifacts under `executor-artifacts/<case-id>/attempt-01/rlm/`.
