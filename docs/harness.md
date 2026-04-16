# Harness

`vtm.harness` is the public execution boundary between the kernel and higher-level evaluation workflows.

It owns the contracts that should stay stable even when benchmark orchestration, prompting, or agent policy changes.

## Public contracts

- `HarnessTaskPack`
  - typed on-disk task definition used by coding executors and local patchers
  - includes optional `retrieval_query` when the benchmark author wants explicit retrieval phrasing
  - includes `execution_style` so patch-oriented and shell-command tasks can share the same suite
- `TaskMemoryContextItem`
  - normalized retrieval context embedded in a task pack
- `ExecutorRequest`
  - typed execution request metadata
  - includes `attempt_index`, `artifact_root`, and `workspace_backend`
- `ExecutorResult`
  - typed execution result and artifact summary
  - includes the normalized `attempt_index`
  - normalizes `workspace_backend` plus optional Docker sandbox metadata

## Workspace contracts

- `WorkspaceBackend`
  - prepares an isolated workspace for a task
- `WorkspaceDriver`
  - terminal execution
  - ripgrep-style search
  - patch capture
  - changed-path capture
  - final verification commands

Built-in workspace backends:

- `LocalWorkspaceBackend`
- `LocalWorkspaceDriver`
- `DockerWorkspaceBackend`
- `DockerWorkspaceDriver`

Docker-backed attempts run one long-lived container per attempt with:

- bind-mounted workspace and artifact roots
- `--network none` by default
- `--read-only` root filesystem by default
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- `--memory 2g`
- `--cpus 2`
- writable `tmpfs` at `/tmp` and `/run` with `noexec`, `nosuid`, and `nodev`
- startup logs persisted at `docker-run.stdout` and `docker-run.stderr`

## Executor contracts

- `SubprocessBenchmarkExecutor`
  - runs a caller-provided external command against a prepared workspace
- `RLMBenchmarkExecutor`
  - runs the vendored upstream `rlm` runtime against the same workspace contract

Both executors currently produce the same case-local artifact backbone:

- `command-events.jsonl`
- `final-git-status.txt`
- `produced.patch`
- `final-verification.stdout`
- `final-verification.stderr`

Vendored-RLM runs also populate benchmark-local artifacts under `rlm/`:

- `response.txt`
- `completion.json`
- optional `trajectory.json`
- `trajectory/`

When coding benchmarks run repeated attempts, the layout is stable:

- canonical task pack: `task-packs/<case-id>.json`
- per-attempt workspace: `workspaces/<mode>/<case-id>/attempt-01`
- per-attempt artifacts: `executor-artifacts/<case-id>/attempt-01`
- aggregate results: `results.jsonl`
- per-attempt results: `attempts.jsonl`

External executor templates may reference:

- `{task_file}`
- `{workspace}`
- `{attempt}`
- `{artifact_root}`

## Task-pack contract

`HarnessTaskPack` is the canonical coding-task file shape written under `task-packs/<case-id>.json`.

Important fields:

- task identity: `case_id`, `repo_name`, `commit_pair_id`
- repo state: `base_ref`, `head_ref`, optional `commit_pair_label`
- task statement: `task_statement`, optional `problem_statement`, optional `hints_text`
- evaluation metadata: `evaluation_backend`, `dataset_name`, `instance_id`
- scoring inputs: `expected_changed_paths`, `target_patch_digest`, optional `gold_test_patch_digest`
- execution settings: `memory_mode`, `top_k`, `coding_executor`
- execution style: `execution_style`
- retrieval override: optional `retrieval_query`
- retrieval context: `memory_context`

`HarnessTaskPack` stays canonical across attempts. Attempt-local data belongs in
`ExecutorRequest`, `ExecutorResult`, and the benchmark runner outputs, not in
the on-disk task-pack file.

## Ownership boundary

- `vtm.harness` owns typed execution contracts and local reference implementations.
- `vtm_rlm` owns the vendored-RLM execution bridge and memory writeback behavior.
- `vtm.benchmarks` owns suite selection, reporting, and manifest-driven orchestration.

New code should import from `vtm.harness`.
