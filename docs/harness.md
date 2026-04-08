# Harness

`vtm.harness` is the public execution boundary between the kernel and higher-level evaluation workflows.

It owns the contracts that should stay stable even when benchmark orchestration, prompting, or agent policy changes.

## Public contracts

- `HarnessTaskPack`
  - typed on-disk task definition used by coding executors and local patchers
  - includes optional `retrieval_query` when the benchmark author wants explicit retrieval phrasing
- `TaskMemoryContextItem`
  - normalized retrieval context embedded in a task pack
- `ExecutorRequest`
  - typed execution request metadata
  - includes `attempt_index` and `artifact_root`
- `ExecutorResult`
  - typed execution result and artifact summary
  - includes the normalized `attempt_index`
- `TraceManifest`
  - stable pointer set for native-agent trace files

## Workspace contracts

- `WorkspaceBackend`
  - prepares an isolated workspace for a task
- `WorkspaceDriver`
  - terminal execution
  - file reads
  - ripgrep-style search
  - patch application
  - patch capture
  - changed-path capture
  - final verification commands

Current local implementation:

- `LocalWorkspaceBackend`
- `LocalWorkspaceDriver`

## Executor contracts

- `SubprocessBenchmarkExecutor`
  - runs a caller-provided external command against a prepared workspace
- `NativeAgentBenchmarkExecutor`
  - runs `TerminalCodingAgent` against the same workspace contract

Both executors currently produce the same case-local artifact backbone:

- `command-events.jsonl`
- `final-git-status.txt`
- `produced.patch`
- `final-verification.stdout`
- `final-verification.stderr`

Native-agent runs also populate a `TraceManifest` over:

- `session.json`
- `turns.jsonl`
- `tool_calls.jsonl`
- `compactions.jsonl`
- `tool-results/`

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
- retrieval override: optional `retrieval_query`
- retrieval context: `memory_context`

`HarnessTaskPack` stays canonical across attempts. Attempt-local data belongs in
`ExecutorRequest`, `ExecutorResult`, and the benchmark runner outputs, not in
the on-disk task-pack file.

## Ownership boundary

- `vtm.harness` owns typed execution contracts and local reference implementations.
- `vtm.agents` owns the native agent loop and tool semantics.
- `vtm.benchmarks` owns suite selection, reporting, and manifest-driven orchestration.

Compatibility shims remain at:

- `vtm.agents.workspace`
- `vtm.benchmarks.executor`

New code should import from `vtm.harness`.
