# Harness

`vtm.harness` is the public execution boundary between the kernel and higher-level evaluation workflows.

It owns the contracts that should stay stable even when benchmark orchestration, prompting, or agent policy changes.

## Public contracts

- `HarnessTaskPack`
  - typed on-disk task definition used by the vendored-RLM coding executor
  - includes `retrieval_query`, which is either author-provided or derived from visible task signals
  - retains `execution_style` for task-pack compatibility, but the maintained paper path is the patch-oriented vendored-RLM executor
  - includes optional `verifier_output` and `localization_notes` for visible, non-oracle task context
- `TaskMemoryContextItem`
  - normalized retrieval context embedded in a task pack
- `ExecutorRequest`
  - typed execution request metadata
  - includes `attempt_index`, `artifact_root`, and `workspace_backend`
- `ExecutorResult`
  - typed execution result and artifact summary
  - includes the normalized `attempt_index`
  - normalizes `workspace_backend`; legacy Docker metadata can still appear in non-maintained runs

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

Legacy/non-maintained backend kept in-tree for compatibility:

- `DockerWorkspaceBackend`
- `DockerWorkspaceDriver`

## Executor contracts

- `RLMBenchmarkExecutor`
  - runs the vendored upstream `rlm` runtime against the local workspace contract
  - is the only maintained built-in coding executor

The RLM executor produces the stable case-local artifact backbone:

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

## Task-pack contract

`HarnessTaskPack` is the canonical coding-task file shape written under `task-packs/<case-id>.json`.

Important fields:

- task identity: `case_id`, `repo_name`, `commit_pair_id`
- repo state: `base_ref`, `head_ref`, optional `commit_pair_label`
- task statement: `task_statement`, optional `problem_statement`, optional `hints_text`
- visible task signals: optional `verifier_output`, optional `localization_notes`
- evaluation metadata: `evaluation_backend`, `dataset_name`, `instance_id`
- scoring inputs: `expected_changed_paths`, `target_patch_digest`, optional `gold_test_patch_digest`
- execution settings: `memory_mode`, `top_k`
- execution style: `execution_style`
- retrieval override: optional `retrieval_query`
- retrieval context: `memory_context`

For external coding tasks, `expected_changed_paths` stays in the canonical task pack for scoring, but prompt builders and the vendored-RLM `TASK` tool hide those oracle hints by default unless `debug_expected_changed_paths=True`.

`HarnessTaskPack` stays canonical across attempts. Attempt-local data belongs in
`ExecutorRequest`, `ExecutorResult`, and the benchmark runner outputs, not in
the on-disk task-pack file.

## Ownership boundary

- `vtm.harness` owns typed execution contracts and local reference implementations.
- `vtm_rlm` owns the vendored-RLM execution bridge and memory writeback behavior.
- `vtm.benchmarks` owns suite selection, reporting, and manifest-driven orchestration.

New code should import from `vtm.harness`.
