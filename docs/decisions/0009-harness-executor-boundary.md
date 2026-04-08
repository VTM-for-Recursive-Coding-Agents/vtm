# ADR 0009: Harness Executor Boundary

## Status

Accepted

## Context

The benchmark runner and native agent runtime both need a stable execution seam, but workspace management, task-pack files, executor results, and scoring should not stay buried inside benchmark orchestration modules or agent-only modules.

## Decision

- Add a public `vtm.harness` package.
- Move typed task-pack, workspace, executor, and scoring contracts into that package.
- Keep `vtm.agents.workspace` and `vtm.benchmarks.executor` as compatibility shims during transition.
- Keep `vtm.benchmarks` responsible for suite selection and reporting, and keep `vtm.agents` responsible for the agent loop and tool behavior.

## Consequences

- The execution boundary is explicit and typed.
- Benchmark orchestration and agent prompting can evolve without redefining the workspace or executor contract every time.
- Local workspace execution remains the default implementation, while remote or sandboxed executors remain future work.
