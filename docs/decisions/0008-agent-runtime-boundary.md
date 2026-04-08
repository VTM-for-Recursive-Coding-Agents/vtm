# ADR 0008: Agent Runtime Boundary

## Status

Accepted

## Context

VTM now needs a native multi-turn coding runtime, but the core kernel is deliberately optimized for typed memory, auditability, and transactional correctness. Embedding the full agent loop directly into the kernel would blur those boundaries, force runtime-policy decisions into stable storage interfaces, and make benchmark-oriented autonomy harder to contain.

## Decision

- Add a separate `vtm.agents` package for the native single-agent terminal runtime.
- Keep `TransactionalMemoryKernel` public methods unchanged.
- Use `VisibilityScope(kind=TASK, scope_id=session_id)` for ephemeral task memory written by the agent during a run.
- Require explicit promotion for durable procedures and other repo-scoped knowledge; task memory is not auto-promoted.
- Keep agent session traces in benchmark/runtime artifact directories, not canonical metadata/event storage.
- Introduce a provider-neutral `AgentModelAdapter` contract plus an OpenAI-compatible reference adapter.
- Let the benchmark harness choose between `external_command` and `native_agent` coding execution without changing retrieval or verification contracts.

## Consequences

- The kernel remains small, typed, and storage-focused.
- Native-agent autonomy, permissions, and prompt policies can evolve without forcing SQLite schema churn.
- Benchmark-native agent runs are reproducible through persisted task packs, run config, and benchmark-local trace artifacts.
- Task-scoped memories stay auditable and queryable during a run, while durable repo knowledge still requires an explicit promotion step.
