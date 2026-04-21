# Current State Audit

This is the maintained snapshot of what still matters for the narrowed paper artifact.

## Guarantees

- The kernel record layer is typed and validated with strict Pydantic models.
- Active transactions, verification writeback, retrieval stats, and event writes are atomic when metadata and events share the same `SqliteMetadataStore`.
- SQLite schema revisions are tracked and future versions are rejected for metadata, cache, and artifact stores.
- Artifact capture stays explicit and auditable through prepared/committed states plus integrity repair helpers.
- Retrieval-time verification can refresh dependency fingerprints, relocate anchors on read, persist relocation evidence, and filter stale memories before they reach the agent.
- External coding prompts no longer expose oracle changed-path hints by default.
- Corrective retry keeps using the same fairness policy, so external retry hints do not inject oracle changed-path targets.
- Controlled coding-drift retrieval derives query and reranking hints from visible task text, tests, verifier output, and deterministic localization notes only; task-pack memory context now retains advisory match metadata for each retrieved memory.
- Coding task packs are stable, executor artifacts are durable, and the maintained coding executor is the DSPy ReAct path.

## Limits

- JSONL event export is derived from SQLite rather than in the same transaction boundary.
- Filesystem artifact writes and SQLite writes still do not share one atomic commit boundary.
- `CommandProcedureValidator` is local-process only; Docker is the only built-in sandbox backend.
- Repeated attempts and `pass@k` reporting exist only for coding suites.
- The maintained benchmark modes are `no_memory`, `naive_lexical`, and `verified_lexical`.
- The maintained benchmark layers are static retrieval, drift verification, drifted retrieval, and controlled coding-drift.
- Historical branches covered embeddings, terminal-only tracks, and Codex execution, but those are outside the maintained surface now.

## Documentation policy

If behavior, CLI contracts, manifests, or artifact layouts change, update:

- `README.md`
- the relevant file under `docs/`
- the affected package README
- the relevant kept ADR when the boundary is durable
