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
- Coding task packs are stable, executor artifacts are durable, and the maintained coding executor is the OpenRouter-backed RLM path.
- Vendored-RLM integration remains thin: fixed OpenRouter-backed execution plus optional reranking.

## Limits

- JSONL event export is derived from SQLite rather than in the same transaction boundary.
- Filesystem artifact writes and SQLite writes still do not share one atomic commit boundary.
- `CommandProcedureValidator` is local-process only; Docker is the only built-in sandbox backend.
- Repeated attempts and `pass@k` reporting exist only for coding suites.
- The maintained benchmark modes are `no_memory`, `naive_lexical`, `verified_lexical`, and optional `lexical_rlm_rerank`.
- The maintained benchmark layers are retrieval, drift, and targeted coding only.
- Embeddings, terminal-only tracks, and Codex execution are outside the maintained surface.

## Documentation policy

If behavior, CLI contracts, manifests, or artifact layouts change, update:

- `README.md`
- the relevant file under `docs/`
- the affected package README
- the relevant kept ADR when the boundary is durable
