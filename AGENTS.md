# Repository Map

- `src/vtm/`: public package. Data models live at the top level; protocols and implementations live under `services/`, `stores/`, `adapters/`, and `benchmarks/`.
- `src/vtm/stores/`: SQLite metadata/events, SQLite cache, SQLite embedding index, and filesystem artifact storage.
- `src/vtm/services/`: transaction orchestration, verification, lexical and embedding retrieval, procedure validation, and deterministic consolidation services.
- `src/vtm/adapters/`: implemented Git, runtime, Tree-sitter, deterministic embedding, optional OpenAI embedding, and optional OpenAI RLM integrations.
- `docs/`: durable architecture, type-system, API, audit, and ADR notes.
- `tests/`: regression coverage for records, storage, migrations, transactions, verification, retrieval, consolidation, benchmarks, adapters, and documentation parity.
