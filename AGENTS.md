# Repository Map

- `src/vtm/`: public package. Data models live at the top level; protocols and implementations live under `services/`, `stores/`, `adapters/`, and `benchmarks/`.
- `src/vtm/stores/`: SQLite metadata/events, SQLite cache, and filesystem artifact storage.
- `src/vtm/services/`: transaction orchestration, verification, lexical retrieval, procedure validation, and deterministic consolidation services.
- `src/vtm/adapters/`: implemented Git, runtime, Tree-sitter, and thin OpenAI-compatible chat/RLM integrations.
- `docs/`: maintained scope, benchmark recipes, audit, harness notes, and ADRs.
- `tests/`: regression coverage for records, storage, migrations, transactions, verification, retrieval, consolidation, harness, and SWE-bench paths.
