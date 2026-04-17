# src/vtm

Purpose: public kernel package.

Start here
- `__init__.py`: stable kernel-first import surface
- `memory_items.py`, `retrieval.py`, `transactions.py`, `verification.py`: core durable records
- `services/memory_kernel.py`: `TransactionalMemoryKernel`
- `stores/` and `adapters/`: concrete infrastructure used to wire the kernel

Contents
- top-level modules: durable records, enums, retrieval, verification, transactions, and artifacts
- `services/`: verification, lexical retrieval, procedures, consolidation, and kernel orchestration
- `stores/`: metadata, cache, and artifact storage protocols plus concrete implementations
- `adapters/`: git, runtime, syntax, and thin OpenAI-compatible chat/RLM integrations
- `harness/`: task-pack, workspace, executor, and scoring boundary
- `benchmarks/`: manifest-driven evaluation orchestration
