# src/vtm

Purpose: public VTM package rooted in the typed memory kernel.

Start here
- `__init__.py`: stable kernel-first import surface for applications.
- `memory_items.py`, `retrieval.py`, `transactions.py`, `verification.py`: core durable records most callers manipulate directly.
- `services/memory_kernel.py`: main `TransactionalMemoryKernel` facade.
- `stores/` and `adapters/`: concrete infrastructure needed to wire the kernel.

Contents
- `__init__.py`: Kernel-first root export surface.
- `anchors.py`, `artifacts.py`, `cache.py`, `consolidation.py`, `embeddings.py`, `events.py`, `evidence.py`, `fingerprints.py`, `memory_items.py`, `retrieval.py`, `transactions.py`, `verification.py`: durable record and enum modules.
- `services/`: Kernel orchestration and retrieval/verification/consolidation services.
- `stores/`: Metadata, cache, embedding, and artifact storage protocols plus concrete implementations.
- `adapters/`: Git, runtime, syntax, embedding, and RLM integrations.
- `harness/`: Typed task-pack, workspace, executor, and scoring boundary.
- `agents/`: Native single-agent runtime, tool registry, and permission policies.
- `benchmarks/`: Manifest-driven evaluation and reporting orchestration.
