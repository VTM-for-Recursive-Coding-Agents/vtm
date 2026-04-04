# src/vtm/services

Purpose: service and orchestration layer for the kernel. These modules turn stores and adapters into the public transactional-memory API.

Contents
- `__init__.py`: Re-exports the public service protocols and default implementations.
- `consolidator.py`: Consolidation protocol, a no-op implementation, and the deterministic consolidator that supersedes duplicates and can emit summary cards.
- `embedding_retriever.py`: Retriever implementation that lazily builds and refreshes derived embeddings for committed memories.
- `fingerprints.py`: Combines live repo and environment collectors into a dependency fingerprint.
- `kernel_artifacts.py`: Artifact capture and code-anchor helper operations with event emission.
- `kernel_mutations.py`: Shared mutation runner that keeps metadata writes and event writes atomic when stores are shared.
- `kernel_retrieval.py`: Retrieval, expansion, cache access, and retrieval-stat persistence helpers.
- `kernel_transactions.py`: Transaction begin, stage, visibility, commit, and rollback operations.
- `kernel_validation.py`: Verification, procedure validation, and procedure-promotion writeback helpers.
- `memory_kernel.py`: `MemoryKernel` protocol and the `TransactionalMemoryKernel` facade that wires collaborators together.
- `procedures.py`: Command-based procedure validator and the protocol it implements.
- `reranking_retriever.py`: Retriever wrapper that reranks lexical candidates through an RLM adapter and optional cache.
- `retriever.py`: Retriever protocol plus deterministic lexical retrieval implementation and shared lexical helpers.
- `verifier.py`: Verifier protocol plus the dependency and anchor-relocation based verifier.
