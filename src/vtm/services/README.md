# src/vtm/services

Purpose: kernel orchestration layer that turns stores and adapters into the public transactional-memory API.

Contents
- `memory_kernel.py`: `MemoryKernel` protocol and the `TransactionalMemoryKernel` facade.
- `kernel_transactions.py`, `kernel_validation.py`, `kernel_retrieval.py`, `kernel_artifacts.py`, `kernel_mutations.py`: focused kernel collaborators.
- `retriever.py`, `embedding_retriever.py`, `reranking_retriever.py`: retrieval implementations.
- `verifier.py`: dependency and anchor-based verification.
- `procedures.py`: command-based procedure validation.
- `consolidator.py`: deterministic consolidation services.
- `fingerprints.py`: dependency fingerprint construction.
