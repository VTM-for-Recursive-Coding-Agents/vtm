# scripts

Purpose: reserved location for developer and maintenance automation that should live with the repository.

Contents
- `generate_code_reference.py`: AST-driven generator for `docs/code-reference.md`.
- `vtm_local_patcher.py`: Single-shot OpenAI-compatible patch generator that reads typed `HarnessTaskPack` files during coding benchmark runs.
- `vtm_scaffold_bridge.py`: Writes richer scaffold-facing task bundles/briefs from `HarnessTaskPack` files and can delegate to an external agent command.
