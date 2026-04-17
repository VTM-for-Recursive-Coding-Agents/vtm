# src/vtm/benchmarks

Purpose: manifest-driven orchestration for the final retrieval, drift, and coding study.

Maintained surface: OpenRouter-only inference, verified lexical memory study first, optional `naive_lexical` and `lexical_rlm_rerank` ablations, no Codex path, no embeddings, no terminal track.

Use this README for package ownership only. For maintained commands, use [`docs/benchmark-recipes.md`](../../../docs/benchmark-recipes.md).

Start here
- `models.py`: manifest, config, and result records
- `runner.py`: public `BenchmarkRunner`
- `suite_execution.py`: retrieval, drift, and coding dispatch
- `run.py`: `vtm-bench`
- `matrix.py`: `vtm-bench-matrix`

Maintained benchmark inputs
- `benchmarks/manifests/synthetic-smoke.json`: local smoke corpus used in tests and quick verification runs
- `benchmarks/manifests/oss-python.json`: pinned OSS retrieval/drift reference corpus
- generated SWE-bench Lite manifests: targeted external coding evaluation inputs created by `vtm-prepare-swebench-lite`
