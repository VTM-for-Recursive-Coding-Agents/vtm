# src/vtm/benchmarks

Purpose: manifest-driven orchestration for the final retrieval, drift, and coding study.

Maintained surface: OpenRouter-only inference, static retrieval, drift verification, drifted retrieval, and controlled coding-drift. Earlier branches explored Codex paths, embeddings, and terminal-only tracks, but those are outside the maintained surface now.

Experimental pilots such as LiveCodeBench DSPy and LongCoT now live under the repo-root `experiments/` directory. The legacy `vtm.benchmarks.*_pilot` import paths remain as thin compatibility loaders so scripts and tests do not need to change.

Use this README for package ownership only. For maintained commands, use [`docs/benchmark-recipes.md`](../../../docs/benchmark-recipes.md).

Start here
- `models.py`: manifest, config, and result records
- `runner.py`: public `BenchmarkRunner`
- `suite_execution.py`: retrieval, drift, and coding dispatch
- `run.py`: `vtm-bench`
- `matrix.py`: `vtm-bench-matrix`

Maintained benchmark inputs
- `benchmarks/manifests/synthetic-smoke.json`: local smoke corpus used in tests and quick verification runs
- `benchmarks/manifests/controlled-coding-drift.json`: maintained coding benchmark for the final paper
- `benchmarks/manifests/oss-python.json`: pinned OSS retrieval/drift reference corpus
