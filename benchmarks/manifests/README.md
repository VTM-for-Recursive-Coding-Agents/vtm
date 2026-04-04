# benchmarks/manifests

Purpose: benchmark input manifests. Each file defines the corpus source, commit pairs, and optional coding-task cases for a repeatable benchmark run.

Contents
- `oss-python.json`: Pinned retrieval benchmark corpus for real open-source Python repositories and optional embedding or reranked retrieval runs.
- `synthetic-smoke.json`: Local synthetic smoke corpus used for deterministic lexical retrieval, embedding retrieval, drift, and a multi-task local coding benchmark suite in tests and quick manual runs.
- Generated SWE-bench Lite manifests are expected to live outside this checked-in directory, for example `.benchmarks/generated/swebench-lite.json`, because they embed local cache paths and generated refs.

The CLI can scope these manifests to selected repos and commit pairs with `--repo` and `--pair` without modifying the checked-in manifest files.
