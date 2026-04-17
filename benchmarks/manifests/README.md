# benchmarks/manifests

Purpose: maintained benchmark manifests. Each file defines the corpus source, commit pairs, and optional coding-task cases for repeatable runs.

Contents
- `synthetic-smoke.json`: deterministic local smoke corpus for retrieval, drift, and quick coding-path verification.
- `oss-python.json`: pinned open-source Python retrieval/drift corpus for paper-style reference runs.

Generated SWE-bench Lite manifests should live outside this checked-in directory, for example under `.benchmarks/generated/`, because they embed local cache paths and prepared refs.
