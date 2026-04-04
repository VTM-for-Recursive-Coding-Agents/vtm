# benchmarks

Purpose: checked-in benchmark assets that feed the benchmark runner. This directory is for durable inputs, not run outputs.

Contents
- `.gitkeep`: Keeps the top-level benchmark asset directory present in git.
- `manifests/`: Versioned benchmark manifests consumed by `python -m vtm.benchmarks.run`.

Benchmark run outputs, task packs, workspaces, executor artifact files, `predictions.jsonl`, and normalized SWE-bench harness artifacts are written under the chosen runtime output directory, not back into this checked-in asset tree.

Maintained benchmark command recipes live in [`../docs/benchmark-recipes.md`](../docs/benchmark-recipes.md).
