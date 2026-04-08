from __future__ import annotations

import re
from pathlib import Path

from vtm.benchmarks import BenchmarkManifest

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
MANIFEST_PATH_RE = re.compile(r"benchmarks/manifests/[A-Za-z0-9._-]+\.json")
IGNORED_MD_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "__pycache__",
}
EXPECTED_CORE_DOCS = (
    Path("docs/harness.md"),
    Path("src/vtm/harness/README.md"),
)
EXPECTED_BENCHMARK_DOCS = (
    Path("docs/benchmark-recipes.md"),
    Path("src/vtm/benchmarks/README.md"),
    Path("docs/decisions/0010-multi-attempt-coding-benchmarks.md"),
)


def _first_python_fence(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```python\n(.*?)```", text, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"no python code fence found in {path}")
    return match.group(1)


def _local_markdown_targets(path: Path) -> tuple[Path, ...]:
    targets: list[Path] = []
    text = path.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK_RE.findall(text):
        target = raw_target.split("#", maxsplit=1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        targets.append((path.parent / target).resolve())
    return tuple(targets)


def _markdown_paths() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(Path(".").rglob("*.md"))
        if not any(part in IGNORED_MD_PARTS for part in path.parts)
    )


def test_runtime_example_code_fence_executes() -> None:
    code = _first_python_fence(Path("docs/runtime-example.md"))
    globals_dict = {"__name__": "__main__"}
    exec(code, globals_dict, globals_dict)


def test_repo_markdown_local_links_resolve() -> None:
    markdown_paths = _markdown_paths()
    missing_targets = [
        target
        for path in markdown_paths
        for target in _local_markdown_targets(path)
        if not target.exists()
    ]
    assert missing_targets == []


def test_markdown_manifest_references_exist_and_load() -> None:
    manifest_paths = sorted(
        {
            manifest_path
            for markdown_path in _markdown_paths()
            for manifest_path in MANIFEST_PATH_RE.findall(
                markdown_path.read_text(encoding="utf-8")
            )
        }
    )

    assert manifest_paths != []
    for manifest_path in manifest_paths:
        resolved = Path(manifest_path)
        assert resolved.exists()
        manifest = BenchmarkManifest.from_path(resolved)
        assert manifest.repos


def test_harness_docs_exist_and_reference_public_contracts() -> None:
    for path in EXPECTED_CORE_DOCS:
        assert path.exists()

    harness_doc = Path("docs/harness.md").read_text(encoding="utf-8")
    assert "HarnessTaskPack" in harness_doc
    assert "ExecutorRequest" in harness_doc
    assert "ExecutorResult" in harness_doc
    assert "TraceManifest" in harness_doc


def test_readme_keeps_kernel_and_harness_boundaries_distinct() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "vtm.harness" in readme
    assert "terminal-smoke" in readme
    assert "from vtm import BenchmarkRunner" not in readme
    assert "from vtm import OpenAIEmbeddingAdapter" not in readme


def test_benchmark_docs_cover_terminal_smoke_attempts_and_pass_k() -> None:
    for path in EXPECTED_BENCHMARK_DOCS:
        assert path.exists()

    recipes = Path("docs/benchmark-recipes.md").read_text(encoding="utf-8")
    assert "benchmarks/manifests/terminal-smoke.json" in recipes
    assert "--attempts" in recipes
    assert "--pass-k" in recipes

    harness_doc = Path("docs/harness.md").read_text(encoding="utf-8")
    assert "attempts.jsonl" in harness_doc
    assert "artifact_root" in harness_doc

    benchmark_readme = Path("src/vtm/benchmarks/README.md").read_text(encoding="utf-8")
    assert "terminal-smoke.json" in benchmark_readme
    assert "attempts.jsonl" in benchmark_readme

    audit = Path("docs/current-state-audit.md").read_text(encoding="utf-8")
    assert "pass@k" in audit
    assert "pass@k controller" not in audit
