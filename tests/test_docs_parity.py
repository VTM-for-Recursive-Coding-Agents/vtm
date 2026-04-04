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
