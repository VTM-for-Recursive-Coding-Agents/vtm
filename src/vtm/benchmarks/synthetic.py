from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class SyntheticPythonSmokeCorpus:
    def materialize(self, repo_root: Path) -> None:
        if repo_root.exists():
            shutil.rmtree(repo_root)
        repo_root.mkdir(parents=True, exist_ok=True)
        self._run(repo_root, "git", "init", "-b", "main")
        self._run(repo_root, "git", "config", "user.name", "VTM Benchmarks")
        self._run(repo_root, "git", "config", "user.email", "vtm-benchmarks@example.com")

        self._write_initial_layout(repo_root)
        self._commit(repo_root, "initial synthetic corpus", "smoke-initial")

        (repo_root / "README.md").write_text(
            "# Synthetic benchmark corpus\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "stable symbol control change", "smoke-stable")

        (repo_root / "whitespace_module.py").write_text(
            "\n"
            "def whitespace_target() -> int:\n"
            '    """Return a value while only moving whitespace."""\n'
            "    return 2\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "whitespace relocation", "smoke-whitespace")

        (repo_root / "relocate_module.py").write_text(
            "def helper() -> int:\n"
            "    return 10\n\n"
            "\n"
            "def relocate_target() -> int:\n"
            '    """Return helper output from a relocated function."""\n'
            "    return helper()\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "structural relocation", "smoke-relocated")

        (repo_root / "semantic_module.py").write_text(
            "def semantic_target() -> int:\n"
            '    """Return a value that later changes semantically."""\n'
            "    return 99\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "semantic change", "smoke-semantic")

        (repo_root / "delete_module.py").write_text(
            "# delete_target intentionally removed in this commit\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "symbol deletion", "smoke-deleted")

        tests_dir = repo_root / "tests"
        tests_dir.mkdir(exist_ok=True)
        self._add_increment_bug_task(repo_root, tests_dir)
        self._add_default_sentinel_task(repo_root, tests_dir)
        self._add_branch_condition_task(repo_root, tests_dir)
        self._add_path_builder_task(repo_root, tests_dir)
        self._add_collection_task(repo_root, tests_dir)

    def _add_increment_bug_task(self, repo_root: Path, tests_dir: Path) -> None:
        (repo_root / "bugfix_module.py").write_text(
            "def buggy_increment(value: int) -> int:\n"
            '    """Return value plus one."""\n'
            "    return value\n",
            encoding="utf-8",
        )
        (tests_dir / "test_bugfix_module.py").write_text(
            "import unittest\n\n"
            "from bugfix_module import buggy_increment\n\n\n"
            "class BugfixModuleTests(unittest.TestCase):\n"
            "    def test_buggy_increment(self) -> None:\n"
            "        self.assertEqual(buggy_increment(3), 4)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "introduce coding benchmark bug", "smoke-bug")

        (repo_root / "bugfix_module.py").write_text(
            "def buggy_increment(value: int) -> int:\n"
            '    """Return value plus one."""\n'
            "    return value + 1\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "fix coding benchmark bug", "smoke-bugfix")

    def _add_default_sentinel_task(self, repo_root: Path, tests_dir: Path) -> None:
        (repo_root / "limit_module.py").write_text(
            "def normalize_limit(value: int | None, default: int = 25) -> int:\n"
            '    """Return an explicit limit, falling back to the default when missing."""\n'
            "    if value is None:\n"
            "        return 0\n"
            "    return value\n",
            encoding="utf-8",
        )
        (tests_dir / "test_limit_module.py").write_text(
            "import unittest\n\n"
            "from limit_module import normalize_limit\n\n\n"
            "class LimitModuleTests(unittest.TestCase):\n"
            "    def test_normalize_limit_uses_default_for_none(self) -> None:\n"
            "        self.assertEqual(normalize_limit(None), 25)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "introduce default sentinel bug", "smoke-sentinel-bug")

        (repo_root / "limit_module.py").write_text(
            "def normalize_limit(value: int | None, default: int = 25) -> int:\n"
            '    """Return an explicit limit, falling back to the default when missing."""\n'
            "    if value is None:\n"
            "        return default\n"
            "    return value\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "fix default sentinel bug", "smoke-sentinel-fix")

    def _add_branch_condition_task(self, repo_root: Path, tests_dir: Path) -> None:
        (repo_root / "status_module.py").write_text(
            "def describe_status(active: bool, suspended: bool) -> str:\n"
            '    """Describe whether an account should be treated as active."""\n'
            "    if active or suspended:\n"
            "        return 'active'\n"
            "    return 'inactive'\n",
            encoding="utf-8",
        )
        (tests_dir / "test_status_module.py").write_text(
            "import unittest\n\n"
            "from status_module import describe_status\n\n\n"
            "class StatusModuleTests(unittest.TestCase):\n"
            "    def test_suspended_account_is_not_active(self) -> None:\n"
            "        self.assertEqual(\n"
            "            describe_status(active=False, suspended=True),\n"
            "            'inactive',\n"
            "        )\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "introduce branch condition bug", "smoke-branch-bug")

        (repo_root / "status_module.py").write_text(
            "def describe_status(active: bool, suspended: bool) -> str:\n"
            '    """Describe whether an account should be treated as active."""\n'
            "    if active and not suspended:\n"
            "        return 'active'\n"
            "    return 'inactive'\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "fix branch condition bug", "smoke-branch-fix")

    def _add_path_builder_task(self, repo_root: Path, tests_dir: Path) -> None:
        (repo_root / "report_module.py").write_text(
            "from pathlib import PurePosixPath\n\n\n"
            "def build_report_path(name: str) -> str:\n"
            '    """Build a stable report path for a named report."""\n'
            "    slug = name.strip().replace(' ', '-')\n"
            "    return str(PurePosixPath('reports') / f'{slug}.txt')\n",
            encoding="utf-8",
        )
        (tests_dir / "test_report_module.py").write_text(
            "import unittest\n\n"
            "from report_module import build_report_path\n\n\n"
            "class ReportModuleTests(unittest.TestCase):\n"
            "    def test_build_report_path_normalizes_name(self) -> None:\n"
            "        self.assertEqual(\n"
            "            build_report_path('Weekly Summary'),\n"
            "            'reports/weekly-summary.json',\n"
            "        )\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "introduce report path bug", "smoke-path-bug")

        (repo_root / "report_module.py").write_text(
            "from pathlib import PurePosixPath\n\n\n"
            "def build_report_path(name: str) -> str:\n"
            '    """Build a stable report path for a named report."""\n'
            "    slug = name.strip().lower().replace(' ', '-')\n"
            "    return str(PurePosixPath('reports') / f'{slug}.json')\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "fix report path bug", "smoke-path-fix")

    def _add_collection_task(self, repo_root: Path, tests_dir: Path) -> None:
        (repo_root / "collection_module.py").write_text(
            "def unique_sorted(values: list[str]) -> list[str]:\n"
            '    """Return values in sorted order without duplicates."""\n'
            "    return sorted(values)\n",
            encoding="utf-8",
        )
        (tests_dir / "test_collection_module.py").write_text(
            "import unittest\n\n"
            "from collection_module import unique_sorted\n\n\n"
            "class CollectionModuleTests(unittest.TestCase):\n"
            "    def test_unique_sorted_removes_duplicates(self) -> None:\n"
            "        self.assertEqual(\n"
            "            unique_sorted(['pear', 'apple', 'pear']),\n"
            "            ['apple', 'pear'],\n"
            "        )\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "introduce collection normalization bug", "smoke-collection-bug")

        (repo_root / "collection_module.py").write_text(
            "def unique_sorted(values: list[str]) -> list[str]:\n"
            '    """Return values in sorted order without duplicates."""\n'
            "    return sorted(set(values))\n",
            encoding="utf-8",
        )
        self._commit(repo_root, "fix collection normalization bug", "smoke-collection-fix")

    def _write_initial_layout(self, repo_root: Path) -> None:
        (repo_root / "stable_module.py").write_text(
            "def stable_target() -> int:\n"
            '    """Return a stable value for retrieval baselines."""\n'
            "    return 1\n",
            encoding="utf-8",
        )
        (repo_root / "whitespace_module.py").write_text(
            "def whitespace_target() -> int:\n"
            '    """Return a value while only moving whitespace."""\n'
            "    return 2\n",
            encoding="utf-8",
        )
        (repo_root / "relocate_module.py").write_text(
            "def relocate_target() -> int:\n"
            '    """Return helper output from a relocated function."""\n'
            "    return helper()\n\n\n"
            "def helper() -> int:\n"
            "    return 10\n",
            encoding="utf-8",
        )
        (repo_root / "semantic_module.py").write_text(
            "def semantic_target() -> int:\n"
            '    """Return a value that later changes semantically."""\n'
            "    return 3\n",
            encoding="utf-8",
        )
        (repo_root / "delete_module.py").write_text(
            "def delete_target() -> str:\n"
            '    """Return a marker that will later disappear."""\n'
            "    return 'keep-me'\n",
            encoding="utf-8",
        )

    def _commit(self, repo_root: Path, message: str, tag: str) -> None:
        self._run(repo_root, "git", "add", ".")
        self._run(repo_root, "git", "commit", "-m", message)
        self._run(repo_root, "git", "tag", "-f", tag)

    def _run(self, cwd: Path, *args: str) -> str:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
