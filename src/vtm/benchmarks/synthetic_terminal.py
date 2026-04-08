"""Deterministic terminal-style coding benchmark corpus."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class SyntheticTerminalSmokeCorpus:
    """Materialize a deterministic local repo with terminal-style coding tasks."""

    def materialize(self, repo_root: Path) -> None:
        if repo_root.exists():
            shutil.rmtree(repo_root)
        repo_root.mkdir(parents=True, exist_ok=True)
        self._run(repo_root, "git", "init", "-b", "main")
        self._run(repo_root, "git", "config", "user.name", "VTM Benchmarks")
        self._run(repo_root, "git", "config", "user.email", "vtm-benchmarks@example.com")
        self._write(
            repo_root / "README.md",
            "# Synthetic terminal benchmark corpus\n",
        )
        self._write(repo_root / "tests" / "__init__.py", "")
        self._write(repo_root / "configs" / ".gitkeep", "")
        self._write(repo_root / "scripts" / ".gitkeep", "")
        self._commit(repo_root, "initial terminal benchmark scaffold", "terminal-initial")

        self._add_export_path_task(repo_root)
        self._add_retry_default_task(repo_root)
        self._add_feature_flag_task(repo_root)
        self._add_unique_hosts_task(repo_root)
        self._add_cleanup_force_task(repo_root)
        self._add_release_notes_task(repo_root)
        self._add_service_runtime_config_task(repo_root)
        self._add_script_locator_task(repo_root)
        self._add_deploy_command_task(repo_root)
        self._add_platform_matrix_task(repo_root)
        self._add_audit_summary_task(repo_root)
        self._add_cleanup_plan_task(repo_root)

    def _add_export_path_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "export_paths.py",
            """def build_export_path(report_name: str) -> str:
    slug = report_name.strip().replace(" ", "-")
    return f"exports/{slug}.txt"
""",
        )
        self._write(
            repo_root / "tests" / "test_export_paths.py",
            """import unittest

from export_paths import build_export_path


class ExportPathsTests(unittest.TestCase):
    def test_build_export_path_normalizes_name(self) -> None:
        self.assertEqual(
            build_export_path("Daily Summary"),
            "exports/daily-summary.json",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce export path bug", "terminal-export-path-bug")
        self._write(
            repo_root / "export_paths.py",
            """def build_export_path(report_name: str) -> str:
    slug = report_name.strip().lower().replace(" ", "-")
    return f"exports/{slug}.json"
""",
        )
        self._commit(repo_root, "fix export path bug", "terminal-export-path-fix")

    def _add_retry_default_task(self, repo_root: Path) -> None:
        self._write(repo_root / "terminal_defaults.py", "DEFAULT_RETRY_LIMIT = 4\n")
        self._write(
            repo_root / "retry_policy.py",
            """from terminal_defaults import DEFAULT_RETRY_LIMIT


def choose_retry_limit(value: int | None) -> int:
    if value is None:
        return 0
    return value
""",
        )
        self._write(
            repo_root / "tests" / "test_retry_policy.py",
            """import unittest

from retry_policy import choose_retry_limit


class RetryPolicyTests(unittest.TestCase):
    def test_choose_retry_limit_uses_default(self) -> None:
        self.assertEqual(choose_retry_limit(None), 4)


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce retry default bug", "terminal-retry-default-bug")
        self._write(
            repo_root / "retry_policy.py",
            """from terminal_defaults import DEFAULT_RETRY_LIMIT


def choose_retry_limit(value: int | None) -> int:
    if value is None:
        return DEFAULT_RETRY_LIMIT
    return value
""",
        )
        self._commit(repo_root, "fix retry default bug", "terminal-retry-default-fix")

    def _add_feature_flag_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "feature_flags.py",
            """def is_feature_enabled(raw: str | None) -> bool:
    if raw is None:
        return False
    return bool(raw.strip())
""",
        )
        self._write(
            repo_root / "tests" / "test_feature_flags.py",
            """import unittest

from feature_flags import is_feature_enabled


class FeatureFlagTests(unittest.TestCase):
    def test_false_string_is_disabled(self) -> None:
        self.assertIs(is_feature_enabled("false"), False)


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce feature flag parsing bug", "terminal-feature-flag-bug")
        self._write(
            repo_root / "feature_flags.py",
            """def is_feature_enabled(raw: str | None) -> bool:
    if raw is None:
        return False
    normalized = raw.strip().lower()
    return normalized in {"1", "true", "yes", "on"}
""",
        )
        self._commit(repo_root, "fix feature flag parsing bug", "terminal-feature-flag-fix")

    def _add_unique_hosts_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "host_index.py",
            """def unique_sorted_hosts(hosts: list[str]) -> list[str]:
    return sorted(hosts)
""",
        )
        self._write(
            repo_root / "tests" / "test_host_index.py",
            """import unittest

from host_index import unique_sorted_hosts


class HostIndexTests(unittest.TestCase):
    def test_unique_sorted_hosts_removes_duplicates(self) -> None:
        self.assertEqual(
            unique_sorted_hosts(["beta", "alpha", "beta"]),
            ["alpha", "beta"],
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce unique host bug", "terminal-unique-hosts-bug")
        self._write(
            repo_root / "host_index.py",
            """def unique_sorted_hosts(hosts: list[str]) -> list[str]:
    return sorted(set(hosts))
""",
        )
        self._commit(repo_root, "fix unique host bug", "terminal-unique-hosts-fix")

    def _add_cleanup_force_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "cleanup_script.py",
            """import json
from pathlib import Path


def build_cleanup_command(config_path: str) -> list[str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    command = ["cleanup", config["target"]]
    if config.get("force"):
        command.append("--force")
    return command
""",
        )
        self._write(
            repo_root / "configs" / "cleanup.json",
            """{
  "force_cleanup": true,
  "target": "build-cache"
}
""",
        )
        self._write(
            repo_root / "tests" / "test_cleanup_script.py",
            """import unittest

from cleanup_script import build_cleanup_command


class CleanupScriptTests(unittest.TestCase):
    def test_build_cleanup_command_uses_force_cleanup_flag(self) -> None:
        self.assertEqual(
            build_cleanup_command("configs/cleanup.json"),
            ["cleanup", "build-cache", "--force"],
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce cleanup force bug", "terminal-cleanup-force-bug")
        self._write(
            repo_root / "cleanup_script.py",
            """import json
from pathlib import Path


def build_cleanup_command(config_path: str) -> list[str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    command = ["cleanup", config["target"]]
    if config.get("force_cleanup"):
        command.append("--force")
    return command
""",
        )
        self._commit(repo_root, "fix cleanup force bug", "terminal-cleanup-force-fix")

    def _add_release_notes_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "slugify.py",
            """def slugify_title(title: str) -> str:
    return title.strip().replace(" ", "-")
""",
        )
        self._write(
            repo_root / "release_notes.py",
            """from slugify import slugify_title


def build_release_notes_path(name: str) -> str:
    return f"notes/{slugify_title(name)}.md"
""",
        )
        self._write(
            repo_root / "tests" / "test_release_notes.py",
            """import unittest

from release_notes import build_release_notes_path


class ReleaseNotesTests(unittest.TestCase):
    def test_build_release_notes_path_normalizes_title(self) -> None:
        self.assertEqual(
            build_release_notes_path("Sprint Update"),
            "notes/sprint-update.md",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce release notes slug bug", "terminal-release-notes-bug")
        self._write(
            repo_root / "slugify.py",
            """def slugify_title(title: str) -> str:
    return title.strip().lower().replace(" ", "-")
""",
        )
        self._commit(repo_root, "fix release notes slug bug", "terminal-release-notes-fix")

    def _add_service_runtime_config_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "service_runtime.py",
            """import json
from pathlib import Path


def load_retry_window_seconds(config_path: str) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return int(config["retry_window_seconds"])
""",
        )
        self._write(
            repo_root / "configs" / "service_runtime.json",
            """{
  "retry_window_seconds": 0
}
""",
        )
        self._write(
            repo_root / "tests" / "test_service_runtime.py",
            """import unittest

from service_runtime import load_retry_window_seconds


class ServiceRuntimeTests(unittest.TestCase):
    def test_load_retry_window_seconds_uses_restored_value(self) -> None:
        self.assertEqual(
            load_retry_window_seconds("configs/service_runtime.json"),
            30,
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce service runtime config bug",
            "terminal-service-runtime-config-bug",
        )
        self._write(
            repo_root / "configs" / "service_runtime.json",
            """{
  "retry_window_seconds": 30
}
""",
        )
        self._commit(
            repo_root,
            "fix service runtime config bug",
            "terminal-service-runtime-config-fix",
        )

    def _add_script_locator_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "workspace_scan.py",
            """from pathlib import Path


def find_task_script(root: str) -> str | None:
    base = Path(root)
    for path in sorted(base.glob("scripts/*.py")):
        if path.name == "sync_task.py":
            return path.relative_to(base).as_posix()
    return None
""",
        )
        self._write(
            repo_root / "scripts" / "tasks" / "sync_task.py",
            "print('sync-task')\n",
        )
        self._write(
            repo_root / "tests" / "test_workspace_scan.py",
            """import unittest

from workspace_scan import find_task_script


class WorkspaceScanTests(unittest.TestCase):
    def test_find_task_script_recurses_under_scripts(self) -> None:
        self.assertEqual(find_task_script("."), "scripts/tasks/sync_task.py")


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce script locator bug", "terminal-script-locator-bug")
        self._write(
            repo_root / "workspace_scan.py",
            """from pathlib import Path


def find_task_script(root: str) -> str | None:
    base = Path(root)
    for path in sorted(base.rglob("*.py")):
        if path.name == "sync_task.py" and "scripts" in path.parts:
            return path.relative_to(base).as_posix()
    return None
""",
        )
        self._commit(repo_root, "fix script locator bug", "terminal-script-locator-fix")

    def _add_deploy_command_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "deploy_paths.py",
            """def build_deploy_log_path(environment: str) -> str:
    return f"logs/{environment}.txt"
""",
        )
        self._write(
            repo_root / "deploy_runner.py",
            """from deploy_paths import build_deploy_log_path


def build_deploy_command(environment: str, dry_run: bool) -> list[str]:
    command = [
        "deploy",
        "--env",
        environment,
        "--log",
        build_deploy_log_path(environment),
    ]
    if dry_run:
        return command
    return command
""",
        )
        self._write(
            repo_root / "tests" / "test_deploy_runner.py",
            """import unittest

from deploy_runner import build_deploy_command


class DeployRunnerTests(unittest.TestCase):
    def test_build_deploy_command_adds_plan_and_log_extension(self) -> None:
        self.assertEqual(
            build_deploy_command("prod", dry_run=True),
            ["deploy", "--env", "prod", "--log", "logs/prod.log", "--plan"],
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce deploy command bug", "terminal-deploy-command-bug")
        self._write(
            repo_root / "deploy_paths.py",
            """def build_deploy_log_path(environment: str) -> str:
    suffix = "log"
    return f"logs/{environment}.{suffix}"
""",
        )
        self._write(
            repo_root / "deploy_runner.py",
            """from deploy_paths import build_deploy_log_path


def build_deploy_command(environment: str, dry_run: bool) -> list[str]:
    command = [
        "deploy",
        "--env",
        environment,
        "--log",
        build_deploy_log_path(environment),
    ]
    if dry_run:
        command.append("--plan")
    return command
""",
        )
        self._commit(repo_root, "fix deploy command bug", "terminal-deploy-command-fix")

    def _add_platform_matrix_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "platform_rules.py",
            """ARCHIVED_TARGETS = {"windows-2019"}
""",
        )
        self._write(
            repo_root / "platform_matrix.py",
            """from platform_rules import ARCHIVED_TARGETS


def active_targets(values: list[str]) -> list[str]:
    selected: list[str] = []
    for value in values:
        if value in ARCHIVED_TARGETS:
            continue
        selected.append(value)
    return sorted(selected)
""",
        )
        self._write(
            repo_root / "tests" / "test_platform_matrix.py",
            """import unittest

from platform_matrix import active_targets


class PlatformMatrixTests(unittest.TestCase):
    def test_active_targets_filters_archived_and_duplicates(self) -> None:
        self.assertEqual(
            active_targets(["linux-2024", "windows-2019", "linux-2024"]),
            ["linux-2024"],
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce platform matrix bug", "terminal-platform-matrix-bug")
        self._write(
            repo_root / "platform_matrix.py",
            """from platform_rules import ARCHIVED_TARGETS


def active_targets(values: list[str]) -> list[str]:
    selected = {
        value
        for value in values
        if value not in ARCHIVED_TARGETS
    }
    return sorted(selected)
""",
        )
        self._commit(repo_root, "fix platform matrix bug", "terminal-platform-matrix-fix")

    def _add_audit_summary_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "audit_sources.py",
            """def load_warning_counts() -> dict[str, dict[str, int]]:
    return {"warnings": {"infra": 2, "lint": 1}}
""",
        )
        self._write(
            repo_root / "audit_report.py",
            """from audit_sources import load_warning_counts


def render_summary() -> str:
    counts = load_warning_counts()
    warnings = counts.get("warnings", {})
    return f"infra={warnings.get('infra', 0)} lint={counts.get('lint', 0)}"
""",
        )
        self._write(
            repo_root / "tests" / "test_audit_report.py",
            """import unittest

from audit_report import render_summary


class AuditReportTests(unittest.TestCase):
    def test_render_summary_reads_nested_warning_counts(self) -> None:
        self.assertEqual(render_summary(), "infra=2 lint=1")


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce audit summary bug", "terminal-audit-summary-bug")
        self._write(
            repo_root / "audit_report.py",
            """from audit_sources import load_warning_counts


def render_summary() -> str:
    counts = load_warning_counts()
    warnings = counts.get("warnings", {})
    return f"infra={warnings.get('infra', 0)} lint={warnings.get('lint', 0)}"
""",
        )
        self._commit(repo_root, "fix audit summary bug", "terminal-audit-summary-fix")

    def _add_cleanup_plan_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "cleanup_plan.py",
            """import json
from pathlib import Path


def build_cleanup_targets(config_path: str) -> list[str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ignored = tuple(config["ignored_prefixes"])
    targets: list[str] = []
    for value in config["targets"]:
        if value.startswith(ignored):
            continue
        targets.append(value)
    return sorted(targets)
""",
        )
        self._write(
            repo_root / "configs" / "cleanup_targets.json",
            """{
  "ignored_prefixes": ["src/", "venv/"],
  "temporary_prefixes": ["generated/", "tmp/"],
  "targets": [
    "src/app.py",
    "generated/report.json",
    "tmp/cache.db",
    "build/output.txt",
    "build/output.txt"
  ]
}
""",
        )
        self._write(
            repo_root / "tests" / "test_cleanup_plan.py",
            """import unittest

from cleanup_plan import build_cleanup_targets


class CleanupPlanTests(unittest.TestCase):
    def test_build_cleanup_targets_ignores_temporary_prefixes_and_duplicates(self) -> None:
        self.assertEqual(
            build_cleanup_targets("configs/cleanup_targets.json"),
            ["build/output.txt"],
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce cleanup plan bug", "terminal-cleanup-plan-bug")
        self._write(
            repo_root / "cleanup_plan.py",
            """import json
from pathlib import Path


def build_cleanup_targets(config_path: str) -> list[str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ignored = tuple(
        list(config["ignored_prefixes"]) + list(config["temporary_prefixes"])
    )
    targets = {
        value
        for value in config["targets"]
        if not value.startswith(ignored)
    }
    return sorted(targets)
""",
        )
        self._commit(repo_root, "fix cleanup plan bug", "terminal-cleanup-plan-fix")

    def _write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

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


__all__ = ["SyntheticTerminalSmokeCorpus"]
