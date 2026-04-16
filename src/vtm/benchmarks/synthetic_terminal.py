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
        self._write(repo_root / "generated" / ".gitkeep", "")
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
        self._add_service_label_task(repo_root)
        self._add_notification_policy_task(repo_root)
        self._add_release_reviewers_task(repo_root)
        self._add_shell_daily_report_task(repo_root)
        self._add_shell_dependency_lock_task(repo_root)
        self._add_shell_runtime_snapshot_task(repo_root)
        self._add_shell_cleanup_inventory_task(repo_root)
        self._add_shell_release_manifest_task(repo_root)
        self._add_shell_docs_index_task(repo_root)
        self._add_shell_service_bundle_task(repo_root)
        self._add_shell_command_matrix_task(repo_root)
        self._add_shell_team_roster_task(repo_root)
        self._add_shell_incident_rollup_task(repo_root)
        self._add_shell_alert_routes_task(repo_root)
        self._add_shell_environment_catalog_task(repo_root)

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

    def _add_service_label_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "service_labels.py",
            """def canonical_service_label(raw: str) -> str:
    return raw.strip().replace(" ", "_")
""",
        )
        self._write(
            repo_root / "tests" / "test_service_labels.py",
            """import unittest

from service_labels import canonical_service_label


class ServiceLabelsTests(unittest.TestCase):
    def test_canonical_service_label_normalizes_case_and_separators(self) -> None:
        self.assertEqual(
            canonical_service_label(" API Gateway "),
            "api-gateway",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(repo_root, "introduce service label bug", "terminal-service-label-bug")
        self._write(
            repo_root / "service_labels.py",
            """def canonical_service_label(raw: str) -> str:
    normalized = raw.strip().lower().replace("_", " ")
    return "-".join(part for part in normalized.split() if part)
""",
        )
        self._commit(repo_root, "fix service label bug", "terminal-service-label-fix")

    def _add_notification_policy_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "notification_policy.py",
            """import json
from pathlib import Path


def load_batch_limit(config_path: str) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return int(config["notify_batch_size"])
""",
        )
        self._write(
            repo_root / "configs" / "notification_policy.json",
            """{
  "notify_batch_size": 5,
  "retry_window_seconds": 30
}
""",
        )
        self._write(
            repo_root / "tests" / "test_notification_policy.py",
            """import unittest

from notification_policy import load_batch_limit


class NotificationPolicyTests(unittest.TestCase):
    def test_load_batch_limit_uses_restored_config_value(self) -> None:
        self.assertEqual(
            load_batch_limit("configs/notification_policy.json"),
            25,
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce notification policy regression",
            "terminal-notification-policy-bug",
        )
        self._write(
            repo_root / "configs" / "notification_policy.json",
            """{
  "notify_batch_size": 25,
  "retry_window_seconds": 30
}
""",
        )
        self._commit(
            repo_root,
            "fix notification policy regression",
            "terminal-notification-policy-fix",
        )

    def _add_release_reviewers_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "team_aliases.py",
            """ALIASES = {
    "platform-api": "api",
    "web-ui": "web",
}


def canonical_team_name(team: str) -> str:
    return team
""",
        )
        self._write(
            repo_root / "release_reviewers.py",
            """import json
from pathlib import Path

from team_aliases import canonical_team_name


def build_release_reviewers(config_path: str) -> list[str]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    reviewers = []
    for team in payload["teams"]:
        if team in payload["archived_teams"]:
            continue
        reviewers.append(f"{canonical_team_name(team)}-oncall")
    return sorted(reviewers)
""",
        )
        self._write(
            repo_root / "configs" / "release_reviewers.json",
            """{
  "archived_teams": ["legacy"],
  "teams": ["web-ui", "platform-api", "web-ui", "legacy"]
}
""",
        )
        self._write(
            repo_root / "tests" / "test_release_reviewers.py",
            """import unittest

from release_reviewers import build_release_reviewers


class ReleaseReviewersTests(unittest.TestCase):
    def test_build_release_reviewers_resolves_aliases_and_deduplicates(self) -> None:
        self.assertEqual(
            build_release_reviewers("configs/release_reviewers.json"),
            ["api-oncall", "web-oncall"],
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce release reviewer alias bug",
            "terminal-release-reviewers-bug",
        )
        self._write(
            repo_root / "team_aliases.py",
            """ALIASES = {
    "platform-api": "api",
    "web-ui": "web",
}


def canonical_team_name(team: str) -> str:
    return ALIASES.get(team, team)
""",
        )
        self._write(
            repo_root / "release_reviewers.py",
            """import json
from pathlib import Path

from team_aliases import canonical_team_name


def build_release_reviewers(config_path: str) -> list[str]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    reviewers = {
        f"{canonical_team_name(team)}-oncall"
        for team in payload["teams"]
        if team not in payload["archived_teams"]
    }
    return sorted(reviewers)
""",
        )
        self._commit(
            repo_root,
            "fix release reviewer alias bug",
            "terminal-release-reviewers-fix",
        )

    def _add_shell_daily_report_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "inputs" / "report_metrics.json",
            """{
  "title": "Daily Ops",
  "passed": 18,
  "failed": 2
}
""",
        )
        self._write(
            repo_root / "scripts" / "build_daily_report.py",
            """import json
from pathlib import Path


def render_report(payload: dict[str, object]) -> str:
    return (
        f"{payload['title']}\\n"
        f"passed={payload['passed']}\\n"
        f"failed={payload['failed']}\\n"
    )


def main() -> None:
    payload = json.loads(
        Path("inputs/report_metrics.json").read_text(encoding="utf-8")
    )
    Path("generated/daily_report.txt").write_text(
        render_report(payload),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "daily_report.txt",
            """Daily Ops
passed=18
failed=0
""",
        )
        self._write(
            repo_root / "tests" / "test_daily_report_refresh.py",
            """import unittest
from pathlib import Path


class DailyReportRefreshTests(unittest.TestCase):
    def test_daily_report_matches_generator_inputs(self) -> None:
        self.assertEqual(
            Path("generated/daily_report.txt").read_text(encoding="utf-8"),
            "Daily Ops\\npassed=18\\nfailed=2\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale daily report output",
            "terminal-shell-daily-report-bug",
        )
        self._run(repo_root, "python3", "scripts/build_daily_report.py")
        self._commit(
            repo_root,
            "refresh daily report output",
            "terminal-shell-daily-report-fix",
        )

    def _add_shell_dependency_lock_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "inputs" / "dependencies.json",
            """{
  "dependencies": [
    {"name": "requests", "version": "2.32.0"},
    {"name": "urllib3", "version": "2.2.1"}
  ]
}
""",
        )
        self._write(
            repo_root / "scripts" / "render_dependency_lock.py",
            """import json
from pathlib import Path


def main() -> None:
    payload = json.loads(
        Path("inputs/dependencies.json").read_text(encoding="utf-8")
    )
    lines = [
        f"{item['name']}=={item['version']}"
        for item in sorted(payload["dependencies"], key=lambda value: value["name"])
    ]
    Path("generated/dependency.lock").write_text(
        "\\n".join(lines) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "dependency.lock",
            "requests==2.31.0\n",
        )
        self._write(
            repo_root / "tests" / "test_dependency_lock_refresh.py",
            """import unittest
from pathlib import Path


class DependencyLockRefreshTests(unittest.TestCase):
    def test_dependency_lock_matches_declared_versions(self) -> None:
        self.assertEqual(
            Path("generated/dependency.lock").read_text(encoding="utf-8"),
            "requests==2.32.0\\nurllib3==2.2.1\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale dependency lock output",
            "terminal-shell-dependency-lock-bug",
        )
        self._run(repo_root, "python3", "scripts/render_dependency_lock.py")
        self._commit(
            repo_root,
            "refresh dependency lock output",
            "terminal-shell-dependency-lock-fix",
        )

    def _add_shell_runtime_snapshot_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "configs" / "runtime_flags.json",
            """{
  "retry_limit": 4,
  "feature_x": true,
  "region": "us-west-2"
}
""",
        )
        self._write(
            repo_root / "scripts" / "render_runtime_snapshot.py",
            """import json
from pathlib import Path


def main() -> None:
    payload = json.loads(
        Path("configs/runtime_flags.json").read_text(encoding="utf-8")
    )
    rendered = (
        f"FEATURE_X={'true' if payload['feature_x'] else 'false'}\\n"
        f"REGION={payload['region']}\\n"
        f"RETRY_LIMIT={payload['retry_limit']}\\n"
    )
    Path("generated/runtime_snapshot.env").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "runtime_snapshot.env",
            """FEATURE_X=false
REGION=us-west-2
RETRY_LIMIT=0
""",
        )
        self._write(
            repo_root / "tests" / "test_runtime_snapshot_refresh.py",
            """import unittest
from pathlib import Path


class RuntimeSnapshotRefreshTests(unittest.TestCase):
    def test_runtime_snapshot_matches_flags(self) -> None:
        self.assertEqual(
            Path("generated/runtime_snapshot.env").read_text(encoding="utf-8"),
            "FEATURE_X=true\\nREGION=us-west-2\\nRETRY_LIMIT=4\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale runtime snapshot output",
            "terminal-shell-runtime-snapshot-bug",
        )
        self._run(repo_root, "python3", "scripts/render_runtime_snapshot.py")
        self._commit(
            repo_root,
            "refresh runtime snapshot output",
            "terminal-shell-runtime-snapshot-fix",
        )

    def _add_shell_cleanup_inventory_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "inputs" / "cleanup_paths.json",
            """{
  "ignored_prefixes": ["cache/", "tmp/"],
  "paths": [
    "tmp/run.log",
    "cache/last-run.db",
    "build/output.log",
    "reports/summary.txt",
    "build/output.log"
  ]
}
""",
        )
        self._write(
            repo_root / "scripts" / "build_cleanup_inventory.py",
            """import json
from pathlib import Path


def main() -> None:
    payload = json.loads(
        Path("inputs/cleanup_paths.json").read_text(encoding="utf-8")
    )
    ignored = tuple(payload["ignored_prefixes"])
    selected = sorted(
        {
            value
            for value in payload["paths"]
            if not value.startswith(ignored)
        }
    )
    Path("generated/cleanup_inventory.txt").write_text(
        "\\n".join(selected) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "cleanup_inventory.txt",
            """build/output.log
reports/summary.txt
tmp/run.log
""",
        )
        self._write(
            repo_root / "tests" / "test_cleanup_inventory_refresh.py",
            """import unittest
from pathlib import Path


class CleanupInventoryRefreshTests(unittest.TestCase):
    def test_cleanup_inventory_filters_ignored_prefixes(self) -> None:
        self.assertEqual(
            Path("generated/cleanup_inventory.txt").read_text(encoding="utf-8"),
            "build/output.log\\nreports/summary.txt\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale cleanup inventory output",
            "terminal-shell-cleanup-inventory-bug",
        )
        self._run(repo_root, "python3", "scripts/build_cleanup_inventory.py")
        self._commit(
            repo_root,
            "refresh cleanup inventory output",
            "terminal-shell-cleanup-inventory-fix",
        )

    def _add_shell_release_manifest_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "inputs" / "releases.json",
            """{
  "version": "2026.04",
  "channels": ["beta", "stable"],
  "artifacts": ["cli", "worker"]
}
""",
        )
        self._write(
            repo_root / "scripts" / "build_release_manifest.py",
            """import json
from pathlib import Path


def main() -> None:
    payload = json.loads(Path("inputs/releases.json").read_text(encoding="utf-8"))
    Path("generated/release_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "release_manifest.json",
            """{
  "artifacts": [
    "cli"
  ],
  "channels": [
    "beta"
  ],
  "version": "2026.04"
}
""",
        )
        self._write(
            repo_root / "tests" / "test_release_manifest_refresh.py",
            """import json
import unittest
from pathlib import Path


class ReleaseManifestRefreshTests(unittest.TestCase):
    def test_release_manifest_matches_inputs(self) -> None:
        self.assertEqual(
            json.loads(Path("generated/release_manifest.json").read_text(encoding="utf-8")),
            {
                "artifacts": ["cli", "worker"],
                "channels": ["beta", "stable"],
                "version": "2026.04",
            },
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale release manifest output",
            "terminal-shell-release-manifest-bug",
        )
        self._run(repo_root, "python3", "scripts/build_release_manifest.py")
        self._commit(
            repo_root,
            "refresh release manifest output",
            "terminal-shell-release-manifest-fix",
        )

    def _add_shell_docs_index_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "docs" / "source_notes" / "alpha.md",
            "# Alpha Notes\n\nAlpha details.\n",
        )
        self._write(
            repo_root / "docs" / "source_notes" / "beta.md",
            "# Beta Notes\n\nBeta details.\n",
        )
        self._write(
            repo_root / "scripts" / "build_docs_index.py",
            """import json
from pathlib import Path


def title_for(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError(f"missing title in {path}")


def main() -> None:
    base = Path("docs/source_notes")
    payload = [
        {
            "path": path.as_posix(),
            "title": title_for(path),
        }
        for path in sorted(base.glob("*.md"))
    ]
    Path("generated/docs_index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "docs_index.json",
            """[
  {
    "path": "docs/source_notes/alpha.md",
    "title": "Alpha Notes"
  }
]
""",
        )
        self._write(
            repo_root / "tests" / "test_docs_index_refresh.py",
            """import json
import unittest
from pathlib import Path


class DocsIndexRefreshTests(unittest.TestCase):
    def test_docs_index_matches_source_titles(self) -> None:
        self.assertEqual(
            json.loads(Path("generated/docs_index.json").read_text(encoding="utf-8")),
            [
                {
                    "path": "docs/source_notes/alpha.md",
                    "title": "Alpha Notes",
                },
                {
                    "path": "docs/source_notes/beta.md",
                    "title": "Beta Notes",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale docs index output",
            "terminal-shell-docs-index-bug",
        )
        self._run(repo_root, "python3", "scripts/build_docs_index.py")
        self._commit(
            repo_root,
            "refresh docs index output",
            "terminal-shell-docs-index-fix",
        )

    def _add_shell_service_bundle_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "configs" / "services" / "api.json",
            """{
  "enabled": true,
  "name": "api",
  "port": 8080
}
""",
        )
        self._write(
            repo_root / "configs" / "services" / "worker.json",
            """{
  "enabled": true,
  "name": "worker",
  "port": 9090
}
""",
        )
        self._write(
            repo_root / "configs" / "services" / "legacy.json",
            """{
  "enabled": false,
  "name": "legacy",
  "port": 6060
}
""",
        )
        self._write(
            repo_root / "scripts" / "build_service_bundle.py",
            """import json
from pathlib import Path


def main() -> None:
    services = []
    for path in sorted(Path("configs/services").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload["enabled"]:
            continue
        services.append(
            {
                "name": payload["name"],
                "port": payload["port"],
            }
        )
    Path("generated/service_bundle.json").write_text(
        json.dumps(services, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    Path("generated/service_targets.txt").write_text(
        "\\n".join(f"{item['name']}:{item['port']}" for item in services) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "service_bundle.json",
            """[
  {
    "name": "api",
    "port": 8080
  }
]
""",
        )
        self._write(
            repo_root / "generated" / "service_targets.txt",
            "api:8080\n",
        )
        self._write(
            repo_root / "tests" / "test_service_bundle_refresh.py",
            """import json
import unittest
from pathlib import Path


class ServiceBundleRefreshTests(unittest.TestCase):
    def test_service_bundle_matches_enabled_service_configs(self) -> None:
        self.assertEqual(
            json.loads(Path("generated/service_bundle.json").read_text(encoding="utf-8")),
            [
                {"name": "api", "port": 8080},
                {"name": "worker", "port": 9090},
            ],
        )
        self.assertEqual(
            Path("generated/service_targets.txt").read_text(encoding="utf-8"),
            "api:8080\\nworker:9090\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale service bundle output",
            "terminal-shell-service-bundle-bug",
        )
        self._run(repo_root, "python3", "scripts/build_service_bundle.py")
        self._commit(
            repo_root,
            "refresh service bundle output",
            "terminal-shell-service-bundle-fix",
        )

    def _add_shell_command_matrix_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "inputs" / "command_matrix_environments.json",
            """{
  "active": ["dev", "prod"],
  "archived": ["qa"]
}
""",
        )
        self._write(
            repo_root / "inputs" / "command_matrix_commands.json",
            """{
  "commands": [
    {"name": "sync", "flags": ["--fast"]},
    {"name": "deploy", "flags": ["--canary", "--audit"]}
  ]
}
""",
        )
        self._write(
            repo_root / "scripts" / "tasks" / "rebuild_command_matrix.py",
            """from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_command_matrix import main


if __name__ == "__main__":
    main()
""",
        )
        self._write(repo_root / "scripts" / "__init__.py", "")
        self._write(
            repo_root / "scripts" / "build_command_matrix.py",
            """import json
from pathlib import Path


def main() -> None:
    environments = json.loads(
        Path("inputs/command_matrix_environments.json").read_text(encoding="utf-8")
    )
    commands = json.loads(
        Path("inputs/command_matrix_commands.json").read_text(encoding="utf-8")
    )["commands"]
    rows = []
    markdown_lines = ["| environment | command | flags |", "| --- | --- | --- |"]
    for environment in sorted(environments["active"]):
        for command in commands:
            flags = " ".join(command["flags"])
            rows.append(
                {
                    "command": command["name"],
                    "environment": environment,
                    "flags": list(command["flags"]),
                }
            )
            markdown_lines.append(
                f"| {environment} | {command['name']} | {flags} |"
            )
    Path("generated/command_matrix.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    Path("generated/command_matrix.md").write_text(
        "\\n".join(markdown_lines) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "command_matrix.json",
            """[
  {
    "command": "sync",
    "environment": "dev",
    "flags": [
      "--fast"
    ]
  }
]
""",
        )
        self._write(
            repo_root / "generated" / "command_matrix.md",
            """| environment | command | flags |
| --- | --- | --- |
| dev | sync | --fast |
""",
        )
        self._write(
            repo_root / "tests" / "test_command_matrix_refresh.py",
            """import json
import unittest
from pathlib import Path


class CommandMatrixRefreshTests(unittest.TestCase):
    def test_command_matrix_matches_inputs(self) -> None:
        self.assertEqual(
            json.loads(Path("generated/command_matrix.json").read_text(encoding="utf-8")),
            [
                {"command": "sync", "environment": "dev", "flags": ["--fast"]},
                {"command": "deploy", "environment": "dev", "flags": ["--canary", "--audit"]},
                {"command": "sync", "environment": "prod", "flags": ["--fast"]},
                {"command": "deploy", "environment": "prod", "flags": ["--canary", "--audit"]},
            ],
        )
        rendered = Path("generated/command_matrix.md").read_text(encoding="utf-8")
        self.assertIn("| prod | deploy | --canary --audit |", rendered)


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale command matrix output",
            "terminal-shell-command-matrix-bug",
        )
        self._run(repo_root, "python3", "scripts/tasks/rebuild_command_matrix.py")
        self._commit(
            repo_root,
            "refresh command matrix output",
            "terminal-shell-command-matrix-fix",
        )

    def _add_shell_team_roster_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "inputs" / "team_roster.json",
            """{
  "teams": [
    {"members": ["alice", "bob"], "name": "api"},
    {"members": ["zoe"], "name": "web"}
  ]
}
""",
        )
        self._write(
            repo_root / "scripts" / "build_team_roster.py",
            """import json
from pathlib import Path


def main() -> None:
    payload = json.loads(Path("inputs/team_roster.json").read_text(encoding="utf-8"))
    lines = [
        f"{item['name']}: {','.join(item['members'])}"
        for item in sorted(payload["teams"], key=lambda value: value["name"])
    ]
    Path("generated/team_roster.txt").write_text(
        "\\n".join(lines) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "team_roster.txt",
            """api: alice
web: zoe
""",
        )
        self._write(
            repo_root / "tests" / "test_team_roster_refresh.py",
            """import unittest
from pathlib import Path


class TeamRosterRefreshTests(unittest.TestCase):
    def test_team_roster_matches_generator_inputs(self) -> None:
        self.assertEqual(
            Path("generated/team_roster.txt").read_text(encoding="utf-8"),
            "api: alice,bob\\nweb: zoe\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale team roster output",
            "terminal-shell-team-roster-bug",
        )
        self._run(repo_root, "python3", "scripts/build_team_roster.py")
        self._commit(
            repo_root,
            "refresh team roster output",
            "terminal-shell-team-roster-fix",
        )

    def _add_shell_incident_rollup_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "inputs" / "incidents.json",
            """{
  "incidents": [
    {"severity": "high", "status": "open"},
    {"severity": "high", "status": "resolved"},
    {"severity": "low", "status": "open"},
    {"severity": "medium", "status": "open"}
  ]
}
""",
        )
        self._write(
            repo_root / "scripts" / "build_incident_rollup.py",
            """import json
from pathlib import Path


def main() -> None:
    payload = json.loads(Path("inputs/incidents.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for item in payload["incidents"]:
        if item["status"] != "open":
            continue
        severity = item["severity"]
        counts[severity] = counts.get(severity, 0) + 1
    Path("generated/incident_rollup.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "incident_rollup.json",
            """{
  "high": 2,
  "medium": 1
}
""",
        )
        self._write(
            repo_root / "tests" / "test_incident_rollup_refresh.py",
            """import json
import unittest
from pathlib import Path


class IncidentRollupRefreshTests(unittest.TestCase):
    def test_incident_rollup_matches_open_incidents(self) -> None:
        self.assertEqual(
            json.loads(Path("generated/incident_rollup.json").read_text(encoding="utf-8")),
            {
                "high": 1,
                "low": 1,
                "medium": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale incident rollup output",
            "terminal-shell-incident-rollup-bug",
        )
        self._run(repo_root, "python3", "scripts/build_incident_rollup.py")
        self._commit(
            repo_root,
            "refresh incident rollup output",
            "terminal-shell-incident-rollup-fix",
        )

    def _add_shell_alert_routes_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "configs" / "alerts" / "api.json",
            """{
  "enabled": true,
  "route": "pager",
  "service": "api"
}
""",
        )
        self._write(
            repo_root / "configs" / "alerts" / "web.json",
            """{
  "enabled": true,
  "route": "slack",
  "service": "web"
}
""",
        )
        self._write(
            repo_root / "configs" / "alerts" / "legacy.json",
            """{
  "enabled": false,
  "route": "email",
  "service": "legacy"
}
""",
        )
        self._write(
            repo_root / "scripts" / "build_alert_routes.py",
            """import json
from pathlib import Path


def main() -> None:
    routes = []
    for path in sorted(Path("configs/alerts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload["enabled"]:
            continue
        routes.append(
            {
                "route": payload["route"],
                "service": payload["service"],
            }
        )
    Path("generated/alert_routes.json").write_text(
        json.dumps(routes, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    Path("generated/alert_routes.md").write_text(
        "\\n".join(f"- {item['service']}: {item['route']}" for item in routes) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "alert_routes.json",
            """[
  {
    "route": "pager",
    "service": "api"
  }
]
""",
        )
        self._write(
            repo_root / "generated" / "alert_routes.md",
            "- api: pager\n",
        )
        self._write(
            repo_root / "tests" / "test_alert_routes_refresh.py",
            """import json
import unittest
from pathlib import Path


class AlertRoutesRefreshTests(unittest.TestCase):
    def test_alert_routes_match_enabled_configs(self) -> None:
        self.assertEqual(
            json.loads(Path("generated/alert_routes.json").read_text(encoding="utf-8")),
            [
                {
                    "route": "pager",
                    "service": "api",
                },
                {
                    "route": "slack",
                    "service": "web",
                },
            ],
        )
        self.assertEqual(
            Path("generated/alert_routes.md").read_text(encoding="utf-8"),
            "- api: pager\\n- web: slack\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale alert routes output",
            "terminal-shell-alert-routes-bug",
        )
        self._run(repo_root, "python3", "scripts/build_alert_routes.py")
        self._commit(
            repo_root,
            "refresh alert routes output",
            "terminal-shell-alert-routes-fix",
        )

    def _add_shell_environment_catalog_task(self, repo_root: Path) -> None:
        self._write(
            repo_root / "configs" / "environments" / "prod.json",
            """{
  "commands": ["deploy", "migrate"],
  "enabled": true,
  "name": "prod"
}
""",
        )
        self._write(
            repo_root / "configs" / "environments" / "staging.json",
            """{
  "commands": ["deploy"],
  "enabled": true,
  "name": "staging"
}
""",
        )
        self._write(
            repo_root / "configs" / "environments" / "legacy.json",
            """{
  "commands": ["archive"],
  "enabled": false,
  "name": "legacy"
}
""",
        )
        self._write(
            repo_root / "scripts" / "tasks" / "rebuild_environment_catalog.py",
            """import json
from pathlib import Path


def main() -> None:
    entries = []
    for path in sorted(Path("configs/environments").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload["enabled"]:
            continue
        entries.append(
            {
                "commands": payload["commands"],
                "name": payload["name"],
            }
        )
    Path("generated/environment_catalog.json").write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    Path("generated/environment_catalog.txt").write_text(
        "\\n".join(
            f"{item['name']}: {','.join(item['commands'])}"
            for item in entries
        )
        + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
        )
        self._write(
            repo_root / "generated" / "environment_catalog.json",
            """[
  {
    "commands": ["deploy", "migrate"],
    "name": "prod"
  }
]
""",
        )
        self._write(
            repo_root / "generated" / "environment_catalog.txt",
            "prod: deploy,migrate\n",
        )
        self._write(
            repo_root / "tests" / "test_environment_catalog_refresh.py",
            """import json
import unittest
from pathlib import Path


class EnvironmentCatalogRefreshTests(unittest.TestCase):
    def test_environment_catalog_matches_enabled_configs(self) -> None:
        self.assertEqual(
            json.loads(
                Path("generated/environment_catalog.json").read_text(encoding="utf-8")
            ),
            [
                {
                    "commands": ["deploy", "migrate"],
                    "name": "prod",
                },
                {
                    "commands": ["deploy"],
                    "name": "staging",
                },
            ],
        )
        self.assertEqual(
            Path("generated/environment_catalog.txt").read_text(encoding="utf-8"),
            "prod: deploy,migrate\\nstaging: deploy\\n",
        )


if __name__ == "__main__":
    unittest.main()
""",
        )
        self._commit(
            repo_root,
            "introduce stale environment catalog output",
            "terminal-shell-environment-catalog-bug",
        )
        self._run(repo_root, "python3", "scripts/tasks/rebuild_environment_catalog.py")
        self._commit(
            repo_root,
            "refresh environment catalog output",
            "terminal-shell-environment-catalog-fix",
        )

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
