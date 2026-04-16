"""Bridge VTM task packs into richer bundles for external coding scaffolds."""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtm.harness.models import HarnessTaskPack


@dataclass(frozen=True)
class ScaffoldBridgeConfig:
    """Controls how much task context is emitted for external scaffolds."""

    max_memory_context_items: int = 8
    max_file_count: int = 10
    max_file_chars: int = 20000


class ScaffoldBridge:
    """Builds a richer task bundle and optional delegate-command handoff."""

    def __init__(self, config: ScaffoldBridgeConfig | None = None) -> None:
        """Create the bridge with bounded file/context limits."""
        self._config = config or ScaffoldBridgeConfig()

    def run(
        self,
        *,
        task_file: str | Path,
        workspace: str | Path,
        artifact_root: str | Path,
        delegate_command: str | None = None,
    ) -> int:
        """Write the scaffold bundle and optionally run a delegate agent command."""
        task_path = Path(task_file).resolve()
        workspace_root = Path(workspace).resolve()
        artifact_dir = Path(artifact_root).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        task = self._load_task_pack(task_path)
        bundle = self.build_bundle(task=task, workspace_root=workspace_root)
        brief = self.build_brief(bundle)
        bundle_path = artifact_dir / "scaffold-bundle.json"
        brief_path = artifact_dir / "scaffold-brief.md"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
        brief_path.write_text(brief, encoding="utf-8")

        if not delegate_command:
            return 0

        formatted = delegate_command.format(
            task_file=str(task_path),
            workspace=str(workspace_root),
            artifact_root=str(artifact_dir),
            scaffold_bundle=str(bundle_path),
            brief_file=str(brief_path),
        )
        completed = subprocess.run(
            shlex.split(formatted),
            cwd=workspace_root,
            check=False,
            capture_output=True,
            text=True,
        )
        (artifact_dir / "delegate.stdout").write_text(completed.stdout, encoding="utf-8")
        (artifact_dir / "delegate.stderr").write_text(completed.stderr, encoding="utf-8")
        return completed.returncode

    def build_bundle(
        self,
        *,
        task: HarnessTaskPack | Mapping[str, Any],
        workspace_root: Path,
    ) -> dict[str, Any]:
        """Build a richer scaffold-facing JSON bundle from a harness task pack."""
        task_pack = self._coerce_task_pack(task)
        candidate_paths = self._candidate_paths(task_pack)
        relevant_files = []
        for relative_path in candidate_paths[: self._config.max_file_count]:
            path = workspace_root / relative_path
            if not path.exists() or not path.is_file():
                continue
            relevant_files.append(
                {
                    "path": relative_path,
                    "content": path.read_text(encoding="utf-8")[: self._config.max_file_chars],
                    "source": self._path_sources(relative_path, task_pack),
                }
            )
        memory_context = [
            {
                "memory_id": candidate.memory_id,
                "title": candidate.title,
                "summary": candidate.summary,
                "score": candidate.score,
                "status": candidate.status,
                "relative_path": candidate.relative_path,
                "symbol": candidate.symbol,
            }
            for candidate in task_pack.memory_context[: self._config.max_memory_context_items]
        ]
        return {
            "schema_version": "1.0",
            "task": {
                "case_id": task_pack.case_id,
                "repo_name": task_pack.repo_name,
                "commit_pair_id": task_pack.commit_pair_id,
                "dataset_name": task_pack.dataset_name,
                "instance_id": task_pack.instance_id,
                "task_statement": task_pack.task_statement,
                "problem_statement": task_pack.problem_statement,
                "hints_text": task_pack.hints_text,
                "memory_mode": task_pack.memory_mode,
                "evaluation_backend": task_pack.evaluation_backend,
                "execution_style": task_pack.execution_style,
            },
            "targets": {
                "expected_changed_paths": list(task_pack.expected_changed_paths),
                "touched_paths": list(task_pack.touched_paths),
                "failing_tests": list(task_pack.failing_tests),
                "fail_to_pass_tests": list(task_pack.fail_to_pass_tests),
                "pass_to_pass_tests": list(task_pack.pass_to_pass_tests),
                "test_command": list(task_pack.test_command),
            },
            "memory_context": memory_context,
            "relevant_files": relevant_files,
            "recommended_workflow": [
                "Inspect the expected changed paths and failing tests first.",
                "Use retrieved memory as hints, but verify against repository files.",
                "Patch the smallest set of files necessary.",
                "Run targeted verification for fail_to_pass tests, then spot-check pass_to_pass tests.",
            ],
        }

    def build_brief(self, bundle: Mapping[str, Any]) -> str:
        """Render a markdown brief from a scaffold bundle."""
        task = dict(bundle.get("task", {}))
        targets = dict(bundle.get("targets", {}))
        lines = [
            "# VTM Scaffold Brief",
            "",
            f"- Case ID: `{task.get('case_id', '')}`",
            f"- Repo: `{task.get('repo_name', '')}`",
            f"- Memory Mode: `{task.get('memory_mode', '')}`",
            "",
            "## Goal",
            str(task.get("problem_statement") or task.get("task_statement") or "").strip(),
            "",
            "## Constraints",
            f"- Expected changed paths: {json.dumps(targets.get('expected_changed_paths', []))}",
            f"- Failing tests: {json.dumps(targets.get('fail_to_pass_tests') or targets.get('failing_tests', []))}",
            f"- Pass-to-pass tests: {json.dumps(targets.get('pass_to_pass_tests', []))}",
            "",
            "## Memory Context",
        ]
        memory_context = bundle.get("memory_context", [])
        if isinstance(memory_context, list) and memory_context:
            for item in memory_context:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    f"- {item.get('title', '')}: {item.get('summary', '')} "
                    f"(path={item.get('relative_path')}, score={item.get('score')})"
                )
        else:
            lines.append("- (none)")
        lines.extend(["", "## Relevant Files"])
        relevant_files = bundle.get("relevant_files", [])
        if isinstance(relevant_files, list) and relevant_files:
            for item in relevant_files:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    f"- `{item.get('path', '')}` from {json.dumps(item.get('source', []))}"
                )
        else:
            lines.append("- (none)")
        lines.extend(["", "## Recommended Workflow"])
        workflow = bundle.get("recommended_workflow", [])
        if isinstance(workflow, list):
            for step in workflow:
                lines.append(f"- {step}")
        return "\n".join(lines).strip() + "\n"

    def _load_task_pack(self, task_path: Path) -> HarnessTaskPack:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return self._coerce_task_pack(payload)
        raise TypeError(f"task file must contain a JSON object: {task_path}")

    def _coerce_task_pack(self, task: HarnessTaskPack | Mapping[str, Any]) -> HarnessTaskPack:
        if isinstance(task, HarnessTaskPack):
            return task
        payload = dict(task)
        raw_memory_context = payload.get("memory_context", ())
        memory_context: list[dict[str, Any]] = []
        if isinstance(raw_memory_context, list):
            for index, item in enumerate(raw_memory_context):
                if not isinstance(item, Mapping):
                    continue
                memory_context.append(
                    {
                        "memory_id": str(item.get("memory_id") or f"memory-{index}"),
                        "title": str(item.get("title") or ""),
                        "summary": str(item.get("summary") or ""),
                        "score": float(item.get("score") or 0.0),
                        "status": str(item.get("status") or "unknown"),
                        "relative_path": item.get("relative_path"),
                        "symbol": item.get("symbol"),
                        "slice_name": item.get("slice_name"),
                        "raw_anchor_path": item.get("raw_anchor_path"),
                    }
                )

        expected_changed_paths = tuple(payload.get("expected_changed_paths", ()) or ())
        touched_paths = tuple(payload.get("touched_paths", ()) or expected_changed_paths)
        task_statement = str(
            payload.get("task_statement") or payload.get("problem_statement") or ""
        )
        problem_statement = payload.get("problem_statement")
        if problem_statement is not None:
            problem_statement = str(problem_statement)

        return HarnessTaskPack.model_validate(
            {
                "case_id": str(payload.get("case_id") or ""),
                "repo_name": str(payload.get("repo_name") or ""),
                "commit_pair_id": str(
                    payload.get("commit_pair_id") or payload.get("case_id") or ""
                ),
                "evaluation_backend": payload.get("evaluation_backend") or "local_subprocess",
                "instance_id": payload.get("instance_id"),
                "dataset_name": payload.get("dataset_name"),
                "base_ref": str(payload.get("base_ref") or ""),
                "head_ref": str(payload.get("head_ref") or ""),
                "commit_pair_label": payload.get("commit_pair_label"),
                "task_statement": task_statement,
                "problem_statement": problem_statement,
                "hints_text": payload.get("hints_text"),
                "failing_tests": tuple(payload.get("failing_tests", ()) or ()),
                "fail_to_pass_tests": tuple(payload.get("fail_to_pass_tests", ()) or ()),
                "pass_to_pass_tests": tuple(payload.get("pass_to_pass_tests", ()) or ()),
                "expected_changed_paths": expected_changed_paths,
                "touched_paths": touched_paths,
                "test_command": tuple(payload.get("test_command", ()) or ()),
                "target_patch_digest": str(payload.get("target_patch_digest") or ""),
                "gold_test_patch_digest": payload.get("gold_test_patch_digest"),
                "memory_mode": payload.get("memory_mode") or "no_memory",
                "top_k": int(payload.get("top_k") or 5),
                "task_kind": payload.get("task_kind"),
                "difficulty": payload.get("difficulty"),
                "execution_style": payload.get("execution_style") or "mixed_patch",
                "memory_context": memory_context,
                "coding_executor": payload.get("coding_executor") or "external_command",
            }
        )

    def _candidate_paths(self, task: HarnessTaskPack) -> tuple[str, ...]:
        selected: list[str] = []
        seen: set[str] = set()
        for relative_path in task.expected_changed_paths:
            self._append_path(relative_path, selected, seen)
        for relative_path in task.touched_paths:
            self._append_path(relative_path, selected, seen)
        for test_id in (*task.fail_to_pass_tests, *task.failing_tests, *task.pass_to_pass_tests):
            test_path = test_id.split("::", 1)[0].strip()
            self._append_path(test_path, selected, seen)
        for candidate in task.memory_context:
            self._append_path(candidate.relative_path, selected, seen)
        return tuple(selected[: self._config.max_file_count])

    def _append_path(
        self,
        relative_path: str | None,
        selected: list[str],
        seen: set[str],
    ) -> None:
        if not relative_path:
            return
        if relative_path in seen:
            return
        seen.add(relative_path)
        selected.append(relative_path)

    def _path_sources(self, relative_path: str, task: HarnessTaskPack) -> list[str]:
        sources: list[str] = []
        if relative_path in task.expected_changed_paths:
            sources.append("expected_changed_path")
        if relative_path in task.touched_paths:
            sources.append("touched_path")
        test_paths = {
            test_id.split("::", 1)[0].strip()
            for test_id in (*task.fail_to_pass_tests, *task.failing_tests, *task.pass_to_pass_tests)
        }
        if relative_path in test_paths:
            sources.append("test_file")
        if any(candidate.relative_path == relative_path for candidate in task.memory_context):
            sources.append("memory_context")
        return sources


__all__ = ["ScaffoldBridge", "ScaffoldBridgeConfig"]
