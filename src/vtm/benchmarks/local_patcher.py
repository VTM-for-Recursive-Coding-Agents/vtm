"""Local LLM patcher used by external-command coding benchmarks."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtm.adapters.openai_chat import OpenAICompatibleChatClient, OpenAICompatibleChatConfig
from vtm.harness.models import HarnessTaskPack


@dataclass(frozen=True)
class LocalPatcherConfig:
    """Configuration for the local OpenAI-compatible patch generator."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 180
    max_output_tokens: int = 8192
    temperature: float = 0.0
    max_memory_context_paths: int = 5
    max_file_chars: int = 20000

    @classmethod
    def from_env(cls) -> LocalPatcherConfig:
        """Load patcher configuration from environment variables."""
        base_url = os.getenv("VTM_LOCAL_LLM_BASE_URL", "").strip()
        model = os.getenv("VTM_LOCAL_LLM_MODEL", "").strip()
        if not base_url:
            raise ValueError("VTM_LOCAL_LLM_BASE_URL must be set")
        if not model:
            raise ValueError("VTM_LOCAL_LLM_MODEL must be set")
        return cls(
            base_url=base_url,
            api_key=os.getenv("VTM_LOCAL_LLM_API_KEY", "vtm-local"),
            model=model,
            timeout_seconds=int(os.getenv("VTM_LOCAL_LLM_TIMEOUT_SECONDS", "180")),
            max_output_tokens=int(os.getenv("VTM_LOCAL_LLM_MAX_OUTPUT_TOKENS", "8192")),
            temperature=float(os.getenv("VTM_LOCAL_LLM_TEMPERATURE", "0.0")),
        )


class LocalOpenAIPatcher:
    """Generates and applies a patch using a local OpenAI-compatible model."""

    def __init__(self, config: LocalPatcherConfig | None = None) -> None:
        """Create the patcher and its chat client."""
        self._config = config or LocalPatcherConfig.from_env()
        self._client = OpenAICompatibleChatClient(
            OpenAICompatibleChatConfig(
                base_url=self._config.base_url,
                api_key=self._config.api_key,
                timeout_seconds=self._config.timeout_seconds,
            )
        )

    def run(self, *, task_file: str | Path, workspace: str | Path) -> int:
        """Load a task pack, request a patch, and apply it in the workspace."""
        task_path = Path(task_file)
        workspace_root = Path(workspace)
        task = self._load_task_pack(task_path)
        artifact_dir = workspace_root / ".vtm-local-patcher"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        prompt = self.build_prompt(task=task, workspace_root=workspace_root)
        (artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        raw_output = self._request_patch(prompt)
        (artifact_dir / "response.txt").write_text(raw_output, encoding="utf-8")

        patch_text = self.extract_patch(raw_output)
        (artifact_dir / "candidate.patch").write_text(patch_text, encoding="utf-8")
        if not patch_text.strip():
            raise RuntimeError(
                f"model did not return a patch; see {(artifact_dir / 'response.txt')}"
            )
        self._apply_patch(
            workspace_root=workspace_root,
            patch_text=patch_text,
            artifact_dir=artifact_dir,
        )
        return 0

    def build_prompt(
        self,
        *,
        task: HarnessTaskPack | Mapping[str, Any],
        workspace_root: Path,
    ) -> str:
        """Build the patch-generation prompt for one task pack."""
        task_pack = self._coerce_task_pack(task)
        candidate_paths = self._candidate_paths(task_pack)
        file_blocks = []
        for relative_path in candidate_paths:
            path = workspace_root / relative_path
            if not path.exists() or not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            file_blocks.append(
                "\n".join(
                    [
                        f"FILE: {relative_path}",
                        "```",
                        content[: self._config.max_file_chars],
                        "```",
                    ]
                )
            )
        memory_blocks = []
        for candidate in task_pack.memory_context[: self._config.max_memory_context_paths]:
            memory_blocks.append(
                json.dumps(
                    {
                        "title": candidate.title,
                        "summary": candidate.summary,
                        "relative_path": candidate.relative_path,
                        "symbol": candidate.symbol,
                        "score": candidate.score,
                    },
                    sort_keys=True,
                )
            )
        sections = [
            "You are fixing a repository task in a local git workspace.",
            (
                "Return only a unified diff patch. "
                "Do not include prose, Markdown fences, or explanations."
            ),
            "",
            f"TASK ID: {task_pack.case_id}",
            f"ISSUE: {task_pack.problem_statement or task_pack.task_statement}",
            f"HINTS: {task_pack.hints_text or 'none'}",
            "FAILING TESTS:",
            json.dumps(list(task_pack.failing_tests), indent=2),
            "EXPECTED CHANGED PATHS:",
            json.dumps(list(task_pack.expected_changed_paths), indent=2),
            "RETRIEVED MEMORY CONTEXT:",
            "\n".join(memory_blocks) if memory_blocks else "(none)",
            "CURRENT FILE CONTENTS:",
            "\n\n".join(file_blocks) if file_blocks else "(none)",
            "PATCH REQUIREMENTS:",
            "- The patch must apply with git apply --3way.",
            "- Modify only files that are necessary.",
            "- Do not change tests unless the task explicitly requires it.",
            "- Output a raw unified diff only.",
        ]
        return "\n".join(sections).strip() + "\n"

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
                "memory_context": memory_context,
                "coding_executor": payload.get("coding_executor") or "external_command",
            }
        )

    def extract_patch(self, raw_output: str) -> str:
        fenced = re.search(
            r"```(?:diff|patch)?\n(?P<body>.*)```",
            raw_output,
            flags=re.DOTALL,
        )
        if fenced is not None:
            return fenced.group("body").strip() + "\n"
        return raw_output.strip() + ("\n" if raw_output.strip() else "")

    def _candidate_paths(self, task: HarnessTaskPack) -> tuple[str, ...]:
        selected: list[str] = []
        seen: set[str] = set()
        for relative_path in task.expected_changed_paths:
            if relative_path not in seen:
                seen.add(relative_path)
                selected.append(relative_path)
        for candidate in task.memory_context:
            candidate_path = candidate.relative_path
            if not candidate_path:
                continue
            if candidate_path in seen:
                continue
            seen.add(candidate_path)
            selected.append(candidate_path)
            max_paths = len(task.expected_changed_paths) + self._config.max_memory_context_paths
            if len(selected) >= max_paths:
                break
        return tuple(selected)

    def _request_patch(self, prompt: str) -> str:
        response_payload = self._client.create_chat_completion(
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_output_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a coding patch generator. Return only a unified diff patch."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return self._client.extract_message_text(response_payload)

    def _apply_patch(self, *, workspace_root: Path, patch_text: str, artifact_dir: Path) -> None:
        patch_path = artifact_dir / "candidate.patch"
        commands = [
            ["git", "apply", "--check", "--3way", str(patch_path)],
            ["git", "apply", "--3way", str(patch_path)],
        ]
        for index, command in enumerate(commands):
            completed = subprocess.run(
                command,
                cwd=workspace_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                (artifact_dir / f"git-apply-{index}.stdout").write_text(
                    completed.stdout,
                    encoding="utf-8",
                )
                (artifact_dir / f"git-apply-{index}.stderr").write_text(
                    completed.stderr,
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "generated patch did not apply cleanly; "
                    f"see {(artifact_dir / f'git-apply-{index}.stderr')}"
                )


__all__ = ["LocalOpenAIPatcher", "LocalPatcherConfig"]
