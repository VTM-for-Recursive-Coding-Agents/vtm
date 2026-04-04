from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class LocalPatcherConfig:
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
    def __init__(self, config: LocalPatcherConfig | None = None) -> None:
        self._config = config or LocalPatcherConfig.from_env()

    def run(self, *, task_file: str | Path, workspace: str | Path) -> int:
        task_path = Path(task_file)
        workspace_root = Path(workspace)
        task = json.loads(task_path.read_text(encoding="utf-8"))
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

    def build_prompt(self, *, task: dict[str, Any], workspace_root: Path) -> str:
        candidate_paths = self._candidate_paths(task)
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
        for candidate in task.get("memory_context", ())[: self._config.max_memory_context_paths]:
            memory_blocks.append(
                json.dumps(
                    {
                        "title": candidate.get("title"),
                        "summary": candidate.get("summary"),
                        "relative_path": candidate.get("relative_path"),
                        "symbol": candidate.get("symbol"),
                        "score": candidate.get("score"),
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
            f"TASK ID: {task.get('case_id')}",
            f"ISSUE: {task.get('problem_statement') or task.get('task_statement')}",
            f"HINTS: {task.get('hints_text') or 'none'}",
            "FAILING TESTS:",
            json.dumps(task.get("failing_tests", []), indent=2),
            "EXPECTED CHANGED PATHS:",
            json.dumps(task.get("expected_changed_paths", []), indent=2),
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

    def extract_patch(self, raw_output: str) -> str:
        fenced = re.search(
            r"```(?:diff|patch)?\n(?P<body>.*)```",
            raw_output,
            flags=re.DOTALL,
        )
        if fenced is not None:
            return fenced.group("body").strip() + "\n"
        return raw_output.strip() + ("\n" if raw_output.strip() else "")

    def _candidate_paths(self, task: dict[str, Any]) -> tuple[str, ...]:
        selected: list[str] = []
        seen: set[str] = set()
        for relative_path in task.get("expected_changed_paths", []):
            path = str(relative_path)
            if path not in seen:
                seen.add(path)
                selected.append(path)
        for candidate in task.get("memory_context", []):
            relative_path = candidate.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                continue
            if relative_path in seen:
                continue
            seen.add(relative_path)
            selected.append(relative_path)
            max_paths = (
                len(task.get("expected_changed_paths", []))
                + self._config.max_memory_context_paths
            )
            if len(selected) >= max_paths:
                break
        return tuple(selected)

    def _request_patch(self, prompt: str) -> str:
        payload = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a coding patch generator. Return only a unified diff patch."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        endpoint = self._chat_endpoint(self._config.base_url)
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self._config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"local model request failed with HTTP {exc.code}: {detail}"
            ) from exc
        content = response_payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            collected = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") in {None, "text"}
            ]
            return "".join(collected)
        raise RuntimeError("local model response contained an unsupported content payload")

    def _chat_endpoint(self, base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

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
