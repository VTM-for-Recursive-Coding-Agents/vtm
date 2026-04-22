"""Optional LongCoT pilot runner using OpenRouter."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Protocol
from urllib import error, request

from vtm.base import utc_now
from vtm.benchmarks.openrouter import execution_model, openrouter_api_key, openrouter_base_url

DEFAULT_OUTPUT_DIR = ".benchmarks/longcot-pilot"
DEFAULT_MAX_COMPLETION_TOKENS = 32768
SOLUTION_PATTERN = re.compile(r"\bsolution\s*=", re.IGNORECASE)


class LongCoTQuestionLike(Protocol):
    """Question fields used by the pilot runner."""

    question_id: str
    domain: str
    difficulty: str
    prompt: str


class ChatCompletionClient(Protocol):
    """Minimal protocol for OpenRouter-compatible chat completion clients."""

    def create_chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        max_completion_tokens: int,
    ) -> dict[str, Any]:
        """Create one chat completion and return the raw response payload."""


@dataclass(frozen=True)
class OpenRouterChatConfig:
    """Connection settings for the optional LongCoT pilot client."""

    base_url: str
    api_key: str
    timeout_seconds: int = 600


class OpenRouterChatClient:
    """Tiny OpenAI-compatible client dedicated to the LongCoT pilot."""

    def __init__(self, config: OpenRouterChatConfig) -> None:
        if not config.base_url.strip():
            raise ValueError("OpenRouter pilot client requires a non-empty base_url")
        if not config.api_key.strip():
            raise ValueError("OpenRouter pilot client requires a non-empty api_key")
        self._config = config

    def create_chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        max_completion_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_completion_tokens": max_completion_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            _chat_endpoint(self._config.base_url),
            data=body,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self._config.timeout_seconds) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter chat request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter chat request failed: {exc.reason}") from exc
        if not isinstance(raw_payload, dict):
            raise RuntimeError("OpenRouter chat response must be a JSON object")
        return raw_payload


def build_parser() -> argparse.ArgumentParser:
    """Build the optional LongCoT pilot CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run an optional LongCoT pilot with one OpenRouter call per question."
    )
    parser.add_argument("--domain", default="cs", help="LongCoT domain filter.")
    parser.add_argument("--difficulty", default="easy", help="LongCoT difficulty filter.")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum number of LongCoT questions to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for responses.jsonl, summary.json, and paper_table.md.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="OpenRouter model id. Falls back to VTM_EXECUTION_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="OpenRouter base URL. Falls back to VTM_OPENROUTER_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Maximum completion tokens for each OpenRouter call.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional LongCoT pilot and write pilot artifacts."""
    args = build_parser().parse_args(argv)
    if args.max_questions < 1:
        raise SystemExit("--max-questions must be at least 1")
    if args.max_completion_tokens < 1:
        raise SystemExit("--max-completion-tokens must be at least 1")

    try:
        longcot = importlib.import_module("longcot")
    except ModuleNotFoundError:
        print(_missing_longcot_message(), file=sys.stderr)
        return 1

    resolved_model = execution_model(args.model or None)
    resolved_base_url = (args.base_url or openrouter_base_url()).strip()
    resolved_api_key = (args.api_key or openrouter_api_key() or "").strip()
    if not resolved_api_key:
        raise SystemExit(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY or pass --api-key."
        )

    questions = list(longcot.load_questions(domain=args.domain, difficulty=args.difficulty))
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
    if not questions:
        raise SystemExit(
            "No LongCoT questions found for "
            f"domain={args.domain!r}, difficulty={args.difficulty!r}."
        )

    client = OpenRouterChatClient(
        OpenRouterChatConfig(base_url=resolved_base_url, api_key=resolved_api_key)
    )
    summary = run_longcot_pilot(
        questions,
        verify_fn=longcot.verify,
        client=client,
        model=resolved_model,
        output_dir=args.output_dir,
        domain=args.domain,
        difficulty=args.difficulty,
        base_url=resolved_base_url,
        max_completion_tokens=args.max_completion_tokens,
    )
    print(
        json.dumps(
            {
                "responses_jsonl": summary["responses_jsonl"],
                "summary_json": summary["summary_json"],
                "paper_table_md": summary["paper_table_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_longcot_pilot(
    questions: Sequence[LongCoTQuestionLike],
    *,
    verify_fn: Callable[[Any, str], bool],
    client: ChatCompletionClient,
    model: str,
    output_dir: str | Path,
    domain: str,
    difficulty: str,
    base_url: str,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Run one OpenRouter completion per question and write pilot artifacts."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = resolved_output_dir / "responses.jsonl"
    summary_path = resolved_output_dir / "summary.json"
    paper_table_path = resolved_output_dir / "paper_table.md"

    response_rows: list[dict[str, Any]] = []
    correct = 0
    incorrect = 0
    failed = 0
    wrong_formatting = 0
    output_tokens: list[int] = []

    for question in questions:
        row = _run_question(
            question,
            verify_fn=verify_fn,
            client=client,
            model=model,
            max_completion_tokens=max_completion_tokens,
        )
        response_rows.append(row)
        status = str(row["status"])
        if status == "correct":
            correct += 1
        elif status == "incorrect":
            incorrect += 1
        else:
            failed += 1
        if bool(row.get("wrong_formatting")):
            wrong_formatting += 1
        token_count = _usage_output_tokens(row.get("usage"))
        if token_count is not None:
            output_tokens.append(token_count)

    total = len(response_rows)
    verified = correct + incorrect
    median_output_tokens = median(output_tokens) if output_tokens else None
    summary = {
        "provider": "openrouter",
        "generated_at": _isoformat_now(),
        "model": model,
        "base_url": base_url,
        "domain": domain,
        "difficulty": difficulty,
        "max_questions": total,
        "max_completion_tokens": max_completion_tokens,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "failed": failed,
        "wrong_formatting": wrong_formatting,
        "accuracy": (correct / verified) if verified else 0.0,
        "overall_accuracy": (correct / total) if total else 0.0,
        "median_output_tokens": median_output_tokens,
        "responses_jsonl": str(responses_path),
        "summary_json": str(summary_path),
        "paper_table_md": str(paper_table_path),
    }

    responses_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in response_rows),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paper_table_path.write_text(_render_paper_table(summary), encoding="utf-8")
    return summary


def _run_question(
    question: LongCoTQuestionLike,
    *,
    verify_fn: Callable[[Any, str], bool],
    client: ChatCompletionClient,
    model: str,
    max_completion_tokens: int,
) -> dict[str, Any]:
    try:
        payload = client.create_chat_completion(
            model=model,
            prompt=question.prompt,
            max_completion_tokens=max_completion_tokens,
        )
        response_text = _extract_message_text(payload)
    except Exception as exc:
        return {
            "question_id": question.question_id,
            "domain": question.domain,
            "difficulty": question.difficulty,
            "successful": False,
            "status": "failed",
            "error": str(exc),
        }

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    wrong_formatting = _has_wrong_formatting(response_text)

    try:
        is_correct = bool(verify_fn(question, response_text))
    except Exception as exc:
        return {
            "question_id": question.question_id,
            "domain": question.domain,
            "difficulty": question.difficulty,
            "successful": True,
            "status": "incorrect",
            "response_text": response_text,
            "usage": usage,
            "wrong_formatting": wrong_formatting,
            "error": str(exc),
        }

    return {
        "question_id": question.question_id,
        "domain": question.domain,
        "difficulty": question.difficulty,
        "successful": True,
        "status": "correct" if is_correct else "incorrect",
        "response_text": response_text,
        "usage": usage,
        "wrong_formatting": wrong_formatting,
    }


def _render_paper_table(summary: dict[str, Any]) -> str:
    pilot_label = "LongCoT-Mini" if summary["difficulty"] == "easy" else "LongCoT"
    median_tokens = summary["median_output_tokens"]
    median_display = "n/a" if median_tokens is None else f"{median_tokens:g}"
    header = (
        "| Pilot | Model | Domain | Difficulty | Total | Correct | Incorrect | Failed "
        "| Wrong formatting | Accuracy | Overall accuracy | Median output tokens |\n"
    )
    divider = (
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: |\n"
    )
    row = (
        f"| {pilot_label} | {summary['model']} | {summary['domain']} | {summary['difficulty']} "
        f"| {summary['total']} | {summary['correct']} | {summary['incorrect']} "
        f"| {summary['failed']} | {summary['wrong_formatting']} | {summary['accuracy']:.1%} "
        f"| {summary['overall_accuracy']:.1%} | {median_display} |\n"
    )
    return (
        "# LongCoT Pilot\n\n"
        "External reasoning pilot only. This is not part of the main VTM retrieval, "
        "drift, or drifted-retrieval evidence.\n\n"
        f"{header}{divider}{row}"
    )


def _extract_message_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter chat response contained no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenRouter chat response contained no message object")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        collected = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {None, "text"}
        ]
        if collected:
            return "".join(collected)
    raise RuntimeError("OpenRouter chat response contained unsupported content")


def _usage_output_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in ("output_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _has_wrong_formatting(response_text: str) -> bool:
    return SOLUTION_PATTERN.search(response_text) is None


def _chat_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _isoformat_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def _missing_longcot_message() -> str:
    return (
        "LongCoT is optional and is not installed in this VTM environment.\n"
        "Clone and install it, then rerun the pilot. Example:\n"
        "  git clone https://github.com/LongHorizonReasoning/longcot.git .vendor/longcot\n"
        "  uv pip install -e ./.vendor/longcot\n"
        "Docs: https://github.com/LongHorizonReasoning/longcot"
    )


__all__ = [
    "DEFAULT_MAX_COMPLETION_TOKENS",
    "DEFAULT_OUTPUT_DIR",
    "OpenRouterChatClient",
    "OpenRouterChatConfig",
    "build_parser",
    "main",
    "run_longcot_pilot",
]
