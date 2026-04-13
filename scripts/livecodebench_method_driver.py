#!/usr/bin/env python3
"""Shared benchmark-time method driver for LiveCodeBench providers.

This module powers three entrypoints:
- scripts/livecodebench_rag_driver.py
- scripts/livecodebench_rlm_driver.py
- scripts/livecodebench_rlm_rag_driver.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LCB_ROOT = PROJECT_ROOT / "benchmarks" / "LiveCodeBench"
RLM_ROOT = PROJECT_ROOT / "rlm"
DEFAULT_RLM_SERVE_STATUS_FILE = PROJECT_ROOT / "logs" / "slurm" / "serve_status.txt"
DEFAULT_RLM_CONTEXT_LIMIT = 32_768
DEFAULT_RLM_COMPLETION_TOKEN_RESERVE = 2_048
DEFAULT_RLM_BACKEND_TIMEOUT = 900.0
DEFAULT_RLM_BACKEND_MAX_RETRIES = 2
CHARS_PER_TOKEN_ESTIMATE = 4
MIN_RLM_PROMPT_TOKEN_BUDGET = 1_024

if str(LCB_ROOT) not in sys.path:
    sys.path.insert(0, str(LCB_ROOT))
if str(RLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLM_ROOT))

@dataclass
class MethodArgs:
    provider: str
    run_id: str
    model: str
    lm_studio_model_id: str
    scenario: str
    release_version: str
    n: int
    temperature: float
    top_p: float
    max_tokens: int
    openai_timeout: int
    multiprocess: int
    evaluate: bool
    num_process_evaluate: int
    timeout_seconds: int
    start_date: str | None
    end_date: str | None
    not_fast: bool
    debug: bool
    max_instances: int | None
    tensor_parallel_size: int
    local_vllm_max_model_len: int | None
    local_vllm_gpu_memory_utilization: float | None
    rag_top_k: int
    rag_max_chars_per_chunk: int
    rlm_max_depth: int
    rlm_max_iterations: int
    rlm_max_timeout: float | None
    rlm_backend_timeout: float | None
    rlm_backend: str | None
    rlm_backend_url: str | None
    rlm_api_key: str | None
    rlm_context_limit: int
    rlm_prompt_token_budget: int | None
    rlm_completion_token_reserve: int


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_positive_int(raw: str | None) -> int | None:
    if raw is None:
        return None

    match = re.search(r"([1-9][0-9]*)", raw)
    if match is None:
        return None
    return int(match.group(1))


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    try:
        return float(stripped)
    except ValueError:
        return None


def _default_tensor_parallel_size() -> int:
    for env_name in (
        "TENSOR_PARALLEL_SIZE",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_NODE",
    ):
        value = _parse_positive_int(os.getenv(env_name))
        if value is not None:
            return value

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices and cuda_visible_devices != "NoDevFiles":
        devices = [item for item in cuda_visible_devices.split(",") if item.strip()]
        if devices:
            return len(devices)

    return 1


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _parse_args(provider: str) -> MethodArgs:
    parser = argparse.ArgumentParser(
        description=f"Run LiveCodeBench provider driver ({provider})"
    )
    parser.add_argument("--provider", default=provider)
    parser.add_argument("--run-id", default=_env("RUN_ID"))
    parser.add_argument("--model", default=_env("MODEL"))
    parser.add_argument("--lm-studio-model-id", default=_env("LM_STUDIO_MODEL_ID", ""))
    parser.add_argument("--scenario", default=_env("SCENARIO", "codegeneration"))
    parser.add_argument("--release-version", default="release_latest")
    parser.add_argument("--n", type=int, default=int(_env("N", "1")))
    parser.add_argument("--temperature", type=float, default=float(_env("TEMPERATURE", "0.2")))
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--openai-timeout", type=int, default=90)
    parser.add_argument("--multiprocess", type=int, default=0)
    parser.add_argument("--evaluate", default="true")
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=6)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--not-fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=_default_tensor_parallel_size())
    parser.add_argument(
        "--local-vllm-max-model-len",
        type=int,
        default=_parse_positive_int(_env("VLLM_MAX_MODEL_LEN")),
    )
    parser.add_argument(
        "--local-vllm-gpu-memory-utilization",
        type=float,
        default=_parse_float(_env("VLLM_GPU_MEMORY_UTILIZATION")),
    )

    parser.add_argument("--rag-top-k", type=int, default=4)
    parser.add_argument("--rag-max-chars-per-chunk", type=int, default=1800)

    parser.add_argument("--rlm-max-depth", type=int, default=1)
    parser.add_argument("--rlm-max-iterations", type=int, default=12)
    parser.add_argument("--rlm-max-timeout", type=float, default=None)
    parser.add_argument("--rlm-backend-timeout", type=float, default=None, help="Backend request timeout in seconds (default: 900s for non-RAG, 1800s for RAG)")
    parser.add_argument("--rlm-backend", default=_env("RLM_BACKEND", ""))
    parser.add_argument("--rlm-backend-url", default=_env("RLM_BACKEND_URL", ""))
    parser.add_argument("--rlm-api-key", default=_env("RLM_API_KEY", ""))
    parser.add_argument(
        "--rlm-context-limit",
        type=int,
        default=int(_env("RLM_CONTEXT_LIMIT", str(DEFAULT_RLM_CONTEXT_LIMIT))),
    )
    parser.add_argument(
        "--rlm-prompt-token-budget",
        type=int,
        default=_parse_positive_int(_env("RLM_PROMPT_TOKEN_BUDGET")),
    )
    parser.add_argument(
        "--rlm-completion-token-reserve",
        type=int,
        default=int(
            _env(
                "RLM_COMPLETION_TOKEN_RESERVE",
                str(DEFAULT_RLM_COMPLETION_TOKEN_RESERVE),
            )
        ),
    )

    raw = parser.parse_args()

    if not raw.run_id:
        raise SystemExit("Missing --run-id (or RUN_ID environment variable).")
    if not raw.model:
        raise SystemExit("Missing --model (or MODEL environment variable).")

    return MethodArgs(
        provider=raw.provider,
        run_id=raw.run_id,
        model=raw.model,
        lm_studio_model_id=raw.lm_studio_model_id,
        scenario=raw.scenario,
        release_version=raw.release_version,
        n=raw.n,
        temperature=raw.temperature,
        top_p=raw.top_p,
        max_tokens=raw.max_tokens,
        openai_timeout=raw.openai_timeout,
        multiprocess=raw.multiprocess,
        evaluate=_parse_bool(raw.evaluate),
        num_process_evaluate=raw.num_process_evaluate,
        timeout_seconds=raw.timeout_seconds,
        start_date=raw.start_date,
        end_date=raw.end_date,
        not_fast=raw.not_fast,
        debug=raw.debug,
        max_instances=raw.max_instances,
        tensor_parallel_size=raw.tensor_parallel_size,
        local_vllm_max_model_len=raw.local_vllm_max_model_len,
        local_vllm_gpu_memory_utilization=raw.local_vllm_gpu_memory_utilization,
        rag_top_k=raw.rag_top_k,
        rag_max_chars_per_chunk=raw.rag_max_chars_per_chunk,
        rlm_max_depth=raw.rlm_max_depth,
        rlm_max_iterations=raw.rlm_max_iterations,
        rlm_max_timeout=raw.rlm_max_timeout,
        rlm_backend_timeout=raw.rlm_backend_timeout,
        rlm_backend=_empty_to_none(raw.rlm_backend),
        rlm_backend_url=_empty_to_none(raw.rlm_backend_url),
        rlm_api_key=_empty_to_none(raw.rlm_api_key),
        rlm_context_limit=max(raw.rlm_context_limit, MIN_RLM_PROMPT_TOKEN_BUDGET),
        rlm_prompt_token_budget=raw.rlm_prompt_token_budget,
        rlm_completion_token_reserve=max(raw.rlm_completion_token_reserve, 0),
    )


def _resolve_model(args: MethodArgs):
    from lcb_runner.lm_styles import LanguageModelStore

    if args.model not in LanguageModelStore:
        known = ", ".join(sorted(LanguageModelStore.keys())[:12])
        raise SystemExit(
            f"Unknown model key '{args.model}'. Sample known keys: {known}"
        )
    return LanguageModelStore[args.model]


def _lmstudio_style() -> Any | None:
    from lcb_runner.lm_styles import LMStyle

    return getattr(LMStyle, "LMStudio", None)


def _uses_lmstudio_backend(model: Any) -> bool:
    lmstudio_style = _lmstudio_style()
    return lmstudio_style is not None and getattr(model, "model_style", None) == lmstudio_style


def _configure_model_env(args: MethodArgs, model: Any) -> None:
    if _uses_lmstudio_backend(model):
        os.environ.setdefault("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        if args.lm_studio_model_id:
            os.environ["LMSTUDIO_MODEL"] = args.lm_studio_model_id

    if args.local_vllm_max_model_len is not None:
        os.environ["VLLM_MAX_MODEL_LEN"] = str(args.local_vllm_max_model_len)
    if args.local_vllm_gpu_memory_utilization is not None:
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.local_vllm_gpu_memory_utilization)


def _effective_prompt_model(model: Any) -> Any:
    if _uses_lmstudio_backend(model):
        from lcb_runner.lm_styles import LMStyle

        return dataclasses.replace(model, model_style=LMStyle.OpenAIChat)

    return model


def _effective_output_model(model: Any, provider: str) -> Any:
    suffix = provider.replace("_", "-")
    return dataclasses.replace(model, model_repr=f"{model.model_repr}-{suffix}")


def _runner_args(args: MethodArgs) -> SimpleNamespace:
    from lcb_runner.utils.scenarios import Scenario

    return SimpleNamespace(
        model=args.model,
        local_model_path=None,
        trust_remote_code=False,
        scenario=Scenario(args.scenario),
        not_fast=args.not_fast,
        release_version=args.release_version,
        cot_code_execution=False,
        n=args.n,
        codegen_n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        multiprocess=args.multiprocess,
        stop=["###"],
        continue_existing=False,
        continue_existing_with_eval=False,
        use_cache=False,
        cache_batch_size=100,
        debug=args.debug,
        evaluate=args.evaluate,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout_seconds,
        openai_timeout=args.openai_timeout,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.local_vllm_max_model_len,
        gpu_memory_utilization=args.local_vllm_gpu_memory_utilization,
        enable_prefix_caching=False,
        custom_output_file=None,
        custom_output_save_name=None,
        dtype="bfloat16",
        start_date=args.start_date,
        end_date=args.end_date,
    )


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _problem_chunks(problem: Any, max_chars: int) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []

    def add_chunk(title: str, content: Any) -> None:
        if content is None:
            return
        if isinstance(content, str):
            text = content.strip()
        else:
            text = _compact_json(content)
        if not text:
            return
        chunks.append((title, text[:max_chars]))

    add_chunk("question", getattr(problem, "question_content", None))
    add_chunk("starter_code", getattr(problem, "starter_code", None))
    add_chunk("public_test_cases", getattr(problem, "public_test_cases", None))
    add_chunk("question_title", getattr(problem, "question_title", None))

    metadata = getattr(problem, "metadata", None)
    if isinstance(metadata, dict):
        safe_meta = {
            key: value
            for key, value in metadata.items()
            if key not in {"private_test_cases", "private_tests", "solutions"}
        }
        add_chunk("metadata", safe_meta)

    if not chunks:
        add_chunk("fallback", str(problem))

    return chunks


def _render_retrieval_context(chunks: list[dict[str, Any]]) -> str:
    lines = ["Relevant local benchmark context:"]
    for chunk in chunks:
        lines.append(f"[{chunk['title']}]\n{chunk['content']}")
    return "\n\n".join(lines)


def _token_set(text: str) -> set[str]:
    return {tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(tok) > 2}


def _retrieve_context(
    problem: Any,
    prompt: str,
    top_k: int,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    chunks = _problem_chunks(problem, max_chars=max_chars)
    prompt_tokens = _token_set(prompt)

    scored: list[tuple[int, str, str]] = []
    for title, chunk in chunks:
        overlap = len(prompt_tokens.intersection(_token_set(chunk)))
        scored.append((overlap, title, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, min(top_k, len(scored)))]
    selected_chunks = [
        {"score": score, "title": title, "content": chunk}
        for score, title, chunk in selected
    ]

    stats = {
        "candidate_chunks": len(scored),
        "selected_chunks": len(selected_chunks),
        "selected_titles": [chunk["title"] for chunk in selected_chunks],
        "selected_scores": [chunk["score"] for chunk in selected_chunks],
    }
    return _render_retrieval_context(selected_chunks), selected_chunks, stats


def _augment_prompt(prompt: str | list[dict[str, str]], context: str, mode: str) -> str | list[dict[str, str]]:
    prefix = (
        "Use the retrieved context below when useful. "
        "Do not copy test outputs or fabricate hidden constraints.\n\n"
        f"{context}\n\n"
    )

    if isinstance(prompt, str):
        return prefix + prompt

    updated = [dict(message) for message in prompt]
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") == "user":
            updated[idx]["content"] = prefix + str(updated[idx].get("content", ""))
            return updated

    updated.append({"role": "user", "content": prefix})
    return updated


def _prompt_to_text(prompt: str | list[dict[str, str]]) -> str:
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for message in prompt:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


def _prompt_messages(prompt: str | list[dict[str, str]]) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(message) for message in prompt]


def _count_tokens_tiktoken(messages: list[dict[str, Any]], model_name: str) -> int | None:
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    total = 0
    tokens_per_message = 3
    tokens_per_name = 1
    for message in messages:
        total += tokens_per_message
        content = message.get("content")
        if isinstance(content, str):
            total += len(encoding.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(encoding.encode(part.get("text", "") or ""))
        elif content is not None and content != "":
            total += len(encoding.encode(str(content)))
        if message.get("name"):
            total += tokens_per_name
    return total


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "") or ""))
            elif part is not None:
                parts.append(str(part))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


@lru_cache(maxsize=8)
def _get_transformers_tokenizer(model_name: str) -> Any | None:
    if not model_name or model_name == "unknown":
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def _count_tokens_transformers(messages: list[dict[str, Any]], model_name: str) -> int | None:
    tokenizer = _get_transformers_tokenizer(model_name)
    if tokenizer is None:
        return None

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(rendered, list):
                return len(rendered)
            if hasattr(rendered, "shape"):
                return int(rendered.shape[-1])
    except Exception:
        pass

    try:
        prompt_text = "\n\n".join(
            f"[{message.get('role', 'user')}] {_message_content_to_text(message.get('content'))}"
            for message in messages
        )
        return len(tokenizer.encode(prompt_text, add_special_tokens=True))
    except Exception:
        return None


def _prefer_tiktoken(model_name: str) -> bool:
    lowered = model_name.lower()
    return lowered.startswith("gpt") or lowered.startswith("o1") or lowered.startswith("o3")


def _count_prompt_tokens(prompt: str | list[dict[str, str]], model_name: str) -> int:
    messages = _prompt_messages(prompt)
    if model_name and model_name != "unknown":
        token_count = None
        if _prefer_tiktoken(model_name):
            token_count = _count_tokens_tiktoken(messages, model_name)
            if token_count is None:
                token_count = _count_tokens_transformers(messages, model_name)
        else:
            token_count = _count_tokens_transformers(messages, model_name)
            if token_count is None:
                token_count = _count_tokens_tiktoken(messages, model_name)
        if token_count is not None:
            return token_count

    total_chars = 0
    for message in messages:
        raw = message.get("content", "") or ""
        total_chars += len(raw) if isinstance(raw, str) else len(str(raw))
    return (total_chars + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE


def _rlm_tokenizer_model_name(model: Any, args: MethodArgs) -> str:
    if _uses_lmstudio_backend(model):
        return os.getenv("LMSTUDIO_MODEL", args.model)
    return model.model_name


def _resolve_rlm_prompt_budget(args: MethodArgs) -> tuple[int, int]:
    context_limit = max(args.rlm_context_limit, MIN_RLM_PROMPT_TOKEN_BUDGET)
    if args.rlm_prompt_token_budget is None:
        prompt_budget = context_limit - args.rlm_completion_token_reserve
    else:
        prompt_budget = args.rlm_prompt_token_budget
    prompt_budget = min(prompt_budget, context_limit)
    prompt_budget = max(prompt_budget, MIN_RLM_PROMPT_TOKEN_BUDGET)
    return context_limit, prompt_budget


def _fit_rlm_retrieval_to_budget(
    prompt: str | list[dict[str, str]],
    retrieval_chunks: list[dict[str, Any]],
    args: MethodArgs,
    model_name: str,
) -> tuple[str | list[dict[str, str]], dict[str, Any]]:
    if not retrieval_chunks:
        prompt_tokens = _count_prompt_tokens(prompt, model_name)
        return prompt, {
            "trimmed_to_budget": False,
            "selected_chunks_before_trim": 0,
            "selected_chunks_after_trim": 0,
            "dropped_titles": [],
            "prompt_tokens_before_trim": prompt_tokens,
            "prompt_tokens_after_trim": prompt_tokens,
        }

    _, prompt_budget = _resolve_rlm_prompt_budget(args)
    kept_chunks = [dict(chunk) for chunk in retrieval_chunks]
    dropped_chunks: list[dict[str, Any]] = []

    full_prompt = _augment_prompt(prompt, _render_retrieval_context(kept_chunks), "rlm_rag")
    prompt_tokens_before_trim = _count_prompt_tokens(full_prompt, model_name)
    candidate_prompt = full_prompt
    candidate_tokens = prompt_tokens_before_trim

    while kept_chunks and candidate_tokens > prompt_budget:
        dropped_chunks.append(kept_chunks.pop())
        if kept_chunks:
            candidate_prompt = _augment_prompt(
                prompt,
                _render_retrieval_context(kept_chunks),
                "rlm_rag",
            )
        else:
            candidate_prompt = prompt
        candidate_tokens = _count_prompt_tokens(candidate_prompt, model_name)

    return candidate_prompt, {
        "trimmed_to_budget": bool(dropped_chunks),
        "fits_budget_after_trim": candidate_tokens <= prompt_budget,
        "selected_chunks_before_trim": len(retrieval_chunks),
        "selected_chunks_after_trim": len(kept_chunks),
        "selected_titles_after_trim": [chunk["title"] for chunk in kept_chunks],
        "dropped_titles": [chunk["title"] for chunk in dropped_chunks],
        "prompt_budget_tokens": prompt_budget,
        "prompt_tokens_before_trim": prompt_tokens_before_trim,
        "prompt_tokens_after_trim": candidate_tokens,
    }


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _discover_vllm_base_url() -> str | None:
    status_path = Path(
        _env("VTM_RLM_SERVE_STATUS_FILE", str(DEFAULT_RLM_SERVE_STATUS_FILE))
    )
    wait_seconds = max(int(_env("VTM_RLM_SERVE_WAIT_SECONDS", "30")), 0)
    poll_seconds = max(float(_env("VTM_RLM_SERVE_WAIT_POLL_SECONDS", "2")), 0.1)
    deadline = time.monotonic() + wait_seconds

    while True:
        if status_path.is_file():
            status = _read_key_value_file(status_path)
            endpoint_file = _empty_to_none(status.get("endpoint_file"))
            if endpoint_file is not None:
                endpoint_path = Path(endpoint_file)
                if endpoint_path.is_file():
                    endpoint_lines = endpoint_path.read_text(encoding="utf-8").splitlines()
                    endpoint_value = _empty_to_none(endpoint_lines[0] if endpoint_lines else None)
                    if endpoint_value is not None and endpoint_value.startswith(("http://", "https://")):
                        return _normalize_openai_base_url(endpoint_value)

            state = (status.get("state", "")).strip().lower()
            if state in {"failed", "error", "cancelled"}:
                raise SystemExit(
                    "Could not auto-discover a healthy local vLLM endpoint because the recorded serve job failed. "
                    "Pass --rlm-backend-url explicitly or relaunch the serving job."
                )

            host = _empty_to_none(status.get("host"))
            port = _empty_to_none(status.get("port"))
            if state == "running" and host is not None and port is not None:
                return _normalize_openai_base_url(f"http://{host}:{port}")

        if time.monotonic() >= deadline:
            return None

        time.sleep(poll_seconds)


def _resolve_rlm_backend_config(model: Any, args: MethodArgs) -> tuple[str, dict[str, Any], dict[str, Any]]:
    uses_lmstudio = _uses_lmstudio_backend(model)
    backend = args.rlm_backend or ("openai" if uses_lmstudio else "vllm")
    base_url = args.rlm_backend_url
    api_key = args.rlm_api_key

    if uses_lmstudio:
        model_name = os.getenv("LMSTUDIO_MODEL", args.model)
    else:
        model_name = model.model_name

    if backend == "vllm":
        base_url = base_url or _discover_vllm_base_url()
        if base_url is None:
            raise SystemExit(
                "RLM backend 'vllm' requires a local OpenAI-compatible base URL. "
                "Pass --rlm-backend-url explicitly or provide a healthy serve status file."
            )
        api_key = api_key or "vtm-local-vllm"
    elif uses_lmstudio:
        base_url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        api_key = api_key or "lm-studio"

    backend_kwargs: dict[str, Any] = {"model_name": model_name}
    metadata: dict[str, Any] = {
        "backend": backend,
        "model_name": model_name,
    }

    if backend in {"vllm", "openai", "openrouter", "vercel", "azure_openai"}:
        client_timeout = _resolve_rlm_backend_timeout(args)
        if client_timeout <= 0:
            raise SystemExit(f"RLM backend timeout must be > 0, got {client_timeout}.")

        backend_kwargs["timeout"] = client_timeout
        backend_kwargs.setdefault("max_retries", DEFAULT_RLM_BACKEND_MAX_RETRIES)
        max_retries = int(backend_kwargs["max_retries"])
        if max_retries < 0:
            raise SystemExit(f"RLM backend max_retries must be >= 0, got {max_retries}.")
        backend_kwargs["max_retries"] = max_retries

        metadata["timeout"] = client_timeout
        metadata["max_retries"] = max_retries

    if base_url is not None:
        normalized_url = _normalize_openai_base_url(base_url)
        backend_kwargs["base_url"] = normalized_url
        metadata["base_url"] = normalized_url
    if api_key is not None:
        backend_kwargs["api_key"] = api_key

    return backend, backend_kwargs, metadata


def _health_check_rlm_backend(backend_meta: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    base_url = backend_meta.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        return {}

    models_url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(models_url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=max(timeout_seconds, 1)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"RLM backend health check failed for {models_url}: {exc}"
        ) from exc

    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models:
        raise SystemExit(
            f"RLM backend health check failed for {models_url}: expected a non-empty model list."
        )

    model_ids = [item.get("id") for item in models if isinstance(item, dict) and item.get("id")]
    return {"models_url": models_url, "available_models": model_ids[:8]}


def _resolve_rlm_backend_timeout(args: MethodArgs) -> float:
    # If explicitly set, use that
    if args.rlm_backend_timeout is not None:
        return max(float(args.rlm_backend_timeout), 1.0)
    
    # Otherwise compute from openai_timeout and provider
    timeout = max(float(args.openai_timeout), DEFAULT_RLM_BACKEND_TIMEOUT)
    
    # RAG requires more time due to retrieval context
    if args.provider in ("rag", "rlm_rag"):
        timeout = max(timeout, 1800.0)  # 30 minutes for RAG
    
    # Use rlm_max_timeout as upper bound if set
    if args.rlm_max_timeout is not None:
        timeout = min(timeout, float(args.rlm_max_timeout))
    
    return max(timeout, 1.0)


def _classify_rlm_error(error_text: str) -> str | None:
    lowered = error_text.lower()
    if (
        "tokenlimitexceedederror" in error_text
        or "maximum context length" in error_text
        or "token limit exceeded" in lowered
        or "safe context window" in lowered
        or "configured prompt budget" in lowered
        or "context_length" in lowered
    ):
        return "rlm_context_limit_exceeded"
    if (
        "apitimeouterror" in lowered
        or "request timed out" in lowered
        or "timeoutexception" in lowered
        or "timeout" in lowered
    ):
        return "rlm_backend_timeout"
    if "attributeerror" in lowered and ("choices" in lowered or "response" in lowered):
        return "rlm_backend_response_error"
    if "backend returned no response object" in lowered:
        return "rlm_backend_response_error"
    return None


def _prompt_preview(prompt_text: str, max_chars: int = 800) -> str:
    if len(prompt_text) <= max_chars:
        return prompt_text
    return prompt_text[:max_chars] + "..."


def _write_rlm_failure_diagnostics(
    run_dir: Path,
    prompt_index: int,
    prompt_text: str,
    backend_meta: dict[str, Any],
    response_meta: dict[str, Any],
    failure_type: str = "all_empty_samples",
) -> Path:
    diagnostics_path = run_dir / "rlm_failure_diagnostics.json"
    payload = {
        "failure_type": failure_type,
        "prompt_index": prompt_index,
        "prompt_preview": _prompt_preview(prompt_text),
        "backend": _json_ready(backend_meta),
        "response_metadata": _json_ready(response_meta),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    diagnostics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return diagnostics_path


def _run_standard_generation(
    runner_args: SimpleNamespace,
    model: Any,
    prompts: list[str | list[dict[str, str]]],
) -> list[list[str]]:
    from lcb_runner.runner.runner_utils import build_runner

    runner = build_runner(runner_args, model)
    return runner.prompts_to_outputs(prompts)


def _build_rlm(model: Any, args: MethodArgs):
    try:
        from rlm import RLM
    except ImportError as exc:
        raise SystemExit(
            "RLM package import failed. Ensure the repo checkout exists at project_root/rlm and run scripts/setup_rlm.sh if needed."
        ) from exc

    backend, backend_kwargs, backend_meta = _resolve_rlm_backend_config(model, args)
    _, prompt_budget = _resolve_rlm_prompt_budget(args)

    return (
        RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment="local",
            max_depth=args.rlm_max_depth,
            max_iterations=args.rlm_max_iterations,
            max_timeout=args.rlm_max_timeout,
            max_prompt_tokens=prompt_budget,
            compaction=True,
            verbose=False,
        ),
        backend_meta,
    )


def _run_rlm_generation(
    model: Any,
    args: MethodArgs,
    prompts: list[str | list[dict[str, str]]],
    run_dir: Path,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    rlm, backend_meta = _build_rlm(model, args)
    tokenizer_model_name = str(backend_meta.get("model_name") or _rlm_tokenizer_model_name(model, args))
    context_limit, prompt_budget = _resolve_rlm_prompt_budget(args)
    healthcheck = _health_check_rlm_backend(backend_meta, min(args.openai_timeout, 15))
    if healthcheck:
        backend_meta = {**backend_meta, "healthcheck": healthcheck}
    all_outputs: list[list[str]] = []
    metadata: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts):
        prompt_text = _prompt_to_text(prompt)
        prompt_tokens = _count_prompt_tokens(prompt, tokenizer_model_name)
        outputs: list[str] = []
        response_meta: dict[str, Any] = {
            "provider": args.provider,
            "rlm_max_depth": args.rlm_max_depth,
            "rlm_max_iterations": args.rlm_max_iterations,
            "rlm_max_prompt_tokens": prompt_budget,
            "rlm_backend": backend_meta.get("backend"),
            "rlm_backend_url": backend_meta.get("base_url"),
            "rlm_backend_timeout": backend_meta.get("timeout"),
            "rlm_backend_max_retries": backend_meta.get("max_retries"),
            "tokenizer_model_name": tokenizer_model_name,
            "context_limit_tokens": context_limit,
            "prompt_budget_tokens": prompt_budget,
            "completion_token_reserve": args.rlm_completion_token_reserve,
            "prompt_tokens": prompt_tokens,
            "samples": [],
        }

        if prompt_tokens > prompt_budget:
            diagnostics_path = _write_rlm_failure_diagnostics(
                run_dir,
                prompt_index,
                prompt_text,
                backend_meta,
                response_meta,
                failure_type="prompt_token_budget_exceeded",
            )
            raise RuntimeError(
                f"RLM prompt exceeded configured token budget on prompt index {prompt_index}. "
                f"prompt_tokens={prompt_tokens} budget_tokens={prompt_budget} context_limit={context_limit} "
                f"diagnostics={diagnostics_path}"
            )

        for _ in range(args.n):
            try:
                completion = rlm.completion(prompt_text)
                sample_meta = {
                    "execution_time": completion.execution_time,
                }
                usage = completion.usage_summary.to_dict() if completion.usage_summary else None
                if usage is not None:
                    sample_meta["usage_summary"] = usage
                if completion.metadata is not None:
                    sample_meta["completion_metadata"] = _json_ready(completion.metadata)
                response_text = (completion.response or "").strip()
                if not response_text:
                    outputs.append("")
                    sample_meta["error"] = "Empty response from RLM backend"
                    if isinstance(completion.metadata, dict) and completion.metadata.get("error"):
                        sample_meta["error"] = str(completion.metadata.get("error"))
                    failure_type = _classify_rlm_error(sample_meta["error"])
                    if failure_type is not None:
                        sample_meta["failure_type"] = failure_type
                    response_meta["samples"].append(sample_meta)
                    continue

                outputs.append(completion.response)
                response_meta["samples"].append(sample_meta)
            except Exception as exc:  # noqa: BLE001
                outputs.append("")
                sample_meta: dict[str, Any] = {"error": repr(exc)}
                sample_meta["exception_type"] = type(exc).__name__
                error_text = repr(exc)
                if hasattr(exc, "tokens_used"):
                    sample_meta["tokens_used"] = getattr(exc, "tokens_used")
                if hasattr(exc, "token_limit"):
                    sample_meta["token_limit"] = getattr(exc, "token_limit")
                if hasattr(exc, "partial_answer") and getattr(exc, "partial_answer"):
                    sample_meta["partial_answer"] = getattr(exc, "partial_answer")
                failure_type = _classify_rlm_error(error_text)
                if failure_type is not None:
                    sample_meta["failure_type"] = failure_type
                response_meta["samples"].append(sample_meta)

        if outputs and all(not output.strip() for output in outputs):
            response_meta["failed_samples_count"] = len(outputs)
            response_meta["total_n_requested"] = args.n
            sample_errors = [
                str(sample.get("error", ""))
                for sample in response_meta["samples"]
                if sample.get("error")
            ]
            failure_type = "all_empty_samples"
            classified_errors = [_classify_rlm_error(error) for error in sample_errors]
            non_null_classifications = [classification for classification in classified_errors if classification is not None]
            if non_null_classifications and len(non_null_classifications) == len(sample_errors):
                unique_classifications = set(non_null_classifications)
                if len(unique_classifications) == 1:
                    failure_type = non_null_classifications[0]
            diagnostics_path = _write_rlm_failure_diagnostics(
                run_dir,
                prompt_index,
                prompt_text,
                backend_meta,
                response_meta,
                failure_type=failure_type,
            )
            raise RuntimeError(
                f"RLM returned empty responses for all {len(outputs)} sample(s) on prompt index {prompt_index}. "
                f"backend={backend_meta.get('backend')} base_url={backend_meta.get('base_url', '<default>')} "
                f"diagnostics={diagnostics_path}"
            )

        all_outputs.append(outputs)
        metadata.append(response_meta)

    return all_outputs, metadata


def _write_metadata_file(
    run_dir: Path,
    args: MethodArgs,
    started_at: str,
    runtime_seconds: float,
) -> None:
    metadata = {
        "run_id": args.run_id,
        "benchmark": "livecodebench",
        "model": args.model,
        "provider": args.provider,
        "scenario": args.scenario,
        "release_version": args.release_version,
        "n": str(args.n),
        "temperature": str(args.temperature),
        "evaluate": str(args.evaluate).lower(),
        "started_at": started_at,
        "driver": "livecodebench_method_driver",
        "tensor_parallel_size": str(args.tensor_parallel_size),
        "local_vllm_max_model_len": str(args.local_vllm_max_model_len),
        "local_vllm_gpu_memory_utilization": str(args.local_vllm_gpu_memory_utilization),
        "rag_top_k": str(args.rag_top_k),
        "rlm_context_limit": str(args.rlm_context_limit),
        "rlm_completion_token_reserve": str(args.rlm_completion_token_reserve),
        "rlm_max_depth": str(args.rlm_max_depth),
        "rlm_max_iterations": str(args.rlm_max_iterations),
    }
    if args.rlm_prompt_token_budget is not None:
        metadata["rlm_prompt_token_budget"] = str(args.rlm_prompt_token_budget)
    if args.lm_studio_model_id:
        metadata["lm_studio_model_id"] = args.lm_studio_model_id
    if args.rlm_backend:
        metadata["rlm_backend"] = args.rlm_backend
    if args.rlm_backend_url:
        metadata["rlm_backend_url"] = args.rlm_backend_url

    lines = [f"{key}={value}" for key, value in metadata.items()]
    (run_dir / "metadata.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "command.log").write_text(
        f"provider={args.provider}\n"
        f"run_id={args.run_id}\n"
        f"runtime {runtime_seconds:.3f}\n",
        encoding="utf-8",
    )


def _append_failure_log(run_dir: Path, exc: BaseException) -> None:
    traceback_text = "".join(traceback.format_exception(exc))
    (run_dir / "command.log").write_text(traceback_text, encoding="utf-8")


def _save_outputs(
    runner_args: SimpleNamespace,
    model: Any,
    benchmark: list[Any],
    combined_results: list[tuple[list[str], list[str]]],
    eval_metadata: list[dict[str, Any]] | None,
) -> list[Path]:
    from lcb_runner.evaluation import extract_instance_results
    from lcb_runner.runner.scenario_router import get_metrics, sort_and_extract_save_results
    from lcb_runner.utils.path_utils import get_output_path
    from lcb_runner.utils.scenarios import Scenario

    output_path = Path(get_output_path(model.model_repr, runner_args))
    eval_file = Path(str(output_path).replace(".json", "_eval.json"))
    eval_all_file = Path(str(output_path).replace(".json", "_eval_all.json"))

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(benchmark, combined_results)
    ]
    save_results, combined_results = sort_and_extract_save_results(
        runner_args.scenario, save_results
    )

    output_path.write_text(json.dumps(save_results, indent=4), encoding="utf-8")

    files = [output_path.resolve()]

    if runner_args.evaluate:
        metrics = get_metrics(runner_args.scenario, runner_args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])

        if runner_args.scenario in {Scenario.codegeneration, Scenario.selfrepair}:
            if eval_metadata is None and len(metrics) > 2:
                eval_metadata = metrics[2]
            if eval_metadata is None:
                eval_metadata = [{} for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark,
                    combined_results,
                    graded,
                    eval_metadata,
                )
            ]
        else:
            save_eval_results = [
                instance.insert_output_evaluation(outputs_list, extracted_list, graded_list)
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark,
                    combined_results,
                    graded,
                )
            ]

        eval_file.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
        eval_all_file.write_text(json.dumps(save_eval_results, indent=4), encoding="utf-8")
        files.extend([eval_file.resolve(), eval_all_file.resolve()])

    return files


def _write_output_manifest(run_dir: Path, files: list[Path]) -> None:
    manifest_lines = [str(path) for path in files]
    (run_dir / "output_files.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def run_provider(provider: str) -> None:
    args = _parse_args(provider)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    run_dir = PROJECT_ROOT / "results" / "raw" / "livecodebench" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = _resolve_model(args)
        _configure_model_env(args, model)
        prompt_model = _effective_prompt_model(model)
        output_model = _effective_output_model(model, provider)
        runner_args = _runner_args(args)

        with _pushd(LCB_ROOT):
            from lcb_runner.runner.scenario_router import (
                build_prompt_benchmark,
                combine_results,
            )

            benchmark, format_prompt = build_prompt_benchmark(runner_args)
            if args.debug:
                benchmark = benchmark[:15]
            if args.max_instances is not None:
                benchmark = benchmark[: args.max_instances]

            prompts: list[str | list[dict[str, str]]] = [
                format_prompt(problem, prompt_model.model_style) for problem in benchmark
            ]

            retrieval_meta: list[dict[str, Any]] = []
            rlm_model_name = _rlm_tokenizer_model_name(model, args)
            if provider in {"rag", "rlm_rag"}:
                updated_prompts: list[str | list[dict[str, str]]] = []
                for problem, prompt in zip(benchmark, prompts):
                    prompt_text = _prompt_to_text(prompt)
                    context, selected_chunks, stats = _retrieve_context(
                        problem,
                        prompt_text,
                        top_k=args.rag_top_k,
                        max_chars=args.rag_max_chars_per_chunk,
                    )
                    if provider == "rlm_rag":
                        adjusted_prompt, trim_meta = _fit_rlm_retrieval_to_budget(
                            prompt,
                            selected_chunks,
                            args,
                            rlm_model_name,
                        )
                        updated_prompts.append(adjusted_prompt)
                        stats.update(trim_meta)
                    else:
                        updated_prompts.append(_augment_prompt(prompt, context, provider))
                    retrieval_meta.append(stats)
                prompts = updated_prompts

            if provider == "rag":
                results = _run_standard_generation(runner_args, model, prompts)
                provider_eval_meta = retrieval_meta if retrieval_meta else None
            elif provider == "baseline":
                results = _run_standard_generation(runner_args, model, prompts)
                provider_eval_meta = None
            elif provider == "rlm":
                results, provider_eval_meta = _run_rlm_generation(model, args, prompts, run_dir)
            elif provider == "rlm_rag":
                results, rlm_meta = _run_rlm_generation(model, args, prompts, run_dir)
                provider_eval_meta = []
                for idx in range(len(results)):
                    merged = {
                        "retrieval": retrieval_meta[idx] if idx < len(retrieval_meta) else {},
                        "rlm": rlm_meta[idx] if idx < len(rlm_meta) else {},
                    }
                    provider_eval_meta.append(merged)
            else:
                raise SystemExit(f"Unsupported provider '{provider}'")

            combined_results = combine_results(
                runner_args.scenario,
                results,
                prompt_model,
                runner_args.cot_code_execution,
            )

            files = _save_outputs(
                runner_args,
                output_model,
                benchmark,
                combined_results,
                provider_eval_meta,
            )
    except Exception as exc:
        _append_failure_log(run_dir, exc)
        raise

    runtime_seconds = time.perf_counter() - started
    _write_metadata_file(run_dir, args, started_at, runtime_seconds)
    _write_output_manifest(run_dir, files)

    print(f"[method-driver] provider={provider} run_id={args.run_id} runtime={runtime_seconds:.2f}s")


if __name__ == "__main__":
    run_provider("rag")
