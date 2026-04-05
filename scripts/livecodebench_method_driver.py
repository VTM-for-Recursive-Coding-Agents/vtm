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
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LCB_ROOT = PROJECT_ROOT / "benchmarks" / "LiveCodeBench"
RLM_ROOT = PROJECT_ROOT / "rlm"

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
    rag_top_k: int
    rag_max_chars_per_chunk: int
    rlm_max_depth: int
    rlm_max_iterations: int
    rlm_max_timeout: float | None


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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

    parser.add_argument("--rag-top-k", type=int, default=4)
    parser.add_argument("--rag-max-chars-per-chunk", type=int, default=1800)

    parser.add_argument("--rlm-max-depth", type=int, default=1)
    parser.add_argument("--rlm-max-iterations", type=int, default=12)
    parser.add_argument("--rlm-max-timeout", type=float, default=None)

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
        rag_top_k=raw.rag_top_k,
        rag_max_chars_per_chunk=raw.rag_max_chars_per_chunk,
        rlm_max_depth=raw.rlm_max_depth,
        rlm_max_iterations=raw.rlm_max_iterations,
        rlm_max_timeout=raw.rlm_max_timeout,
    )


def _resolve_model(args: MethodArgs):
    from lcb_runner.lm_styles import LanguageModelStore

    if args.model not in LanguageModelStore:
        known = ", ".join(sorted(LanguageModelStore.keys())[:12])
        raise SystemExit(
            f"Unknown model key '{args.model}'. Sample known keys: {known}"
        )
    return LanguageModelStore[args.model]


def _configure_model_env(args: MethodArgs, model: Any) -> None:
    from lcb_runner.lm_styles import LMStyle

    if model.model_style == LMStyle.LMStudio:
        os.environ.setdefault("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        if args.lm_studio_model_id:
            os.environ["LMSTUDIO_MODEL"] = args.lm_studio_model_id


def _effective_prompt_model(model: Any) -> Any:
    from lcb_runner.lm_styles import LMStyle

    if model.model_style == LMStyle.LMStudio:
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
        tensor_parallel_size=1,
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


def _token_set(text: str) -> set[str]:
    return {tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(tok) > 2}


def _retrieve_context(problem: Any, prompt: str, top_k: int, max_chars: int) -> tuple[str, dict[str, Any]]:
    chunks = _problem_chunks(problem, max_chars=max_chars)
    prompt_tokens = _token_set(prompt)

    scored: list[tuple[int, str, str]] = []
    for title, chunk in chunks:
        overlap = len(prompt_tokens.intersection(_token_set(chunk)))
        scored.append((overlap, title, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, min(top_k, len(scored)))]

    lines = ["Relevant local benchmark context:"]
    for _, title, chunk in selected:
        lines.append(f"[{title}]\n{chunk}")

    stats = {
        "candidate_chunks": len(scored),
        "selected_chunks": len(selected),
        "selected_titles": [title for _, title, _ in selected],
    }
    return "\n\n".join(lines), stats


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
            "RLM package import failed. Ensure dependencies are installed in the active environment."
        ) from exc

    backend_kwargs: dict[str, Any] = {}
    from lcb_runner.lm_styles import LMStyle

    if model.model_style == LMStyle.LMStudio:
        backend_kwargs["base_url"] = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        backend_kwargs["api_key"] = "lm-studio"
        backend_kwargs["model_name"] = os.getenv("LMSTUDIO_MODEL", args.model)
    else:
        backend_kwargs["model_name"] = model.model_name

    return RLM(
        backend="openai",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=args.rlm_max_depth,
        max_iterations=args.rlm_max_iterations,
        max_timeout=args.rlm_max_timeout,
        verbose=False,
    )


def _run_rlm_generation(
    model: Any,
    args: MethodArgs,
    prompts: list[str | list[dict[str, str]]],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    rlm = _build_rlm(model, args)
    all_outputs: list[list[str]] = []
    metadata: list[dict[str, Any]] = []

    for prompt in prompts:
        prompt_text = _prompt_to_text(prompt)
        outputs: list[str] = []
        response_meta: dict[str, Any] = {
            "provider": args.provider,
            "rlm_max_depth": args.rlm_max_depth,
            "rlm_max_iterations": args.rlm_max_iterations,
            "samples": [],
        }

        for _ in range(args.n):
            try:
                completion = rlm.completion(prompt_text)
                outputs.append(completion.response)
                sample_meta = {
                    "execution_time": completion.execution_time,
                }
                usage = completion.usage_summary.to_dict() if completion.usage_summary else None
                if usage is not None:
                    sample_meta["usage_summary"] = usage
                response_meta["samples"].append(sample_meta)
            except Exception as exc:  # noqa: BLE001
                outputs.append("")
                response_meta["samples"].append({"error": repr(exc)})

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
        "rag_top_k": str(args.rag_top_k),
        "rlm_max_depth": str(args.rlm_max_depth),
        "rlm_max_iterations": str(args.rlm_max_iterations),
    }
    if args.lm_studio_model_id:
        metadata["lm_studio_model_id"] = args.lm_studio_model_id

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
            if provider in {"rag", "rlm_rag"}:
                updated_prompts: list[str | list[dict[str, str]]] = []
                for problem, prompt in zip(benchmark, prompts):
                    prompt_text = _prompt_to_text(prompt)
                    context, stats = _retrieve_context(
                        problem,
                        prompt_text,
                        top_k=args.rag_top_k,
                        max_chars=args.rag_max_chars_per_chunk,
                    )
                    updated_prompts.append(_augment_prompt(prompt, context, provider))
                    retrieval_meta.append(stats)
                prompts = updated_prompts

            if provider == "rag":
                results = _run_standard_generation(runner_args, model, prompts)
                provider_eval_meta = retrieval_meta if retrieval_meta else None
            elif provider == "rlm":
                results, provider_eval_meta = _run_rlm_generation(model, args, prompts)
            elif provider == "rlm_rag":
                results, rlm_meta = _run_rlm_generation(model, args, prompts)
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
