#!/usr/bin/env python3
"""Build and optionally run an external LiveCodeBench baseline command."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SRC_ROOT: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from vtm.benchmarks.openrouter import (
        DEFAULT_EXECUTION_MODEL,
        DEFAULT_OPENROUTER_BASE_URL,
        execution_model,
        openrouter_api_key,
        openrouter_base_url,
    )
except Exception:
    DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_EXECUTION_MODEL = "google/gemma-4-31b-it:free"

    def openrouter_base_url() -> str:
        return os.getenv("VTM_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()

    def openrouter_api_key() -> str | None:
        value = os.getenv("OPENROUTER_API_KEY", "").strip()
        return value or None

    def execution_model(explicit: str | None = None) -> str:
        resolved = (explicit or os.getenv("VTM_EXECUTION_MODEL") or DEFAULT_EXECUTION_MODEL).strip()
        if not resolved:
            raise ValueError("execution model id must not be empty")
        return resolved


DEFAULT_OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / ".benchmarks" / "livecodebench"
DEFAULT_SUMMARY_ROOT: Final[Path] = (
    PROJECT_ROOT / ".benchmarks" / "paper-tables" / "livecodebench-baselines"
)
DEFAULT_RELEASE_VERSION: Final[str] = "release_v1"
DEFAULT_SCENARIO: Final[str] = "codegeneration"
DEFAULT_MODEL_MATRIX_NAME: Final[str] = "openrouter-baselines"
OPENROUTER_MODEL_MATRIX: Final[tuple[str, ...]] = (
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
)
SMOKE_START_DATE: Final[str] = "2023-05-01"
SMOKE_END_DATE: Final[str] = "2023-05-31"


@dataclass(frozen=True)
class BaselineConfig:
    model: str
    base_url: str
    api_key: str | None
    run_id: str
    output_root: Path
    summary_root: Path
    benchmark_root: Path
    scenario: str
    release_version: str
    n: int
    temperature: float
    evaluate: bool
    start_date: str | None
    end_date: str | None
    smoke: bool
    execute: bool
    debug: bool = False

    @property
    def dry_run(self) -> bool:
        return not self.execute


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an external LiveCodeBench baseline with VTM OpenRouter defaults."
    )
    parser.add_argument(
        "--model",
        default="",
        help="Baseline model id. Defaults to VTM_EXECUTION_MODEL.",
    )
    parser.add_argument(
        "--model-matrix",
        default="",
        help=(
            "Optional model matrix to expand. Use "
            f"'{DEFAULT_MODEL_MATRIX_NAME}' for the maintained OpenRouter baseline trio."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="OpenRouter base URL. Defaults to VTM_OPENROUTER_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument("--run-id", default="", help="Run id suffix. Defaults to a timestamp.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for raw baseline artifacts.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=DEFAULT_SUMMARY_ROOT,
        help="Summary output root for paper-table ready baseline metadata.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "LiveCodeBench",
        help="LiveCodeBench checkout root.",
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--release-version", default=DEFAULT_RELEASE_VERSION)
    parser.add_argument("--n", type=int, default=10, help="Sample count for full runs.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--evaluate",
        default="true",
        help="Whether to ask LiveCodeBench to evaluate after generation.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass through LiveCodeBench debug mode to cap the run to the first 15 filtered problems.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small baseline-only smoke profile: n=1, evaluate=false, fixed date window.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the external LiveCodeBench command. Default behavior is dry-run only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated no-op kept for compatibility. Dry-run is now the default.",
    )
    return parser


def select_python(benchmark_root: Path) -> str:
    candidates = [
        benchmark_root / ".venv" / "bin" / "python",
        Path(os.getenv("VIRTUAL_ENV", "")) / "bin" / "python" if os.getenv("VIRTUAL_ENV") else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable or "python3"


def normalize_config(config: BaselineConfig) -> BaselineConfig:
    if not config.smoke:
        return config
    return BaselineConfig(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        run_id=config.run_id,
        output_root=config.output_root,
        summary_root=config.summary_root,
        benchmark_root=config.benchmark_root,
        scenario=config.scenario,
        release_version=config.release_version,
        n=1,
        temperature=config.temperature,
        evaluate=False,
        start_date=config.start_date or SMOKE_START_DATE,
        end_date=config.end_date or SMOKE_END_DATE,
        smoke=True,
        execute=config.execute,
        debug=config.debug,
    )


def build_openrouter_env(*, model: str, base_url: str, api_key: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["VTM_EXECUTION_MODEL"] = model
    env["VTM_OPENROUTER_BASE_URL"] = base_url
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_BASE"] = base_url
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_KEY"] = api_key
    return env


def resolve_models(*, explicit_model: str, model_matrix: str) -> tuple[str, ...]:
    matrix_name = model_matrix.strip()
    if matrix_name:
        if matrix_name != DEFAULT_MODEL_MATRIX_NAME:
            raise ValueError(
                f"unknown model matrix: {matrix_name}. Expected '{DEFAULT_MODEL_MATRIX_NAME}'."
            )
        return OPENROUTER_MODEL_MATRIX
    return (execution_model(explicit_model or None),)


def build_livecodebench_command(python_bin: str, config: BaselineConfig) -> list[str]:
    command = [
        python_bin,
        "-m",
        "lcb_runner.runner.main",
        "--model",
        config.model,
        "--scenario",
        config.scenario,
        "--release_version",
        config.release_version,
        "--n",
        str(config.n),
        "--temperature",
        str(config.temperature),
    ]
    if config.evaluate:
        command.append("--evaluate")
    if config.start_date:
        command.extend(["--start_date", config.start_date])
    if config.end_date:
        command.extend(["--end_date", config.end_date])
    if config.debug:
        command.append("--debug")
    return command


def normalized_run_dir(config: BaselineConfig) -> Path:
    return config.output_root / model_slug(config.model) / config.run_id


def normalized_summary_path(config: BaselineConfig) -> Path:
    return config.summary_root / f"{config.run_id}__{model_slug(config.model)}.json"


def _baseline_output_dir(benchmark_root: Path) -> Path:
    return benchmark_root / "output"


def _snapshot_output_files(benchmark_root: Path) -> dict[Path, int]:
    output_dir = _baseline_output_dir(benchmark_root)
    if not output_dir.exists():
        return {}
    return {
        path: path.stat().st_mtime_ns
        for path in output_dir.rglob("*.json")
        if path.is_file()
    }


def _candidate_output_files(benchmark_root: Path, config: BaselineConfig) -> list[Path]:
    output_dir = _baseline_output_dir(benchmark_root)
    if not output_dir.exists():
        return []
    prefix = f"{config.scenario}_{config.n}_{config.temperature}"
    return sorted(
        (
            path
            for path in output_dir.rglob("*.json")
            if path.is_file() and path.name.startswith(prefix)
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )


def _resolve_official_artifact_paths(
    benchmark_root: Path,
    *,
    config: BaselineConfig,
    before_snapshot: dict[Path, int],
) -> dict[str, str | None]:
    candidates = _candidate_output_files(benchmark_root, config)
    updated = [
        path
        for path in candidates
        if path.stat().st_mtime_ns > before_snapshot.get(path, -1)
    ]
    selected = updated or candidates
    raw_output = next(
        (path for path in reversed(selected) if not path.name.endswith(("_eval.json", "_eval_all.json"))),
        None,
    )
    eval_file = next(
        (path for path in reversed(selected) if path.name.endswith("_eval.json")),
        None,
    )
    eval_all_file = next(
        (path for path in reversed(selected) if path.name.endswith("_eval_all.json")),
        None,
    )
    return {
        "raw_output_file": str(raw_output) if raw_output is not None else None,
        "eval_file": str(eval_file) if eval_file is not None else None,
        "eval_all_file": str(eval_all_file) if eval_all_file is not None else None,
    }


def _extract_official_metrics(artifacts: dict[str, str | None]) -> dict[str, Any]:
    eval_file = artifacts.get("eval_file")
    eval_all_file = artifacts.get("eval_all_file")
    payload: dict[str, Any] = {
        "official_metrics_available": False,
        "official_metric_source": None,
        "official_pass_at_1": None,
        "official_pass_at_5": None,
        "official_problem_count": None,
        "raw_output_file": artifacts.get("raw_output_file"),
        "eval_file": eval_file,
        "eval_all_file": eval_all_file,
    }
    if not eval_file:
        return payload
    eval_path = Path(eval_file)
    if not eval_path.exists():
        return payload
    try:
        metrics_blob = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return payload
    metrics: Mapping[str, Any] | None = None
    if isinstance(metrics_blob, list) and metrics_blob and isinstance(metrics_blob[0], dict):
        metrics = metrics_blob[0]
    elif isinstance(metrics_blob, dict):
        metrics = metrics_blob
    if not isinstance(metrics, dict):
        return payload
    pass_at_1 = metrics.get("pass@1")
    pass_at_5 = metrics.get("pass@5")
    detail = metrics.get("detail")
    problem_count = None
    if isinstance(detail, dict):
        for key in ("pass@1", "pass@5"):
            values = detail.get(key)
            if isinstance(values, dict):
                problem_count = len(values)
                break
    if problem_count is None and eval_all_file:
        eval_all_path = Path(eval_all_file)
        if eval_all_path.exists():
            try:
                eval_all_payload = json.loads(eval_all_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                eval_all_payload = None
            if isinstance(eval_all_payload, list):
                problem_count = len(eval_all_payload)
    payload.update(
        {
            "official_metrics_available": isinstance(pass_at_1, int | float) or isinstance(pass_at_5, int | float),
            "official_metric_source": "livecodebench_eval_json",
            "official_pass_at_1": float(pass_at_1) if isinstance(pass_at_1, int | float) else None,
            "official_pass_at_5": float(pass_at_5) if isinstance(pass_at_5, int | float) else None,
            "official_problem_count": problem_count,
        }
    )
    return payload


def preflight_report(config: BaselineConfig) -> dict[str, str]:
    benchmark_exists = "true" if config.benchmark_root.exists() else "false"
    api_key_configured = "true" if config.api_key else "false"
    return {
        "mode": "execute" if config.execute else "dry_run",
        "benchmark_checkout_exists": benchmark_exists,
        "api_key_configured": api_key_configured,
        "base_url": config.base_url,
        "model": config.model,
        "output_root": str(config.output_root),
        "summary_root": str(config.summary_root),
        "debug": "true" if config.debug else "false",
    }


def resolve_config(args: argparse.Namespace) -> BaselineConfig:
    base_url = (args.base_url or openrouter_base_url()).strip()
    api_key = (args.api_key or openrouter_api_key() or "").strip() or None
    run_id = args.run_id or f"lcb_baseline_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    models = resolve_models(explicit_model=args.model, model_matrix=args.model_matrix)
    configs = tuple(
        normalize_config(
            BaselineConfig(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                run_id=run_id,
                output_root=args.output_root,
                summary_root=args.summary_root,
                benchmark_root=args.benchmark_root,
                scenario=args.scenario,
                release_version=args.release_version,
                n=args.n,
                temperature=args.temperature,
                evaluate=_parse_bool(args.evaluate),
                start_date=args.start_date,
                end_date=args.end_date,
                smoke=bool(args.smoke),
                execute=bool(args.execute),
                debug=bool(args.debug),
            )
        )
        for model_name in models
    )
    return configs


def write_metadata(
    run_dir: Path,
    config: BaselineConfig,
    command: list[str],
    *,
    status: str,
    exit_code: int | None,
    official_metrics: Mapping[str, Any] | None = None,
) -> None:
    metrics = dict(official_metrics or {})
    metadata = [
        f"run_id={config.run_id}",
        "benchmark=livecodebench",
        "mode=baseline_only",
        f"status={status}",
        f"model={config.model}",
        f"base_url={config.base_url}",
        f"scenario={config.scenario}",
        f"release_version={config.release_version}",
        f"n={config.n}",
        f"temperature={config.temperature}",
        f"evaluate={str(config.evaluate).lower()}",
        f"start_date={config.start_date or ''}",
        f"end_date={config.end_date or ''}",
        f"smoke={str(config.smoke).lower()}",
        f"debug={str(config.debug).lower()}",
        f"benchmark_root={config.benchmark_root}",
        f"output_dir={run_dir}",
        f"summary_path={normalized_summary_path(config)}",
        f"exit_code={'' if exit_code is None else exit_code}",
        f"command={shlex.join(command)}",
        f"started_at={dt.datetime.now(dt.UTC).isoformat()}",
        f"official_metrics_available={str(bool(metrics.get('official_metrics_available'))).lower()}",
        f"official_pass_at_1={'' if metrics.get('official_pass_at_1') is None else metrics.get('official_pass_at_1')}",
        f"official_pass_at_5={'' if metrics.get('official_pass_at_5') is None else metrics.get('official_pass_at_5')}",
        f"official_problem_count={'' if metrics.get('official_problem_count') is None else metrics.get('official_problem_count')}",
        f"raw_output_file={metrics.get('raw_output_file') or ''}",
        f"eval_file={metrics.get('eval_file') or ''}",
        f"eval_all_file={metrics.get('eval_all_file') or ''}",
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.txt").write_text("\n".join(metadata) + "\n", encoding="utf-8")


def write_summary(
    summary_path: Path,
    *,
    config: BaselineConfig,
    command: list[str],
    run_dir: Path,
    exit_code: int | None,
    official_metrics: Mapping[str, Any] | None = None,
) -> None:
    metrics = dict(official_metrics or {})
    payload = {
        "benchmark": "livecodebench",
        "kind": "baseline_model_evaluation",
        "mode": "baseline_only",
        "run_id": config.run_id,
        "model": config.model,
        "scenario": config.scenario,
        "release_version": config.release_version,
        "smoke": config.smoke,
        "debug": config.debug,
        "evaluate": config.evaluate,
        "output_dir": str(run_dir),
        "command": command,
        "command_text": shlex.join(command),
        "base_url": config.base_url,
        "exit_code": exit_code,
        "status": "planned" if exit_code is None else ("passed" if exit_code == 0 else "failed"),
        "official_metrics_available": bool(metrics.get("official_metrics_available")),
        "official_metric_source": metrics.get("official_metric_source"),
        "official_pass_at_1": metrics.get("official_pass_at_1"),
        "official_pass_at_5": metrics.get("official_pass_at_5"),
        "official_problem_count": metrics.get("official_problem_count"),
        "raw_output_file": metrics.get("raw_output_file"),
        "eval_file": metrics.get("eval_file"),
        "eval_all_file": metrics.get("eval_all_file"),
        "paper_note": (
            "LiveCodeBench is a baseline model coding benchmark only. "
            "It is not a VTM memory-drift result."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def run(
    config: BaselineConfig,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[object]] | None = None,
) -> int:
    benchmark_exists = config.benchmark_root.exists()
    if not benchmark_exists and config.execute:
        raise SystemExit(
            f"LiveCodeBench checkout not found at {config.benchmark_root}. "
            "Run bash scripts/livecodebench/setup_livecodebench.sh first."
        )

    if benchmark_exists:
        python_bin = select_python(config.benchmark_root)
    else:
        python_bin = sys.executable or "python3"
    command = build_livecodebench_command(python_bin, config)
    env = build_openrouter_env(model=config.model, base_url=config.base_url, api_key=config.api_key)
    run_dir = normalized_run_dir(config)
    summary_path = normalized_summary_path(config)
    official_metrics: dict[str, Any] = {
        "official_metrics_available": False,
        "official_metric_source": None,
        "official_pass_at_1": None,
        "official_pass_at_5": None,
        "official_problem_count": None,
        "raw_output_file": None,
        "eval_file": None,
        "eval_all_file": None,
    }

    print("[livecodebench] baseline-only runner")
    print(f"[livecodebench] run_id={config.run_id}")
    print(f"[livecodebench] benchmark_root={config.benchmark_root}")
    print(f"[livecodebench] output_dir={run_dir}")
    print(f"[livecodebench] summary_path={summary_path}")
    print(f"[livecodebench] model={config.model}")
    print(f"[livecodebench] mode={'execute' if config.execute else 'dry-run'}")
    print(f"[livecodebench] command={shlex.join(command)}")
    for key, value in preflight_report(config).items():
        print(f"[livecodebench] preflight_{key}={value}")
    if not benchmark_exists:
        print("[livecodebench] benchmark checkout missing; dry-run only")

    write_metadata(
        run_dir,
        config,
        command,
        status="planned",
        exit_code=None,
        official_metrics=official_metrics,
    )

    if config.dry_run:
        write_summary(
            summary_path,
            config=config,
            command=command,
            run_dir=run_dir,
            exit_code=None,
            official_metrics=official_metrics,
        )
        return 0

    if not config.api_key:
        raise SystemExit(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY or pass --api-key."
        )

    before_snapshot = _snapshot_output_files(config.benchmark_root)
    runner = subprocess.run if command_runner is None else command_runner
    completed = runner(
        command,
        cwd=config.benchmark_root,
        env=env,
        check=False,
    )
    exit_code = int(completed.returncode)
    if config.evaluate:
        artifacts = _resolve_official_artifact_paths(
            config.benchmark_root,
            config=config,
            before_snapshot=before_snapshot,
        )
        official_metrics = _extract_official_metrics(artifacts)
    write_metadata(
        run_dir,
        config,
        command,
        status="passed" if exit_code == 0 else "failed",
        exit_code=exit_code,
        official_metrics=official_metrics,
    )
    write_summary(
        summary_path,
        config=config,
        command=command,
        run_dir=run_dir,
        exit_code=exit_code,
        official_metrics=official_metrics,
    )
    if official_metrics.get("official_metrics_available"):
        print(
            "[livecodebench] official_pass@1="
            f"{official_metrics.get('official_pass_at_1')}"
        )
        print(
            "[livecodebench] official_pass@5="
            f"{official_metrics.get('official_pass_at_5')}"
        )
    return exit_code


def main() -> int:
    args = build_parser().parse_args()
    resolved = resolve_config(args)
    exit_codes = [run(config) for config in resolved]
    return max(exit_codes, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
