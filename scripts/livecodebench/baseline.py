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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
    return command


def normalized_run_dir(config: BaselineConfig) -> Path:
    return config.output_root / model_slug(config.model) / config.run_id


def normalized_summary_path(config: BaselineConfig) -> Path:
    return config.summary_root / f"{config.run_id}__{model_slug(config.model)}.json"


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
) -> None:
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
        f"benchmark_root={config.benchmark_root}",
        f"output_dir={run_dir}",
        f"summary_path={normalized_summary_path(config)}",
        f"exit_code={'' if exit_code is None else exit_code}",
        f"command={shlex.join(command)}",
        f"started_at={dt.datetime.now(dt.UTC).isoformat()}",
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
) -> None:
    payload = {
        "benchmark": "livecodebench",
        "kind": "baseline_model_evaluation",
        "mode": "baseline_only",
        "run_id": config.run_id,
        "model": config.model,
        "scenario": config.scenario,
        "release_version": config.release_version,
        "smoke": config.smoke,
        "evaluate": config.evaluate,
        "output_dir": str(run_dir),
        "command": command,
        "command_text": shlex.join(command),
        "base_url": config.base_url,
        "exit_code": exit_code,
        "status": "planned" if exit_code is None else ("passed" if exit_code == 0 else "failed"),
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

    write_metadata(run_dir, config, command, status="planned", exit_code=None)

    if config.dry_run:
        write_summary(summary_path, config=config, command=command, run_dir=run_dir, exit_code=None)
        return 0

    if not config.api_key:
        raise SystemExit(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY or pass --api-key."
        )

    runner = subprocess.run if command_runner is None else command_runner
    completed = runner(
        command,
        cwd=config.benchmark_root,
        env=env,
        check=False,
    )
    exit_code = int(completed.returncode)
    write_metadata(
        run_dir,
        config,
        command,
        status="passed" if exit_code == 0 else "failed",
        exit_code=exit_code,
    )
    write_summary(
        summary_path,
        config=config,
        command=command,
        run_dir=run_dir,
        exit_code=exit_code,
    )
    return exit_code


def main() -> int:
    args = build_parser().parse_args()
    resolved = resolve_config(args)
    exit_codes = [run(config) for config in resolved]
    return max(exit_codes, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
