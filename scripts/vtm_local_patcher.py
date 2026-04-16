#!/usr/bin/env python3
"""CLI wrapper for the local OpenAI-compatible benchmark patcher."""

from __future__ import annotations

import argparse

from vtm.benchmarks.local_patcher import LocalOpenAIPatcher


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one local patching attempt."""
    parser = argparse.ArgumentParser(description="Run the local VTM OpenAI-compatible patcher.")
    parser.add_argument("--task-file", required=True, help="Path to the benchmark task pack JSON.")
    parser.add_argument("--workspace", required=True, help="Path to the writable git workspace.")
    return parser


def main() -> int:
    """Run the patcher from parsed CLI arguments."""
    args = build_parser().parse_args()
    return LocalOpenAIPatcher().run(task_file=args.task_file, workspace=args.workspace)


if __name__ == "__main__":
    raise SystemExit(main())
