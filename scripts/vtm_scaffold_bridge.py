#!/usr/bin/env python3
"""Build a scaffold-facing task bundle and optionally delegate to an external agent."""

from __future__ import annotations

import argparse

from vtm.benchmarks.scaffold_bridge import ScaffoldBridge


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one scaffold-bridge invocation."""
    parser = argparse.ArgumentParser(
        description="Build a VTM scaffold bundle and optional external-agent handoff."
    )
    parser.add_argument("--task-file", required=True, help="Path to the benchmark task pack JSON.")
    parser.add_argument("--workspace", required=True, help="Path to the writable git workspace.")
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Directory where scaffold artifacts and optional delegate logs are written.",
    )
    parser.add_argument(
        "--delegate-command",
        default="",
        help=(
            "Optional command template to run after writing the bundle. Supports "
            "{task_file}, {workspace}, {artifact_root}, {scaffold_bundle}, and {brief_file}."
        ),
    )
    return parser


def main() -> int:
    """Build the bundle, then optionally run a delegate command."""
    args = build_parser().parse_args()
    return ScaffoldBridge().run(
        task_file=args.task_file,
        workspace=args.workspace,
        artifact_root=args.artifact_root,
        delegate_command=args.delegate_command or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
