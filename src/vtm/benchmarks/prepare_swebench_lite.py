"""CLI entrypoint for generating a SWE-bench Lite manifest."""

from __future__ import annotations

import argparse

from vtm.benchmarks.swebench import SWEbenchLitePreparer


def build_parser() -> argparse.ArgumentParser:
    """Build the SWE-bench Lite preparation CLI parser."""
    parser = argparse.ArgumentParser(description="Prepare a SWE-bench Lite manifest for VTM.")
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Optional local SWE-bench Lite dataset path (.json or .jsonl).",
    )
    parser.add_argument(
        "--dataset-name",
        default="princeton-nlp/SWE-bench_Lite",
        help="Dataset identifier used for loading and downstream harness evaluation.",
    )
    parser.add_argument(
        "--cache-root",
        default=".benchmarks/swebench-lite",
        help="Directory for local repo caches and generated refs.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Path to write the generated benchmark manifest.",
    )
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="Optional SWE-bench instance filter. Repeat to select multiple instances.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Optional prepared repo filter (for example astropy__astropy). Repeat as needed.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional limit for smoke or development runs.",
    )
    parser.add_argument(
        "--skip-failed-instances",
        action="store_true",
        help="Skip instance preparation failures, write a preparation report, and continue.",
    )
    return parser


def main() -> int:
    """Generate a manifest and print it to stdout."""
    args = build_parser().parse_args()
    manifest = SWEbenchLitePreparer().prepare_manifest(
        dataset_name=args.dataset_name,
        cache_root=args.cache_root,
        output_manifest=args.output_manifest,
        dataset_path=args.dataset_path or None,
        repo_filters=tuple(args.repo),
        instance_filters=tuple(args.instance),
        max_instances=args.max_instances,
        skip_failed_instances=args.skip_failed_instances,
    )
    print(manifest.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
