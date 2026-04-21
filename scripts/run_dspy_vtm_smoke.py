#!/usr/bin/env python3
"""Dry-run smoke script for the optional DSPy plus VTM integration layer."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.enums import MemoryKind, ScopeKind, ValidityStatus
from vtm.fingerprints import DependencyFingerprint, EnvFingerprint, RepoFingerprint, ToolVersion
from vtm.memory_items import ClaimPayload, MemoryItem, ValidityState, VisibilityScope
from vtm.services.memory_kernel import TransactionalMemoryKernel
from vtm.services.procedures import CommandProcedureValidator
from vtm.services.retriever import LexicalRetriever
from vtm.services.verifier import BasicVerifier
from vtm.stores.artifact_store import FilesystemArtifactStore
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.sqlite_store import SqliteMetadataStore
from vtm_dspy.config import DSPyOpenRouterConfig
from vtm_dspy.react_agent import VTMReActCodingAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run smoke script for the optional DSPy integration. "
            "No OpenRouter call is made unless --run-model is passed."
        )
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used for controlled file and git tools.",
    )
    parser.add_argument(
        "--query",
        default="How should I set up the VTM repository?",
        help="Query used to exercise the memory tools.",
    )
    parser.add_argument(
        "--task",
        default="Summarize the available repository memory before editing anything.",
        help="Task passed into DSPy ReAct when --run-model is supplied.",
    )
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Actually instantiate DSPy and call the configured OpenRouter-backed model.",
    )
    return parser.parse_args()


def build_dependency(workspace_root: Path) -> DependencyFingerprint:
    return DependencyFingerprint(
        repo=RepoFingerprint(
            repo_root=str(workspace_root),
            branch="josh-testing",
            head_commit="smoke-head",
            tree_digest="smoke-tree",
            dirty_digest="smoke-dirty",
        ),
        env=EnvFingerprint(
            python_version=platform.python_version(),
            platform=platform.platform(),
            tool_versions=(ToolVersion(name="uv", version="smoke"),),
        ),
        dependency_ids=("smoke:artifact",),
        input_digests=("smoke-input",),
    )


def build_smoke_kernel(
    workspace_root: Path,
    state_root: Path,
) -> tuple[TransactionalMemoryKernel, SqliteMetadataStore, VisibilityScope, DependencyFingerprint]:
    metadata_store = SqliteMetadataStore(
        state_root / "metadata.sqlite",
        event_log_path=state_root / "events.jsonl",
    )
    artifact_store = FilesystemArtifactStore(state_root / "artifacts")
    cache_store = SqliteCacheStore(state_root / "cache.sqlite", event_store=metadata_store)
    anchor_builder = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    kernel = TransactionalMemoryKernel(
        metadata_store=metadata_store,
        event_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        verifier=BasicVerifier(relocator=anchor_builder),
        retriever=LexicalRetriever(metadata_store),
        anchor_adapter=anchor_builder,
        procedure_validator=CommandProcedureValidator(artifact_store),
    )
    scope = VisibilityScope(kind=ScopeKind.BRANCH, scope_id="smoke")
    dependency = build_dependency(workspace_root)
    record = kernel.capture_artifact(
        b"Use `uv sync --dev` for the base VTM environment. DSPy remains optional.",
        content_type="text/plain",
        tool_name="run_dspy_vtm_smoke",
        metadata={"purpose": "smoke"},
    )
    evidence = kernel.artifact_evidence(
        record,
        label="setup_note",
        summary="Base setup uses uv sync --dev. DSPy is optional.",
    )
    transaction = kernel.begin_transaction(scope)
    kernel.stage_memory_item(
        transaction.tx_id,
        MemoryItem(
            kind=MemoryKind.CLAIM,
            title="Base VTM setup",
            summary="Use uv sync --dev for the base environment; add DSPy only when needed.",
            payload=ClaimPayload(
                claim="Base VTM setup uses uv sync --dev, and DSPy is an optional extra.",
            ),
            evidence=(evidence,),
            tags=("setup", "dspy"),
            visibility=scope,
            validity=ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=dependency,
            ),
        ),
    )
    kernel.commit_transaction(transaction.tx_id)
    return kernel, metadata_store, scope, dependency


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    model_config = DSPyOpenRouterConfig.from_env()
    with tempfile.TemporaryDirectory(prefix="vtm-dspy-smoke-") as temp_dir:
        kernel, metadata_store, scope, dependency = build_smoke_kernel(
            workspace_root,
            Path(temp_dir),
        )
        agent = VTMReActCodingAgent(
            kernel=kernel,
            scopes=(scope,),
            workspace_root=workspace_root,
            dependency_provider=lambda: dependency,
            memory_lookup=metadata_store.get_memory_item,
            model_config=model_config,
        )
        payload: dict[str, object] = {
            "dry_run": not args.run_model,
            "agent": agent.describe(),
            "config": model_config.summary(),
            "query": args.query,
            "naive_results": agent.memory_tools.search_naive_memory(args.query),
            "verified_results": agent.memory_tools.search_verified_memory(args.query),
        }
        if args.run_model:
            payload["result"] = agent.run(args.task)
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
