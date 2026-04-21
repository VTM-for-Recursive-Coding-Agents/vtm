from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.anchors import CodeAnchor
from vtm.enums import EvidenceKind, MemoryKind, ScopeKind, ValidityStatus
from vtm.evidence import ArtifactRef, EvidenceRef
from vtm.fingerprints import DependencyFingerprint, EnvFingerprint, RepoFingerprint, ToolVersion
from vtm.harness.models import HarnessTaskPack
from vtm.memory_items import (
    ClaimPayload,
    MemoryItem,
    ProcedurePayload,
    ProcedureStep,
    SummaryCardPayload,
    ValidatorSpec,
    ValidityState,
    VisibilityScope,
)
from vtm.services.memory_kernel import TransactionalMemoryKernel
from vtm.services.procedures import CommandProcedureValidator
from vtm.services.retriever import LexicalRetriever
from vtm.services.verifier import BasicVerifier
from vtm.stores.artifact_store import FilesystemArtifactStore
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.sqlite_store import SqliteMetadataStore

DEFAULT_PROCEDURE_VALIDATOR = object()


@pytest.fixture
def repo_fp(tmp_path: Path) -> RepoFingerprint:
    return RepoFingerprint(
        repo_root=str(tmp_path),
        branch="main",
        head_commit="abc123",
        tree_digest="tree-1",
        dirty_digest="dirty-1",
    )


@pytest.fixture
def env_fp() -> EnvFingerprint:
    return EnvFingerprint(
        python_version="3.12.8",
        platform="darwin-arm64",
        tool_versions=(ToolVersion(name="pytest", version="8.3.4"),),
    )


@pytest.fixture
def dep_fp(repo_fp: RepoFingerprint, env_fp: EnvFingerprint) -> DependencyFingerprint:
    return DependencyFingerprint(
        repo=repo_fp,
        env=env_fp,
        dependency_ids=("artifact:123",),
        input_digests=("input-1",),
    )


@pytest.fixture
def scope() -> VisibilityScope:
    return VisibilityScope(kind=ScopeKind.BRANCH, scope_id="main")


@pytest.fixture
def artifact_evidence() -> EvidenceRef:
    return EvidenceRef(
        kind=EvidenceKind.ARTIFACT,
        ref_id="artifact:1",
        artifact_ref=ArtifactRef(
            artifact_id="art_existing",
            sha256="deadbeef",
            content_type="text/plain",
        ),
        summary="tool output",
    )


@pytest.fixture
def anchor_evidence() -> EvidenceRef:
    return EvidenceRef(
        kind=EvidenceKind.CODE_ANCHOR,
        ref_id="anchor:1",
        code_anchor=CodeAnchor(
            path="src/example.py",
            symbol="target",
            kind="function",
            language="python",
            ast_digest="ast-1",
            context_digest="ctx-1",
            start_line=10,
            end_line=12,
            start_byte=100,
            end_byte=160,
        ),
        summary="source anchor",
    )


@pytest.fixture
def memory_factory(
    scope: VisibilityScope,
    dep_fp: DependencyFingerprint,
    artifact_evidence: EvidenceRef,
) -> Callable[..., MemoryItem]:
    def _make(
        *,
        title: str = "Parser claim",
        summary: str = "Parser output is stable",
        evidence: tuple[EvidenceRef, ...] | None = None,
        validity_status: ValidityStatus = ValidityStatus.VERIFIED,
        dependency: DependencyFingerprint | None = None,
        tags: tuple[str, ...] = ("parser",),
    ) -> MemoryItem:
        status = validity_status
        dependency_fingerprint = dependency
        if (
            status in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}
            and dependency_fingerprint is None
        ):
            dependency_fingerprint = dep_fp
        evidence_refs = evidence if evidence is not None else (artifact_evidence,)
        return MemoryItem(
            kind=MemoryKind.CLAIM,
            title=title,
            summary=summary,
            payload=ClaimPayload(claim=summary),
            evidence=evidence_refs,
            tags=tags,
            visibility=scope,
            validity=ValidityState(status=status, dependency_fingerprint=dependency_fingerprint),
        )

    return _make


@pytest.fixture
def procedure_factory(
    scope: VisibilityScope,
    dep_fp: DependencyFingerprint,
    artifact_evidence: EvidenceRef,
) -> Callable[..., MemoryItem]:
    def _make(
        *,
        title: str = "Parser procedure",
        summary: str = "Run the parser validation flow",
        goal: str = "Run the parser validation flow",
        steps: tuple[ProcedureStep, ...] | None = None,
        validator: ValidatorSpec | None | object = DEFAULT_PROCEDURE_VALIDATOR,
        evidence: tuple[EvidenceRef, ...] | None = None,
        validity_status: ValidityStatus = ValidityStatus.PENDING,
        dependency: DependencyFingerprint | None = None,
        tags: tuple[str, ...] = ("procedure",),
        metadata: dict[str, object] | None = None,
    ) -> MemoryItem:
        status = validity_status
        dependency_fingerprint = dependency
        if (
            status in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}
            and dependency_fingerprint is None
        ):
            dependency_fingerprint = dep_fp

        procedure_steps = steps or (ProcedureStep(order=0, instruction="Run parser"),)
        evidence_refs = evidence
        if evidence_refs is None:
            evidence_refs = (
                (artifact_evidence,)
                if status in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}
                else ()
            )

        return MemoryItem(
            kind=MemoryKind.PROCEDURE,
            title=title,
            summary=summary,
            payload=ProcedurePayload(
                goal=goal,
                steps=procedure_steps,
                validator=(
                    ValidatorSpec(
                        name="parser-check",
                        kind="command",
                        config={"command": ["python3", "-c", "print('ok')"]},
                    )
                    if validator is DEFAULT_PROCEDURE_VALIDATOR
                    else validator
                ),
            ),
            evidence=evidence_refs,
            tags=tags,
            visibility=scope,
            validity=ValidityState(status=status, dependency_fingerprint=dependency_fingerprint),
            metadata=dict(metadata or {}),
        )

    return _make


@pytest.fixture
def summary_card(scope: VisibilityScope, artifact_evidence: EvidenceRef) -> MemoryItem:
    return MemoryItem(
        kind=MemoryKind.SUMMARY_CARD,
        title="Parser summary card",
        summary="Summarized parser state",
        payload=SummaryCardPayload(
            summary="Summarized parser state",
            supporting_memory_ids=("mem_a",),
        ),
        evidence=(artifact_evidence,),
        visibility=scope,
    )


@pytest.fixture
def metadata_store(tmp_path: Path) -> SqliteMetadataStore:
    store = SqliteMetadataStore(
        tmp_path / "metadata.sqlite",
        event_log_path=tmp_path / "events.jsonl",
    )
    yield store
    store.close()


@pytest.fixture
def artifact_store(tmp_path: Path) -> FilesystemArtifactStore:
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    yield store
    store.close()


@pytest.fixture
def cache_store(tmp_path: Path, metadata_store: SqliteMetadataStore) -> SqliteCacheStore:
    store = SqliteCacheStore(tmp_path / "cache.sqlite", event_store=metadata_store)
    yield store
    store.close()


@pytest.fixture
def kernel(
    metadata_store: SqliteMetadataStore,
    artifact_store: FilesystemArtifactStore,
    cache_store: SqliteCacheStore,
) -> TransactionalMemoryKernel:
    anchor_builder = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    return TransactionalMemoryKernel(
        metadata_store=metadata_store,
        event_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        verifier=BasicVerifier(relocator=anchor_builder),
        retriever=LexicalRetriever(metadata_store),
        anchor_adapter=anchor_builder,
        procedure_validator=CommandProcedureValidator(artifact_store),
    )


@pytest.fixture
def install_fake_benchmark_agent(monkeypatch: pytest.MonkeyPatch):
    def _install(
        *,
        apply_workspace_update: Callable[[Any, Path, Path], None] | None = None,
        response: str = "FINAL(Applied synthetic benchmark update.)",
    ) -> None:
        def _apply_update(task_pack, workspace_root: Path, artifact_root: Path) -> None:
            if apply_workspace_update is None:
                diff_result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--binary",
                        "--no-ext-diff",
                        f"{task_pack.base_ref}..{task_pack.head_ref}",
                    ],
                    cwd=workspace_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if diff_result.stdout.strip():
                    subprocess.run(
                        ["git", "apply", "--whitespace=nowarn"],
                        cwd=workspace_root,
                        input=diff_result.stdout,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
            else:
                apply_workspace_update(task_pack, workspace_root, artifact_root)

        def fake_run_dspy_agent(self, agent, prompt: str):  # noqa: ANN001
            del agent, prompt
            task_pack = HarnessTaskPack.model_validate_json(
                Path(self._active_request.task_file).read_text(encoding="utf-8")
            )
            artifact_root = Path(self._active_artifact_root)
            workspace_root = Path(self._active_workspace_root)
            _apply_update(task_pack, workspace_root, artifact_root)
            return {
                "response": {"response": response},
                "trajectory": {
                    "execution_mode": "react",
                    "diagnostics": {
                        "lm_call_count": 1,
                        "tool_call_count": 1,
                        "total_lm_duration_ms": 25.0,
                        "total_prompt_tokens": 10,
                        "total_completion_tokens": 5,
                        "truncated_lm_call_count": 0,
                    },
                },
            }

        monkeypatch.setattr(
            "vtm.harness.executors.DSPyReActBenchmarkExecutor._run_agent",
            fake_run_dspy_agent,
        )

    return _install


@pytest.fixture
def fake_docker_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state_dir = tmp_path / "fake-docker-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_path / "fake-docker"
    script_path.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import subprocess
            import sys
            from pathlib import Path


            STATE_DIR = Path(os.environ["FAKE_DOCKER_STATE_DIR"])
            STATE_DIR.mkdir(parents=True, exist_ok=True)


            def _container_state(name: str) -> Path:
                return STATE_DIR / f"{name}.json"


            def _load_state(name: str) -> dict[str, object]:
                return json.loads(_container_state(name).read_text(encoding="utf-8"))


            def _write_state(name: str, payload: dict[str, object]) -> None:
                _container_state(name).write_text(
                    json.dumps(payload, sort_keys=True),
                    encoding="utf-8",
                )


            def _run(args: list[str]) -> int:
                name = None
                network = "none"
                workdir = None
                image = None
                read_only_rootfs = False
                pids_limit = None
                memory_limit = None
                cpu_limit = None
                tmpfs_mounts = []
                bind_mounts = []
                security_opts = []
                cap_drops = []
                index = 0
                while index < len(args):
                    token = args[index]
                    if token == "-d":
                        index += 1
                    elif token == "--read-only":
                        read_only_rootfs = True
                        index += 1
                    elif token in {
                        "--name",
                        "--network",
                        "--cap-drop",
                        "--security-opt",
                        "--tmpfs",
                        "--pids-limit",
                        "--memory",
                        "--cpus",
                        "--user",
                        "-v",
                        "-w",
                    }:
                        value = args[index + 1]
                        if token == "--name":
                            name = value
                        elif token == "--network":
                            network = value
                        elif token == "--cap-drop":
                            cap_drops.append(value)
                        elif token == "--security-opt":
                            security_opts.append(value)
                        elif token == "--tmpfs":
                            tmpfs_mounts.append(value)
                        elif token == "--pids-limit":
                            pids_limit = value
                        elif token == "--memory":
                            memory_limit = value
                        elif token == "--cpus":
                            cpu_limit = value
                        elif token == "-w":
                            workdir = value
                        elif token == "-v":
                            bind_mounts.append(value)
                        index += 2
                    else:
                        image = token
                        break
                if name is None or workdir is None or image is None:
                    raise SystemExit("fake docker run missing required arguments")
                _write_state(
                    name,
                    {
                        "container_id": f"fake-{name}",
                        "image": image,
                        "network": network,
                        "workdir": workdir,
                        "read_only_rootfs": read_only_rootfs,
                        "pids_limit": pids_limit,
                        "memory_limit": memory_limit,
                        "cpu_limit": cpu_limit,
                        "tmpfs_mounts": tmpfs_mounts,
                        "bind_mounts": bind_mounts,
                        "security_opts": security_opts,
                        "cap_drops": cap_drops,
                    },
                )
                sys.stdout.write(f"fake-{name}\\n")
                return 0


            def _exec(args: list[str]) -> int:
                interactive = False
                workdir = None
                index = 0
                while index < len(args):
                    token = args[index]
                    if token == "-i":
                        interactive = True
                        index += 1
                    elif token == "-w":
                        workdir = args[index + 1]
                        index += 2
                    else:
                        break
                name = args[index]
                command = args[index + 1 :]
                if not command:
                    raise SystemExit("fake docker exec missing command")
                state = _load_state(name)
                cwd = Path(workdir or str(state["workdir"]))
                env = {
                    **os.environ,
                    "PS1": "",
                    "TERM": "dumb",
                }
                if interactive:
                    os.chdir(cwd)
                    os.execvpe(command[0], command, env)
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                sys.stdout.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                return completed.returncode


            def _rm(args: list[str]) -> int:
                names = [value for value in args if value != "-f"]
                for name in names:
                    state_path = _container_state(name)
                    if state_path.exists():
                        state_path.unlink()
                return 0


            def main() -> int:
                if len(sys.argv) < 2:
                    raise SystemExit("fake docker requires a subcommand")
                command = sys.argv[1]
                args = sys.argv[2:]
                if command == "run":
                    return _run(args)
                if command == "exec":
                    return _exec(args)
                if command == "rm":
                    return _rm(args)
                raise SystemExit(f"unsupported fake docker command: {command}")


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_STATE_DIR", str(state_dir))
    return script_path
