from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

import vtm_dspy
from vtm.enums import ValidityStatus
from vtm.fingerprints import DependencyFingerprint
from vtm.retrieval import RetrieveCandidate, RetrieveExplanation, RetrieveRequest, RetrieveResult
from vtm.verification import VerificationResult
from vtm_dspy.config import DSPyOpenRouterConfig
from vtm_dspy.react_agent import VTMReActCodingAgent
from vtm_dspy.tools import VTMMemoryTools

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeKernel:
    def __init__(self, *, item, dep_fp: DependencyFingerprint) -> None:
        self._item = item
        self._dep_fp = dep_fp
        self.retrieve_requests: list[RetrieveRequest] = []
        self.expand_calls: list[str] = []
        self.verify_calls: list[tuple[str, DependencyFingerprint]] = []

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        self.retrieve_requests.append(request)
        candidate = RetrieveCandidate(
            memory=self._item,
            score=0.95,
            explanation=RetrieveExplanation(
                matched_tokens=("parser",),
                matched_fields=("title",),
                score=0.95,
                reason="fake lexical hit",
            ),
            evidence=self._item.evidence,
        )
        return RetrieveResult(
            request=request,
            candidates=(candidate,),
            total_candidates=1,
            verified_count=1,
        )

    def expand(self, memory_id: str):
        self.expand_calls.append(memory_id)
        return self._item.evidence

    def verify_memory(self, memory_id: str, current_dependency: DependencyFingerprint):
        self.verify_calls.append((memory_id, current_dependency))
        return self._item, VerificationResult(
            memory_id=memory_id,
            previous_status=self._item.validity.status,
            current_status=ValidityStatus.VERIFIED,
            dependency_changed=False,
            reasons=("memory remains valid",),
            updated_validity=self._item.validity,
            updated_evidence=self._item.evidence,
        )


def test_importing_vtm_dspy_stays_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "dspy":
            raise ImportError("missing dspy")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="optional 'dspy' dependency"):
        vtm_dspy.require_dspy()


def test_vtm_memory_tools_work_with_fake_kernel(
    memory_factory,
    anchor_evidence,
    dep_fp: DependencyFingerprint,
    scope,
) -> None:
    item = memory_factory(evidence=(anchor_evidence,))
    kernel = FakeKernel(item=item, dep_fp=dep_fp)
    tools = VTMMemoryTools(
        kernel=kernel,
        scopes=(scope,),
        dependency_provider=lambda: dep_fp,
        memory_lookup=lambda memory_id: item if memory_id == item.memory_id else None,
    )

    verified = tools.search_verified_memory("parser", k=3)
    naive = tools.search_naive_memory("parser", k=2)
    expanded = tools.expand_memory_evidence(item.memory_id)
    checked = tools.verify_memory(item.memory_id)

    assert verified[0]["id"] == item.memory_id
    assert verified[0]["status"] == "verified"
    assert verified[0]["path"] == "src/example.py"
    assert verified[0]["symbol"] == "target"
    assert verified[0]["verification_mode"] == "verify_on_read"
    assert naive[0]["id"] == item.memory_id
    assert expanded["evidence"][0]["path"] == "src/example.py"
    assert checked["verification_reasons"] == ["memory remains valid"]
    assert kernel.retrieve_requests[0].verify_on_read is True
    assert kernel.retrieve_requests[0].return_verified_only is True
    assert kernel.retrieve_requests[1].verify_on_read is False
    assert kernel.expand_calls == [item.memory_id]
    assert kernel.verify_calls[0][0] == item.memory_id


def test_react_agent_constructs_tools_without_running_a_model(
    memory_factory,
    dep_fp: DependencyFingerprint,
    scope,
    tmp_path: Path,
) -> None:
    item = memory_factory()
    kernel = FakeKernel(item=item, dep_fp=dep_fp)
    agent = VTMReActCodingAgent(
        kernel=kernel,
        scopes=(scope,),
        workspace_root=tmp_path,
        dependency_provider=lambda: dep_fp,
        memory_lookup=lambda memory_id: item if memory_id == item.memory_id else None,
    )

    assert "search_verified_memory" in agent.tool_names()
    assert "search_naive_memory" in agent.tool_names()
    assert "read_file" in agent.tool_names()
    assert "run_command" in agent.tool_names()
    description = agent.describe()
    assert description["workspace_tools_enabled"] is True
    assert description["memory_tools_enabled"] is True
    assert description["workspace_root"] == str(tmp_path.resolve())


def test_openrouter_config_maps_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("VTM_OPENROUTER_BASE_URL", "https://openrouter.example/api/v1")
    monkeypatch.setenv("VTM_EXECUTION_MODEL", "google/gemma-test")
    monkeypatch.setenv("VTM_RERANK_MODEL", "nvidia/nemotron-test")
    monkeypatch.delenv("VTM_DSPY_MODEL", raising=False)

    config = DSPyOpenRouterConfig.from_env()

    assert config.base_url == "https://openrouter.example/api/v1"
    assert config.api_key == "openrouter-test-key"
    assert config.execution_model == "google/gemma-test"
    assert config.rerank_model == "nvidia/nemotron-test"
    assert config.dspy_model == "openrouter/google/gemma-test"
    assert config.lm_model_name() == "openai/google/gemma-test"
    assert config.as_env()["VTM_DSPY_MODEL"] == "openrouter/google/gemma-test"


def test_docs_frame_dspy_and_livecodebench_correctly() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    final_scope = (REPO_ROOT / "docs" / "final-scope.md").read_text(encoding="utf-8")
    dspy_doc = (REPO_ROOT / "docs" / "dspy-integration.md").read_text(encoding="utf-8")

    assert "DSPy is the recommended forward-facing agent and programming interface" in readme
    assert "LiveCodeBench support is available for baseline model coding ability checks" in readme
    assert "main VTM evidence remains retrieval, drift, and drifted retrieval" in readme
    assert "DSPy is the recommended forward-facing agent interface for VTM memory" in final_scope
    assert (
        "controlled_coding_drift remains the small maintained agent-loop benchmark" in final_scope
    )
    assert (
        "DSPy is the recommended forward-facing agent and programming interface for VTM"
        in dspy_doc
    )
