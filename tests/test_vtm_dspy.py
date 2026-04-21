from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

import vtm_dspy
from vtm.benchmarks.livecodebench_dspy_pilot import (
    describe_method_runtime,
    open_memory_session,
    open_persistent_memory_session,
    run_dspy_attempt,
)
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
    original_import_module = importlib.import_module

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

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "dspy":
            raise ImportError("missing dspy")
        return original_import_module(name, package)

    monkeypatch.delitem(sys.modules, "dspy", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

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


def test_vtm_memory_tools_buffer_write_proposals(
    memory_factory,
    dep_fp: DependencyFingerprint,
    scope,
) -> None:
    item = memory_factory()
    kernel = FakeKernel(item=item, dep_fp=dep_fp)
    tools = VTMMemoryTools(
        kernel=kernel,
        scopes=(scope,),
        dependency_provider=lambda: dep_fp,
        memory_lookup=lambda memory_id: item if memory_id == item.memory_id else None,
    )

    lesson = tools.propose_memory_lesson(
        "Off-by-one repair",
        "When public tests are off by one, re-check inclusive bounds before returning.",
        rationale="Observed during repair after a public mismatch.",
        transfer_terms="bounds,off_by_one",
        function_name="solve",
        repair_kind="public_test_logic_mismatch",
        interface_mode="top_level_function",
    )
    failure = tools.propose_failure_pattern(
        "expected 5 actual 4",
        "Mismatched totals usually indicate the subtraction branch survived instead of the sum branch.",
        transfer_terms="logic_mismatch,sum",
    )
    drained = tools.drain_write_proposals()

    assert lesson["status"] == "buffered"
    assert failure["status"] == "buffered"
    assert len(drained) == 2
    assert drained[0]["memory_role"] == "repair_lesson"
    assert drained[0]["function_name"] == "solve"
    assert drained[0]["transfer_terms"] == ["bounds", "off_by_one"]
    assert drained[1]["proposal_kind"] == "failure_pattern"
    assert tools.drain_write_proposals() == []


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
    assert "propose_memory_lesson" in agent.tool_names()
    assert "propose_failure_pattern" in agent.tool_names()
    assert "propose_solution_pattern" in agent.tool_names()
    assert "read_file" in agent.tool_names()
    assert "run_command" in agent.tool_names()
    description = agent.describe()
    assert description["workspace_tools_enabled"] is True
    assert description["memory_tools_enabled"] is True
    assert description["workspace_root"] == str(tmp_path.resolve())


def test_react_agent_uses_plain_predict_when_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeProgram:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.lm = None

        def set_lm(self, lm) -> None:
            self.lm = lm

    class FakeDSPy:
        @staticmethod
        def Predict(signature: str) -> FakeProgram:
            calls.append(f"predict:{signature}")
            return FakeProgram("predict")

        @staticmethod
        def ReAct(
            signature: str,
            tools: list[object],
            max_iters: int | None = None,
        ) -> FakeProgram:
            calls.append(f"react:{signature}:{len(tools)}")
            assert max_iters is not None
            return FakeProgram("react")

    agent = VTMReActCodingAgent(kernel=None, scopes=(), workspace_root=None)
    monkeypatch.setattr("vtm_dspy.react_agent.require_dspy", lambda: FakeDSPy)
    monkeypatch.setattr(agent, "create_lm", lambda **_: object())

    program = agent.create_program()

    assert isinstance(program, FakeProgram)
    assert program.kind == "predict"
    assert calls == ["predict:task -> response"]


def test_react_agent_uses_react_when_tools_are_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class FakeProgram:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.lm = None

        def set_lm(self, lm) -> None:
            self.lm = lm

    class FakeDSPy:
        @staticmethod
        def Predict(signature: str) -> FakeProgram:
            calls.append(f"predict:{signature}")
            return FakeProgram("predict")

        @staticmethod
        def ReAct(
            signature: str,
            tools: list[object],
            max_iters: int | None = None,
        ) -> FakeProgram:
            calls.append(f"react:{signature}:{len(tools)}")
            assert max_iters is not None
            return FakeProgram("react")

    agent = VTMReActCodingAgent(kernel=None, scopes=(), workspace_root=tmp_path)
    monkeypatch.setattr("vtm_dspy.react_agent.require_dspy", lambda: FakeDSPy)
    monkeypatch.setattr(agent, "create_lm", lambda **_: object())

    program = agent.create_program()

    assert isinstance(program, FakeProgram)
    assert program.kind == "react"
    assert calls == ["react:task -> response:4"]

def test_livecodebench_dspy_vtm_runtime_exposes_memory_tools() -> None:
    runtime = describe_method_runtime(
        "dspy_vtm",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
    )

    assert runtime.uses_dspy is True
    assert runtime.uses_vtm_memory is True
    assert runtime.memory_tools_enabled is True
    assert "search_verified_memory" in runtime.tool_names
    assert "verify_memory" in runtime.tool_names


def test_livecodebench_dspy_vtm_local_only_runtime_exposes_memory_tools() -> None:
    runtime = describe_method_runtime(
        "dspy_vtm_local_only",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
    )

    assert runtime.uses_dspy is True
    assert runtime.uses_vtm_memory is True
    assert runtime.memory_tools_enabled is True
    assert "search_verified_memory" in runtime.tool_names


def test_livecodebench_dspy_vtm_persistent_only_runtime_exposes_memory_tools() -> None:
    runtime = describe_method_runtime(
        "dspy_vtm_persistent_only",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
    )

    assert runtime.uses_dspy is True
    assert runtime.uses_vtm_memory is True
    assert runtime.memory_tools_enabled is True
    assert "search_verified_memory" in runtime.tool_names


def test_livecodebench_dspy_vtm_attempt_exposes_memory_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, task: str, *, signature: str = "task -> response") -> dict[str, Any]:
        del task, signature
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        captured["max_iters"] = self.max_iters
        return {
            "response": {"response": "```python\npass\n```"},
            "trajectory": {"execution_mode": "react", "diagnostics": {}},
        }

    monkeypatch.setattr(VTMReActCodingAgent, "run", fake_run)
    session = open_memory_session(
        state_root=tmp_path / "pilot-session",
        problem_id="example-problem",
        workspace_root=tmp_path,
    )
    try:
        payload = run_dspy_attempt(
            prompt="solve it",
            method="dspy_vtm",
            session=session,
            model_config=DSPyOpenRouterConfig(
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
                execution_model="google/gemma-test",
                rerank_model="google/rerank-test",
                dspy_model="openrouter/google/gemma-test",
            ),
            attempt_index=2,
        )
    finally:
        session.close()

    assert captured["memory_tools_enabled"] is True
    assert captured["max_iters"] == 10
    assert "search_verified_memory" in captured["tool_names"]
    assert "search_naive_memory" in captured["tool_names"]
    assert "expand_memory_evidence" in captured["tool_names"]
    assert "verify_memory" in captured["tool_names"]
    assert payload["response_text"] == "```python\npass\n```"


def test_livecodebench_dspy_vtm_local_only_attempt_exposes_memory_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, task: str, *, signature: str = "task -> response") -> dict[str, Any]:
        del task, signature
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        captured["max_iters"] = self.max_iters
        return {
            "response": {"response": "```python\npass\n```"},
            "trajectory": {"execution_mode": "react", "diagnostics": {}},
        }

    monkeypatch.setattr(VTMReActCodingAgent, "run", fake_run)
    session = open_memory_session(
        state_root=tmp_path / "pilot-session",
        problem_id="example-problem",
        workspace_root=tmp_path,
    )
    try:
        payload = run_dspy_attempt(
            prompt="solve it",
            method="dspy_vtm_local_only",
            session=session,
            model_config=DSPyOpenRouterConfig(
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
                execution_model="google/gemma-test",
                rerank_model="google/rerank-test",
                dspy_model="openrouter/google/gemma-test",
            ),
            attempt_index=2,
        )
    finally:
        session.close()

    assert captured["memory_tools_enabled"] is True
    assert captured["max_iters"] == 10
    assert "search_verified_memory" in captured["tool_names"]
    assert payload["response_text"] == "```python\npass\n```"


def test_livecodebench_dspy_vtm_persistent_only_attempt_exposes_memory_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, task: str, *, signature: str = "task -> response") -> dict[str, Any]:
        del task, signature
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        captured["max_iters"] = self.max_iters
        return {
            "response": {"response": "```python\npass\n```"},
            "trajectory": {"execution_mode": "react", "diagnostics": {}},
        }

    monkeypatch.setattr(VTMReActCodingAgent, "run", fake_run)
    session = open_persistent_memory_session(
        state_root=tmp_path / "persistent-session",
        scenario="self_repair",
        model="qwen/qwen3-coder-next",
        workspace_root=tmp_path,
    )
    try:
        payload = run_dspy_attempt(
            prompt="solve it",
            method="dspy_vtm_persistent_only",
            session=session,
            model_config=DSPyOpenRouterConfig(
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
                execution_model="google/gemma-test",
                rerank_model="google/rerank-test",
                dspy_model="openrouter/google/gemma-test",
            ),
            attempt_index=2,
        )
    finally:
        session.close()

    assert captured["memory_tools_enabled"] is True
    assert captured["max_iters"] == 10
    assert "search_verified_memory" in captured["tool_names"]
    assert payload["response_text"] == "```python\npass\n```"


def test_livecodebench_dspy_vtm_attempt_one_keeps_write_tools_but_hides_lookup_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, task: str, *, signature: str = "task -> response") -> dict[str, Any]:
        del task, signature
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        captured["max_iters"] = self.max_iters
        return {
            "response": {"response": "```python\npass\n```"},
            "trajectory": {"execution_mode": "react", "diagnostics": {}},
        }

    monkeypatch.setattr(VTMReActCodingAgent, "run", fake_run)
    session = open_memory_session(
        state_root=tmp_path / "pilot-session",
        problem_id="example-problem",
        workspace_root=tmp_path,
    )
    try:
        payload = run_dspy_attempt(
            prompt="solve it",
            method="dspy_vtm",
            session=session,
            model_config=DSPyOpenRouterConfig(
                base_url="https://openrouter.example/api/v1",
                api_key="openrouter-test-key",
                execution_model="google/gemma-test",
                rerank_model="google/rerank-test",
                dspy_model="openrouter/google/gemma-test",
            ),
            attempt_index=1,
        )
    finally:
        session.close()

    assert captured["memory_tools_enabled"] is True
    assert captured["max_iters"] == 8
    assert "search_verified_memory" not in captured["tool_names"]
    assert "search_naive_memory" not in captured["tool_names"]
    assert "expand_memory_evidence" not in captured["tool_names"]
    assert "verify_memory" not in captured["tool_names"]
    assert "propose_memory_lesson" in captured["tool_names"]
    assert "propose_failure_pattern" in captured["tool_names"]
    assert "propose_solution_pattern" in captured["tool_names"]
    assert payload["response_text"] == "```python\npass\n```"


def test_livecodebench_dspy_baseline_runtime_hides_memory_tools() -> None:
    runtime = describe_method_runtime(
        "dspy_baseline",
        model="qwen/qwen3-coder-next",
        base_url="https://openrouter.example/api/v1",
        api_key="openrouter-test-key",
    )

    assert runtime.uses_dspy is True
    assert runtime.uses_vtm_memory is False
    assert runtime.memory_tools_enabled is False
    assert runtime.tool_names == ()


def test_livecodebench_dspy_baseline_attempt_hides_memory_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, task: str, *, signature: str = "task -> response") -> dict[str, Any]:
        del task, signature
        captured["tool_names"] = self.tool_names()
        captured["memory_tools_enabled"] = self.memory_tools.enabled
        captured["max_iters"] = self.max_iters
        return {
            "response": {"response": "```python\npass\n```"},
            "trajectory": {"execution_mode": "predict", "diagnostics": {}},
        }

    monkeypatch.setattr(VTMReActCodingAgent, "run", fake_run)

    payload = run_dspy_attempt(
        prompt="solve it",
        method="dspy_baseline",
        session=None,
        model_config=DSPyOpenRouterConfig(
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
            execution_model="google/gemma-test",
            rerank_model="google/rerank-test",
            dspy_model="openrouter/google/gemma-test",
        ),
    )

    assert captured["memory_tools_enabled"] is False
    assert captured["max_iters"] == 8
    assert captured["tool_names"] == ()
    assert payload["response_text"] == "```python\npass\n```"


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
    assert config.timeout_seconds == 180.0
    assert config.lm_model_name() == "openai/google/gemma-test"
    assert config.as_env()["VTM_DSPY_MODEL"] == "openrouter/google/gemma-test"


def test_openrouter_config_includes_optional_lm_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("VTM_OPENROUTER_BASE_URL", "https://openrouter.example/api/v1")

    config = DSPyOpenRouterConfig.from_env(
        execution_model_name="google/gemma-test",
        dspy_model_name="google/gemma-test",
        temperature=0.25,
        max_tokens=4096,
        extra_body={
            "provider": {
                "only": ["ionstream/fp8"],
                "allow_fallbacks": False,
            }
        },
    )

    assert config.lm_kwargs() == {
        "api_base": "https://openrouter.example/api/v1",
        "model_type": "chat",
        "api_key": "openrouter-test-key",
        "temperature": 0.25,
        "max_tokens": 4096,
        "timeout": 180.0,
        "extra_body": {
            "provider": {
                "only": ["ionstream/fp8"],
                "allow_fallbacks": False,
            }
        },
    }


def test_react_agent_create_lm_forwards_sampling_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeDSPy:
        @staticmethod
        def LM(model: str, **kwargs: object) -> object:
            calls.append((model, dict(kwargs)))
            return object()

    agent = VTMReActCodingAgent(
        kernel=None,
        scopes=(),
        model_config=DSPyOpenRouterConfig(
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
            execution_model="google/gemma-test",
            rerank_model="google/rerank-test",
            dspy_model="openrouter/google/gemma-test",
            temperature=0.25,
            max_tokens=4096,
            extra_body={
                "provider": {
                    "only": ["ionstream/fp8"],
                    "allow_fallbacks": False,
                }
            },
        ),
    )
    monkeypatch.setattr("vtm_dspy.react_agent.require_dspy", lambda: FakeDSPy)

    agent.create_lm()

    assert calls == [
        (
            "openai/google/gemma-test",
            {
                "api_base": "https://openrouter.example/api/v1",
                "model_type": "chat",
                "api_key": "openrouter-test-key",
                "temperature": 0.25,
                "max_tokens": 4096,
                "timeout": 180.0,
                "extra_body": {
                    "provider": {
                        "only": ["ionstream/fp8"],
                        "allow_fallbacks": False,
                    }
                },
            },
        )
    ]


def test_openrouter_config_allows_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("VTM_OPENROUTER_BASE_URL", "https://openrouter.example/api/v1")
    monkeypatch.setenv("VTM_DSPY_TIMEOUT_SECONDS", "75")

    config = DSPyOpenRouterConfig.from_env(
        execution_model_name="google/gemma-test",
        dspy_model_name="google/gemma-test",
    )

    assert config.timeout_seconds == 75.0
    assert config.lm_kwargs()["timeout"] == 75.0
    assert config.as_env()["VTM_DSPY_TIMEOUT_SECONDS"] == "75.0"


def test_docs_frame_dspy_and_livecodebench_correctly() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    final_scope = (REPO_ROOT / "docs" / "final-scope.md").read_text(encoding="utf-8")
    dspy_doc = (REPO_ROOT / "docs" / "dspy-integration.md").read_text(encoding="utf-8")
    recipes = (REPO_ROOT / "docs" / "benchmark-recipes.md").read_text(encoding="utf-8")

    assert "DSPy is the recommended forward-facing agent and programming interface" in readme
    assert (
        "Main benchmark layers: static retrieval, drift verification, drifted retrieval, "
        "controlled coding-drift"
    ) in readme
    assert "LiveCodeBench support is available for baseline model coding ability checks" in readme
    assert (
        "main VTM evidence remains retrieval, drift verification, drifted retrieval, "
        "and controlled coding-drift"
    ) in readme
    assert "DSPy is the recommended forward-facing agent interface for VTM memory" in final_scope
    assert "it remains optional" in final_scope
    assert (
        "controlled_coding_drift remains the small maintained agent-loop benchmark" in final_scope
    )
    assert (
        "DSPy is the recommended forward-facing agent and programming interface for VTM"
        in dspy_doc
    )
    assert "drifted retrieval, and controlled coding-drift" in dspy_doc
    assert "it remains optional" in recipes
    assert "uv run python scripts/run_dspy_vtm_smoke.py --workspace-root ." in recipes
