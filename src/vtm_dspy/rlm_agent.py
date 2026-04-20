"""Minimal DSPy RLM wrapper for optional VTM long-context coding experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.services.memory_kernel import MemoryKernel

from . import require_dspy
from .config import DSPyOpenRouterConfig
from .rlm_adapter import VTMRLMContextAdapter, make_vtm_rlm


class VTMRLMCodingAgent:
    """Optional DSPy RLM wrapper that keeps VTM as the verified-memory layer."""

    def __init__(
        self,
        *,
        kernel: MemoryKernel | None,
        scopes: Sequence[VisibilityScope] = (),
        dependency_provider: Callable[[], DependencyFingerprint | None] | None = None,
        memory_lookup: Callable[[str], MemoryItem | None] | None = None,
        model_config: DSPyOpenRouterConfig | None = None,
        max_context_cards: int = 5,
        max_iterations: int = 8,
        max_llm_calls: int = 16,
        verbose: bool = False,
    ) -> None:
        self.model_config = model_config or DSPyOpenRouterConfig.from_env()
        self.context_adapter = VTMRLMContextAdapter.from_kernel(
            kernel=kernel,
            scopes=scopes,
            dependency_provider=dependency_provider,
            memory_lookup=memory_lookup,
            model_config=self.model_config,
            max_cards=max_context_cards,
        )
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.verbose = verbose

    def tool_mapping(self) -> dict[str, Callable[..., Any]]:
        """Return the named tool callables exposed to the RLM interpreter."""
        return dict(self.context_adapter.memory_tools.tool_mapping())

    def tool_names(self) -> tuple[str, ...]:
        """Return the stable ordered tool names for dry-run inspection."""
        return tuple(self.tool_mapping().keys())

    def describe(self) -> dict[str, Any]:
        """Return dry-run metadata describing the configured RLM surface."""
        return {
            "tool_names": list(self.tool_names()),
            "model": self.model_config.summary(),
            "memory_tools_enabled": self.context_adapter.memory_tools.enabled,
            "max_context_cards": self.context_adapter.max_cards,
            "max_iterations": self.max_iterations,
            "max_llm_calls": self.max_llm_calls,
            "execution_mode": "rlm",
        }

    def create_lm(self) -> Any:
        """Instantiate the configured DSPy LM using OpenRouter-compatible settings."""
        dspy = require_dspy()
        lm_kwargs = dict(self.model_config.lm_kwargs())
        lm_kwargs.setdefault("api_key", self.model_config.require_api_key())
        return dspy.LM(self.model_config.lm_model_name(), **lm_kwargs)

    def create_program(self, *, signature: str = "task, context -> response") -> Any:
        """Construct a DSPy RLM module backed by the current VTM tool surface."""
        dspy = require_dspy()
        lm = self.create_lm()
        program = make_vtm_rlm(
            adapter=self.context_adapter,
            signature=signature,
            query="",
            tools=list(self.tool_mapping().values()),
            sub_lm=lm,
            max_iterations=self.max_iterations,
            max_llm_calls=self.max_llm_calls,
            verbose=self.verbose,
        )
        if hasattr(program, "set_lm"):
            program.set_lm(lm)
            return program
        if hasattr(dspy, "configure"):
            dspy.configure(lm=lm)
        return program

    def run(
        self,
        task: str,
        *,
        query: str | None = None,
        signature: str = "task, context -> response",
    ) -> dict[str, Any]:
        """Execute one DSPy RLM trajectory and capture the resulting response."""
        if not task.strip():
            raise ValueError("task must be non-empty")
        resolved_query = (query or task).strip()
        context = self.context_adapter.build_context(resolved_query)
        program = self.create_program(signature=signature)
        prediction = program(task=task, context=context)
        return {
            "response": self._serialize_prediction(prediction),
            "trajectory": {
                **self.describe(),
                "task": task,
                "query": resolved_query,
                "context_card_count": len(context["memory_cards"]),
            },
        }

    def _serialize_prediction(self, prediction: Any) -> Any:
        if hasattr(prediction, "model_dump"):
            return prediction.model_dump()
        if hasattr(prediction, "toDict"):
            return prediction.toDict()
        if hasattr(prediction, "__dict__"):
            return dict(prediction.__dict__)
        return prediction


__all__ = ["VTMRLMCodingAgent"]
