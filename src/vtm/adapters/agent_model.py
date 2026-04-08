"""Provider-neutral model-turn contract for the native agent runtime."""

from __future__ import annotations

from typing import Protocol

from vtm.agents.models import AgentModelTurnRequest, AgentModelTurnResponse


class AgentModelAdapter(Protocol):
    """Minimal interface required to drive `TerminalCodingAgent`."""

    @property
    def model_id(self) -> str: ...

    def complete_turn(self, request: AgentModelTurnRequest) -> AgentModelTurnResponse: ...


__all__ = ["AgentModelAdapter"]
