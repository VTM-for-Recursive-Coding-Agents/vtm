"""Transcript compaction strategies for the native agent runtime."""

from __future__ import annotations

from typing import Protocol

from vtm.agents.models import AgentConversationMessage, CompactionRecord
from vtm.base import utc_now


class ContextCompactor(Protocol):
    """Compacts long conversations into a smaller message window."""

    def compact(
        self,
        *,
        messages: tuple[AgentConversationMessage, ...],
        turn_index: int,
        window: int,
    ) -> tuple[tuple[AgentConversationMessage, ...], CompactionRecord | None]: ...


class DeterministicContextCompactor:
    """Simple compactor that keeps the head and tail with a synthetic summary."""

    def compact(
        self,
        *,
        messages: tuple[AgentConversationMessage, ...],
        turn_index: int,
        window: int,
    ) -> tuple[tuple[AgentConversationMessage, ...], CompactionRecord | None]:
        """Deterministically compact messages when they exceed the window."""
        if len(messages) <= window:
            return messages, None

        head = list(messages[:2])
        tail = list(messages[-(window - len(head) - 1) :])
        dropped = messages[len(head) : len(messages) - len(tail)]
        summary_lines = [
            f"{message.role}:{message.tool_name or '-'}:{message.content[:120]}"
            for message in dropped
        ]
        summary = "\n".join(summary_lines).strip() or "compacted prior context"
        summary_message = AgentConversationMessage(
            role="assistant",
            content=f"[Compacted context]\n{summary}",
        )
        compacted = tuple(head + [summary_message] + tail)
        record = CompactionRecord(
            compaction_id=f"compact-{turn_index}",
            turn_index=turn_index,
            created_at=utc_now().isoformat(),
            trigger_message_count=len(messages),
            dropped_message_count=len(dropped),
            kept_message_count=len(compacted),
            summary=summary,
        )
        return compacted, record


__all__ = ["ContextCompactor", "DeterministicContextCompactor"]
