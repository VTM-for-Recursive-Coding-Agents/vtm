"""Transaction records for staging and committing memory changes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from vtm.base import VTMModel, utc_now
from vtm.enums import TxState
from vtm.ids import new_transaction_id
from vtm.memory_items import VisibilityScope


class TransactionRecord(VTMModel):
    """Durable record of a memory transaction lifecycle."""

    tx_id: str = Field(default_factory=new_transaction_id)
    parent_tx_id: str | None = None
    state: TxState = TxState.ACTIVE
    visibility: VisibilityScope
    opened_at: datetime = Field(default_factory=utc_now)
    committed_at: datetime | None = None
    rolled_back_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_state(self) -> TransactionRecord:
        """Require the appropriate terminal timestamp for each terminal state."""
        if self.state is TxState.COMMITTED and self.committed_at is None:
            raise ValueError("committed transactions require committed_at")
        if self.state is TxState.ROLLED_BACK and self.rolled_back_at is None:
            raise ValueError("rolled back transactions require rolled_back_at")
        return self
