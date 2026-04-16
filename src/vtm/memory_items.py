"""Durable memory-item payloads and supporting state records."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from vtm.base import VTMModel, utc_now
from vtm.enums import (
    ClaimStrength,
    DetailLevel,
    EvidenceKind,
    MemoryKind,
    ScopeKind,
    ValidityStatus,
)
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.ids import new_memory_id


class ProcedureStep(VTMModel):
    """One ordered instruction inside a procedure memory."""

    order: int = Field(ge=0)
    instruction: str
    expected_outcome: str | None = None


class CommandValidatorConfig(VTMModel):
    """Typed configuration for `ValidatorSpec(kind="command")`."""

    command: tuple[str, ...]
    cwd: str | None = None
    env: dict[str, str] | None = None
    env_allowlist: tuple[str, ...] | None = None
    env_denylist: tuple[str, ...] | None = None
    expected_exit_code: int = 0
    timeout_seconds: float | None = None
    max_output_bytes: int | None = None
    inherit_parent_env: bool = True
    restrict_cwd_to_repo: bool = False
    rlimit_cpu_seconds: int | None = None
    rlimit_memory_bytes: int | None = None
    rlimit_process_count: int | None = None
    rlimit_file_size_bytes: int | None = None

    @field_validator("command", mode="before")
    @classmethod
    def validate_command(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple) or not value or not all(
            isinstance(part, str) and part for part in value
        ):
            raise ValueError("command validator config requires a non-empty command list[str]")
        return tuple(value)

    @field_validator("cwd", mode="before")
    @classmethod
    def validate_cwd(cls, value: object) -> object:
        if value is not None and not isinstance(value, str):
            raise ValueError("command validator cwd must be a string")
        return value

    @field_validator("env", mode="before")
    @classmethod
    def validate_env(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) and isinstance(entry, str) for key, entry in value.items()
        ):
            raise ValueError("command validator env must be a dict[str, str]")
        return dict(value)

    @field_validator("env_allowlist", "env_denylist", mode="before")
    @classmethod
    def validate_env_name_list(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            return value
        if not isinstance(value, list | tuple) or not all(
            isinstance(entry, str) and entry for entry in value
        ):
            raise ValueError(f"command validator {info.field_name} must be a list[str]")
        return tuple(value)

    @field_validator("inherit_parent_env", "restrict_cwd_to_repo", mode="before")
    @classmethod
    def validate_bool_fields(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, bool):
            raise ValueError(f"command validator {info.field_name} must be a bool")
        return value

    @field_validator("expected_exit_code", mode="before")
    @classmethod
    def validate_expected_exit_code(cls, value: object) -> object:
        if not isinstance(value, int):
            raise ValueError("command validator expected_exit_code must be an int")
        return value

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_seconds(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, int | float) or value <= 0:
            raise ValueError("command validator timeout_seconds must be a positive number")
        return float(value)

    @field_validator("max_output_bytes", mode="before")
    @classmethod
    def validate_max_output_bytes(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, int) or value <= 0:
            raise ValueError("command validator max_output_bytes must be a positive int")
        return value

    @field_validator(
        "rlimit_cpu_seconds",
        "rlimit_memory_bytes",
        "rlimit_process_count",
        "rlimit_file_size_bytes",
        mode="before",
    )
    @classmethod
    def validate_resource_limit(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            return value
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"command validator {info.field_name} must be a positive int")
        return value

    def resource_limits(self) -> dict[str, int]:
        """Return the configured POSIX resource limits as a compact mapping."""
        limits: dict[str, int] = {}
        for field_name in (
            "rlimit_cpu_seconds",
            "rlimit_memory_bytes",
            "rlimit_process_count",
            "rlimit_file_size_bytes",
        ):
            raw_value = getattr(self, field_name)
            if raw_value is not None:
                limits[field_name] = raw_value
        return limits


class ValidatorSpec(VTMModel):
    """Validator configuration attached to a procedure payload."""

    name: str
    kind: str
    config: dict[str, Any] | CommandValidatorConfig = Field(default_factory=dict)

    def command_config(self) -> CommandValidatorConfig:
        """Return the typed command-validator config for this validator."""
        if self.kind != "command":
            raise ValueError(f"unsupported validator kind: {self.kind}")
        if isinstance(self.config, CommandValidatorConfig):
            return self.config
        return CommandValidatorConfig.model_validate(self.config)


class ClaimPayload(VTMModel):
    """Payload for a claim memory item."""

    kind: Literal["claim"] = "claim"
    claim: str
    strength: ClaimStrength = ClaimStrength.SUPPORTED


class ProcedurePayload(VTMModel):
    """Payload for an executable or reviewable procedure."""

    kind: Literal["procedure"] = "procedure"
    goal: str
    steps: tuple[ProcedureStep, ...] = Field(default_factory=tuple)
    validator: ValidatorSpec | None = None


class ConstraintPayload(VTMModel):
    """Payload describing a durable constraint or policy."""

    kind: Literal["constraint"] = "constraint"
    statement: str
    severity: str = "info"


class DecisionPayload(VTMModel):
    """Payload describing a recorded decision and rationale."""

    kind: Literal["decision"] = "decision"
    summary: str
    rationale: str | None = None
    supersedes: tuple[str, ...] = Field(default_factory=tuple)


class SummaryCardPayload(VTMModel):
    """Payload for a synthetic summary created from lower-level memory."""

    kind: Literal["summary_card"] = "summary_card"
    summary: str
    detail_level: DetailLevel = DetailLevel.SUMMARY
    supporting_memory_ids: tuple[str, ...] = Field(default_factory=tuple)


MemoryPayload = Annotated[
    ClaimPayload | ProcedurePayload | ConstraintPayload | DecisionPayload | SummaryCardPayload,
    Field(discriminator="kind"),
]


class VisibilityScope(VTMModel):
    """Namespace that determines where a memory item is visible."""

    kind: ScopeKind
    scope_id: str


class ValidityState(VTMModel):
    """Verification state attached to a memory item."""

    status: ValidityStatus = ValidityStatus.PENDING
    dependency_fingerprint: DependencyFingerprint | None = None
    checked_at: datetime | None = None
    reason: str | None = None


class LineageEdge(VTMModel):
    """Directed relationship between two memory items."""

    parent_id: str
    child_id: str
    edge_type: str
    tx_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MemoryStats(VTMModel):
    """Derived usage counters maintained by retrieval and verification flows."""

    retrieval_count: int = Field(default=0, ge=0)
    verification_count: int = Field(default=0, ge=0)
    last_retrieved_at: datetime | None = None
    last_verified_at: datetime | None = None


RETRIEVABLE_KINDS = {
    MemoryKind.CLAIM,
    MemoryKind.PROCEDURE,
    MemoryKind.CONSTRAINT,
    MemoryKind.DECISION,
}
VERIFIED_STATUSES = {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}


class MemoryItem(VTMModel):
    """Canonical durable memory record stored by the kernel."""

    memory_id: str = Field(default_factory=new_memory_id)
    kind: MemoryKind
    title: str
    summary: str
    payload: MemoryPayload
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    tags: tuple[str, ...] = Field(default_factory=tuple)
    visibility: VisibilityScope
    validity: ValidityState = Field(default_factory=ValidityState)
    lineage: tuple[LineageEdge, ...] = Field(default_factory=tuple)
    stats: MemoryStats = Field(default_factory=MemoryStats)
    tx_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_invariants(self) -> MemoryItem:
        """Enforce cross-field invariants for durable memory records."""
        if self.kind.value != self.payload.kind:
            raise ValueError("memory kind must match payload kind")

        if (
            self.validity.status in VERIFIED_STATUSES
            and self.validity.dependency_fingerprint is None
        ):
            raise ValueError("verified and relocated memories require a dependency fingerprint")

        if (
            self.kind in RETRIEVABLE_KINDS
            and self.validity.status in VERIFIED_STATUSES
            and not self.evidence
        ):
            raise ValueError("verified or relocated claim-like memories require evidence")

        if self.kind is MemoryKind.SUMMARY_CARD:
            if not any(
                evidence.kind in {EvidenceKind.ARTIFACT, EvidenceKind.MEMORY}
                for evidence in self.evidence
            ):
                raise ValueError(
                    "summary cards must reference raw artifacts or lower-level memories"
                )

        return self
