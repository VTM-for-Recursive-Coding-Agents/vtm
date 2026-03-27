from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    ARTIFACT = "ARTIFACT"
    CLAIM = "CLAIM"
    PROCEDURE = "PROCEDURE"


class VerifyStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VALID = "VALID"
    INVALID = "INVALID"
    STALE = "STALE"


class Verifier(str, Enum):
    SCRIPT = "SCRIPT"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str  # the identifier of the artifact
    tool_name: str  # the name of the tool to be used by the agent
    tool_version: str | None = None  # the version of the tool used by the agent
    env_hash: str | None = (
        None  # the ev_hash is used to document changes to the artifact
    )
    dependencies: list[str] | None = (
        None  # the list of dependencies (artifacts, code anchors, etc.) required for the artifact to run
    )


@dataclass(slots=True)
class CodeAnchor:
    path: str  # the path to the code file
    symbol: str | None  # Discription of the code
    kind: str | None  # function, class, etc pulled from like treat sitter
    # verifier: Verifier | None #decide whether code verifier should be stored in with the anchor or the verifier should take in a code anchor to verify
    language: str | None  # what coding language idk if will be important long term
    hash_ast: str  # hash of AST parser tree
    hash_ctx: str  # a hash of the surrounding context of the anchor could be usfeul in relcation stuff
    start_line: int  # the line number where the anchor starts
    end_line: int  # the line number where the anchor ends
    start_byte: int  # the byte offset where the anchor starts
    end_byte: int  # the byte offset where the anchor ends


@dataclass(slots=True)
class VerifcationResult:
    memory_id: str  # the unique identifier for the results of a verified artifact
    status: VerifyStatus  # the status of the verification results.
    evidence_results: list[dict[str, Any]] = field(default_factory=list)  # the


@dataclass(slots=True)
class Memory:
    memory_id: str  # the unique identifier for the memory
    memory_type: MemoryType
    evidence: list[dict[str, Any]] = field(
        default_factory=list
    )  # the evidence for the memory
    status: VerifyStatus = VerifyStatus.UNVERIFIED
    verifier_script: str | None = None
