# VTM Types

This document defines the type-layer contract for VTM. The goal is to keep
verification, parser, and storage logic aligned around one shared model.

## Purpose of the Type Layer

The VTM type layer models four core concepts:
- Memory objects that VTM tracks over time.
- Artifact references that identify tool-produced or external objects.
- Code anchors that bind memories to source locations and hashes.
- Verification results that describe the latest verification outcome.

These types are intentionally small and serializable so they can be passed
between parser code, verifier code, and future persistence adapters.

## Core Concepts

Memory
- The top-level tracked entity in VTM.

Artifact
- A reference to an external object associated with a memory.

Code Anchor
- A location in source code with both structural and contextual hashes.

Verification Result
- A normalized result from running verification against one memory.

Verifier
- The execution mechanism used to produce verification results.

## Enumerations

### MemoryType

Defined in [vtm/types.py](vtm/types.py).

Values:
- ARTIFACT: Tracks an artifact-level object.
- CLAIM: Tracks a claim that can be proven or disproven by evidence.
- PROCEDURE: Tracks a process or method used to validate behavior.

### VerifyStatus

Defined in [vtm/types.py](vtm/types.py).

Values:
- UNVERIFIED: Memory exists but has not been evaluated yet.
- VALID: Current evidence supports the memory.
- INVALID: Current evidence does not support the memory.
- STALE: Previously verified but dependencies or context have changed.

Status lifecycle guidance:
- New memories should default to UNVERIFIED.
- Verification should transition to VALID or INVALID.
- Dependency or source drift may transition to STALE.

### AnchorKind

Defined in [vtm/types.py](vtm/types.py).

Values:
- ROOT_LINK: Anchor points to the root location for a tracked subject.
- SOURCE_LINK: Anchor points to a concrete source location.

### Verifier

Defined in [vtm/types.py](vtm/types.py).

Values:
- SCRIPT: Verification is executed via a script entrypoint.

Notes:
- This enum is intentionally minimal for now.
- Additional verifier strategies can be added as implementation expands.

## Data Records

### ArtifactRef

Defined in [vtm/types.py](vtm/types.py).

Role:
- Identifies an artifact and the environment/tooling used to produce it.

Fields:
- artifact_id (str): Stable artifact identifier within VTM.
- tool_name (str): Name of the producing or validating tool.
- tool_version (str | None): Optional tool version.
- env_hash (str | None): Optional environment fingerprint.
- dependencies (list[str]): IDs of upstream artifacts or anchors.

Invariants:
- artifact_id must be unique in the relevant scope.
- dependencies should contain normalized identifier strings.

### CodeAnchor

Defined in [vtm/types.py](vtm/types.py).

Role:
- Locates a code subject and tracks both semantic and contextual drift.

Fields:
- path (str): Path to the source file.
- hash_ast (str): Structural hash of anchored code.
- hash_ctx (str): Hash of surrounding context.
- start_line (int): First line of anchored span.
- end_line (int): Last line of anchored span.
- start_byte (int): First byte offset of anchored span.
- end_byte (int): Last byte offset of anchored span.
- symbol (str | None): Optional symbol name.
- kind (str | None): Optional parser-derived kind (function, class, etc).
- anchor_kind (AnchorKind): Root vs source anchor classification.
- language (str | None): Optional language identifier.

Invariants:
- start_line <= end_line.
- start_byte <= end_byte.
- hash_ast should change on semantic structure changes.
- hash_ctx may change when neighboring code changes.

### VerificationResult

Defined in [vtm/types.py](vtm/types.py).

Role:
- Captures the latest verification outcome for one memory.

Fields:
- memory_id (str): The memory that was verified.
- status (VerifyStatus): Verification status at evaluation time.
- evidence_results (list[dict[str, Any]]): Structured evidence output.

Invariants:
- memory_id must reference an existing memory.
- evidence_results entries should be serializable dictionaries.

### Memory

Defined in [vtm/types.py](vtm/types.py).

Role:
- Represents a tracked VTM unit that may accumulate evidence over time.

Fields:
- memory_id (str): Stable memory identifier.
- memory_type (MemoryType): Category of memory.
- evidence (list[dict[str, Any]]): Collected supporting or failing evidence.
- status (VerifyStatus): Current verification state.
- verifier_script (str | None): Optional script entrypoint for verification.

Invariants:
- memory_id should be unique in memory storage.
- status defaults to UNVERIFIED for newly created memories.

## Type Relationships

- Memory is the root tracked object.
- VerificationResult references Memory by memory_id.
- Memory evidence may include data linked to ArtifactRef and CodeAnchor.
- Verifier selects the execution mechanism used to produce result evidence.

## Open Design Decisions

- Define a stricter schema for evidence and evidence_results records.
- Decide whether verifier_script remains free-form or becomes typed metadata.
- Decide identifier prefix rules for mixed dependencies (artifact vs anchor IDs).
- Decide whether kind should move from free-form string to enum.
