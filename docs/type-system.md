# Type System

The type system is designed to make incorrect memory states hard to represent.

## Core enums

- `MemoryKind`: `claim`, `procedure`, `constraint`, `decision`, `summary_card`
- `ValidityStatus`: `pending`, `verified`, `relocated`, `unknown`, `stale`, `refuted`, `superseded`, `quarantined`
- `EvidenceKind`: `artifact`, `code_anchor`, `memory`
- `ArtifactCaptureState`: `prepared`, `committed`, `abandoned`
- `ScopeKind`: `call`, `tx`, `task`, `branch`, `repo`, `user`, `global`
- `TxState`, `DetailLevel`, `EvidenceBudget`, and `FreshnessMode` support execution, retrieval, consolidation, and cache policy.

## Fingerprints

- `RepoFingerprint`: repo-root and revision state for the repository view that produced a memory.
- `EnvFingerprint`: Python/runtime/tool version context.
- `DependencyFingerprint`: the combined state used by verification and cache invalidation.

## Evidence

- `ArtifactRef`: stable pointer to a stored artifact blob.
- `CodeAnchor`: semantic location in source code with parser-agnostic `symbol_digest` plus compatibility loading for legacy `ast_digest`.
- `EvidenceRef`: discriminated reference to an artifact, code anchor, or lower-level memory.

## Memory records

- Payloads are discriminated by `kind`: `ClaimPayload`, `ProcedurePayload`, `ConstraintPayload`, `DecisionPayload`, `SummaryCardPayload`.
- `MemoryItem` is the canonical persisted memory record and carries visibility, validity, evidence, lineage, and retrieval stats.
- `ValidatorSpec` is the durable procedure-validator envelope.
- `CommandValidatorConfig` is the typed config model behind `ValidatorSpec(kind="command")`.
- `VisibilityScope` is a single write scope. Retrieval can query across multiple scopes.
- `ValidityState` stores the current status and the dependency fingerprint it was checked against.
- `LineageEdge` captures provenance between transactions and committed memories.
- `ArtifactRecord` carries first-class capture lifecycle and provenance fields: state, capture group, actor, and session id.
- `ArtifactIntegrityReport` is a non-mutating diagnostic record for prepared captures, committed missing blobs, orphaned blob paths, and abandoned-capture summaries by reason/origin.
- `ArtifactRepairReport` records one repair pass over the store: audit before, audit after, abandoned prepared captures, removed orphaned blobs, and unresolved committed-missing-blob cases.

## Retrieval, embedding, and consolidation records

- `RetrieveRequest`, `RetrieveExplanation`, `RetrieveCandidate`, and `RetrieveResult` define the retrieval boundary.
- `EmbeddingIndexEntry` stores derived vector rows keyed by `memory_id` and `adapter_id`.
- `ConsolidationAction` records superseding and summary-card creation actions.
- `ConsolidationRunResult` records scanned counts, action counts, and emitted consolidation actions.

## RLM reranking records

- `RLMRankRequest` is the narrow auditable request sent to an RLM adapter.
- `RLMRankedCandidate` carries retrieval-facing candidate metadata plus lexical and RLM scores.
- `RLMRankResponse` stores the reranked candidate list together with optional model and usage metadata.

## Benchmark records

- `RepoSpec` and `CommitPair` describe the pinned corpus inputs.
- `RetrievalCase`, `DriftCase`, and `CodingTaskCase` are deterministic evaluation cases derived from those inputs.
- `RetrievalCase.slice_name` distinguishes harder task-oriented retrieval prompts from smoke identity lookups.
- `CodingTaskCase.evaluation_backend` distinguishes local subprocess scoring from official SWE-bench harness scoring, and SWE-bench cases can additionally carry `instance_id`, `dataset_name`, `problem_statement`, `hints_text`, `fail_to_pass_tests`, `pass_to_pass_tests`, and `gold_test_patch_digest`.
- `BenchmarkRunConfig` carries optional repo and commit-pair filters in addition to suite, mode, output settings, and optional SWE-bench harness configuration.
- `BenchmarkRunResult` defines the execution boundary and persisted summary for a benchmark run.

## Invariants encoded in validators

- `verified` and `relocated` memories require a dependency fingerprint.
- `verified` and `relocated` claim/procedure/constraint/decision memories require at least one evidence reference.
- `summary_card` items must point to raw artifacts or lower-level memories.
- Evidence references must match their declared evidence kind.
- RLM rank requests require unique candidate IDs.
- Git-backed benchmark repo specs require a `remote_url`; synthetic repo specs must not set one.
