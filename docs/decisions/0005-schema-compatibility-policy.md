# ADR 0005: Schema Compatibility Policy

## Status

Accepted

## Context

VTM now carries explicit schema versions and ordered migrations for metadata, cache, and
artifact stores. That is enough for local evolution, but it is not enough for operators or
maintainers unless the compatibility contract is explicit and release-backed.

## Decision

- Treat SQLite schema versions as forward-only upgrade contracts.
- Require every supported schema revision to have a checked-in fixture under
  `tests/fixtures/migrations/`.
- Require every new schema revision to add:
  - an ordered migration step
  - a fixture snapshot for the new revision
  - regression coverage proving upgrade from every supported older revision
  - future-version rejection coverage
- Do not promise downgrades for SQLite stores.
- Keep persisted record JSON backward-loadable when possible, but treat store schema versioning
  as the authoritative compatibility boundary.

## Consequences

- Releases must update migration fixtures as part of schema work.
- Operators can upgrade old supported stores in place, but not downgrade them.
- Unsupported future schema versions fail fast instead of being opened optimistically.
