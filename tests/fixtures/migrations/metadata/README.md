# tests/fixtures/migrations/metadata

Purpose: checked-in metadata-store schema fixtures used to test event and transaction schema upgrades.

Contents
- `v1.sql`: Metadata store revision 1 snapshot with memory, lineage, transaction, staged-memory, and event tables.
- `v2.sql`: Metadata store revision 2 snapshot that adds `event_export_state` and exported-event cursor data.
