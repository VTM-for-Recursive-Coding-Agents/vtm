# tests/fixtures/migrations/artifact

Purpose: checked-in artifact-store schema fixtures used to test upgrade behavior across supported revisions.

Contents
- `v1.sql`: Artifact store revision 1 snapshot with the original artifact table layout.
- `v2.sql`: Artifact store revision 2 snapshot with capture-state, capture-group, actor, and session metadata columns.
