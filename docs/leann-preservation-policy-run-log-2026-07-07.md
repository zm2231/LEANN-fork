# LENS-2 LEANN Preservation Policy Run Log

Date: 2026-07-07
Lane: LENS
Slice: LENS-2 / R-F LEANN-fork preservation policy doc
Repo: `/Volumes/4/GitHub/LEANN-fork`

## Scope

- Added `docs/leann-preservation-policy.md`.
- Preservation set documented:
  - `flat` backend exact search with `--no-recompute`.
  - `build-jsonl --incremental-by-id` stable row updates.
  - MCP index targeting by name or explicit `index_path`.
- Live dependency counts recorded as operational context: `fable-sessions` 10,962 docs, `sessions-claude` 263,733 docs, `sessions-codex` 286,087 docs, `sessions-pi` 40,802 docs.
- Upstream merge rule documented: run the preservation check before landing upstream merges or retrieval refactors that touch flat backend, incremental JSONL, MCP targeting, metadata filtering, or build-config replay.

## Bare Rerun Command

```bash
uv run pytest \
  tests/test_incremental_jsonl_by_id.py::test_flat_incremental_by_id_uses_cache_and_updates_exact_vectors \
  tests/test_prefilter_integration.py::test_flat_auto_prefilter_scores_dense_filter_subset \
  tests/test_prefilter_integration.py::test_prefilter_always_forces_prefilter_on_dense_filter \
  packages/leann-core/tests/test_mcp_server.py::test_leann_search_uses_python_api_not_cli_subprocess \
  packages/leann-core/tests/test_mcp_server.py::test_leann_inspect_rejects_ambiguous_index_name \
  packages/leann-core/tests/test_mcp_server.py::test_leann_search_accepts_explicit_index_path_directory
```

Result: exit 0, `6 passed, 1 warning`.

## Boundaries

- No live index rebuilds were run.
- No embedding model changes.
- No personal-tier data was moved.
- No training.
