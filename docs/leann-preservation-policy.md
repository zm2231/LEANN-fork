# LEANN Preservation Policy

Status: active. Owner: retrieval infrastructure. Last updated: 2026-07-07.

This fork preserves three local capabilities that downstream indexes depend on:

1. `flat` backend exact search with `--no-recompute` (preserved from `1f23715`).
2. Stable JSONL updates with `build-jsonl --incremental-by-id` (preserved from `1f23715`).
3. MCP searches that target the intended index by name or explicit `index_path` (preserved from `ab07d0a`).

Do not regress these while merging upstream or refactoring retrieval internals. Four live indexes currently depend on this set: `fable-sessions` (10,962 docs), `sessions-claude` (263,733 docs), `sessions-codex` (286,087 docs), and `sessions-pi` (40,802 docs). Treat those counts as operational context, not test fixtures.

## Non-Regressions

- A `flat` index must keep exact vector search semantics and support metadata-prefiltered searches without ANN post-filter false-zero behavior.
- `build-jsonl --incremental-by-id` must add, update, remove, and no-op by stable row ID without full rebuilds when text is unchanged.
- Incremental updates must keep passages, offsets, BM25 state, stable IDs, metadata IDs, and stored vectors consistent.
- MCP tools must resolve ambiguous names safely, accept explicit physical `index_path`, and call the Python search API rather than shelling through the CLI.
- `rebuild` must replay stored build config, including backend, embedding mode/model/options, `--no-recompute`, and `--incremental-by-id`.

## Required Check Before Upstream Merges

Run this check before landing any upstream merge or retrieval refactor that touches `packages/leann-core/src/leann/api.py`, `packages/leann-core/src/leann/cli.py`, `packages/leann-core/src/leann/mcp.py`, `packages/leann-backend-flat/`, metadata filtering, or build-config replay:

```bash
uv run pytest \
  tests/test_incremental_jsonl_by_id.py::test_flat_incremental_by_id_uses_cache_and_updates_exact_vectors \
  tests/test_prefilter_integration.py::test_flat_auto_prefilter_scores_dense_filter_subset \
  tests/test_prefilter_integration.py::test_prefilter_always_forces_prefilter_on_dense_filter \
  packages/leann-core/tests/test_mcp_server.py::test_leann_search_uses_python_api_not_cli_subprocess \
  packages/leann-core/tests/test_mcp_server.py::test_leann_inspect_rejects_ambiguous_index_name \
  packages/leann-core/tests/test_mcp_server.py::test_leann_search_accepts_explicit_index_path_directory
```

Expected result: exit 0.

If this check fails, stop the merge and fix the fork behavior before continuing. Do not rebuild the four live indexes as a workaround unless there is a stated reason, a queue check, and a latency note for live-search impact.

## Optional Smoke

When a change touches CLI argument replay or installed package entry points, also run:

```bash
uv run pytest tests/test_cli_build_config.py tests/test_incremental_build.py -q
```

This is broader and slower, but it catches build-config drift that the narrow preservation check may not exercise.
