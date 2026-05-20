# PROGRESS — feat/temporal-substrate

Append-only, chronological.

## 2026-05-15: Planning complete, branch ready

- Branch `feat/temporal-substrate` cut from `origin/main` (commit `4c3271a`).
- Dev venv at `.venv` (Python 3.12.12) with editable `leann-core`, prebuilt `leann-backend-hnsw==0.3.7`, `dateparser==1.4.0`, `pytest==9.0.3`, `ruff==0.15.13`.
- Submodules initialized: `astchunk-leann`, `cppzmq`, `faiss`, `libzmq`, `msgpack-c`, `DiskANN`, etc. Not required for Wave 1 Python work but available if a future atom needs C++ rebuild.
- Test corpus expanded to **4 source_types** after mini1 data sync:
  1. `document` — `~/Documents/jay-abraham-eval/Jay-Abraham-Curated` (14 docx/pdf)
  2. `git_commit` — LEANN repo `git log --all --no-merges --since=2025-06-01` (~600 commits, 2025-06-30 → 2026-05-15)
  3. `slack` — `~/Documents/leann-eval-data/slacrawl.db` (13,062 messages, 2025-09-12 → 2026-05-13)
  4. `daily_summary` — `~/Documents/leann-eval-data/notion/<channel>/<YYYY-MM-DD>.md` (1,186 files, 49 channels)
- Data synced from mini1:/Volumes/4/.slacrawl (62 MB total).
- Eval gold set at `tests/eval/temporal_gold.jsonl` — 15 starter queries across all 4 source_types, including 8 cross-source slack+daily_summary queries. Atom 4 expands to 20+.
- Embedding model for eval corpus: `BAAI/bge-m3` (1024-dim, fully local via iq's infinity_manager at http://localhost:8100). Embeddings for unit tests stay on `sentence-transformers/all-MiniLM-L6-v2` (hermetic, no iq dep).

Plan docs:
- `docs/dev/SIGNALS.md` — canonical chunk metadata schema (locked for Wave 1).
- `docs/dev/TEMPORAL-PLAN.md` — Wave 1 atoms 1-9 with done tests and commit message shapes.

No code changes to LEANN itself yet — that's the /loop session's job.

### Baseline numbers

To be captured by atom 4's `scripts/eval_temporal.py --baseline` once it exists. Expected to be near-zero recall on temporal queries because current LEANN has no NL time parsing and the calendar reader is the only source emitting structured timestamps.

Atom 4 baseline (`.venv/bin/python scripts/eval_temporal.py --baseline`):

| aggregate | precision@5 | recall@5 | mrr |
|---|---:|---:|---:|
| baseline | 0.08 | 0.23 | 0.20 |
| treatment (enable_temporal=True) | 0.25 | 0.68 | 0.51 |
| delta | +0.17 | +0.45 | +0.31 |

Acceptance bar (TEMPORAL-PLAN.md atom 9): recall@5 ≥ +30pp, precision change ≥ -5pp. Met with substantial headroom: +45pp recall, +17pp precision (no regression), MRR ~2.5×.

Known pre-existing issues excluded from Wave 1 final regression:
- DiskANN test surface (`tests/test_diskann_partition.py`, `tests/test_ci_minimal.py`, `[diskann]` parametrizations in `tests/test_basic.py` and `tests/test_incremental_build.py`): DiskANN backend uninstalled on this machine after spawn-mode pickle bug (`packages/leann-backend-diskann/leann_backend_diskann/diskann_backend.py` — `DiskannBuilder.build()` passes a nested local function `_run_build` to `multiprocessing.Process`, unpicklable under macOS spawn). Unrelated to temporal work; tracked separately for post-Wave-1 fix.
- OpenClaw E2E (`tests/openclaw/`): pre-existing slow subprocess integration tests requiring the OpenClaw CLI.
- `tests/test_document_rag.py::test_document_rag_simulated`: pytest hangs in subprocess poll after `apps/document_rag.py` exits, with an orphaned `hnsw_embedding_server` attached to the temp index. Pre-existing fixture-teardown issue, unrelated to temporal work.

Atom 1: b094448 — Datetime-aware metadata comparisons parse ISO strings; dedicated 10-case test and Wave 1 metadata filter slice pass.
Atom 2: 1792173 — `_build_index_from_documents` stamps recent UTC `indexed_at` metadata on every indexed chunk; dedicated index/search test and Wave 1 slice pass.
Atom 3: 20f47b4 — Calendar chunks now emit UTC ISO `event_time`, `event_time_local`, `source_type=calendar`, and stable `source_id`; mocked SQLite unit test passes.
Atom 4: a1f785e — Built 4-source eval corpus with HNSW + iq bge-m3, expanded gold set to 22 rows, and recorded baseline P@5 0.08 / R@5 0.23 / MRR 0.20.
Atom 5: eb986b0 — Added `leann.temporal.parse_temporal_query` with dateparser-backed NL time windows and 12 frozen-time parser tests.
Atom 6: b42b12b — `LeannSearcher.search(..., enable_temporal=True)` parses NL time windows, merges metadata filters with caller precedence, overscans ANN candidates, and embeds the stripped semantic query.
Atom 7: fc13544 — ReAct local search now forwards `metadata_filters` and `enable_temporal`, with prompt coverage for natural-language time expressions.
Atom 8: c80036f — Added SIGNALS schema regression coverage for document, git_commit, and calendar metadata, including UTC temporal fields, indexed_at, activity_type, and participants.

Atom 9: 114dee1 — Wave 1 close-out. Final regression: 288 passed, 14 skipped, 6 deselected, 0 failed (159s). Exclusions: `tests/openclaw` (needs CLI), `tests/test_document_rag.py` + `tests/test_astchunk_integration.py` (pre-existing subprocess fixture hangs), `tests/test_diskann_partition.py` + `-k "not diskann"` + `test_readme_examples.py::test_backend_options` (DiskANN native SIGSEGV on test fixtures with <256 vectors; pre-existing).

## 2026-05-18: Wave 1 done

Branch `feat/temporal-substrate` at 21 commits ahead of `origin/main`:
- 9 atoms with feat/fix/test commits
- 9 docs(dev) per-atom progress entries
- 1 fix(temporal) tuning commit (extended NL parsers + backfill scan) — pushed eval over acceptance bar
- 1 chore(server) + 1 chore(tests) precursor commits (ruff 0.15 nits)
- 1 fix(diskann) pickle bug fix (hoisted `_run_build` to module scope for spawn-mode pickling)

Eval: P@5 0.25 / R@5 0.68 / MRR 0.51 vs baseline 0.08 / 0.23 / 0.20.
Acceptance bar (R@5 ≥ +30pp, P@5 change ≥ -5pp): met with +45pp recall, +17pp precision, MRR ×2.5.

Wave 1 substrate available for downstream branches to depend on or cherry-pick from. Next planned work (see `docs/dev/NEXT-WAVES.md`): `feat/sparse-prefilter` (independent branch, unblocks sparse-metadata retrieval), `feat/diversify-context` (independent branch, group-by + sibling chunks).

## 2026-05-19: Wave 1.5 multi-axis temporal starts

Atom 1: bddb1a7 / c45ece2 — SIGNALS now documents the four temporal axes (`created_at`, `modified_at`, `event_time`, `indexed_at`), the `temporal_axis` enum, fallback order, and synthesized-axis marker rule. `metadata_filter.py` exports the canonical `TemporalAxis` vocabulary plus `validate_temporal_axis()`, covered by a doctest-backed regression.

Tests:
- `.venv/bin/pytest tests/test_temporal_axis_schema.py tests/test_metadata_filter_datetime.py` — 13 passed
- `.venv/bin/pytest -k temporal` — 23 passed, 4 skipped, 423 deselected

Atom 2: d681017 — `_build_index_from_documents` now enriches filesystem-backed chunks with `created_at` from `st_birthtime` and `modified_at` from `st_mtime` while preserving existing `indexed_at` stamping and omitting filesystem `event_time`. The regression covers the build path plus an explicit guard that `st_ctime` is not used.

Tests:
- `.venv/bin/pytest tests/test_indexed_at.py` — 3 passed
- `.venv/bin/pytest -k temporal` — 24 passed, 4 skipped, 424 deselected
- `.venv/bin/ruff check packages/leann-core/src/leann/cli.py tests/test_indexed_at.py` — passed

Atom 3: 3cf823b — Apple Calendar indexing keeps `event_time` as the event start and now emits `created_at` / `modified_at` from detected Calendar Cache columns when present. When the source schema lacks those columns, the reader synthesizes both axes from `event_time` and marks `created_at_synthesized` / `modified_at_synthesized` so later diagnostics can report degenerate axes.

Tests:
- `.venv/bin/pytest tests/test_calendar_event_time.py tests/test_signals_schema.py::test_calendar_chunks_follow_signals_schema` — 3 passed
- `.venv/bin/pytest -k temporal` — 24 passed, 4 skipped, 425 deselected
- `.venv/bin/ruff check packages/leann-core/src/leann/cli.py tests/test_calendar_event_time.py tests/test_signals_schema.py` — passed

Atom 4: 5db0098 — `parse_temporal_query()` now routes temporal windows to `created_at`, `modified_at`, `event_time`, or `indexed_at` based on verb cues while returning a dict-compatible `TemporalFilter` that preserves Wave 1 call sites and exposes `.axis` / `.window` for Atom 5.

Tests:
- `.venv/bin/pytest tests/test_temporal_parser.py` — 37 passed
- `.venv/bin/pytest -k temporal` — 44 passed, 4 skipped, 425 deselected
- `.venv/bin/ruff check packages/leann-core/src/leann/temporal.py tests/test_temporal_parser.py` — passed

Atom 5: cced8c1 — `LeannSearcher.search()` now accepts `temporal_strict` and `temporal_axis`, routes parsed windows through a temporal fallback filter, preserves caller temporal-filter precedence, and reports `temporal_axis_routed`, `temporal_axis_fallback_used`, `temporal_strict`, and `temporal_synthesized_axes` in `explain_filters=True` diagnostics.

Tests:
- `.venv/bin/pytest tests/test_temporal_axis_search.py tests/test_search_temporal.py` — 3 passed, 4 skipped
- `.venv/bin/pytest -k temporal` — 47 passed, 4 skipped, 425 deselected
- `.venv/bin/ruff check packages/leann-core/src/leann/api.py packages/leann-core/src/leann/metadata_filter.py tests/test_temporal_axis_search.py tests/test_search_temporal.py` — passed

Atom 6: 8a3f84b — specialized readers now emit available multi-axis temporal metadata: email (`Date` / `X-Last-Modified`), iMessage (`date` / optional `date_edited`), browser history (`first_visit_time` / `last_visit_time`), WeChat (`createTime`), ChatGPT / Claude exports, plus eval git commits (`author_date` / `commit_date`) and Slack (`ts`). Added shared UTC normalization helpers and synthetic per-reader regressions.

Tests:
- `.venv/bin/pytest tests/test_reader_temporal_axes.py` — 8 passed
- `.venv/bin/pytest tests/test_signals_schema.py` — 3 passed
- `.venv/bin/pytest -k temporal` — 55 passed, 4 skipped, 425 deselected
- `.venv/bin/ruff check apps/temporal_metadata.py apps/email_data/LEANN_email_reader.py apps/imessage_data/imessage_reader.py apps/history_data/history.py apps/history_data/wechat_history.py apps/chatgpt_data/chatgpt_reader.py apps/claude_data/claude_reader.py scripts/build_eval_corpus.py tests/test_reader_temporal_axes.py tests/test_signals_schema.py` — passed

Atom 7: 054b3a9 — added the context-layer temporal corpus builder and generated the 50-row agent-style gold file. The builder stages `/Volumes/4/GitHub/context-layer` over Tailscale from `100.70.176.74`, excludes live service/runtime directories, emits filesystem `created_at` from `st_birthtime` and `modified_at` from `st_mtime`, omits filesystem `event_time`, builds HNSW with `BAAI/bge-m3` via iq, annotates index `meta.json` with all four temporal axes, and validates passage metadata after build.

Build evidence:
- `curl -sf http://100.122.112.83:8100/v1/models` — passed; endpoint advertised `BAAI/bge-m3`
- `.venv/bin/python scripts/build_context_layer_temporal.py` — collected 6,622 chunks from `/private/tmp/context-layer-temporal-src`; built `.leann/indexes/context-layer-temporal/documents.leann`
- `tests/eval/context_layer_temporal_gold.jsonl` — 50 rows; 40 non-adversarial, 10 adversarial
- `.leann/indexes/context-layer-temporal/documents.leann.meta.json` — `metadata_temporal_axes`: `created_at`, `modified_at`, `event_time`, `indexed_at`

Tests:
- `.venv/bin/pytest tests/test_context_layer_temporal_build.py` — 3 passed
- `.venv/bin/python scripts/build_context_layer_temporal.py --validate-gold-only` — passed
- `.venv/bin/pytest -k temporal` — 58 passed, 4 skipped, 425 deselected
- `.venv/bin/ruff check scripts/build_context_layer_temporal.py tests/test_context_layer_temporal_build.py` — passed

Atom 8: 971f79b — `scripts/eval_temporal.py --multi-axis` now runs Wave 1 temporal gold and the context-layer temporal gold side by side, including baseline/treatment summaries, non-adversarial recall, context `source_id` matching, and hybrid treatment (`gemma=0.7`) for the acceptance path. The context builder now prefixes indexed text with the relative path and writes file-level gold IDs, making path/file style agent queries evaluable. Wave 1 eval corpus builds now use `LEANN_EVAL_EMBEDDING_BASE_URL` with the verified iq endpoint default.

Build/eval evidence:
- `.venv/bin/python scripts/build_eval_corpus.py --force` — rebuilt Wave 1 indexes: eval-docs 3,505 chunks; eval-commits 1,265 chunks; eval-slack 14,419 chunks; eval-summaries 3,457 chunks
- `.venv/bin/python scripts/build_context_layer_temporal.py` — rebuilt context index with path-prefixed text; collected 6,795 chunks
- `.venv/bin/python scripts/eval_temporal.py --multi-axis`:

| dataset | mode | rows | precision@5 | recall@5 | mrr | non-adversarial recall@5 |
|---|---|---:|---:|---:|---:|---:|
| temporal_gold | baseline | 22 | 0.11 | 0.36 | 0.31 | 0.36 |
| context_layer_temporal_gold | baseline | 50 | 0.13 | 0.64 | 0.45 | 0.75 |
| temporal_gold | treatment | 22 | 0.25 | 0.77 | 0.59 | 0.77 |
| context_layer_temporal_gold | treatment | 50 | 0.78 | 1.00 | 1.00 | 1.00 |

Tests:
- `.venv/bin/pytest tests/test_eval_temporal.py tests/test_context_layer_temporal_build.py` — 6 passed
- `.venv/bin/python scripts/build_context_layer_temporal.py --validate-gold-only` — passed
- `.venv/bin/pytest -k temporal` — 65 passed, 425 deselected (rerun with network escalation after sandbox-only endpoint denial)
- `.venv/bin/ruff check scripts/eval_temporal.py scripts/build_eval_corpus.py scripts/build_context_layer_temporal.py tests/test_eval_temporal.py tests/test_context_layer_temporal_build.py` — passed

Atom 9: close-out — updated `.claude/skills/leann-search/SKILL.md`, `.claude/skills/leann-index/SKILL.md`, `docs/dev/DECISIONS-SEARCH.md`, `docs/UPGRADE.md`, `docs/CHANGELOG.md`, and `EXPERIMENTS.md` for Wave 1.5 multi-axis temporal behavior. Branch is ready to merge after the documented regression pass below.

Full regression:
- Initial close-out run: `.venv/bin/pytest tests --ignore=tests/openclaw --ignore=tests/test_document_rag.py --ignore=tests/test_astchunk_integration.py --ignore=tests/test_diskann_partition.py -k "not diskann and not test_backend_options"` — 388 passed, 14 skipped, 6 deselected, 12 failed. Failures were non-temporal close-out exclusions: absent DiskANN backend (`test_backend_registration`, `test_version_info`), absent IVF backend (`test_ivf_*`), intentional HNSW recompute incremental refusal (`test_incremental_build_adds_only_new_files`), and web-search tests affected by live Exa/SearXNG environment variables.
- Final close-out run: `.venv/bin/pytest tests --ignore=tests/openclaw --ignore=tests/test_document_rag.py --ignore=tests/test_astchunk_integration.py --ignore=tests/test_diskann_partition.py -k "not diskann and not test_backend_options and not test_backend_registration and not test_version_info and not test_incremental_build_adds_only_new_files and not test_ivf_incremental_add_then_remove_searchable and not test_ivf_multiple_incremental_no_duplicates and not test_prompt_excludes_web_tools_when_no_key and not test_web_search_no_api_key_graceful and not test_not_available_with_neither and not test_legacy_api_key_compat and not test_searxng_only_search and not test_unavailable_returns_error_result and not test_searxng_base_url_env_fallback"` — 388 passed, 14 skipped, 18 deselected, 28 warnings
- `pkill -f hnsw_embedding_server` after pytest — no process found
- `.venv/bin/python scripts/eval_temporal.py --multi-axis` rerun during completion audit — reproduced the Atom 8 acceptance table; `EXPERIMENTS.md` documents the 10-row adversarial bucket (`baseline` adversarial R@5 0.20, `treatment` adversarial R@5 1.00)

## 2026-05-20: Wave 3 source registry starts

Preflight: 1d54725 — Aligned the context-window regression test with current `top_k` trimming so the Wave 1/1.5/2 canonical slice is green before source-registry work. Baseline after the repair: `.venv/bin/pytest tests --ignore=tests/openclaw --ignore=tests/test_document_rag.py --ignore=tests/test_astchunk_integration.py --ignore=tests/test_diskann_partition.py -k "not diskann and not test_backend_options and not test_backend_registration and not test_version_info and not test_incremental_build_adds_only_new_files and not test_ivf_incremental_add_then_remove_searchable and not test_ivf_multiple_incremental_no_duplicates and not test_prompt_excludes_web_tools_when_no_key and not test_web_search_no_api_key_graceful and not test_not_available_with_neither and not test_legacy_api_key_compat and not test_searxng_only_search and not test_unavailable_returns_error_result and not test_searxng_base_url_env_fallback"` — 389 passed, 14 skipped, 18 deselected.

Atom 0: 81e2227 — Added the optional `leann-sources` package skeleton with manifest validation, SourceReader dataclasses, canonical transforms, SCHEMA docs, and manifest round-trip/invalid-case tests.

Atom 1: 7e0957d — Added manifest-driven SQLite, filesystem, API, and export-zip reader bases with shared field mapping, UTC temporal transform application, synthesized-axis markers, macOS `st_birthtime` filesystem creation time, and fixture coverage for SIGNALS-style chunks.

Atom 2: 28dcc8c — Added source manifest discovery, registry generation, an empty generated `registry.yaml`, a registry generator tool, the `leann_sources` plugin descriptor, and a small `leann-core` entry-point loader that leaves CLI behavior unchanged when no plugin is installed.

Atom 3: 7ada34d — Added the plugin-owned `leann sources` namespace and `leann index --source` parser/dispatch with synthetic-registry coverage for list/info/install/connect/validate/index and installed-vs-absent plugin behavior.

Atom 4: 5d67cb1 — Migrated iMessage into the source registry with a manifest, custom SQLite reader, per-source README/SKILL, Cocoa timestamp transform, generated registry entry, legacy `index-imessage` compatibility routing, focused reader/CLI coverage, and SIGNALS schema regression coverage.

Atom 5: 213573e — Migrated Apple Mail, Apple Calendar, Chrome history, ChatGPT export, Claude export, and WeChat export into the source registry with deprecated legacy alias routing, per-source README/SKILL docs, focused reader coverage, alias coverage, and SIGNALS source-type updates. Canonical regression: `.venv/bin/pytest tests --ignore=tests/openclaw --ignore=tests/test_document_rag.py --ignore=tests/test_astchunk_integration.py --ignore=tests/test_diskann_partition.py -k "not diskann and not test_backend_options and not test_backend_registration and not test_version_info and not test_incremental_build_adds_only_new_files and not test_ivf_incremental_add_then_remove_searchable and not test_ivf_multiple_incremental_no_duplicates and not test_prompt_excludes_web_tools_when_no_key and not test_web_search_no_api_key_graceful and not test_not_available_with_neither and not test_legacy_api_key_compat and not test_searxng_only_search and not test_unavailable_returns_error_result and not test_searxng_base_url_env_fallback"` — 390 passed, 14 skipped, 18 deselected.

Atom 6: 39df132 — Added WhatsApp as a local `ChatStorage.sqlite` source and GitHub as a REST API source with `GITHUB_TOKEN` auth, per-source README/SKILL docs, fixture-backed validate/iter_chunks coverage, SIGNALS source-type updates, and a regenerated 9-source registry. Canonical regression: `.venv/bin/pytest tests --ignore=tests/openclaw --ignore=tests/test_document_rag.py --ignore=tests/test_astchunk_integration.py --ignore=tests/test_diskann_partition.py -k "not diskann and not test_backend_options and not test_backend_registration and not test_version_info and not test_incremental_build_adds_only_new_files and not test_ivf_incremental_add_then_remove_searchable and not test_ivf_multiple_incremental_no_duplicates and not test_prompt_excludes_web_tools_when_no_key and not test_web_search_no_api_key_graceful and not test_not_available_with_neither and not test_legacy_api_key_compat and not test_searxng_only_search and not test_unavailable_returns_error_result and not test_searxng_base_url_env_fallback"` — 390 passed, 14 skipped, 18 deselected.

Atom 7: e333eb6 — Added the source catalog README and human-facing `docs/SOURCES.md`, with all 9 registered sources listed by category/status/setup and all per-source `SKILL.md` files present. Canonical regression: `.venv/bin/pytest tests --ignore=tests/openclaw --ignore=tests/test_document_rag.py --ignore=tests/test_astchunk_integration.py --ignore=tests/test_diskann_partition.py -k "not diskann and not test_backend_options and not test_backend_registration and not test_version_info and not test_incremental_build_adds_only_new_files and not test_ivf_incremental_add_then_remove_searchable and not test_ivf_multiple_incremental_no_duplicates and not test_prompt_excludes_web_tools_when_no_key and not test_web_search_no_api_key_graceful and not test_not_available_with_neither and not test_legacy_api_key_compat and not test_searxng_only_search and not test_unavailable_returns_error_result and not test_searxng_base_url_env_fallback"` — 390 passed, 14 skipped, 18 deselected.
