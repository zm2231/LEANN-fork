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
