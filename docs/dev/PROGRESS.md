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

Known pre-existing issues excluded from Wave 1 final regression: DiskANN spawn-mode pickle bug on macOS (`packages/leann-backend-diskann`, unrelated to temporal work), and OpenClaw local model/subprocess tests on this machine.

Atom 1: b094448 — Datetime-aware metadata comparisons parse ISO strings; dedicated 10-case test and Wave 1 metadata filter slice pass.
Atom 2: 1792173 — `_build_index_from_documents` stamps recent UTC `indexed_at` metadata on every indexed chunk; dedicated index/search test and Wave 1 slice pass.
Atom 3: 20f47b4 — Calendar chunks now emit UTC ISO `event_time`, `event_time_local`, `source_type=calendar`, and stable `source_id`; mocked SQLite unit test passes.
Atom 4: a1f785e — Built 4-source eval corpus with HNSW + iq bge-m3, expanded gold set to 22 rows, and recorded baseline P@5 0.08 / R@5 0.23 / MRR 0.20.
Atom 5: eb986b0 — Added `leann.temporal.parse_temporal_query` with dateparser-backed NL time windows and 12 frozen-time parser tests.
Atom 6: b42b12b — `LeannSearcher.search(..., enable_temporal=True)` parses NL time windows, merges metadata filters with caller precedence, overscans ANN candidates, and embeds the stripped semantic query.
