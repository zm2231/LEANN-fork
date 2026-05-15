# PROGRESS — feat/temporal-substrate

Append-only, chronological.

## 2026-05-15: Planning complete, branch ready

- Branch `feat/temporal-substrate` cut from `origin/main` (commit `4c3271a`).
- Dev venv at `.venv` (Python 3.12.12) with editable `leann-core`, prebuilt `leann-backend-hnsw==0.3.7`, `dateparser==1.4.0`, `pytest==9.0.3`, `ruff==0.15.13`.
- Submodules initialized: `astchunk-leann`, `cppzmq`, `faiss`, `libzmq`, `msgpack-c`, `DiskANN`, etc. Not required for Wave 1 Python work but available if a future atom needs C++ rebuild.
- Test corpus decision: hybrid local — `~/Documents/jay-abraham-eval/Jay-Abraham-Curated` (14 docx/pdf) + `git log --all` from this repo (2025-06-30 → 2026-05-15, ~11 months). Two `source_type`s sufficient for cross-source temporal eval without mini1 sync.
- Eval gold set seeded at `tests/eval/temporal_gold.jsonl` — 7 starter queries (5 git_commit + 2 document). Atom 4 of TEMPORAL-PLAN.md expands this to 15+ during the /loop session.

Plan docs:
- `docs/dev/SIGNALS.md` — canonical chunk metadata schema (locked for Wave 1).
- `docs/dev/TEMPORAL-PLAN.md` — Wave 1 atoms 1-9 with done tests and commit message shapes.

No code changes to LEANN itself yet — that's the /loop session's job.

### Baseline numbers

To be captured by atom 4's `scripts/eval_temporal.py --baseline` once it exists. Expected to be near-zero recall on temporal queries because current LEANN has no NL time parsing and the calendar reader is the only source emitting structured timestamps.
