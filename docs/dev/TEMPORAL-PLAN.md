# TEMPORAL-PLAN — Wave 1

**Status:** active. Branch `feat/temporal-substrate` off `origin/main`.

**Goal:** end-to-end temporal awareness in LEANN — natural-language time queries return time-filtered, time-ranked results across heterogeneous sources — with the metadata schema (see `SIGNALS.md`) already shaped for non-temporal connectors to plug into later.

**Driver philosophy:** atoms are sized for a single commit. Each has a binary done-or-not test. The /loop's job is to drive atoms 1→9 in order, committing after each, and stopping cleanly when the eval shows the target gain.

**Out of scope for Wave 1:** other readers (email, iMessage, browser, WeChat, Qwen, Gemini), federated cross-index search, decay scoring, conversation history, MCP server passthrough beyond `metadata_filters`. Each is its own future wave.

---

## Glossary

- `MetadataFilterEngine` — `packages/leann-core/src/leann/metadata_filter.py`. Applies filters POST-search. Currently has no datetime semantics (`_numeric_compare` at line 209 casts to float).
- `LeannSearcher.search` — `packages/leann-core/src/leann/api.py:1155`. Entry point; takes `metadata_filters` kwarg; applies them at line 1372.
- `_build_index_from_documents` — `packages/leann-core/src/leann/cli.py:2885`. Shared ingest for all `index-*` CLI commands; the right place to stamp `indexed_at`.
- `ReActAgent.search` — `packages/leann-core/src/leann/react_agent.py:176`. Wraps `LeannSearcher.search`; currently drops `metadata_filters`.
- `MCP server` — `packages/leann-core/src/leann/mcp.py:102`. Shells out to the `leann search` CLI; currently doesn't pass `--metadata-filters`.
- **Temporal overscan** — when a temporal filter is present, fetch `top_k * overscan` ANN candidates BEFORE applying the date filter, so the filter doesn't drain the result set to zero.
- **Eval harness** — `tests/eval/temporal_gold.jsonl` (gold) + `scripts/eval_temporal.py` (runner). Acceptance bar lives here.

---

## Test corpus (already decided)

Two source_types, both local, built once at the start of Wave 1:

1. **Documents** — `/Users/zain/Documents/jay-abraham-eval/Jay-Abraham-Curated` (14 docx/pdf). Index name: `eval-docs`. `source_type=document`, `event_time` from filesystem `creation_date`/`last_modified_date`.
2. **Commits** — git log of the LEANN repo itself (`git log --all` over the last 3 years). Index name: `eval-commits`. `source_type=git_commit`, `event_time` from committer date (UTC), `author` from `git log --pretty`, `mentioned_refs` from `#NNN` regex.

A new script `scripts/build_eval_corpus.py` builds both indexes. Re-runnable; idempotent on existing indexes via `--force` flag.

---

## Atoms

Each atom: **what** + **where** + **done test** + **commit message shape**.

### Atom 1 — Datetime-aware MetadataFilterEngine

**What:** make `<`, `<=`, `>`, `>=` operators accept ISO 8601 strings. When both operands parse as ISO datetimes, compare as `datetime` objects (timezone-aware, naive treated as UTC with a logged warning). Fall back to existing float comparison if not parseable.

**Where:**
- `packages/leann-core/src/leann/metadata_filter.py` — add `_try_parse_datetime(value) -> datetime | None`. Modify `_numeric_compare` to attempt datetime parse first, fall through to float.
- Keep behavior backward-compatible for numeric/string fields.

**Done test:** `tests/test_metadata_filter_datetime.py` (NEW) — 10 cases:
1. ISO with `T`, UTC offset, range filter matches inside
2. Same, range filter excludes outside
3. ISO with `Z` suffix (Zulu)
4. ISO with negative offset (`-04:00`)
5. Date-only string (`2026-05-15`)
6. Mixed: filter is datetime, field is float → falls back to numeric path
7. Mixed: filter is datetime, field is unparseable string → returns False (no exception)
8. Equality (`==`) on datetime strings (existing string compare path, regression check)
9. Two naive datetimes: compared as UTC (warning logged once per process)
10. SQLite calendar format (`2026-05-15 14:30:00` with space) parses correctly

All pass: `.venv/bin/pytest tests/test_metadata_filter_datetime.py -v`.

**Commit:** `feat(filter): datetime-aware comparison operators`

---

### Atom 2 — `indexed_at` stamping

**What:** `_build_index_from_documents` writes `indexed_at` (UTC ISO) into every chunk's metadata at ingest. Once per `add_text` call.

**Where:** `packages/leann-core/src/leann/cli.py:2885` (`_build_index_from_documents`). Compute `indexed_at = datetime.now(timezone.utc).isoformat()` once at function entry; merge into each `doc.metadata` before `builder.add_text(...)`.

**Done test:** `tests/test_indexed_at.py` (NEW) — build a 3-doc test index via the function, search, assert every result has `indexed_at` matching ISO regex and within last 60 seconds of test start.

**Commit:** `feat(cli): stamp indexed_at on every chunk during index-*`

---

### Atom 3 — Calendar reader timestamp fix

**What:** Calendar reader emits `event_time` (UTC ISO) instead of `start` (SQLite localtime string with space). Add `event_time_local` for display.

**Where:** `packages/leann-core/src/leann/cli.py:2965` (`index_calendar`). Replace `datetime(start_date + 978307200, 'unixepoch', 'localtime')` with two SELECTs: one for `datetime(..., 'unixepoch')` (UTC, the canonical one) → `event_time`; one for `localtime` → `event_time_local`. Also emit `source_type=calendar`, `source_id=event_id`, `event_time` in metadata.

**Done test:** unit test that mocks the SQLite cursor with one event, asserts metadata has `event_time` parseable by `datetime.fromisoformat()` AND `source_type=="calendar"`.

**Commit:** `fix(cli): calendar event_time as UTC ISO + source_type tag`

---

### Atom 4 — Eval harness scaffolding

**What:** `scripts/build_eval_corpus.py` + `tests/eval/temporal_gold.jsonl` + `scripts/eval_temporal.py`.

**Where:**
- `scripts/build_eval_corpus.py` — builds `eval-docs` index from jay-abraham-eval and `eval-commits` index from `git log --all --pretty=format:'%H|%aI|%an|%s%n%b' --no-merges` of the LEANN repo (last 3 years). Stamps full SIGNALS.md schema on each chunk.
- `tests/eval/temporal_gold.jsonl` — 15 hand-written queries. Each row: `{query, expected_event_time_range: [iso, iso], expected_source_types: [...], expected_min_results: int}`.
- `scripts/eval_temporal.py` — runs each gold query against both indexes (baseline = no temporal handling, treatment = with temporal pre-processing once atoms 5-6 land). Prints precision@5, recall@5, mean reciprocal rank.

**Done test:** `python scripts/build_eval_corpus.py` produces two indexes; `python scripts/eval_temporal.py --baseline` prints a table. Both indexes searchable. Baseline numbers recorded in `docs/dev/PROGRESS.md`.

**Commit:** `test(eval): temporal eval corpus + gold set + harness scaffold`

---

### Atom 5 — `leann.temporal` NL parser module

**What:** new module `packages/leann-core/src/leann/temporal.py`. Function `parse_temporal_query(query: str, now: datetime | None = None) -> tuple[str, dict | None]` returns `(query_with_time_stripped, metadata_filters_or_None)`.

Uses `dateparser` for absolute/relative parsing. Recognizes:
- "N {hours,days,weeks,months,years} ago"
- "last/this {week,month,year,Monday,…}"
- "in January", "on March 5th", "since 2026"
- "between X and Y"
- "yesterday", "today", "tomorrow"

Returns filters as `{"event_time": {">=": iso_start, "<=": iso_end}}`. If no time expression, returns `(query, None)`.

**Where:** new file. Add `dateparser>=1.2` to `packages/leann-core/pyproject.toml` as required dep (it's small and pure Python).

**Done test:** `tests/test_temporal_parser.py` (NEW) — 12 cases with frozen `now`:
1-3. Relative ago expressions
4-6. "last Tuesday" / "last week" / "last month"
7-8. Absolute dates
9. "between" range
10. No time → returns original query, None
11. "yesterday" → 24h window
12. Stripped query has time tokens removed but semantic content intact ("what was I working on last Tuesday" → "what was I working on")

**Commit:** `feat(temporal): NL time-expression parser`

---

### Atom 6 — `LeannSearcher.search` temporal integration

**What:** `LeannSearcher.search` gains opt-in `enable_temporal=False` kwarg (off by default — no behavior change for existing callers). When `True`:
1. Pre-process `query` via `parse_temporal_query`
2. If a temporal filter was found, merge into `metadata_filters` (caller-supplied filters take precedence on key conflict)
3. Multiply `top_k` by `temporal_overscan` (default 10) for the ANN call, then let post-search filter narrow
4. Use the stripped query for embedding

**Where:** `packages/leann-core/src/leann/api.py:1155` (`search`). Add the two params at the end of the signature. Implement logic at the top of the method body, before the existing ANN call.

**Done test:** `tests/test_search_temporal.py` (NEW) — uses the eval corpus from atom 4:
1. `search("commits about Slack", enable_temporal=False)` returns N results across all dates
2. `search("commits about Slack last month", enable_temporal=True)` returns ≤N results, all with `event_time` in the last month
3. Caller-supplied `metadata_filters={"source_type": {"==": "git_commit"}}` is preserved alongside parsed temporal filter
4. Stripped query is used for embedding (verify via a wrapped embedding fn that records the input)

**Commit:** `feat(search): opt-in temporal query pre-processing`

---

### Atom 7 — ReActAgent metadata_filters passthrough

**What:** `ReActAgent.search` accepts `metadata_filters` and `enable_temporal` and forwards to `LeannSearcher.search`. ReAct system prompt mentions the temporal capability in one sentence.

**Where:** `packages/leann-core/src/leann/react_agent.py:176`. Two-arg addition. Update prompt template.

**Done test:** `tests/test_react_temporal.py` (NEW) — mock searcher, assert agent forwards both params correctly.

**Commit:** `feat(react): pass metadata_filters + enable_temporal to searcher`

---

### Atom 8 — SIGNALS schema regression test

**What:** `tests/test_signals_schema.py` — for each source type that has a reader (`document` via `leann build`, `git_commit` via the eval builder, `calendar` via mocked CI_EVENT data), assert the emitted chunks comply with `SIGNALS.md`:
- `event_time` parses as ISO 8601 with UTC offset
- `source_type` is in the documented enum
- `activity_type` is in the documented enum if `author` is set
- `indexed_at` set, UTC, recent
- `participant_ids` is a list and includes `author` when both present

**Done test:** all 3 source-type subtests pass.

**Commit:** `test(signals): schema regression across source types`

---

### Atom 9 — Run eval, measure, commit numbers

**What:** Run `python scripts/eval_temporal.py` (now using `enable_temporal=True` from atom 6). Compare to baseline from atom 4. Write the before/after table into `docs/dev/PROGRESS.md`.

**Acceptance bar (this is the /loop's go/no-go):**
- Recall@5 on temporal_gold.jsonl: **≥ +30 percentage points** vs baseline
- Precision@5: **no regression** (within ±5pp)
- All temporal queries return at least 1 result that falls in `expected_event_time_range`
- No test in `tests/` regresses (`.venv/bin/pytest tests/ -x` green)

If acceptance bar not met:
- If recall short by <10pp: inspect failing gold rows, may need to widen overscan or improve `dateparser` settings (atom 5 tweak)
- If precision regressed: parsed time filter is too aggressive; revisit "last Tuesday" anchoring in atom 5
- If exception: stop, do NOT keep iterating blindly

**Commit:** `docs(dev): wave 1 eval results — recall@5 +Xpp`

---

## /loop driver prompt

When you start the second Claude session, paste this:

```
/loop

You are executing Wave 1 of docs/dev/TEMPORAL-PLAN.md on branch
feat/temporal-substrate. Work atoms 1 through 9 in strict order.

Before starting atom N:
  - read docs/dev/TEMPORAL-PLAN.md and docs/dev/SIGNALS.md
  - read the atom's "Where" file at the cited line numbers
  - state in one sentence what you'll change

After completing atom N:
  - run the atom's done test; if it fails, fix and re-run
  - run .venv/bin/pytest tests/ -x to check for regressions
  - run .venv/bin/ruff check --fix and .venv/bin/ruff format
  - commit with the documented commit message shape
  - append a one-line PROGRESS.md entry: "Atom N done — <result>"
  - move to atom N+1

Stop and ask the user when:
  - atom 5 NL parser hits ambiguous anchoring (e.g., "last Tuesday"
    on a Tuesday — does it mean today or 7 days ago?)
  - atom 9 acceptance bar is missed by more than 10pp
  - any atom's done test fails after 3 fix attempts
  - you discover a needed change OUTSIDE the cited file
  - any test outside the new ones regresses

Do NOT:
  - touch other readers (email, iMessage, browser, etc.)
  - add federated multi-index search
  - add decay scoring
  - add LeannChat conversation history
  - touch the MCP server
  - rename existing public APIs

Use the .venv at /Users/zain/Documents/LEANN/.venv for all python/pytest/ruff.
```

---

## Out of scope — captured for future waves

- **Wave 2:** other reader migrations (email, iMessage, browser, WeChat, Qwen, Gemini) — one PR per reader once Wave 1 schema is locked
- **Wave 3:** MCP `metadata_filters` passthrough + `LeannChat` conversation history (two small independent PRs)
- **Wave 4:** federated multi-index temporal merge (`leann search a,b "q" --merge-by event_time`)
- **Wave 5:** temporal decay scoring (`gemma_temporal` param on `search`)
- **Wave 6:** connector layer + sidecar edge store for non-temporal connections (referential / participant / hierarchical)
- **Separate repo:** `leann-memory` — Dakera-style importance/decay/TTL/consolidation as a wrapper around LeannSearcher, NOT a modification to leann-core
