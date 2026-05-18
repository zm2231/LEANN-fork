# METADATA-AWARE-PLAN — Wave 2

**Status:** active. Branch `feat/metadata-aware-search` off `origin/main` (commit `d99714d`), in worktree `/Users/zain/Documents/LEANN-metadata/`.

**Goal:** make sparse-metadata-filter retrieval correct and explainable, and add document-level result diversity and chunk-context expansion. Unblocks the Rubio/CF-SAI use case where filtered subsets are 1-5% of the corpus and current ANN+post-filter returns false zeros.

**Parent of upstream:** picks up #315 (browser_rag metadata + broader chunking_utils propagation) and #316 (`--metadata-filters` on `leann ask`). Independent of `feat/temporal-substrate` — the two branches merge cleanly into an integration branch when both are done.

**Driver philosophy:** same as Wave 1 (TEMPORAL-PLAN.md). Atoms sized for a single commit, binary done-or-not test, /loop drives in order.

**Out of scope for Wave 2:** route planners, named filter profiles, citation rendering, cross-source event linking, leann-memory. Each is a future wave or app-layer.

---

## Glossary

- **Prefilter** — apply metadata filters to the corpus BEFORE ANN retrieval, by selecting matching passage IDs and vector-scoring just that subset. Inverse of current LEANN behavior (ANN → post-filter), which gives false zeros on sparse families.
- **Filter selectivity** — `filter_matches / total_passages`. Below the prefilter threshold (default 5%) → prefilter path activates; above → ANN+post-filter (current behavior) stays the right tool.
- **Facet counts** — per-field-value occurrence counter over all stored passages. Used for explain output and app-level corpus introspection.
- **Diversify_by** — post-score group key. Top-k results are walked in score order; any whose group-key bucket is full are dropped, the next-best is taken instead.
- **Context window** — sibling chunks from the same `source_document_id` adjacent (by `chunk_seq`) to a hit. Useful for evidence review where a single chunk is missing surrounding context.
- **`PassageManager`** — `packages/leann-core/src/leann/api.py:248`. Indexes passages.jsonl files, applies post-filters today. New methods land here.
- **`LeannSearcher.search`** — `packages/leann-core/src/leann/api.py:1155`. Entry point; gains `prefilter`, `prefilter_threshold`, `explain_filters`, `diversify_by`, `max_per_group`, `context_window` kwargs.

---

## SIGNALS additions (new reserved fields)

These are documented in this wave for use by readers and tested by the regression suite, but no reader migration happens here — Wave 3 (deferred) migrates readers to emit them.

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `source_document_id` | string | recommended | `"sha256-abc123"`, `"hearing_2025_05_20_sfrc_budget"` | Stable identifier for the source document this chunk came from. Multiple chunks from the same PDF / video / hearing transcript share one `source_document_id`. Drives diversify_by + context_window. |
| `chunk_seq` | integer | recommended when `source_document_id` is set | `0`, `1`, `2`, ... | Ordering within `source_document_id`. Use sequential ints from 0; gaps allowed. |
| `event_id` | string | optional | `"hearing_2025_05_20_sfrc_budget"` | Cross-source event identifier — a hearing video, transcript, and press release can share this. Reserved for connector layer (deferred wave); included now so readers can start emitting it. |
| `source_url`, `event_date`, `speaker`, `public_citation_allowed`, `review_required` | per-type | optional | varies | Citation/provenance fields reserved as a group. Readers emit what makes sense for their source. No enforcement. |

Edit `docs/dev/SIGNALS.md` to add these in atom 0.

---

## Test corpus

The CF-SAI test case `/Volumes/4/CF/public-figure-pilots/rubio-2025/.leann/indexes/rubio-raw-sources-bge-metadata` exists on the other Mac and is the *real* validation surface. We can't reach it from this Mac.

For local atoms, build a synthetic sparse corpus:

- Take Wave 1's `eval-slack` index (13,062 slack messages) — already on this Mac at `~/Documents/leann-eval-data/slacrawl.db`
- Subset to two channels with very different volume: high-density (e.g., `C0A6AUU729W` = 3,146 msgs) and low-density (e.g., `C09U06NEHMW` = 532 msgs)
- Build `eval-sparse-corpus` index with `source_type=slack`, `author=user_id`, `parent_ref="channel:<name>"`, `chunk_seq=<index in channel timeline>`, `source_document_id="<channel_id>:<thread_ts or msg_ts>"`
- Sparse filter test queries: `{"parent_ref": {"==": "channel:C09U06NEHMW"}}` (4% of corpus) and `{"author": {"==": "<rare user>"}}` (≤1% of corpus)

`scripts/build_metadata_eval_corpus.py` orchestrates the build. Re-runnable.

---

## Atoms

Each atom: **what** + **where** + **done test** + **commit message shape**.

### Atom 0 — SIGNALS additions (docs only)

**What:** Extend SIGNALS.md with the 5 new reserved fields above. No code change.

**Where:** `docs/dev/SIGNALS.md` — append a "## Document hierarchy (Wave 2 additions)" section with the field table.

**Done test:** `grep -q source_document_id docs/dev/SIGNALS.md && grep -q chunk_seq docs/dev/SIGNALS.md`.

**Commit:** `docs(signals): reserve source_document_id, chunk_seq, event_id, citation fields`

---

### Atom 1 — Facet counts

**What:** `PassageManager.facets(fields: list[str]) -> dict[str, dict[str, int]]`. Single pass over all `passages.jsonl` files, build counters per field, return `{field_name: {value: count, ...}, ...}`. Sort by count descending. Cap each field's distinct-values at 1000 to bound memory (configurable via `max_values_per_field` kwarg).

**Where:**
- `packages/leann-core/src/leann/api.py` — new method on `PassageManager`, around line 320
- `LeannSearcher.facets()` wrapper that delegates to `self.passage_manager.facets()`

**Done test:** `tests/test_facets.py` (NEW) — 5 cases:
1. Single field, returns correct value counts
2. Multiple fields, each independent
3. Missing field returns empty dict (not an error)
4. `max_values_per_field=2` caps the result
5. Mixed types (string + int) handled in same field gracefully

**Commit:** `feat(search): facet counts over passage metadata`

---

### Atom 2 — Score filtered subset

**What:** `PassageManager.score_filtered_subset(query_embedding, metadata_filters, top_k, exclude_ids=None) -> list[SearchResult]`. Identifies passages matching filters via metadata scan, recomputes embeddings for those passages via the existing embedding pipeline, dot-products against query embedding, returns top_k sorted by score (descending). Returns SearchResult with real `score` (not `-inf` like Wave 1's `filter_all_passages`).

**Where:**
- `packages/leann-core/src/leann/api.py` — new method on `PassageManager`
- Delete `PassageManager.filter_all_passages` (Wave 1's unranked version) — refactor its caller in `LeannSearcher.search` to use the new scored path

**Done test:** `tests/test_score_filtered_subset.py` (NEW) — 6 cases:
1. Tiny corpus (10 docs), filter selects 3, returns 3 ranked by similarity
2. Filter selects 0 → returns empty list
3. `top_k=2` with 5 matches returns top 2
4. `exclude_ids={"0", "1"}` skips those
5. Filter is a list operator (`{"in": [...]}`)
6. Score is a real float in `[-1, 1]` range, not `-inf`

**Commit:** `feat(search): brute-force scored prefilter for sparse subsets`

---

### Atom 3 — Wire prefilter into LeannSearcher.search

**What:** Add `prefilter: Literal["auto","always","never"]="auto"` and `prefilter_threshold: float=0.05` kwargs to `LeannSearcher.search`. Logic:

```
if metadata_filters and prefilter != "never":
    selectivity = passage_manager.estimate_selectivity(metadata_filters)
    if prefilter == "always" or (prefilter == "auto" and selectivity < prefilter_threshold):
        return passage_manager.score_filtered_subset(
            query_embedding, metadata_filters, top_k
        )
# else: fall through to current ANN + post-filter path
```

Default `prefilter="auto"` is a **behavioral change** but a strict improvement — no callers see false zeros on sparse filters. Document in docstring.

**Where:**
- `packages/leann-core/src/leann/api.py:1155` — `LeannSearcher.search` signature
- Add `PassageManager.estimate_selectivity(metadata_filters) -> float` (count matches / total, fast scan)

**Done test:** `tests/test_prefilter_integration.py` (NEW) — 5 cases:
1. Sparse filter (3/100 corpus) with `prefilter="auto"` returns 3 ranked results (was 0 with `prefilter="never"`)
2. Dense filter (60/100) with `prefilter="auto"` falls back to ANN path (verified via mock)
3. `prefilter="always"` forces prefilter even on dense filters
4. `prefilter="never"` matches current Wave 1 behavior exactly
5. No `metadata_filters` → prefilter parameter ignored, ANN path used

**Commit:** `feat(search): prefilter mode with selectivity-aware auto-routing`

---

### Atom 4 — Explain filters diagnostic

**What:** Add `explain_filters: bool=False` kwarg to `LeannSearcher.search`. When True, return `(results, diagnostics)` tuple instead of `results`. Diagnostics dict:

```python
{
  "total_passages": int,
  "filter_matches": int,           # passages matching metadata_filters
  "filter_selectivity": float,     # filter_matches / total_passages
  "prefilter_mode_used": Literal["bruteforce_filtered_subset", "ann_postfilter", "no_filter"],
  "ann_candidates_requested": int, # top_k or top_k * overscan
  "ann_candidates_returned": int,
  "postfilter_survivors": int,
  "results_returned": int,
}
```

Backward compat: when `explain_filters=False` (default), return type is `list[SearchResult]` as today.

**Where:** Same `LeannSearcher.search` in api.py.

**Done test:** `tests/test_explain_filters.py` (NEW) — 4 cases:
1. Sparse filter + `prefilter="auto"` returns diagnostics with `prefilter_mode_used="bruteforce_filtered_subset"`
2. Dense filter returns `"ann_postfilter"`
3. No filter returns `"no_filter"`
4. Backward compat: `explain_filters=False` returns `list[SearchResult]` directly

**Commit:** `feat(search): explain_filters diagnostic mode`

---

### Atom 5 — Diversify by group + max_per_group

**What:** Add `diversify_by: Union[str, list[str], None]=None` and `max_per_group: int=2` kwargs. After scoring (ANN or prefilter path), walk results in descending score order; group-key = tuple of values for fields in `diversify_by`. Drop any result whose group bucket has hit `max_per_group`. Keep going until `top_k` survivors collected or candidate pool exhausted.

**Where:** Same `LeannSearcher.search` — add diversification as a post-scoring step before returning results.

**Done test:** `tests/test_diversify.py` (NEW) — 4 cases:
1. 10 results from 2 docs, `diversify_by="source_document_id", max_per_group=2, top_k=4` → 2 from each doc
2. `diversify_by=["source_document_id", "author"]` (list) → group key is composite tuple
3. `diversify_by=None` → no diversification, results unchanged (regression)
4. Pool exhausted before top_k reached → returns fewer than top_k results (no infinite loop)

**Commit:** `feat(search): diversify_by post-score group cap`

---

### Atom 6 — Context window expansion

**What:** Add `context_window: int=0` kwarg. When > 0, after retrieval, for each hit, fetch up to `context_window` chunks before AND `context_window` chunks after by `(source_document_id, chunk_seq)` adjacency. Returned as a sibling list on the SearchResult, NOT as separate results. Also expose `LeannSearcher.expand_context(hit, before=1, after=1) -> SearchResult` for app-level explicit calls.

Requires the new SIGNALS fields. If a hit lacks `source_document_id` or `chunk_seq`, expansion is a no-op for that hit (empty sibling list).

**Where:**
- `SearchResult` dataclass — add optional `siblings: list[SearchResult] | None = None` field
- `PassageManager.fetch_siblings(source_document_id, chunk_seq, before, after) -> list[SearchResult]`
- `LeannSearcher.search` applies if `context_window > 0`
- `LeannSearcher.expand_context` post-call helper

**Done test:** `tests/test_context_window.py` (NEW) — 5 cases:
1. `context_window=1` returns each hit with 1 prev + 1 next sibling
2. First chunk in doc returns 0 prev + 1 next
3. Hit without `source_document_id` returns `siblings=None` (or `[]`, decision time)
4. Two hits from same doc don't deduplicate (each gets its own context; consumer's job to dedup if desired)
5. `expand_context(hit, before=2, after=0)` post-call returns 2 prev, 0 next

**Commit:** `feat(search): chunk context expansion via siblings`

---

### Atom 7 — Eval harness + sparse gold set

**What:**
- `scripts/build_metadata_eval_corpus.py` — builds `eval-sparse-corpus` from slacrawl.db subset (described in Test corpus section)
- `tests/eval/metadata_gold.jsonl` — 15+ queries spanning sparse-filter, diversify, context cases. Schema:
  ```json
  {
    "query": "...",
    "metadata_filters": {...},
    "prefilter": "auto",
    "diversify_by": "source_document_id" | null,
    "max_per_group": 2,
    "context_window": 0,
    "expected_min_results": int,
    "expected_filter_match": str,    # all results must satisfy this
    "expected_max_per_group": int,   # no group exceeds this in results
  }
  ```
- `scripts/eval_metadata.py` — runs gold set against `eval-sparse-corpus` and the real `eval-slack` index. Two modes:
  - `--baseline`: `prefilter="never"`, no diversify, no context (Wave 1 behavior)
  - `--treatment`: `prefilter="auto"`, diversify_by + max_per_group per gold row, context_window per gold row

Acceptance bar:
- **False-zero rate** (queries returning 0 results when expected ≥1): **< 5%** (was ~30% with sparse filters in baseline)
- **Group-cap compliance** (no result group exceeds `max_per_group`): **100%**
- **Filter-correctness** (every returned result satisfies `metadata_filters`): **100%**

**Where:** `scripts/`, `tests/eval/`.

**Done test:** `python scripts/build_metadata_eval_corpus.py && python scripts/eval_metadata.py --baseline && python scripts/eval_metadata.py --treatment`. Both modes produce a results table. Treatment hits the acceptance bar.

**Commit:** `test(eval): sparse-prefilter + diversify gold set and runner`

---

### Atom 8 — Wave 2 close

**What:** Final regression with the same exclusion pattern as Wave 1. PROGRESS.md entry summarizing eval table + commits + branch state.

**Done test:** Final regression passes. PROGRESS.md updated.

**Commit:** `docs(dev): wave 2 closed — metadata-aware search`

---

## /loop driver prompt

Paste in a fresh Claude Code session opened at `/Users/zain/Documents/LEANN-metadata`:

```
/loop

Goal: execute Wave 2 of docs/dev/METADATA-AWARE-PLAN.md on branch
feat/metadata-aware-search. Atoms 0 through 8 in strict order.

Required reading before atom 1:
- docs/dev/METADATA-AWARE-PLAN.md (atoms, done-tests, commit shapes)
- docs/dev/SIGNALS.md (canonical metadata schema; Wave 2 adds to it in atom 0)
- /Users/zain/Documents/LEANN/docs/dev/PROGRESS.md (Wave 1 context — same metadata filter primitives we extend)

Setup invariants — halt if any fail:
- cwd = /Users/zain/Documents/LEANN-metadata
- git branch = feat/metadata-aware-search
- .venv/bin/python -c "import leann, dateparser" succeeds
- pkill -f hnsw_embedding_server 2>/dev/null || true
- ls /Users/zain/Documents/leann-eval-data/slacrawl.db

Embeddings:
- Unit tests: sentence-transformers/all-MiniLM-L6-v2, hermetic.
- Eval corpus (atom 7): BAAI/bge-m3 via iq at http://localhost:8100/v1.
  If iq is down, halt and ask. Same pattern as Wave 1.

LLM policy: no atom in Wave 2 needs LLM chat. Halt if you think you do.

Per-atom loop:
1. Read the atom + cited Where files.
2. State in one sentence what you'll change.
3. Edit + write test file.
4. .venv/bin/pytest <new test> -v
5. .venv/bin/pytest tests/test_facets.py tests/test_score_filtered_subset.py
   tests/test_prefilter_integration.py tests/test_explain_filters.py
   tests/test_diversify.py tests/test_context_window.py -x
   (Only include test files that exist; add as atoms create them.)
6. .venv/bin/ruff check --fix packages/ tests/ scripts/ &&
   .venv/bin/ruff format packages/ tests/ scripts/
7. git add <files>; git commit -m "<documented msg>"
8. Append to docs/dev/PROGRESS.md: "Atom N: <sha> — <one-line result>"
9. Next atom.

Stop and ask the user when:
- An atom's done-test fails 3 times
- The edit needs to touch a file NOT in the atom's Where section
- Existing tests outside the Wave 2 set regress
- Atom 7 acceptance bar (false-zero rate < 5%, group-cap 100%,
  filter-correctness 100%) is missed by more than 5pp
- iq becomes unreachable mid-eval

Hard NO list:
- Touching readers (Wave 3 work)
- Touching MCP server
- Adding route planners or named filter profiles
- Renaming any existing public API
- Touching anything in packages/leann-backend-*
- pip installing new deps without asking
- Re-implementing #315's chunking_utils metadata propagation (upstream
  already did this; build on it)

Wave done criteria:
- 9 commits with conventional prefixes (feat/test/docs)
- Final regression green (Wave 1 exclusion pattern)
- docs/dev/PROGRESS.md has 9 atom entries + final eval table
- Branch ahead of origin/main by 9 commits

Tools: .venv/bin/{python,pytest,ruff}.
```

---

## Out of scope — for the wave after this

- **Reader migrations** — Slack reader, calendar reader, code reader emitting canonical SIGNALS metadata including the Wave 2 additions (`source_document_id`, `chunk_seq`, citation fields). Browser reader already partially done by upstream #315.
- **Route planner / multi_search primitive** — parallel-query batching (`searcher.multi_search([...])`). Useful for CF/SAI balanced extraction but app-layer composable on top of Wave 2 primitives.
- **Cross-source event linking** — `searcher.find_related(hit, relation="same_event")`. Connector-layer wave.
- **Temporal extensions** — temporal+metadata interaction (e.g., `prefilter` activated alongside `enable_temporal`). Integration happens when both branches merge.
- **HNSW masked search** — push filter mask into C++ ANN graph traversal. Order-of-magnitude faster than brute-force on medium-sparsity (5-30%); needs backend changes. Defer.
- **`leann-memory`** — separate repo, importance/decay/TTL/consolidation wrapping LeannSearcher.
